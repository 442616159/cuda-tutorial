// ============================================================================
//  code/lessons/ch06_advanced/vmm/vmm_alloc.cu
//
//  虚拟内存管理 API（VMM）：cuMemCreate / cuMemAddressReserve / cuMemMap
//
//  // 需要 Compute Capability >= 6.0 且驱动/设备支持虚拟地址管理
//
//  编译: nvcc -O3 -arch=sm_70 vmm_alloc.cu -o vmm_alloc -lcuda
//        （Windows 上写 -lcuda 等价于链接 cuda.lib）
//  运行: ./vmm_alloc
//
//  为什么要这么麻烦？
//    cudaMalloc 是"物理显存和虚拟地址一次性绑死"的简单接口。
//    但在下面这些场景里它不够用：
//      * 想先把一大段虚拟地址预留出来，之后再一块一块把物理内存映射进去
//        （可增长的数据结构，比如动态扩容的显存池）；
//      * 想让**两段物理上不连续**的显存，在虚拟地址上看起来是连续的
//        （外部库给你两块碎片，但你的 kernel 只接受一个连续指针）；
//      * 想把同一块物理内存映射到多个不同的虚拟地址（别名，节省拷贝）；
//      * 想和别的进程/设备共享显存（配合 cuMemExportToShareableHandle）。
//    这就是 VMM API 存在的理由：把"虚拟地址"和"物理分配"解耦。
//
//  典型四步：
//    1) cuMemAddressReserve  预留一段空闲的虚拟地址（此时还没有物理内存）
//    2) cuMemCreate          申请一块物理分配，拿到一个 handle
//    3) cuMemMap             把 handle 映射到预留地址的某个偏移上
//    4) cuMemSetAccess       指定哪些设备可以访问这段映射
//  释放顺序反过来：cuMemUnmap -> cuMemRelease -> cuMemAddressFree
// ============================================================================
#include "../../../common/cuda_check.h"

#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

// 驱动 API 的错误检查宏：多一步"把 CUresult 转成字符串"
#define CU_CHECK(expr)                                                          \
    do {                                                                        \
        CUresult _r = (expr);                                                   \
        if (_r != CUDA_SUCCESS) {                                               \
            const char* _s = nullptr;                                           \
            cuGetErrorString(_r, &_s);                                          \
            fprintf(stderr, "\n[DRIVER ERROR] %s:%d\n  call : %s\n  msg  : %s\n", \
                    __FILE__, __LINE__, #expr, _s ? _s : "(unknown)");          \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// 把两块物理显存分别填上 1.0f 和 2.0f；
// 因为它们在虚拟地址上是连续的，所以可以用**一个**指针线性访问
__global__ void fillTwoHalves(float* p, size_t halfFloats) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= halfFloats * 2) return;
    p[i] = (i < halfFloats) ? 1.0f : 2.0f;
}

int main() {
    printDeviceInfo();

    // ---- 0. 初始化驱动 API 并确保有一个当前上下文 ----
    CU_CHECK(cuInit(0));
    // 用运行时 API 触发主上下文创建；之后驱动 API 的调用都作用在这个上下文上。
    // 混用 runtime 和 driver API 时，这是最省事的做法。
    CUDA_CHECK(cudaFree(nullptr));

    CUdevice dev = 0;
    CU_CHECK(cuDeviceGet(&dev, 0));

    int vmmSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(&vmmSupported,
                                  CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                                  dev));
    if (!vmmSupported) {
        printf("\n[跳过] 本设备不支持虚拟地址管理（需要 CC >= 6.0 且驱动 >= 440）。\n");
        return 0;
    }

    // ---- 1. 查询最小分配粒度（通常是 2 MB）----
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;      // 显存必须用 PINNED 类型
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    size_t granularity = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    printf("\n[粒度] 本设备的最小 VMM 分配粒度 = %.2f MB\n",
           granularity / 1024.0 / 1024.0);

    const size_t segBytes = granularity;       // 每块物理分配 1 个粒度
    const size_t totalBytes = segBytes * 2;    // 虚拟地址上要连续的两块
    const size_t halfFloats = segBytes / sizeof(float);

    // ---- 2. 预留虚拟地址（此时还没有物理内存）----
    CUdeviceptr va = 0;
    CU_CHECK(cuMemAddressReserve(&va, totalBytes, 0 /*对齐要求*/, 0 /*建议地址*/, 0));
    printf("[地址] 预留虚拟地址 0x%llx，长度 %.2f MB\n",
           (unsigned long long)va, totalBytes / 1024.0 / 1024.0);

    // ---- 3. 创建两块独立的物理分配，映射到虚拟地址的两个偏移上 ----
    CUmemGenericAllocationHandle handles[2] = {};
    for (int i = 0; i < 2; ++i) {
        CU_CHECK(cuMemCreate(&handles[i], segBytes, &prop, 0));
        CU_CHECK(cuMemMap(va + i * segBytes, segBytes, 0, handles[i], 0));
    }
    printf("[映射] 已把 2 块独立的物理分配映射到 0x%llx 和 0x%llx\n",
           (unsigned long long)va, (unsigned long long)(va + segBytes));

    // ---- 4. 授予访问权限（不设置的话 kernel 一碰就崩）----
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(va, totalBytes, &access, 1));

    // ---- 5. 像用普通指针一样用它 ----
    float* d_ptr = reinterpret_cast<float*>(va);
    const size_t totalFloats = totalBytes / sizeof(float);
    const int grid = (int)((totalFloats + 255) / 256);
    fillTwoHalves<<<grid, 256>>>(d_ptr, halfFloats);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(totalFloats, 0.f);
    CU_CHECK(cuMemcpyDtoH(host.data(), va, totalBytes));

    // 校验 1：前半段全是 1.0，后半段全是 2.0
    int bad = 0;
    for (size_t i = 0; i < totalFloats; ++i) {
        const float expect = (i < halfFloats) ? 1.0f : 2.0f;
        if (host[i] != expect) ++bad;
    }
    // 校验 2：跨越两块物理分配的那个边界必须是"1.0 -> 2.0"的突变
    const bool boundaryOK = (host[halfFloats - 1] == 1.0f) && (host[halfFloats] == 2.0f);

    printf("\n[结果] 元素总数 = %zu，不匹配 = %d  %s\n",
           totalFloats, bad, bad == 0 ? "[OK]" : "[FAIL]");
    printf("[结果] 跨物理边界的取值 (%.1f -> %.1f) %s\n",
           host[halfFloats - 1], host[halfFloats],
           boundaryOK ? "[OK] 说明两块不连续的物理内存在虚拟地址上确实拼成了连续区间"
                      : "[FAIL]");

    // ---- 6. 释放：顺序必须和创建时相反 ----
    CU_CHECK(cuMemUnmap(va, segBytes));            // 先解映射
    CU_CHECK(cuMemUnmap(va + segBytes, segBytes));
    CU_CHECK(cuMemRelease(handles[0]));            // 再释放物理分配
    CU_CHECK(cuMemRelease(handles[1]));
    CU_CHECK(cuMemAddressFree(va, totalBytes));    // 最后归还虚拟地址
    printf("[清理] 已按 cuMemUnmap -> cuMemRelease -> cuMemAddressFree 的顺序释放\n");

    printf("\nVMM 的其他常见用法：\n");
    printf("  * 显存池：一次预留 40 GB 虚拟地址，按需 cuMemMap 小块进去，\n");
    printf("    池管理器自己负责分配，不再受 cudaMalloc 的碎片化影响；\n");
    printf("  * 别名映射：把同一块物理 handle cuMemMap 到两个虚拟地址，\n");
    printf("    实现零拷贝的“同一份数据两个视图”；\n");
    printf("  * 跨进程共享：cuMemExportToShareableHandle + cuMemImportFromShareableHandle；\n");
    printf("  * 与 NCCL / CUDA IPC 配合做大模型的多卡张量切分。\n");
    printf("\n注意：这一段 API 完全绕过运行时 API 的分配器和缓存，\n");
    printf("      一旦忘记 cuMemRelease 就会一直占着显存直到进程退出。\n");

    return 0;
}
