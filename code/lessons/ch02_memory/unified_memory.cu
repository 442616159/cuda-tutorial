// ============================================================
//  code/lessons/ch02_memory/unified_memory.cu
//  统一内存（Unified Memory）演示：cudaMallocManaged
//
//  统一内存的关键价值：**你写代码时不用再区分"这份数据在 CPU 上
//  还是在 GPU 上"**。一个指针，两边都能用。CUDA 驱动负责在需要时
//  把页面在 CPU 和 GPU 之间迁移。
//
//  四个实验：
//      A. 传统 cudaMalloc + cudaMemcpy（对照组）
//      B. cudaMallocManaged 直接使用（最省事）
//      C. cudaMallocManaged + cudaMemPrefetchAsync 预取（最省事+快）
//      D. 演示 cudaMemAdvise 的用法
//
//  编译（在 code/lessons/ch02_memory 目录下）：
//      nvcc -O3 -arch=sm_75 unified_memory.cu -o unified_memory
//  运行：
//      ./unified_memory        (Linux)
//      unified_memory.exe      (Windows)
// ============================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.h"
#include "../../common/cuda_timer.h"

__global__ void scaleKernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            float k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * k + 1.0f;
    }
}

int main() {
    printDeviceInfo();

    // 检查设备是否支持统一内存。
    // 支持的判据是"设备是否可以和主机共享同一套虚拟地址空间"，
    // CC 6.0+ 的 64 位系统上都支持。
    int pageableAccessOk = 0, concurrentManaged = 0, managed = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&managed,
        cudaDevAttrManagedMemory, 0));
    if (managed) {
        CUDA_CHECK(cudaDeviceGetAttribute(&pageableAccessOk,
            cudaDevAttrPageableMemoryAccess, 0));
        CUDA_CHECK(cudaDeviceGetAttribute(&concurrentManaged,
            cudaDevAttrConcurrentManagedAccess, 0));
    }
    printf("设备是否支持统一内存      : %s\n", managed ? "是" : "否");
    if (!managed) {
        printf("本设备不支持统一内存，程序退出。\n");
        return 0;
    }
    printf("支持直接访问可分页主机内存: %s\n", pageableAccessOk ? "是" : "否");
    printf("支持并发托管访问(CPU/GPU 同时) : %s\n", concurrentManaged ? "是" : "否");

    const int N = 1 << 24;                 // 64 MB
    const size_t bytes = (size_t)N * sizeof(float);
    const int block = 256;
    const int grid = (N + block - 1) / block;

    std::vector<float> hRef(N);
    for (int i = 0; i < N; ++i) hRef[i] = 1.0f * i;

    printf("\n数据规模: %d 个 float = %.1f MB\n", N, bytes / 1024.0 / 1024.0);
    printf("================================================================================\n");

    // ---------------- A. 传统方式 ----------------
    {
        float *hIn = nullptr, *hOut = nullptr;
        // cudaMallocHost 分配的是"页锁定内存"（pinned memory），
        // 它不会被操作系统换出，因此可以直接被 DMA 引擎读写，
        // 拷贝速度比普通 malloc 的内存快得多（有时快 2 倍以上）。
        CUDA_CHECK(cudaMallocHost(&hIn, bytes));
        CUDA_CHECK(cudaMallocHost(&hOut, bytes));
        for (int i = 0; i < N; ++i) hIn[i] = 1.0f * i;

        float *dIn = nullptr, *dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dIn, bytes));
        CUDA_CHECK(cudaMalloc(&dOut, bytes));

        CudaTimer t;
        t.start();
        CUDA_CHECK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));
        scaleKernel<<<grid, block>>>(dIn, dOut, 2.0f, N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
        float ms = t.stopMs();

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (fabs(hOut[i] - (hRef[i] * 2.0f + 1.0f)) > 1e-3f) { ok = false; break; }
        }
        printf("A. cudaMalloc + cudaMemcpy（含两次拷贝）  总耗时 %8.3f ms   %s\n",
               ms, ok ? "正确" : "错误");

        CUDA_CHECK(cudaFree(dIn));
        CUDA_CHECK(cudaFree(dOut));
        CUDA_CHECK(cudaFreeHost(hIn));
        CUDA_CHECK(cudaFreeHost(hOut));
    }

    // ---------------- B. 统一内存，不做预取 ----------------
    {
        float *uIn = nullptr, *uOut = nullptr;
        // cudaMallocManaged 在"统一地址空间"里分配内存：
        // 返回的指针在 CPU 和 GPU 上含义相同，可以直接两处都用。
        CUDA_CHECK(cudaMallocManaged(&uIn, bytes));
        CUDA_CHECK(cudaMallocManaged(&uOut, bytes));
        for (int i = 0; i < N; ++i) uIn[i] = 1.0f * i;

        CudaTimer t;
        t.start();
        // 没有 cudaMemcpy！kernel 直接用 uIn / uOut。
        // 这一次 kernel 访问时会触发"缺页"，驱动把页面从主机内存
        // 迁移到显存 —— 这个过程是逐步发生的，代价不小。
        scaleKernel<<<grid, block>>>(uIn, uOut, 2.0f, N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = t.stopMs();

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (fabs(uOut[i] - (hRef[i] * 2.0f + 1.0f)) > 1e-3f) { ok = false; break; }
        }
        printf("B. cudaMallocManaged（无预取）              总耗时 %8.3f ms   %s\n",
               ms, ok ? "正确" : "错误");

        CUDA_CHECK(cudaFree(uIn));
        CUDA_CHECK(cudaFree(uOut));
    }

    // ---------------- C. 统一内存 + 预取 ----------------
    {
        float *uIn = nullptr, *uOut = nullptr;
        CUDA_CHECK(cudaMallocManaged(&uIn, bytes));
        CUDA_CHECK(cudaMallocManaged(&uOut, bytes));
        for (int i = 0; i < N; ++i) uIn[i] = 1.0f * i;

        // 告诉驱动："这块数据接下来主要在设备 0 上被访问"，
        // 驱动就会提前把整块页面迁移到显存，而不是等缺页时一页一页搬。
        // 这是统一内存"性能不输显式拷贝"的关键。
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
#if CUDART_VERSION >= 13000
        // CUDA 13 起 cudaMemPrefetchAsync 的第 3 个参数由 device id 变成 cudaMemLocation
        cudaMemLocation prefetchLoc{};
        prefetchLoc.type = cudaMemLocationTypeDevice;
        prefetchLoc.id   = dev;
        CUDA_CHECK(cudaMemPrefetchAsync(uIn, bytes, prefetchLoc, 0, nullptr));
#else
        CUDA_CHECK(cudaMemPrefetchAsync(uIn, bytes, dev, nullptr));
#endif

        CudaTimer t;
        t.start();
        scaleKernel<<<grid, block>>>(uIn, uOut, 2.0f, N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = t.stopMs();

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (fabs(uOut[i] - (hRef[i] * 2.0f + 1.0f)) > 1e-3f) { ok = false; break; }
        }
        printf("C. cudaMallocManaged + Prefetch           总耗时 %8.3f ms   %s\n",
               ms, ok ? "正确" : "错误");
        printf("   注意：cudaMemPrefetchAsync 是异步的，正式计时时应把它也放进\n");
        printf("   计时段并用 cudaStreamSynchronize 等待，否则数字会偏乐观。\n");

        CUDA_CHECK(cudaFree(uIn));
        CUDA_CHECK(cudaFree(uOut));
    }

    // ---------------- D. cudaMemAdvise ----------------
    {
        float* uIn = nullptr;
        CUDA_CHECK(cudaMallocManaged(&uIn, bytes));
        for (int i = 0; i < N; ++i) uIn[i] = 1.0f * i;

        // cudaMemAdvise 是"给驱动的建议"，不是强制命令。
        // cudaMemAdviseSetReadMostly：这块内存基本只读，
        //   驱动可以为它建立只读副本，提高命中率。
#if CUDART_VERSION >= 13000
        // CUDA 13 起 cudaMemAdvise 的第 4 个参数由 device id 变成 cudaMemLocation
        cudaMemLocation adviseLoc{};
        adviseLoc.type = cudaMemLocationTypeDevice;
        adviseLoc.id   = 0;
        const cudaMemLocation adviseTarget = adviseLoc;
#else
        const int adviseTarget = 0;
#endif
        CUDA_CHECK(cudaMemAdvise(uIn, bytes, cudaMemAdviseSetReadMostly, adviseTarget));

        // cudaMemAdviseSetPreferredLocation：优先放在某个设备上。
        // cudaMemAdviseSetAccessedBy：允许某个设备直接访问这块内存，
        //   避免每次都建立映射。
        CUDA_CHECK(cudaMemAdvise(uIn, bytes, cudaMemAdviseSetPreferredLocation, adviseTarget));

        printf("D. cudaMemAdvise 调用成功（ReadMostly + PreferredLocation）\n");
        printf("   这类建议对【只读大数组】、【多 GPU 共享】等场景有明显收益，\n");
        printf("   但对单卡、单次访问的小数组几乎没有影响。\n");

        CUDA_CHECK(cudaFree(uIn));
    }

    printf("================================================================================\n");
    printf("\n怎么解读：\n");
    printf("  1. B 通常比 A 慢，因为缺页迁移是一页一页发生的，\n");
    printf("     而且可能伴随 CPU/GPU 之间的往返同步。\n");
    printf("  2. C 通常能追平甚至超过 A —— 预取把【迁移】变成了\n");
    printf("     一次可调度的大块传输，效率高得多。\n");
    printf("  3. 统一内存最大的价值是**可编程性**：代码更短、更容易移植。\n");
    printf("     性能上，只要你用 cudaMemPrefetchAsync，就能做到接近显式拷贝。\n");
    printf("  4. 在你的机器上实测约为多少可能不同，请以自己测到的为准。\n");
    printf("  5. 一个常见陷阱：如果托管内存在 CPU 上被频繁修改，\n");
    printf("     就会反复触发页面迁移（抖动），性能会崩。\n");
    printf("     对策：确定数据归谁用之后，一次性 prefetch 过去。\n");

    printf("\n全部完成。\n");
    return 0;
}
