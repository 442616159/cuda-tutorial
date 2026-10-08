// ============================================================
//  code/lessons/ch04_timing/bandwidth/bandwidth_test.cu
//  第 4 章示例 1：把「有效带宽」真正测出来
//
//  这个程序做四件事：
//    1. 从设备属性里算出理论峰值带宽
//    2. 分别测「纯拷贝 / 缩放 / 向量加 / 只读」的有效带宽
//    3. 用「读字节 + 写字节」而不是「数组大小」来算带宽
//    4. 把实测值换算成「占理论峰值的百分比」
//
//  有效带宽公式：
//      有效带宽(GB/s) = 实际搬运字节数 / 耗时(s) / 1e9
//  这里的关键是「实际搬运字节数」：
//      copy    : 读 N*4 + 写 N*4 = 2*N*4
//      scale   : 读 N*4 + 写 N*4 = 2*N*4
//      add     : 读 2*N*4 + 写 N*4 = 3*N*4
//      只读求和 : 读 N*4
//  初学者最常犯的错就是只按「数组大小」算，结果带宽虚高一大截。
//
//  编译： nvcc -O3 -arch=sm_70 bandwidth_test.cu -o bandwidth_test
//  运行： ./bandwidth_test
// ============================================================
#include "../../../common/cuda_check.h"
#include "../../../common/cuda_bench.h"
#include <cstdio>
#include <cmath>

#define BLOCK 256
#define N     (1 << 23)          // 8M 个 float = 32 MB

// ------------------------------------------------------------
// 四种典型的访存模式
// ------------------------------------------------------------
__global__ void copyKernel(const float* __restrict__ in,
                           float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];                  // 读 1 写 1
}

__global__ void scaleKernel(const float* __restrict__ in,
                            float* __restrict__ out, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * in[i];              // 读 1 写 1，多了一次乘法
}

__global__ void addKernel(const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];              // 读 2 写 1
}

// 只读求和：用网格步长循环 + warp 归约，防止编译器把"读了没用"的代码优化掉
__global__ void readOnlySum(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        v += in[i];
    }
    // warp 内先归约，再让每个 warp 的 0 号 lane 做一次全局原子加
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    if ((threadIdx.x & 31) == 0) atomicAdd(out, v);
}

// ------------------------------------------------------------
// 理论峰值带宽的估算公式（与 CUDA samples 里的 bandwidthTest 一致）
//   memoryClockRate 单位 kHz，memoryBusWidth 单位 bit
//   DDR/GDDR 每个时钟周期传输 2 次，所以要乘 2
//   结果是 GB/s（10 进制）
//
//   这里用 cudaDeviceGetAttribute 而不是 cudaDeviceProp 的对应字段：
//   后者的时钟相关成员在不同 CUDA 版本里被移除过，属性接口更稳定。
// ------------------------------------------------------------
static double theoreticalPeakGBs(double memClockKHz, int busWidth) {
    return 2.0 * memClockKHz * ((double)busWidth / 8.0) / 1.0e6;
}

// ------------------------------------------------------------
// 打印一行结果
// ------------------------------------------------------------
static void report(const char* name, double bytesMoved, double ms, double peak) {
    double bw = gbps(bytesMoved, ms);
    printf("  %-22s | %10.4f | %9.1f | %6.1f%%\n",
           name, ms, bw, bw / peak * 100.0);
}

int main() {
    printf("=== 有效带宽测试 (N = %d 个 float = %.1f MB/数组) ===\n",
           N, (double)N * 4 / 1024.0 / 1024.0);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int memClockKHz = 0, busWidth = 0;
    // CUDA 13 起 cudaDevAttrMemoryBusWidth 改名为 cudaDevAttrGlobalMemoryBusWidth
#if CUDART_VERSION >= 13000
    const cudaDeviceAttr kBusWidthAttr = cudaDevAttrGlobalMemoryBusWidth;
#else
    const cudaDeviceAttr kBusWidthAttr = cudaDevAttrMemoryBusWidth;
#endif
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&busWidth,    kBusWidthAttr, dev));
    double peak = theoreticalPeakGBs(memClockKHz, busWidth);

    printf("\n设备：%s\n", prop.name);
    printf("  显存时钟      : %.0f MHz\n", memClockKHz / 1000.0);
    printf("  显存位宽      : %d bit\n", busWidth);
    printf("  理论峰值带宽  : %.1f GB/s（公式估算，实际以规格书为准）\n", peak);
    printf("  SM 数量       : %d\n\n", prop.multiProcessorCount);

    // ---------- 分配 ----------
    size_t bytes = (size_t)N * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_a, 0, bytes));       // 内容是零，对带宽没有影响
    CUDA_CHECK(cudaMemset(d_b, 0, bytes));

    int grid = (N + BLOCK - 1) / BLOCK;          // "每线程一个元素"的经典网格
    int gridStride = prop.multiProcessorCount * 32;   // 网格步长循环用的网格

    printf("  %-22s | %10s | %9s | %7s\n", "操作", "耗时(ms)", "带宽(GB/s)", "占峰值");
    printf("  -----------------------+------------+-----------+--------\n");

    // ---------- 1. 纯拷贝：读 N*4，写 N*4 ----------
    double tCopy = medianMs([&] {
        copyKernel<<<grid, BLOCK>>>(d_a, d_c, N);
    }, 5, 30);
    report("copy (读1写1)", 2.0 * bytes, tCopy, peak);

    // ---------- 2. 缩放：同样是读 1 写 1 ----------
    double tScale = medianMs([&] {
        scaleKernel<<<grid, BLOCK>>>(d_a, d_c, 2.0f, N);
    }, 5, 30);
    report("scale (读1写1)", 2.0 * bytes, tScale, peak);

    // ---------- 3. 向量加：读 2 写 1 ----------
    double tAdd = medianMs([&] {
        addKernel<<<grid, BLOCK>>>(d_a, d_b, d_c, N);
    }, 5, 30);
    report("add (读2写1)", 3.0 * bytes, tAdd, peak);

    // ---------- 4. 只读求和：读 N*4 ----------
    double tRead = medianMs([&] {
        readOnlySum<<<gridStride, BLOCK>>>(d_a, d_sum, N);
    }, 5, 30);
    report("readOnly (读1)", 1.0 * bytes, tRead, peak);

    // ---------- 校验 ----------
    float h_sum = -1.f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    printf("\n校验：只读求和结果 = %.1f（数组全为 0，所以应当是 0）\n", h_sum);

    // ---------- 结论提示 ----------
    printf("\n怎么看这张表：\n");
    printf("  1. copy / scale / add 的带宽应该彼此接近 —— 它们都是「流式访存」，\n");
    printf("     瓶颈在显存带宽而不是计算。\n");
    printf("  2. 如果 add 的带宽明显低于 copy，说明读两路数据时合并访存没做好。\n");
    printf("  3. readOnly 能跑出比 copy 更高的数字是正常的：\n");
    printf("     它只有读、没有写，而写往往要占用额外的带宽。\n");
    printf("  4. 实测能到理论峰值的 70%%~90%% 就算相当健康了，\n");
    printf("     达不到 100%% 不是你的代码有问题。\n");
    printf("  5. 以上数字在你的机器上会不一样，重点是「同一台机器上怎么比」。\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_sum));
    return 0;
}
