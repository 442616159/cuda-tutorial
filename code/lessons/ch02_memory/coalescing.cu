// ============================================================
//  code/lessons/ch02_memory/coalescing.cu
//  合并访存（Coalesced Access）的量化实验。
//
//  这些 kernel 都在做同一件事：把一个大数组读一遍、累加起来。
//  唯一的区别是"哪个线程读哪个地址"，而耗时会差好几倍。
//
//      A coalescedKernel ：相邻线程读相邻地址   —— 合并
//      B stridedKernel   ：相邻线程跨 32 个元素 —— 不合并
//      C reversedKernel  ：顺序颠倒但区间连续   —— 仍然合并
//      D aosKernel       ：结构体数组 AoS，只读其中 1/3 的字节
//      E soaKernel       ：拆成独立数组 SoA
//
//  编译（在 code/lessons/ch02_memory 目录下）：
//      nvcc -O3 -arch=sm_75 coalescing.cu -o coalescing
//  运行：
//      ./coalescing        (Linux)
//      coalescing.exe      (Windows)
// ============================================================

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../../common/cuda_check.h"
#include "../../common/cuda_timer.h"

constexpr int BLOCK = 256;

// ------------------------------------------------------------
// A. 合并访存（正确示范）
//
// 线程 0 读 a[0]，线程 1 读 a[1]，...，线程 31 读 a[31]。
// 一个 warp 需要的 32 个 float 恰好是 128 字节且完全连续，
// 硬件用一个（或两个）128 字节的内存事务就能全部取回。
//
// grid-stride 循环保证：无论数组多大，每个元素都恰好被读一次。
// ------------------------------------------------------------
__global__ void coalescedKernel(const float* __restrict__ a,
                                float* __restrict__ out,
                                int n) {
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        sum += a[i];
    }
    // 只让每个块的第 0 个线程写一个值，避免"写"的开销混进对比
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// ------------------------------------------------------------
// B. 跨步访存（错误示范）
//
// 全局编号 g 的线程读 a[g * 32]。
// 那么同一个 warp 里，线程 0 读 a[0]、线程 1 读 a[32]、
// 线程 2 读 a[64] ...  地址之间相差 32 个 float = 128 字节。
//
// 结果：一个 warp 的 32 次访问落在 32 个不同的 128 字节段里，
// 本来 1 次事务能搞定的活变成了 32 次。
//
// 注意：读到的"有用字节数"和 A 完全一样（每个元素还是读一次），
// 但事务利用率从 100% 掉到约 1/32，所以有效带宽会掉一个数量级。
// ------------------------------------------------------------
__global__ void stridedKernel(const float* __restrict__ a,
                              float* __restrict__ out,
                              int n) {
    float sum = 0.0f;
    const int S = 32;   // 故意取 32：正好让一个 warp 完全散开
    int stride = 1;
    if (blockDim.x >= 1) stride = blockDim.x * gridDim.x * S;
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * S;
         i < n;
         i += stride) {
        sum += a[i];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// ------------------------------------------------------------
// C. 反向合并访存
//
// 线程 t 读 a[n - 1 - t]。看起来"倒着读"，但对一个 warp 来说，
// 这 32 个地址仍然落在一个连续的 128 字节区间里，
// 硬件照样能用一次事务取完。
//
// 结论：合并访存关心的是"一个 warp 覆盖的地址范围是否连续"，
//      而不是"编号小的线程读的地址是否也小"。
// ------------------------------------------------------------
__global__ void reversedKernel(const float* __restrict__ a,
                               float* __restrict__ out,
                               int n) {
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // n 是 32 的倍数时，n-1-i 仍然覆盖 [0, n) 的每个下标恰好一次；
        // 但为了任何 n 都安全，还是加一个上界判断。
        int j = n - 1 - i;
        if (j >= 0) sum += a[j];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// ------------------------------------------------------------
// D. AoS（Array of Structs）的浪费
//
// struct { float x, y, z; } 每个元素 12 字节。
// 线程 t 读 p[t].x，相邻线程的地址差是 12 字节。
// 一个 warp 覆盖 32*12 = 384 字节 = 3 个 128 字节段，
// 但实际只用了其中 1/3 的字节（只取 x）。
//
// 于是"有用带宽"掉到约 1/3 —— 注意这不是"没合并"，
// 而是合并了但每个事务里有 2/3 的字节被浪费掉了。
// ------------------------------------------------------------
struct ParticleAoS {
    float x, y, z;
};

__global__ void aosKernel(const ParticleAoS* __restrict__ p,
                          float* __restrict__ out,
                          int n) {
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        sum += p[i].x;   // 只用了 12 字节里的 4 字节
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// ------------------------------------------------------------
// E. SoA（Struct of Arrays）：把 x / y / z 拆成三个独立数组
//
// 只要访问模式是"只用 x"，SoA 就是完美合并：
// 一个 warp 读 32 个连续的 float = 128 字节，零浪费。
//
// 代价是：如果你经常需要同时访问 x/y/z，SoA 会变成 3 次独立访问；
// 所以到底用 AoS 还是 SoA，取决于你的访问模式。
// ------------------------------------------------------------
__global__ void soaKernel(const float* __restrict__ px,
                          float* __restrict__ out,
                          int n) {
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        sum += px[i];    // 连续的 float 数组，完美合并
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

static double benchFloat1(const char* label,
                          void (*k)(const float*, float*, int),
                          const float* dIn, float* dOut,
                          int n, int grid, int block,
                          double touchedBytes) {
    double ms = CudaTimer::measure([&](double& m, int) {
        CudaTimer t;
        t.start();
        k<<<grid, block>>>(dIn, dOut, n);
        CUDA_CHECK_KERNEL();
        m = t.stopMs();
    }, 10, 2);

    printf("  %-32s 耗时 %8.4f ms   吞吐 %8.1f GB/s\n",
           label, ms, CudaTimer::bandwidthGBs(touchedBytes, ms));
    return ms;
}

static double benchAoS(const char* label,
                       const ParticleAoS* dIn, float* dOut,
                       int n, int grid, int block,
                       double touchedBytes) {
    double ms = CudaTimer::measure([&](double& m, int) {
        CudaTimer t;
        t.start();
        aosKernel<<<grid, block>>>(dIn, dOut, n);
        CUDA_CHECK_KERNEL();
        m = t.stopMs();
    }, 10, 2);

    printf("  %-32s 耗时 %8.4f ms   吞吐 %8.1f GB/s\n",
           label, ms, CudaTimer::bandwidthGBs(touchedBytes, ms));
    return ms;
}

int main() {
    printDeviceInfo();

    // N = 1<<24 = 16777216（约 1677 万）个 float = 64 MB。
    // 这个规模足够让带宽成为唯一瓶颈。
    const int N = 1 << 24;
    const size_t bytes = (size_t)N * sizeof(float);
    const size_t pcount = (size_t)N / 3;   // AoS 的粒子数（保证 AoS 总字节数小于 64 MB）

    std::vector<float> hIn(N);
    for (int i = 0; i < N; ++i) hIn[i] = (float)(i % 100) * 0.01f;

    float* dIn = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, bytes));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    // ---- AoS 数据 ----
    std::vector<ParticleAoS> hAoS(pcount);
    std::vector<float> hSoA(pcount);
    for (size_t i = 0; i < pcount; ++i) {
        hAoS[i].x = (float)(i % 100) * 0.01f;
        hAoS[i].y = 1.0f;
        hAoS[i].z = 2.0f;
        hSoA[i] = hAoS[i].x;     // 单独抽出 x 分量
    }
    ParticleAoS* dAoS = nullptr;
    float* dSoA = nullptr;
    CUDA_CHECK(cudaMalloc(&dAoS, pcount * sizeof(ParticleAoS)));
    CUDA_CHECK(cudaMalloc(&dSoA, pcount * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dAoS, hAoS.data(), pcount * sizeof(ParticleAoS),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSoA, hSoA.data(), pcount * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int block = BLOCK;
    const int grid = 1024;   // 固定网格，靠 grid-stride 循环覆盖全部数据
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, (grid + 1) * sizeof(float)));

    printf("\n数组大小: %d 个 float = %.1f MB\n", N, bytes / 1024.0 / 1024.0);
    printf("网格: %d 个块 x %d 线程 = %d 个线程，每个线程循环约 %d 次\n",
           grid, block, grid * block, N / (grid * block));
    printf("--------------------------------------------------------------------------------\n");

    benchFloat1("A. 合并访存 (coalesced)", coalescedKernel,
                dIn, dOut, N, grid, block, (double)bytes);
    benchFloat1("B. 跨步访存 (strided x32)", stridedKernel,
                dIn, dOut, N, grid, block, (double)bytes);
    benchFloat1("C. 反向合并访存 (reversed)", reversedKernel,
                dIn, dOut, N, grid, block, (double)bytes);

    printf("--------------------------------------------------------------------------------\n");

    // D / E：AoS vs SoA。两个 kernel 的"有用字节数"一样（都是 pcount * 4），
    // 但 AoS 实际被拉进缓存的字节数是 pcount * 12。
    {
        double touched = (double)pcount * sizeof(ParticleAoS);
        double useful = (double)pcount * 4.0;

        double msAoS = benchAoS("D. AoS 只读 x 分量", dAoS, dOut,
                                (int)pcount, grid, block, touched);
        double msSoA = benchFloat1("E. SoA 只读 x 分量", soaKernel,
                                   dSoA, dOut, (int)pcount, grid, block, useful);

        printf("\n  换算成【同样算力下的等效带宽】（都按 pcount*4 有用字节算）：\n");
        printf("    AoS: %8.1f GB/s\n", CudaTimer::bandwidthGBs(useful, msAoS));
        printf("    SoA: %8.1f GB/s\n", CudaTimer::bandwidthGBs(useful, msSoA));
        printf("    倍率: %.2f x\n", msAoS / msSoA);
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("\n怎么解读：\n");
    printf("  1. A 是基准，其有效带宽应接近显卡理论带宽的 60%%~85%%（受限于 ECC、刷新等）。\n");
    printf("  2. B 的吞吐应该比 A 低一个数量级。注意 B 读的总元素数和 A 完全一样，\n");
    printf("     慢纯粹是因为一个 warp 的访问落在 32 个不同的事务里。\n");
    printf("  3. C 的吞吐应该和 A 差不多，因为一个 warp 覆盖的地址区间仍然连续。\n");
    printf("  4. D/E 的差别来自【每个事务里有多少字节是有用的】：\n");
    printf("     AoS 每个 12 字节只用 4 字节，SoA 每个 4 字节用满。\n");
    printf("  5. 以上都是定性规律。在你的机器上实测约为多少可能不同，以自测为准。\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dAoS));
    CUDA_CHECK(cudaFree(dSoA));
    CUDA_CHECK(cudaFree(dOut));
    printf("\n全部完成。\n");
    return 0;
}
