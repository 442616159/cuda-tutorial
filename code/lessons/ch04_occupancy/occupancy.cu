// ============================================================
//  code/lessons/ch04_occupancy/occupancy.cu
//  第 4 章示例 2：寄存器、共享内存与占用率
//
//  占用率（Occupancy）= 实际驻留在 SM 上的 warp 数 / SM 能容纳的最大 warp 数。
//
//  它为什么重要：GPU 靠「同时挂很多 warp」来隐藏访存延迟。
//  某个 warp 去等显存（几百个周期）时，调度器立刻切到别的 warp。
//  能同时挂的 warp 越多，等待就越容易被填满。
//
//  它为什么不是越高越好：占了更多寄存器的 kernel 往往每条指令干的活更多，
//  强行压低寄存器会导致溢出（spill）到本地内存，反而更慢。
//
//  本程序用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 等 API，
//  把这些数字直接打印出来 —— 不需要 Nsight 也能看。
//
//  编译：
//    nvcc -O3 -arch=sm_70 -Xptxas -v occupancy.cu -o occupancy
//    （-Xptxas -v 会顺便打印每个 kernel 的寄存器数、shared memory 用量、spill 情况，
//      这份输出请和程序自己打印的数字对照着看）
//  运行： ./occupancy
// ============================================================
#include "../../common/cuda_check.h"
#include <cstdio>

#define BLOCK 256
#define N     (1 << 20)

// ------------------------------------------------------------
// kernel A：轻量型，寄存器需求小
// ------------------------------------------------------------
__global__ void lightKernel(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

// ------------------------------------------------------------
// kernel B：重度使用寄存器，故意制造寄存器压力
//   展开 16 个累加器 + 8 个临时量，编译器会需要几十个寄存器。
// ------------------------------------------------------------
__global__ void heavyKernel(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float acc[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = in[i] * (float)(k + 1);
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = acc[k] * 1.0001f + 0.5f;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) s += acc[k];
    out[i] = s;
}

// ------------------------------------------------------------
// kernel C：用 __launch_bounds__ 给编译器下"硬指标"
//   __launch_bounds__(最大线程数, 每 SM 最少要能放下的块数)
//   第二个参数等于告诉编译器："你必须把寄存器压到能同时放下 4 个块"，
//   编译器会据此减少寄存器用量（可能带来轻微 spill，换来更高占用率）。
// ------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK, 4)
boundedKernel(const float* __restrict__ in,
              float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float acc[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = in[i] * (float)(k + 1);
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = acc[k] * 1.0001f + 0.5f;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) s += acc[k];
    out[i] = s;
}

// ------------------------------------------------------------
// kernel D：大量使用共享内存，占用率被 shared memory 卡住
// ------------------------------------------------------------
__global__ void sharedHeavyKernel(const float* __restrict__ in,
                                  float* __restrict__ out, int n) {
    // 每块 48 KB 共享内存 —— 大多数 SM 只有 96~100 KB，
    // 于是每个 SM 最多只能放 2 个这样的块。
    __shared__ float s[12 * 1024];          // 12K * 4B = 48 KB
    int t = threadIdx.x;
    s[t] = in[blockIdx.x * blockDim.x + t];
    __syncthreads();
    float v = 0.f;
    for (int k = t; k < 12 * 1024; k += blockDim.x) v += s[k];
    out[blockIdx.x * blockDim.x + t] = v;
}

int main() {
    printf("=== 占用率与寄存器 / 共享内存演示 ===\n");

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("设备：%s（CC %d.%d）\n", prop.name, prop.major, prop.minor);
    printf("  SM 数量                  : %d\n", prop.multiProcessorCount);
    printf("  每 SM 寄存器总数         : %d\n", prop.regsPerMultiprocessor);
    printf("  每 SM 共享内存           : %.1f KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
    printf("  每块最大共享内存         : %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("  每 SM 最大线程数         : %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  每 SM 最大 warp 数       : %d\n\n",
           prop.maxThreadsPerMultiProcessor / prop.warpSize);

    const int maxWarpsPerSm = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    // 一个通用的小报告函数（写成 lambda，避免函数指针模板推导的麻烦）
    auto report = [&](const char* name, const cudaFuncAttributes& attr, int blockSize) {
        // 注意：这里假定没有动态共享内存；静态共享内存已经在 attr 里
        int blocksPerSm = 0;
        // 用带属性重载不方便，直接对每个 kernel 单独调用（见下面各段）
        printf("  ---- %s ----\n", name);
        printf("    寄存器/线程      : %d\n", attr.numRegs);
        printf("    静态共享内存     : %d 字节\n", (int)attr.sharedSizeBytes);
        printf("    静态本地内存     : %d 字节（>0 说明有 spill）\n", (int)attr.localSizeBytes);
        printf("    编译时最大线程数 : %d\n", attr.maxThreadsPerBlock);
        printf("    要求块大小       : %d\n", blockSize);
        (void)blocksPerSm;
        (void)maxWarpsPerSm;
    };

    cudaFuncAttributes attrA{}, attrB{}, attrC{}, attrD{};

    CUDA_CHECK(cudaFuncGetAttributes(&attrA, lightKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&attrB, heavyKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&attrC, boundedKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&attrD, sharedHeavyKernel));

    report("lightKernel", attrA, BLOCK);
    report("heavyKernel", attrB, BLOCK);
    report("boundedKernel (__launch_bounds__(256,4))", attrC, BLOCK);
    report("sharedHeavyKernel", attrD, BLOCK);

    // ------------------------------------------------------------
    // 逐 kernel 计算占用率
    // ------------------------------------------------------------
    printf("\n--- 占用率（blockSize = %d）---\n", BLOCK);

    int bA = 0, bB = 0, bC = 0, bD = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bA, lightKernel, BLOCK, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bB, heavyKernel, BLOCK, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bC, boundedKernel, BLOCK, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bD, sharedHeavyKernel, BLOCK, 0));

    auto occLine = [&](const char* name, int blocksPerSm) {
        int warps = blocksPerSm * BLOCK / prop.warpSize;
        double occ = 100.0 * warps / maxWarpsPerSm;
        printf("  %-34s 每 SM %2d 块 = %3d warp，占用率 %5.1f%%\n",
               name, blocksPerSm, warps, occ);
    };
    occLine("lightKernel", bA);
    occLine("heavyKernel", bB);
    occLine("boundedKernel", bC);
    occLine("sharedHeavyKernel", bD);

    printf("\n  寄存器估算：每 SM 有 %d 个寄存器，分给 %d 个线程就是\n",
           prop.regsPerMultiprocessor, prop.maxThreadsPerMultiProcessor);
    printf("  每人最多 %d 个。lightKernel 只用 %d 个，所以寄存器不是瓶颈；\n",
           prop.regsPerMultiprocessor / prop.maxThreadsPerMultiProcessor, attrA.numRegs);
    printf("  heavyKernel 用了 %d 个，寄存器就成了限制占用率的头号因素。\n",
           attrB.numRegs);

    // ------------------------------------------------------------
    // 用 cudaOccupancyMaxPotentialBlockSize 让运行时帮你选块大小
    // ------------------------------------------------------------
    {
        int minGridSize = 0, blockSize = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, heavyKernel, 0, 0));
        printf("\n--- cudaOccupancyMaxPotentialBlockSize(heavyKernel) ---\n");
        printf("  推荐块大小 = %d，对应最小网格 = %d 个块\n", blockSize, minGridSize);
        printf("  这只是「起点建议」，真实最优值仍要靠测量。\n");
    }

    // ------------------------------------------------------------
    // 同一 kernel 在不同块大小下的占用率
    // ------------------------------------------------------------
    {
        printf("\n--- lightKernel 在不同块大小下的占用率 ---\n");
        const int sizes[] = {32, 64, 128, 256, 512, 1024};
        for (int k = 0; k < 6; ++k) {
            int b = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &b, lightKernel, sizes[k], 0));
            int warps = b * sizes[k] / prop.warpSize;
            printf("  blockSize = %4d : 每 SM %2d 块 = %3d warp，占用率 %5.1f%%\n",
                   sizes[k], b, warps, 100.0 * warps / maxWarpsPerSm);
        }
        printf("  注意 blockSize = 32 时占用率反而很低：\n");
        printf("  每个 SM 能同时驻留的 block 数有上限（通常 16~32 个），\n");
        printf("  块太小会导致 warp 数量上不去。\n");
    }

    // ------------------------------------------------------------
    // 实测：占用率不同，性能差多少？
    // ------------------------------------------------------------
    {
        size_t bytes = (size_t)N * sizeof(float);
        float *d_in = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemset(d_in, 0, bytes));
        int grid = (N + BLOCK - 1) / BLOCK;

        cudaEvent_t st, sp;
        CUDA_CHECK(cudaEventCreate(&st));
        CUDA_CHECK(cudaEventCreate(&sp));

        auto timeIt = [&](auto launch) {
            launch();                                   // 预热
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(st));
            for (int r = 0; r < 20; ++r) launch();
            CUDA_CHECK(cudaEventRecord(sp));
            CUDA_CHECK(cudaEventSynchronize(sp));
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, st, sp));
            return ms / 20.0f;
        };

        float tA = timeIt([&] { lightKernel<<<grid, BLOCK>>>(d_in, d_out, N); });
        float tB = timeIt([&] { heavyKernel<<<grid, BLOCK>>>(d_in, d_out, N); });
        float tC = timeIt([&] { boundedKernel<<<grid, BLOCK>>>(d_in, d_out, N); });

        printf("\n--- 实测耗时（各跑 20 次取平均）---\n");
        printf("  lightKernel   : %8.4f ms\n", tA);
        printf("  heavyKernel   : %8.4f ms\n", tB);
        printf("  boundedKernel : %8.4f ms\n", tC);
        printf("  这个 kernel 是带宽受限的，所以占用率从 100%% 掉到 50%%\n");
        printf("  也未必明显变慢 —— 这正是「不能只看占用率」的原因。\n");
        printf("  真正的判据是 ncu 给出的 「Memory Throughput」和\n");
        printf("  「Warp Stall Reasons」：如果 stall 主要在长记分板等待，\n");
        printf("  那提高占用率就有用；如果已经在带宽上限附近，就没用。\n");

        CUDA_CHECK(cudaEventDestroy(st));
        CUDA_CHECK(cudaEventDestroy(sp));
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    printf("\n完成。\n");
    return 0;
}
