// ============================================================
//  v3_shared_tile.cu —— SGEMM 版本 3：共享内存分块
//
//  三个问题：
//    1) v2 慢在哪？A 的同一行数据被块内 16 个线程各读了一遍，
//       B 的同一列也被各读了一遍 —— 读的是同一份数据，却走了 16 次全局内存。
//    2) 为什么共享内存能解决？共享内存的带宽比全局内存高一个数量级，
//       而且同一块内的线程可以共享它。把数据"搬一次、用很多次"，
//       全局内存的访问量就降下来了。
//    3) 怎么证明？看 GFLOPS 和 ncu 里的 "Memory Throughput" 指标。
//
//  这是矩阵乘法优化里最重要的一次跳跃。
//
//  编译： nvcc -O3 -arch=native v3_shared_tile.cu -o sgemm_v3
// ============================================================

#include "sgemm_common.h"

constexpr int TILE = 32;   // 32x32 的块，正好一个 warp 覆盖一行

// ------------------------------------------------------------
// 线程块 32x32 = 1024 线程，每个线程算 1 个输出元素。
// 分块后：读一个 TILE x TILE 的数据，被 TILE 个线程各用一次。
//
// 共享内存大小 = 2 * 32 * 33 * 4 字节 ≈ 8.4 KB，远小于 48 KB 上限。
// ------------------------------------------------------------
__global__ void sgemmSharedTile(int M, int N, int K,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C) {
    // [TILE][TILE + 1]：多出来的那一列是经典的"错位填充"，
    // 用来打散 bank。没有它，读 Bs[k][tx] 时一个 warp 内 tx=0..31
    // 会落在 stride=32 的地址上，全部撞到同一个 bank（32 路冲突），
    // 共享内存的优势会被完全吃掉。
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.f;

    const int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        // ---- 装载阶段：相邻 tx 读相邻地址，全局访存完全合并 ----
        // 越界填 0 而不是 return：块内所有线程都必须走到 __syncthreads。
        const int aCol = t * TILE + tx;
        As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.f;

        const int bRow = t * TILE + ty;
        Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.f;

        // 屏障 1（生产者-消费者）：保证共享内存写完了，所有人才能开始读
        __syncthreads();

        // ---- 计算阶段：全部数据来自共享内存 ----
        // As[ty][k] 在同一个 warp 内是同一个地址 -> 广播，不冲突
        // Bs[k][tx] 在 warp 内是连续地址 -> 也不冲突（padding 起了作用）
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        // 屏障 2（防覆盖）：保证所有人都读完了，下一轮才能覆盖共享内存
        __syncthreads();
    }

    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    sgemm::printBandwidthReference();
    printf("\n=== v3 共享内存分块 (TILE=%d): %d x %d x %d ===\n", TILE, M, N, K);

    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    std::vector<float> hC((size_t)M * N, 0.f), hRef((size_t)M * N, 0.f);
    sgemm::initMatrix(hA, 1);
    sgemm::initMatrix(hB, 2);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    auto launch = [&]() {
        sgemmSharedTile<<<grid, block>>>(M, N, K, dA, dB, dC);
        CUDA_CHECK_KERNEL();
    };

    launch();
    CUDA_CHECK_SYNC();
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    printf("正在计算 CPU 参考结果...\n");
    sgemm::cpuSgemm(M, N, K, hA.data(), hB.data(), hRef.data());
    const bool ok = sgemm::verify(hRef, hC);
    printf("正确性: %s\n", ok ? "通过 ✔" : "失败 ✘");

    const sgemm::Result r = sgemm::benchmark(launch, M, N, K);
    sgemm::printTableHeader();
    sgemm::printRow("v3 共享内存分块", r);

    printf("\n结论：通常比 v2 有明显提升，但还会遇到两个新问题：\n");
    printf("  1) 计算/访存比仍然只有 TILE 倍 —— 每从共享内存读 2 个 float 才做 1 次乘加；\n");
    printf("     共享内存带宽也是有限的，继续加大 TILE 会被共享内存容量卡住。\n");
    printf("  2) 每次装载都要两次 __syncthreads，块内线程越多，同步代价越大。\n");
    printf("下一步：让每个线程算 4x4 的输出块（v4 线程级分块），\n");
    printf("        把共享内存的数据先搬进寄存器，让每个共享读服务于多次乘加。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
