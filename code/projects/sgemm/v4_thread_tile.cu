// ============================================================
//  v4_thread_tile.cu —— SGEMM 版本 4：共享内存 + 线程级分块
//
//  三个问题：
//    1) v3 慢在哪？每做 1 次乘加要读 2 次共享内存（As 和 Bs 各一次）。
//       共享内存虽快，但一条 LDS 指令的吞吐是有限的（每个 SM 每周期只能
//       处理有限次共享内存访问），所以瓶颈从全局内存转移到了共享内存。
//    2) 怎么解决？把共享内存里的数据先读进寄存器，再让寄存器里的值
//       参与多次乘加。每个线程负责 4x4 = 16 个输出：
//         读 4 个 A 值 + 4 个 B 值（8 次 LDS）→ 做 16 次 FMA
//       访存/计算比从 2:1 降到 0.5:1，理论上提升 4 倍。
//    3) 怎么证明？比较 v3/v4 的 GFLOPS，并用 ncu 看
//       "Shared Memory Throughput" 是否下降。
//
//  这一版的布局是后面所有高级版本的基础。
//
//  编译： nvcc -O3 -arch=native v4_thread_tile.cu -o sgemm_v4
// ============================================================

#include "sgemm_common.h"

// 块级分块大小
constexpr int BM = 64;   // 一个线程块负责的输出行数
constexpr int BN = 64;   // 一个线程块负责的输出列数
constexpr int BK = 16;   // K 方向一次装载的深度
// 线程级分块大小：每个线程算 TM x TN 个输出
constexpr int TM = 4;
constexpr int TN = 4;

// 线程块 (BN/TN) x (BM/TM) = 16 x 16 = 256 线程
constexpr int THREADS_X = BN / TN;
constexpr int THREADS_Y = BM / TM;

__global__ void sgemmThreadTile(int M, int N, int K,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C) {
    // padding 1 列，作用同 v3
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // 每个线程负责的 4x4 输出子块的左上角
    const int rowStart = blockRow + ty * TM;
    const int colStart = blockCol + tx * TN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int numTiles = (K + BK - 1) / BK;

    for (int t = 0; t < numTiles; ++t) {
        // --------------------------------------------------
        // 装载 As：BM x BK = 64 x 16 = 1024 个元素，256 线程 -> 每人 4 个
        // 用 一维编号 -> 二维坐标 的展开方式，保证同一时刻相邻线程
        // 访问的全局地址是连续的（合并访存）。
        // --------------------------------------------------
#pragma unroll
        for (int i = 0; i < BM * BK / (THREADS_X * THREADS_Y); ++i) {
            const int idx = tid + i * (THREADS_X * THREADS_Y);
            const int r = idx / BK;          // 0..BM-1
            const int c = idx % BK;          // 0..BK-1
            const int gr = blockRow + r;
            const int gc = t * BK + c;
            As[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.f;
        }
        // --------------------------------------------------
        // 装载 Bs：BK x BN = 16 x 64 = 1024 个元素，同样每人 4 个。
        // Bs 的行是 K 方向，列是 N 方向，和 B 在全局内存里的布局一致，
        // 所以这里也是合并访存。
        // --------------------------------------------------
#pragma unroll
        for (int i = 0; i < BK * BN / (THREADS_X * THREADS_Y); ++i) {
            const int idx = tid + i * (THREADS_X * THREADS_Y);
            const int r = idx / BN;          // 0..BK-1
            const int c = idx % BN;          // 0..BN-1
            const int gr = t * BK + r;
            const int gc = blockCol + c;
            Bs[r][c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.f;
        }
        __syncthreads();

        // --------------------------------------------------
        // 计算：每次 k 只读 TM + TN = 8 个共享内存值，
        // 却做了 TM * TN = 16 次乘加。这就是线程级分块的全部价值。
        // --------------------------------------------------
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float aReg[TM], bReg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[ty * TM + i][k];
#pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }

    // 写回：4x4 = 16 次全局写。相邻线程写相邻地址 -> 合并
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = rowStart + i, c = colStart + j;
            if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
        }
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    sgemm::printBandwidthReference();
    printf("\n=== v4 线程级分块 (块 %dx%d, 线程级 %dx%d): %d x %d x %d ===\n",
           BM, BN, TM, TN, M, N, K);

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

    const dim3 block(THREADS_X, THREADS_Y);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    auto launch = [&]() {
        sgemmThreadTile<<<grid, block>>>(M, N, K, dA, dB, dC);
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
    sgemm::printRow("v4 共享+线程分块", r);

    printf("\n结论：这一版开始接近合格的 GPU 矩阵乘法。剩下的瓶颈是：\n");
    printf("  1) 每次装载用 4 条 32 位全局读指令（LDG.E.32），\n");
    printf("     而硬件一次事务能搬 128 字节 —— 用 float4 一次搬 16 字节更划算；\n");
    printf("  2) 装载下一块数据和计算当前块是串行的，\n");
    printf("     全局内存的延迟没有被计算掩盖（v6 用双缓冲解决）。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
