// ============================================================
//  v7_reg_tile_8x8.cu —— SGEMM 版本 7：大线程块 + 8x8 寄存器分块
//
//  三个问题：
//    1) v4~v6 慢在哪？线程级分块只有 4x4，每次 k 读 8 个共享内存值、
//       只做 16 次乘加。而共享内存的带宽也是有限的，
//       访存/计算比还不够低，而且 C 的数据复用次数不够。
//    2) 这一版做了什么？
//       - 把线程级分块放大到 8x8：读 16 个值做 64 次乘加，比值再降 4 倍
//       - 块级分块放大到 128x128：一次装载的数据被 256 个线程共享，
//         全局内存的重复读取次数进一步下降
//       - 用 __launch_bounds__ 告诉编译器"我要 256 线程/块"，
//         让它敢用更多寄存器（避免把累加器溢出到本地内存 —— 那是最惨的情况）
//    3) 怎么证明？ncu 里 "Compute (SM) Throughput" 应该接近 100%，
//       而 "Memory Throughput" 不再是瓶颈 —— 说明进入计算受限区间了。
//
//  代价：寄存器用量大幅上升（累加器就占 64 个），占用率会降到 25% 左右。
//  这很正常：**现代 GPU 上高寄存器用量 + 低占用率 + 高 ILP 是 GEMM 的常态**，
//  因为每个线程有 64 个独立的 FMA 可以发射，已经足够填满流水线了。
//
//  编译： nvcc -O3 -arch=native -Xptxas -v v7_reg_tile_8x8.cu -o sgemm_v7
// ============================================================

#include "sgemm_common.h"

constexpr int BM = 128, BN = 128, BK = 8;
constexpr int TM = 8, TN = 8;
// 行跨度加 4：既保证 16 字节对齐（float4 需要），又缓解 bank conflict
constexpr int LD_A = BK + 4;    // 12
constexpr int LD_B = BN + 4;    // 132

constexpr int THREADS_X = BN / TN;              // 16
constexpr int THREADS_Y = BM / TM;              // 16
constexpr int NUM_THREADS = THREADS_X * THREADS_Y;   // 256

__global__ __launch_bounds__(NUM_THREADS, 1) void sgemmRegTile(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {
    __shared__ __align__(16) float As[BM * LD_A];   // 128 x 12
    __shared__ __align__(16) float Bs[BK * LD_B];   // 8 x 132

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // 每个线程负责 8x8 的输出子块
    const int threadRow = ty * TM;   // 0..120
    const int threadCol = tx * TN;   // 0..120

    // 装载索引：
    //   As = 128x8 = 1024 float = 256 个 float4 -> 每线程 1 个
    //     每行 8 个 float = 2 个 float4，所以 row = tid/2，col4 = (tid%2)*4
    //   Bs = 8x128 = 1024 float = 256 个 float4 -> 每线程 1 个
    //     每行 128 个 float = 32 个 float4，所以 row = tid/32，col4 = (tid%32)*4
    const int aRow = tid / (BK / 4);
    const int aCol4 = (tid % (BK / 4)) * 4;
    const int bRow = tid / (BN / 4);
    const int bCol4 = (tid % (BN / 4)) * 4;

    // 64 个累加器全在寄存器里。这是这一版的核心资产，
    // 也是为什么必须用 __launch_bounds__ 保证编译器不去 spill。
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int numTiles = (K + BK - 1) / BK;

    for (int t = 0; t < numTiles; ++t) {
        // ---- 装载：每线程恰好 1 个 float4，完美分摊 ----
        {
            const int gr = blockRow + aRow;
            const int gc = t * BK + aCol4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < M && gc + 3 < K)
                v = *reinterpret_cast<const float4*>(A + (size_t)gr * K + gc);
            *reinterpret_cast<float4*>(As + aRow * LD_A + aCol4) = v;
        }
        {
            const int gr = t * BK + bRow;
            const int gc = blockCol + bCol4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < K && gc + 3 < N)
                v = *reinterpret_cast<const float4*>(B + (size_t)gr * N + gc);
            *reinterpret_cast<float4*>(Bs + bRow * LD_B + bCol4) = v;
        }
        __syncthreads();

        // ---- 计算：8 次 A 读 + 2 次 128 位 B 读，换来 64 次 FMA ----
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = As[(threadRow + i) * LD_A + k];

            // 用 float4 读 B 的一行：一条 LDS.128 指令拿 4 个值。
            // 如果改成逐元素读 b[j] = Bs[k*LD_B + threadCol + j]，
            // 相邻线程的地址间隔是 8 个 float，会撞成 4 路 bank conflict，
            // 共享内存吞吐直接掉到 1/4。
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[k * LD_B + threadCol]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[k * LD_B + threadCol + 4]);
            const float b[TN] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};

#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = blockRow + threadRow + i;
            const int c = blockCol + threadCol + j;
            if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
        }
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    sgemm::printBandwidthReference();
    printf("\n=== v7 大块 + 8x8 寄存器分块 (%dx%d, K 步 %d): %d x %d x %d ===\n",
           BM, BN, BK, M, N, K);
    printf("共享内存用量 = %.1f KB\n",
           sizeof(float) * (BM * LD_A + BK * LD_B) / 1024.0);

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
        sgemmRegTile<<<grid, block>>>(M, N, K, dA, dB, dC);
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
    sgemm::printRow("v7 8x8 寄存器分块", r);

    printf("\n用 -Xptxas -v 编译时，你应该能看到类似输出：\n");
    printf("    Used 90+ registers, ...  -- 高于 4x4 版本，属于预期\n");
    printf("    0 bytes spill stores       -- 这个必须是 0！出现 spill 说明寄存器不够，\n");
    printf("                                  累加器被丢到本地内存（等价于全局内存），性能暴跌\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
