// ============================================================
//  v6_double_buffer.cu —— SGEMM 版本 6：双缓冲 / 软件流水线
//
//  三个问题：
//    1) v5 慢在哪？循环体里三件事是严格串行的：
//         发起全局读 -> __syncthreads 等待 -> 用共享内存计算
//       在"等待"那一段，所有线程都闲着，SM 的 FMA 单元空转。
//       全局内存延迟大约几百个时钟周期，而一个 BK 块的计算只有几十个周期，
//       所以等待时间占了循环的大部分。
//    2) 双缓冲怎么解决？准备两份共享内存缓冲，A 用着的时候 B 在装：
//         - 把下一块数据**提前读到寄存器里**（这一步不用等数据到达，
//           指令发射完线程就能继续往下走）
//         - 然后用当前缓冲做计算 —— 计算的这几十个周期正好用来
//           掩盖寄存器装载的几百周期延迟
//         - 计算完了再把寄存器写进另一份共享缓冲
//       这就是"软件流水线（software pipelining）"。
//    3) 怎么证明？ncu 里看 "Warp Stall Reasons" 中的
//       Long Scoreboard（等全局内存）占比下降，GFLOPS 上升。
//
//  为什么用"寄存器中转"而不是直接 cp.async？
//    因为 cp.async 需要 CC >= 8.0，而寄存器中转在所有架构上都能用，
//    也更能帮你看清"预取"这件事的本质。第 6 章讲过的 cuda::pipeline
//    是同一思想的库化版本。
//
//  编译： nvcc -O3 -arch=native v6_double_buffer.cu -o sgemm_v6
// ============================================================

#include "sgemm_common.h"

constexpr int BM = 64, BN = 64, BK = 16;
constexpr int TM = 4, TN = 4;
constexpr int LD_A = BK + 4;   // 20
constexpr int LD_B = BN + 4;   // 68
constexpr int THREADS_X = BN / TN;
constexpr int THREADS_Y = BM / TM;

__global__ __launch_bounds__(256) void sgemmDoubleBuffer(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {
    // 两份缓冲：As[0]/Bs[0] 和 As[1]/Bs[1] 轮流当"当前块"
    __shared__ __align__(16) float As[2][BM * LD_A];
    __shared__ __align__(16) float Bs[2][BK * LD_B];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int aRow = tid / (BK / 4);
    const int aCol4 = (tid % (BK / 4)) * 4;
    const int bRow = tid / (BN / 4);
    const int bCol4 = (tid % (BN / 4)) * 4;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int numTiles = (K + BK - 1) / BK;

    // --------------------------------------------------------
    // 序言：先把第 0 块装进缓冲 0。
    // 流水线总要有一次"填不满"的启动过程，这是任何软件流水线的固有开销。
    // --------------------------------------------------------
    {
        float4 va = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 vb = make_float4(0.f, 0.f, 0.f, 0.f);
        const int gr = blockRow + aRow, gc = aCol4;
        if (gr < M && gc + 3 < K) va = *reinterpret_cast<const float4*>(A + (size_t)gr * K + gc);
        const int grb = bRow, gcb = blockCol + bCol4;
        if (grb < K && gcb + 3 < N) vb = *reinterpret_cast<const float4*>(B + (size_t)grb * N + gcb);
        *reinterpret_cast<float4*>(As[0] + aRow * LD_A + aCol4) = va;
        *reinterpret_cast<float4*>(Bs[0] + bRow * LD_B + bCol4) = vb;
    }
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        const int cur = t & 1;
        const int nxt = (t + 1) & 1;

        // ----------------------------------------------------
        // 第 1 步：预取下一块到寄存器。
        // 关键点：这里只是"发射"读指令，线程不会停在原地等数据；
        // 真正使用寄存器里的值是在下面第 3 步，中间隔着整个计算过程，
        // 几百周期的全局内存延迟就被这段计算盖住了。
        // ----------------------------------------------------
        float4 va = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 vb = make_float4(0.f, 0.f, 0.f, 0.f);
        if (t + 1 < numTiles) {
            const int gr = blockRow + aRow;
            const int gc = (t + 1) * BK + aCol4;
            if (gr < M && gc + 3 < K)
                va = *reinterpret_cast<const float4*>(A + (size_t)gr * K + gc);

            const int grb = (t + 1) * BK + bRow;
            const int gcb = blockCol + bCol4;
            if (grb < K && gcb + 3 < N)
                vb = *reinterpret_cast<const float4*>(B + (size_t)grb * N + gcb);
        }

        // ----------------------------------------------------
        // 第 2 步：用"当前"缓冲做计算 —— 这段时间就是预取的隐藏期
        // ----------------------------------------------------
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float aReg[TM], bReg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[cur][(ty * TM + i) * LD_A + k];
#pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[cur][k * LD_B + tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += aReg[i] * bReg[j];
        }

        // ----------------------------------------------------
        // 第 3 步：等所有人都算完（屏障 A），再把预取的数据写进另一份缓冲，
        // 然后再一次壁垒（屏障 B）保证下一轮所有人都能看到新数据。
        // 屏障 A 必须在写之前：否则会覆盖别人还没读完的旧缓冲。
        // ----------------------------------------------------
        __syncthreads();
        if (t + 1 < numTiles) {
            *reinterpret_cast<float4*>(As[nxt] + aRow * LD_A + aCol4) = va;
            *reinterpret_cast<float4*>(Bs[nxt] + bRow * LD_B + bCol4) = vb;
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = blockRow + ty * TM + i;
            const int c = blockCol + tx * TN + j;
            if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
        }
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    sgemm::printBandwidthReference();
    printf("\n=== v6 双缓冲软件流水线: %d x %d x %d ===\n", M, N, K);
    printf("共享内存用量 = %zu 字节（两份缓冲）\n",
           sizeof(float) * (2 * BM * LD_A + 2 * BK * LD_B));

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
        sgemmDoubleBuffer<<<grid, block>>>(M, N, K, dA, dB, dC);
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
    sgemm::printRow("v6 双缓冲流水线", r);

    printf("\n注意：双缓冲不一定总是更快。\n");
    printf("  - K 很小（循环次数少）时，流水线的序言/尾声开销占比高，可能变慢；\n");
    printf("  - 寄存器用量上升会压低占用率，占用率太低时连计算都盖不住延迟。\n");
    printf("  这就是为什么优化必须用数据说话，不能凭感觉。\n");
    printf("  建议用 ncu 对比 v5/v6 的 Long Scoreboard 停顿占比来确认。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
