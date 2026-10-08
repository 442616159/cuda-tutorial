// ============================================================
//  v5_float4.cu —— SGEMM 版本 5：向量化访存（float4）
//
//  三个问题：
//    1) v4 慢在哪？装载共享内存时每个线程发 4 条 32 位读指令。
//       硬件搬运数据是按"内存事务"（32/64/128 字节）来的，
//       指令条数多意味着更多指令周期、更多地址计算、
//       更高的 LSU（Load-Store Unit）占用。
//    2) 为什么 float4 会快？一条 LDG.E.128 指令能搬 16 字节，
//       同样搬 1024 个 float，指令数变成原来的 1/4。
//       而且半个 warp（16 个线程 x 16 字节）正好铺满 256 字节，
//       内存事务的利用率接近 100%。
//    3) 怎么证明？ncu 里看 "L1/TEX Load Throughput" 和指令数
//       （Inst Executed）会明显下降，GFLOPS 上升。
//
//  代价：数据必须 16 字节对齐，且尺寸要能整除分块大小，
//        否则要写边界处理代码。生产代码里这一步最容易出隐藏 bug。
//
//  编译： nvcc -O3 -arch=native v5_float4.cu -o sgemm_v5
// ============================================================

#include "sgemm_common.h"

constexpr int BM = 64, BN = 64, BK = 16;
constexpr int TM = 4, TN = 4;

// 共享内存的"行跨度"。BN=64 时用 68：既保持 16 字节对齐
// （68 是 4 的倍数），又能顺带缓解 bank conflict。
constexpr int LD_A = BK + 4;    // 20
constexpr int LD_B = BN + 4;    // 68

constexpr int THREADS_X = BN / TN;   // 16
constexpr int THREADS_Y = BM / TM;   // 16

__global__ __launch_bounds__(256) void sgemmFloat4(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {
    // __align__(16)：float4 的读写要求 16 字节对齐。
    // 声明成 __shared__ float 时编译器不保证 16 字节对齐，必须显式声明。
    __shared__ __align__(16) float As[BM * LD_A];
    __shared__ __align__(16) float Bs[BK * LD_B];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    // 装载索引：每个线程负责 1 个 float4。
    // As 是 64x16 = 1024 个 float = 256 个 float4，正好每人一个。
    const int aRow = tid / (BK / 4);      // tid / 4  -> 0..63
    const int aCol4 = (tid % (BK / 4)) * 4;  // 0,4,8,12
    // Bs 是 16x64 = 1024 个 float = 256 个 float4
    const int bRow = tid / (BN / 4);      // tid / 16 -> 0..15
    const int bCol4 = (tid % (BN / 4)) * 4;  // 0,4,...,60

    const int numTiles = (K + BK - 1) / BK;

    for (int t = 0; t < numTiles; ++t) {
        // ---- 装载 A 的一块 ----
        {
            const int gr = blockRow + aRow;
            const int gc = t * BK + aCol4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            // 先判断"整个 float4 都在界内"再读，这样既安全又能保住向量化指令。
            // 如果改成逐元素判断，编译器会退化成 4 条标量读，白忙一场。
            if (gr < M && gc + 3 < K) {
                v = *reinterpret_cast<const float4*>(A + (size_t)gr * K + gc);
            }
            *reinterpret_cast<float4*>(As + aRow * LD_A + aCol4) = v;
        }
        // ---- 装载 B 的一块 ----
        {
            const int gr = t * BK + bRow;
            const int gc = blockCol + bCol4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < K && gc + 3 < N) {
                v = *reinterpret_cast<const float4*>(B + (size_t)gr * N + gc);
            }
            *reinterpret_cast<float4*>(Bs + bRow * LD_B + bCol4) = v;
        }
        __syncthreads();

        // ---- 计算部分和 v4 完全一样：访存优化不该改动计算逻辑 ----
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float aReg[TM], bReg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[(ty * TM + i) * LD_A + k];
#pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[k * LD_B + tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += aReg[i] * bReg[j];
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
    printf("\n=== v5 float4 向量化: %d x %d x %d ===\n", M, N, K);

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
        sgemmFloat4<<<grid, block>>>(M, N, K, dA, dB, dC);
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
    sgemm::printRow("v5 float4 向量化", r);

    printf("\n提示：如果 M/N/K 不是 64 的整数倍，这一版的边界处理会生效，\n");
    printf("      但性能会因为大量标量回退而下降 —— 这正是练习 4 要你解决的。\n");
    printf("\n结论：剩下的瓶颈是装载与计算串行。每次循环里，\n");
    printf("      所有线程先等全局内存的数据到达共享内存，再开始算，\n");
    printf("      而这段等待时间（几百个周期）完全没有被利用。v6 用双缓冲解决。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
