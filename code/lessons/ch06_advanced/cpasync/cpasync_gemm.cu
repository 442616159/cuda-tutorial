// ============================================================================
//  code/lessons/ch06_advanced/cpasync/cpasync_gemm.cu
//
//  异步拷贝 cp.async 与多阶段软件流水线
//
//  ⚠ // 需要 Compute Capability >= 8.0（Ampere / Ada / Hopper）
//    在更低的架构上 __pipeline_memcpy_async 会退化成普通的同步 ld/st，
//    程序仍然能编过、结果也对，但拿不到任何加速。
//
//  编译: nvcc -O3 -arch=sm_80 cpasync_gemm.cu -o cpasync_gemm
//  运行: ./cpasync_gemm
//
//  要解决什么问题？
//    传统的"加载到共享内存"必须经过寄存器：
//        global --ld--> register --st--> shared
//    这两条指令既占满指令发射槽，又让数据在寄存器文件里多绕一圈；
//    更糟的是，LSU（load-store unit）必须等数据真的回来才能继续。
//    cp.async 允许：
//        global --cp.async--> shared
//    数据完全不经过寄存器，发起之后线程可以去干别的，
//    到需要的时候再用 __pipeline_wait_prior() 等它就绪。
//    这正是"多阶段软件流水线"能成立的前提。
// ============================================================================
#include "../../../common/cuda_check.h"

#include <cuda_pipeline.h>  // __pipeline_memcpy_async / __pipeline_commit / __pipeline_wait_prior

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// 分块参数（都是 16 的倍数，方便后面和 WMMA 对接）
// ---------------------------------------------------------------------------
constexpr int BM     = 64;   // 每个块负责 C 的 64 行
constexpr int BN     = 64;   // 每个块负责 C 的 64 列
constexpr int BK     = 16;   // K 方向每次搬 16 个
constexpr int TPB    = 256;  // 256 线程 = 16x16，每线程算 4x4
constexpr int STAGES = 2;    // 双缓冲 = 2 阶段流水线

constexpr int TM = BM / 16;  // 每线程负责的行数 = 4
constexpr int TN = BN / 16;  // 每线程负责的列数 = 4

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;
constexpr int REPEAT = 100;

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

// ---------------------------------------------------------------------------
// 发起一次 A tile 的异步拷贝：BM x BK
//   注意第 4 个参数是 size，cp.async 只支持 4/8/16 字节。
//   要做 16 字节拷贝就先在全局内存里保证 16 字节对齐、并且用 float4 的粒度搬，
//   这里为了讲清楚结构，用 4 字节（单个 float）。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void loadATileAsync(float (*As)[BK],
                                               const float* __restrict__ A,
                                               int Kdim, int by, int k0, int tid) {
    static_assert(BM * BK % TPB == 0, "tile 大小必须能被线程数整除");
#pragma unroll
    for (int i = 0; i < BM * BK / TPB; ++i) {
        const int idx = tid + i * TPB;
        const int r = idx / BK;   // tile 内的行
        const int c = idx % BK;   // tile 内的列
        // 全局地址：A[by*BM + r][k0 + c]
        // 一个 warp 内 tid 连续 -> A 的地址连续 -> 合并访存没有被破坏
        __pipeline_memcpy_async(&As[r][c],
                                &A[(size_t)(by * BM + r) * Kdim + (k0 + c)],
                                sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// 发起一次 B tile 的异步拷贝：BK x BN
// ---------------------------------------------------------------------------
__device__ __forceinline__ void loadBTileAsync(float (*Bs)[BN],
                                               const float* __restrict__ B,
                                               int Ndim, int bx, int k0, int tid) {
    static_assert(BK * BN % TPB == 0, "tile 大小必须能被线程数整除");
#pragma unroll
    for (int i = 0; i < BK * BN / TPB; ++i) {
        const int idx = tid + i * TPB;
        const int r = idx / BN;
        const int c = idx % BN;
        __pipeline_memcpy_async(&Bs[r][c],
                                &B[(size_t)(k0 + r) * Ndim + (bx * BN + c)],
                                sizeof(float));
    }
}

// ===========================================================================
// 双缓冲流水线 GEMM
//   时间轴（S=共享内存缓冲下标）：
//     k=0:  发起 S0 的 cp.async ── 等待 S0 ── 计算 S0 ── 同时发起 S1
//     k=1:  等待 S1 ── 计算 S1 ── 同时发起 S0(k=2)
//     ...
//   关键：**发起下一块的拷贝** 和 **计算当前块** 是重叠的。
//   硬件上 cp.async 由 LSU 直接写共享内存，不占用计算单元的发射槽。
// ===========================================================================
__global__ void gemmCpAsync(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C) {
    // STAGES 个缓冲：As[s] 是 BM x BK，Bs[s] 是 BK x BN
    __shared__ float As[STAGES][BM][BK];
    __shared__ float Bs[STAGES][BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid % 16;   // 线程在 16x16 网格里的列
    const int ty = tid / 16;   // 行

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // 每线程 4x4 的寄存器累加器 —— 寄存器分块，提升算术强度与 ILP
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    // ---- 预取第 0 块 ----
    loadATileAsync(As[0], A, K, by, 0, tid);
    loadBTileAsync(Bs[0], B, N, bx, 0, tid);
    __pipeline_commit();  // 把上面所有 cp.async 打包成一个"提交组"

    int stage = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int next = k0 + BK;
        if (next < K) {
            // 先把下一块发出去（此刻还没等它，线程立刻继续往下走）
            loadATileAsync(As[stage ^ 1], A, K, by, next, tid);
            loadBTileAsync(Bs[stage ^ 1], B, N, bx, next, tid);
            __pipeline_commit();
            // 允许 1 个提交组未完成 —— 也就是只等"当前这一块"就绪，
            // 刚发出去的那一块继续在后台搬
            __pipeline_wait_prior(1);
        } else {
            // 最后一轮没有新的提交组，等全部完成
            __pipeline_wait_prior(0);
        }

        // __pipeline_wait_prior 只保证**本线程**发起的拷贝完成了，
        // 但计算阶段要读别的线程搬进来的数据，所以还需要一次块级同步
        __syncthreads();

        // ---- 计算当前块 ----
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float aReg[TM], bReg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[stage][ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[stage][kk][tx * TN + j];
            // 外积形式：TM*TN 次 FMA 只需要 TM+TN 次共享内存读
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += aReg[i] * bReg[j];
        }

        // 计算完成前不能让下一轮覆盖这块缓冲
        __syncthreads();
        stage ^= 1;
    }

    // ---- 写回 C ----
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int grow = by * BM + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gcol = bx * BN + tx * TN + j;
            C[(size_t)grow * N + gcol] = acc[i][j];
        }
    }
}

int main() {
    printDeviceInfo();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 8) {
        printf("\n[提示] 本机 CC = %d.%d < 8.0：cp.async 会退化成同步拷贝，\n",
               prop.major, prop.minor);
        printf("       程序仍可运行，但看不到流水线带来的收益。\n");
    } else {
        printf("\n[提示] 本机 CC = %d.%d，支持 cp.async 异步拷贝。\n",
               prop.major, prop.minor);
    }

    // 规模必须被 BM/BN/BK 整除，这里 1024 都满足
    static_assert(M % BM == 0 && N % BN == 0 && K % BK == 0, "尺寸必须能被分块整除");

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N, 0.f);
    srand(11);
    auto rnd = [] { return (float)((rand() % 2001) - 1000) / 1000.0f; };
    for (auto& v : hA) v = rnd();
    for (auto& v : hB) v = rnd();

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TPB);
    dim3 grid(M / BM, N / BN);
    printf("[配置] block = %d 线程，grid = (%d, %d)，每线程 %dx%d 累加器，%d 阶段流水线\n",
           TPB, grid.x, grid.y, TM, TN, STAGES);

    // 预热
    gemmCpAsync<<<grid, block>>>(dA, dB, dC);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    float best = 1e30f;
    for (int r = 0; r < REPEAT; ++r) {
        CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
        timer.start();
        gemmCpAsync<<<grid, block>>>(dA, dB, dC);
        const float ms = timer.stop();
        if (ms < best) best = ms;
    }
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- CPU 参考（1024^3 用 double 大约要 3~5 秒，这里只抽查前 64 行） ----
    printf("[校验] 抽查前 64 行的相对误差（完整跑 CPU 参考太慢）……\n");
    double maxRel = 0.0;
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < N; ++j) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += (double)hA[(size_t)i * K + k] * (double)hB[(size_t)k * N + j];
            const double got = (double)hC[(size_t)i * N + j];
            const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1e-6);
            if (rel > maxRel) maxRel = rel;
        }
    }

    const double flops = 2.0 * (double)M * N * K;
    printf("\n[正确性] 前 64 行最大相对误差 = %.3e  %s\n", maxRel,
           maxRel < 1e-4 ? "[OK]" : "[FAIL]");
    printf("[性能]   %dx%dx%d FP32 GEMM (cp.async 双缓冲): %.4f ms\n", M, N, K, best);
    printf("         算力 = %.2f TFLOPS (FP32)\n", flops / ((double)best * 1e-3) / 1e12);
    printf("\n[实验]   把 STAGES 改成 1（等于关掉流水线，每次都要先等拷贝完再算），\n");
    printf("         重新编译跑一次，你会看到性能明显下降 —— 这就是「重叠」的价值。\n");
    printf("         在 RTX 3060 上，双缓冲相对单缓冲通常有 20%%~50%% 的提升，\n");
    printf("         你的机器可能不同。\n");
    printf("\n[进阶]   libcu++ 的写法（等价，但更安全、能和 CUDA Graph 一起用）：\n");
    printf("           #include <cuda/pipeline>\\n");
    printf("           cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline();\\n");
    printf("           pipe.producer_acquire();\\n");
    printf("           cuda::memcpy_async(group, dst, src, size, pipe);\\n");
    printf("           pipe.producer_commit();\\n");
    printf("           pipe.consumer_wait();  ... compute ...  pipe.consumer_release();\\n");
    printf("         它和 __pipeline_* 系列是同一套底层的两种封装。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
