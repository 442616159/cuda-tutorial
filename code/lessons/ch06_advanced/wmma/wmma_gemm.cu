// ============================================================================
//  code/lessons/ch06_advanced/wmma/wmma_gemm.cu
//
//  张量核心（Tensor Core）与 WMMA API：用 nvcuda::wmma 做 FP16 矩阵乘
//
//  ⚠ // 需要 Compute Capability >= 7.0（Volta / Turing / Ampere / Ada / Hopper）
//    编译时必须指定 >= sm_70，否则 <mma.h> 里的 fragment 无法实例化。
//
//  编译: nvcc -O3 -arch=sm_70 wmma_gemm.cu -o wmma_gemm
//        支持 Ampere 的卡建议改成 -arch=sm_80
//  运行: ./wmma_gemm
//
//  核心概念：fragment（片段）
//    张量核心不接受"某个线程负责哪些元素"这种传统映射。
//    一次 mma 由**一整个 warp** 协作完成，累加器分散在该 warp 32 个线程的
//    寄存器里，具体布局是硬件细节 —— 你不应该、也不需要关心它。
//    fragment 就是这个"分布式的、不透明的寄存器集合"的抽象。
// ============================================================================
#include "../../../common/cuda_check.h"

#include <cuda_fp16.h>
#include <mma.h>
// WMMA 的 fragment / load_matrix_sync / store_matrix_sync 都在 nvcuda::wmma 命名空间下
using namespace nvcuda;

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// 一次 WMMA 操作的形状：16 x 16 x 16
//   注意：这是 **warp 级别** 的形状，不是块级别，更不是线程级别。
//   一个 warp 一次吃 16x16 的 A、16x16 的 B，攒出 16x16 的 C。
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int M = 512;  // A: M x K
constexpr int N = 512;  // B: K x N
constexpr int K = 512;  // C: M x N
constexpr int REPEAT = 200;

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

// ===========================================================================
// WMMA 版本的矩阵乘
//   块组织：blockDim = (32, 4)
//     threadIdx.x 恰好覆盖一个 warp 的 32 条 lane  -> 一个 warp 一个输出 tile
//     threadIdx.y 是"每个块里有几个 warp"
//   每个 warp 负责输出矩阵 C 中一个 16x16 的小块。
//
//   为什么 K 方向要循环：一次 mma_sync 只能吃掉 16 个 K。
//   为了不对每个 k 都单独做一次 global load，生产代码会先把 A/B 的 tile
//   用 cp.async 搬进共享内存（见 lessons/ch06_advanced/cpasync/cpasync_gemm.cu）。
//   这里为了把 WMMA 本身讲清楚，先直接对全局内存做 load_matrix_sync。
// ===========================================================================
__global__ void wmmaGemmKernel(const __half* __restrict__ a,
                               const __half* __restrict__ b,
                               float* __restrict__ c) {
    // blockIdx.x 决定"第几个 16 行块"，每个 warp 独占一个 tile
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // 1) 声明 fragment：这一步只是在寄存器里预留空间，不访问内存
    //    matrix_a 行主序；matrix_b 列主序 —— 后者搭配"行主序存储的 B 数组"
    //    使用，等价于读取 B 的转置，这是官方样例的标准写法。
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    //    累加器用 float：即使输入是 FP16，累加也在 FP32 里做。
    //    否则 K 一大精度就崩 —— 这就是"混合精度"这个词的由来。
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 2) 累加器清零（fill_fragment 也是 per-warp 协作的）
    wmma::fill_fragment(c_frag, 0.0f);

    // 3) 沿 K 方向循环，每次吃 16 个
    for (int k = 0; k < K; k += WMMA_K) {
        // load_matrix_sync：把一块数据从内存（或共享内存）搬进 fragment
        //   第三个参数 ldm 是行/列跨度（leading dimension）
        wmma::load_matrix_sync(a_frag, a + (size_t)warpM * WMMA_M * K + k, K);
        wmma::load_matrix_sync(b_frag, b + (size_t)k * N + warpN * WMMA_N, N);

        // mma_sync：一条指令完成 16x16x16 的乘加 D = A*B + C
        //   四个 fragment 的线程分布必须一致 —— WMMA 保证了这一点
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 4) 把累加器写回全局内存
    wmma::store_matrix_sync(c + (size_t)warpM * WMMA_M * N + warpN * WMMA_N,
                            c_frag, N, wmma::mem_row_major);
}

int main() {
    printDeviceInfo();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        printf("\n[跳过] 本机 CC = %d.%d，WMMA 需要 CC >= 7.0。\n", prop.major, prop.minor);
        return 0;
    }

    // ---- 准备数据：FP16 输入，FP32 输出 ----
    std::vector<__half> hA((size_t)M * K), hB((size_t)K * N);
    std::vector<float> hGot((size_t)M * N, 0.f);
    srand(7);
    auto rnd = [] { return (float)((rand() % 2001) - 1000) / 1000.0f; };
    for (auto& v : hA) v = __float2half(rnd() * 0.1f);
    for (auto& v : hB) v = __float2half(rnd() * 0.1f);

    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));

    // 布局：grid.x 覆盖 M/16 个行块，grid.y 覆盖 N/(16*blockDim.y) 个列块
    dim3 block(32, 4);
    dim3 grid(M / WMMA_M, N / (WMMA_N * (int)block.y));
    printf("\n[配置] block = (%d, %d)，grid = (%d, %d)，共 %d 个 warp\n",
           block.x, block.y, grid.x, grid.y, grid.x * grid.y * block.y);

    // ---- 预热一次，排除首次 launch 的固定开销 ----
    wmmaGemmKernel<<<grid, block>>>(dA, dB, dC);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    float best = 1e30f;
    for (int r = 0; r < REPEAT; ++r) {
        CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
        timer.start();
        wmmaGemmKernel<<<grid, block>>>(dA, dB, dC);
        const float ms = timer.stop();
        if (ms < best) best = ms;
    }
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(hGot.data(), dC, (size_t)M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ---- CPU 参考：用 double 完整算一遍（512^3 次乘加，约 0.5 秒） ----
    printf("[校验] 正在用 CPU(double) 计算参考结果，请稍候……\n");
    std::vector<double> ref((size_t)M * N, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const double aik = (double)__half2float(hA[(size_t)i * K + k]);
            const __half* brow = &hB[(size_t)k * N];
            double* crow = &ref[(size_t)i * N];
            for (int j = 0; j < N; ++j)
                crow[j] += aik * (double)__half2float(brow[j]);
        }
    }

    // FP16 输入 + FP32 累加，相对误差用"绝对值量级"衡量更合适，
    // 因为 C 的元素可能非常接近 0，逐元素相对误差会失去意义
    double maxAbsErr = 0.0, sumAbs = 0.0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        const double e = std::fabs((double)hGot[i] - ref[i]);
        if (e > maxAbsErr) maxAbsErr = e;
        sumAbs += std::fabs(ref[i]);
    }
    const double relL1 = maxAbsErr * (double)M * N / (sumAbs + 1e-12);

    const double flops = 2.0 * (double)M * N * K;
    printf("\n[正确性] 最大绝对误差 = %.4e，相对 L1 误差量级 = %.3e  %s\n",
           maxAbsErr, relL1, relL1 < 1e-3 ? "[OK]" : "[FAIL]");
    printf("[性能]   %dx%dx%d FP16->FP32 GEMM: %.4f ms\n", M, N, K, best);
    printf("         算力 = %.2f TFLOPS (FP16 输入 + FP32 累加)\n",
           flops / ((double)best * 1e-3) / 1e12);
    printf("\n[对照]   同样规模在 CPU 上单线程约需 0.5 秒（约 0.0003 TFLOPS）；\n");
    printf("         这个朴素 WMMA 版本在 RTX 3060 上约 2~5 TFLOPS，你的机器可能不同；\n");
    printf("         同样硬件上 cuBLAS 能到 20+ TFLOPS —— 差距来自三件事：\n");
    printf("           1) 共享内存分块（减少全局访存）；\n");
    printf("           2) cp.async 多阶段流水线（把访存和计算重叠起来）；\n");
    printf("           3) 每线程多个累加器（寄存器分块，提升指令级并行）。\n");
    printf("\n[下一步] 用 ncu 看张量管线的利用率：\n");
    printf("         ncu --metrics "
           "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active ./wmma_gemm\n");
    printf("         这个数低于 50%% 时，瓶颈通常就是那两次 load_matrix_sync。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
