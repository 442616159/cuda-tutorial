// ============================================================
//  cublas_sgemm.cu —— 用 cuBLAS 做单精度矩阵乘法 C = A * B
//
//  本程序重点演示三件事：
//    1) 行主序（C 语言风格）的数据，如何正确地喂给"列主序"的 cuBLAS
//    2) cublasSgemm 每一个参数到底是什么意思
//    3) 如何与 CPU 参考实现对比、如何测 GFLOPS
//
//  编译：
//    nvcc -O3 -arch=native cublas_sgemm.cu -o cublas_sgemm -lcublas
//  （老版本 nvcc 不支持 -arch=native，请换成 -arch=sm_75 之类）
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include <cublas_v2.h>
#include "../../../common/cuda_check.h"

// ------------------------------------------------------------
// cuBLAS 的状态码不是 cudaError_t，所以需要自己的检查宏。
// 这是工程里的常见做法：每个库配一个 CHECK 宏。
// ------------------------------------------------------------
#define CUBLAS_CHECK(expr)                                                        \
    do {                                                                          \
        cublasStatus_t _st = (expr);                                              \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "\n[CUBLAS ERROR] %s:%d\n  call : %s\n  code : %d\n", \
                    __FILE__, __LINE__, #expr, (int)_st);                         \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// ------------------------------------------------------------
// 统一的 GPU 计时器（与 ch00 中的 CudaTimer 一致）
// 为什么不用 CPU 上的 clock()？因为 kernel 是异步启动的，
// CPU 计时会把"启动耗时"当成"计算耗时"，测出来完全没有意义。
// ------------------------------------------------------------
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
    void start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
    // 同步等待 stop 事件，返回毫秒
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

// ------------------------------------------------------------
// CPU 参考实现：行主序 C = A * B，尺寸 M x N = (M x K) * (K x N)
// 只用来验证正确性，不做任何优化。
// ------------------------------------------------------------
static void cpuSgemm(int M, int N, int K,
                     const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                // A 行主序：A[i][k] 在 A[i*K + k]；B[k][j] 在 B[k*N + j]
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

// 相对误差比较，浮点结果绝不能用 == 比较
static bool verify(const float* ref, const float* got, int n, float tol = 1e-3f) {
    float maxAbs = 0.f, maxRel = 0.f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(ref[i] - got[i]);
        maxAbs = fmaxf(maxAbs, diff);
        float denom = fmaxf(1.f, fabsf(ref[i]));
        maxRel = fmaxf(maxRel, diff / denom);
    }
    printf("  最大绝对误差 = %.3e，最大相对误差 = %.3e\n", maxAbs, maxRel);
    return maxRel < tol;
}

int main() {
    printDeviceInfo();

    // 用小尺寸，保证任何机器都跑得动
    const int M = 1024, N = 1024, K = 1024;
    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    printf("\n问题规模: C(%d x %d) = A(%d x %d) * B(%d x %d)\n\n", M, N, M, K, K, N);

    // ---------- 1. 准备数据（行主序，和普通 C 数组完全一样） ----------
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    // dC 先清零，因为下面 beta=0 时 cuBLAS 不保证不去读旧值（见第 3 节）
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));

    // ---------- 2. 创建句柄 ----------
    // 句柄里保存着 stream、workspace、算法选择等上下文，
    // 创建一次反复使用即可。频繁创建/销毁会带来不必要的开销。
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    // 让 cuBLAS 在默认流（stream 0）上干活，和我们的 cudaMemcpy 顺序一致
    CUBLAS_CHECK(cublasSetStream(handle, 0));

    // ---------- 3. 调用 GEMM ----------
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    CudaTimer timer;

    // 预热一次：第一次调用会包含库内部初始化，不计入正式计时
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             dB, N,
                             dA, K,
                             &beta,
                             dC, N));
    CUDA_CHECK_SYNC();

    const int repeat = 10;
    timer.start();
    for (int r = 0; r < repeat; ++r) {
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 dB, N,
                                 dA, K,
                                 &beta,
                                 dC, N));
    }
    float ms = timer.stop() / repeat;

    // ---------- 4. 取回结果并与 CPU 对比 ----------
    std::vector<float> hC((size_t)M * N), hRef((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    printf("[cuBLAS] 平均耗时 %.3f ms，等效算力 %.1f GFLOPS\n",
           ms, 2.0 * M * N * K / (ms * 1e-3) / 1e9);
    printf("  算力公式: 2*M*N*K 次浮点运算（一次乘 + 一次加）除以耗时\n");

    cpuSgemm(M, N, K, hA.data(), hB.data(), hRef.data());
    bool ok = verify(hRef.data(), hC.data(), M * N);
    printf("  正确性: %s\n", ok ? "通过 ✔" : "失败 ✘");

    // ---------- 5. 收尾 ----------
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    printf("\n完成。\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
