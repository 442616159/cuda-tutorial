// ============================================================
//  cublas_strided_batch.cu —— 用 cublasSgemmStridedBatched 一次算一批小矩阵
//
//  场景：深度学习里的"多张图片同时做全连接"、批量最小二乘等，
//  都是"同一个形状的矩阵乘法要做几百上千次"。
//  逐个调用 cublasSgemm 会被 CPU 端启动开销拖死，
//  stride 批量接口只发一次调用，让 GPU 端自己循环。
//
//  编译：
//    nvcc -O3 -arch=native cublas_strided_batch.cu -o cublas_strided_batch -lcublas
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include <cublas_v2.h>
#include "../../../common/cuda_check.h"

#define CUBLAS_CHECK(expr)                                                        \
    do {                                                                          \
        cublasStatus_t _st = (expr);                                              \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "\n[CUBLAS ERROR] %s:%d\n  call : %s\n  code : %d\n", \
                    __FILE__, __LINE__, #expr, (int)_st);                         \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
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

int main() {
    printDeviceInfo();

    // batch = 256 个 64x64 的小矩阵乘法
    const int batch = 256;
    const int M = 64, N = 64, K = 64;
    const size_t elemsA = (size_t)M * K;
    const size_t elemsB = (size_t)K * N;
    const size_t elemsC = (size_t)M * N;

    printf("\n批量 GEMM: %d 组 C(%d x %d) = A(%d x %d) * B(%d x %d)\n\n",
           batch, M, N, M, K, K, N);

    // 行主序的连续存储：第 b 个矩阵从 b * elems 开始
    std::vector<float> hA(elemsA * batch), hB(elemsB * batch), hRef(elemsC * batch);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hRef.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, hRef.size() * sizeof(float)));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.f, beta = 0.f;
    // stride 的单位是"元素个数"，不是字节！
    // 这是最常见的坑之一：写成字节数会导致越界访问或结果错乱。
    const long long strideA = (long long)elemsA;
    const long long strideB = (long long)elemsB;
    const long long strideC = (long long)elemsC;

    CudaTimer timer;
    // 预热
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           N, M, K,
                                           &alpha,
                                           dB, N, strideB,
                                           dA, K, strideA,
                                           &beta,
                                           dC, N, strideC,
                                           batch));
    CUDA_CHECK_SYNC();

    const int repeat = 10;
    timer.start();
    for (int r = 0; r < repeat; ++r) {
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
                                               CUBLAS_OP_N, CUBLAS_OP_N,
                                               N, M, K,
                                               &alpha,
                                               dB, N, strideB,
                                               dA, K, strideA,
                                               &beta,
                                               dC, N, strideC,
                                               batch));
    }
    float ms = timer.stop() / repeat;

    double flops = 2.0 * M * N * K * (double)batch;
    printf("[StridedBatched] 平均耗时 %.3f ms，等效算力 %.1f GFLOPS\n",
           ms, flops / (ms * 1e-3) / 1e9);

    // ---------- 验证：逐 batch 用 CPU 算一遍 ----------
    std::vector<float> hC(hRef.size());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float maxRel = 0.f;
    for (int b = 0; b < batch; ++b) {
        const float* A = hA.data() + (size_t)b * elemsA;
        const float* B = hB.data() + (size_t)b * elemsB;
        float* R = hRef.data() + (size_t)b * elemsC;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float acc = 0.f;
                for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
                R[i * N + j] = acc;
            }
    }
    for (size_t i = 0; i < hC.size(); ++i) {
        float d = fabsf(hC[i] - hRef[i]);
        maxRel = fmaxf(maxRel, d / fmaxf(1.f, fabsf(hRef[i])));
    }
    printf("  最大相对误差 = %.3e  ->  %s\n", maxRel, maxRel < 1e-3f ? "通过 ✔" : "失败 ✘");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return maxRel < 1e-3f ? EXIT_SUCCESS : EXIT_FAILURE;
}
