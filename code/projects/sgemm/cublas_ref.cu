// ============================================================
//  cublas_ref.cu —— 用 cuBLAS 作为"天花板"参照
//
//  所有 SGEMM 版本都要和它比。注意：
//    * 这是"参考上限"，不是"必须超过的目标"。
//      手写 kernel 能达到 cuBLAS 的 60%~85% 已经是很不错的成绩。
//    * cuBLAS 会针对不同形状选不同的实现，所以不同 M/N/K 下的
//      相对表现会变化 —— 报告里要写清楚测试形状。
//
//  编译：
//    nvcc -O3 -arch=native cublas_ref.cu -o sgemm_cublas -lcublas
// ============================================================

#include "sgemm_common.h"
#include <cublas_v2.h>

#define CUBLAS_CHECK(expr)                                                          \
    do {                                                                            \
        cublasStatus_t _st = (expr);                                                \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                         \
            fprintf(stderr, "[CUBLAS ERROR] %s:%d %s -> %d\n",                      \
                    __FILE__, __LINE__, #expr, (int)_st);                           \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    sgemm::printBandwidthReference();
    printf("\n=== cuBLAS 参照: %d x %d x %d ===\n", M, N, K);

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

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, 0));

    const float alpha = 1.f, beta = 0.f;
    // 列主序陷阱的解法：把 C = A*B 的行主序问题换算成
    // "C^T = B^T * A^T 的列主序问题"，于是第一个参数传 B，第二个传 A，
    // 维度顺序变成 (N, M, K)，lda 分别是 N 和 K，ldc = N。
    auto launch = [&]() {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha, dB, N, dA, K, &beta, dC, N));
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
    sgemm::printRow("cuBLAS SGEMM", r);

    if (r.gbps > 0) {
        printf("\n提示：cuBLAS 的 GB/s 是按理论最小访存算的，所以它看起来\n");
        printf("      可能超过物理带宽 —— 这说明它靠分块把重复访存吃掉了。\n");
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
