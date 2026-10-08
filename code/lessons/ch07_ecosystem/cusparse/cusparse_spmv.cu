// ============================================================
//  cusparse_spmv.cu —— 用 cuSPARSE 做稀疏矩阵向量乘 y = A * x
//
//  稀疏矩阵用 CSR 格式存：
//    csrRowPtr : 长度为 M+1，第 i 行的非零元在 [csrRowPtr[i], csrRowPtr[i+1])
//    csrColInd : 每个非零元的列号
//    csrVal    : 每个非零元的值
//  CSR 是稀疏计算里最常用的格式，本程序手工构造一个三对角矩阵来演示。
//
//  编译（CUDA 11.0 及以上，用 generic API）：
//    nvcc -O3 -arch=native cusparse_spmv.cu -o cusparse_spmv -lcusparse
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cusparse.h>
#include "../../../common/cuda_check.h"
#include "../../../common/cuda_timer.h"   // 提供 CudaTimer（cudaEvent 计时 + 取中位数）

#define CUSPARSE_CHECK(expr)                                                        \
    do {                                                                            \
        cusparseStatus_t _s = (expr);                                               \
        if (_s != CUSPARSE_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "\n[CUSPARSE ERROR] %s:%d\n  call : %s\n  code : %d\n", \
                    __FILE__, __LINE__, #expr, (int)_s);                            \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

int main() {
    printDeviceInfo();

    // 构造一个 M x M 的三对角矩阵：主对角 = 2，上下次对角 = -1
    // 这是 1D 拉普拉斯算子的离散形式，是稀疏求解器最经典的测试用例
    const int M = 4096;
    std::vector<int> hRowPtr(M + 1);
    std::vector<int> hColInd;
    std::vector<float> hVal;
    hColInd.reserve(3 * M);
    hVal.reserve(3 * M);

    for (int i = 0; i < M; ++i) {
        hRowPtr[i] = (int)hColInd.size();
        if (i > 0) { hColInd.push_back(i - 1); hVal.push_back(-1.f); }
        hColInd.push_back(i);
        hVal.push_back(2.f);
        if (i + 1 < M) { hColInd.push_back(i + 1); hVal.push_back(-1.f); }
    }
    hRowPtr[M] = (int)hColInd.size();
    const int nnz = (int)hColInd.size();
    printf("\n稀疏矩阵: %d x %d，非零元 %d 个，密度 %.3f%%\n",
           M, M, nnz, 100.0 * nnz / ((double)M * M));

    // 向量 x：取 x[i] = i % 7，方便手算校验
    std::vector<float> hX(M), hY(M, 0.f);
    for (int i = 0; i < M; ++i) hX[i] = (float)(i % 7);

    // ---------- 1. 上传数据到显存 ----------
    int *dRowPtr = nullptr, *dColInd = nullptr;
    float *dVal = nullptr, *dX = nullptr, *dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dRowPtr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dColInd, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dVal, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dX, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dRowPtr, hRowPtr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dColInd, hColInd.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVal, hVal.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dY, 0, M * sizeof(float)));

    // ---------- 2. 创建句柄与矩阵/向量描述符 ----------
    cusparseHandle_t handle = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    // 默认就是 HOST 指针模式：alpha/beta 是主机上的普通变量。
    // 如果你想让 alpha 也放在显存里（例如它本身是一个 kernel 算出来的结果），
    // 就改成 CUSPARSE_POINTER_MODE_DEVICE，并把 &alpha 换成设备指针。
    CUSPARSE_CHECK(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, M, M, nnz,
                                     dRowPtr, dColInd, dVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, M, dX, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, M, dY, CUDA_R_32F));

    // ---------- 3. 申请临时工作区 ----------
    // 和 CUB 一样的两步走：先问大小，再分配。
    // 不同稀疏格式/算法需要的临时空间差别很大，必须按库的答复来分配。
    const float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecY,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                           &bufferSize));
    void* dBuffer = nullptr;
    if (bufferSize > 0) CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
    printf("cuSPARSE 请求的临时工作区 = %zu 字节\n", bufferSize);

    // ---------- 4. 执行 SpMV ----------
    CudaTimer timer;
    CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    CUDA_CHECK_SYNC();

    const int repeat = 50;
    timer.start();
    for (int r = 0; r < repeat; ++r) {
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY,
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    }
    float ms = timer.stop() / repeat;

    CUDA_CHECK(cudaMemcpy(hY.data(), dY, M * sizeof(float), cudaMemcpyDeviceToHost));

    // 有效带宽：SpMV 是典型的内存带宽受限算法
    double bytes = (double)nnz * (sizeof(float) + sizeof(int)) + (double)M * 2 * sizeof(float);
    printf("平均耗时 %.4f ms，有效带宽 %.1f GB/s\n", ms, bytes / (ms * 1e-3) / 1e9);

    // ---------- 5. 正确性校验（CPU 端按 CSR 逐行算一遍） ----------
    float maxAbs = 0.f;
    for (int i = 0; i < M; ++i) {
        float acc = 0.f;
        for (int p = hRowPtr[i]; p < hRowPtr[i + 1]; ++p)
            acc += hVal[p] * hX[hColInd[p]];
        maxAbs = fmaxf(maxAbs, fabsf(acc - hY[i]));
    }
    printf("与 CPU 结果的最大绝对误差 = %.3e  ->  %s\n", maxAbs, maxAbs < 1e-3f ? "通过 ✔" : "失败 ✘");
    printf("抽样：y[1] = %.4f（期望 2*x[1]-x[0]-x[2] = %.4f）\n", hY[1], 2.f * hX[1] - hX[0] - hX[2]);

    // ---------- 6. 收尾：描述符和句柄都要销毁 ----------
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    if (dBuffer) CUDA_CHECK(cudaFree(dBuffer));
    CUDA_CHECK(cudaFree(dRowPtr));
    CUDA_CHECK(cudaFree(dColInd));
    CUDA_CHECK(cudaFree(dVal));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    return EXIT_SUCCESS;
}
