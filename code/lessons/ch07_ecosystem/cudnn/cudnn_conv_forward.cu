// ============================================================
//  cudnn_conv_forward.cu —— 用 cuDNN 做一次二维卷积前向
//
//  你需要先安装 cuDNN（与 CUDA 版本匹配），然后：
//    Linux : nvcc -O3 -arch=native cudnn_conv_forward.cu -o cudnn_conv \
//              -lcudnn -I/usr/local/cuda/include
//    Windows(开发者命令提示符，假设 CUDA_PATH 已设置):
//      nvcc -O3 -arch=native cudnn_conv_forward.cu -o cudnn_conv.exe ^
//        -I"%CUDA_PATH%\include" -L"%CUDA_PATH%\lib\x64" -lcudnn
//
//  cuDNN 的 API 有个固定套路，记住这 5 步就够用了：
//    1) 建句柄              cudnnCreate
//    2) 建张量/滤波器/卷积描述符（描述"形状"，不描述数据）
//    3) 让库帮你挑算法      cudnnGetConvolutionForwardAlgorithm_v7
//    4) 问算法要多少显存    cudnnGetConvolutionForwardWorkspaceSize
//    5) 真正执行            cudnnConvolutionForward
//
//  本文件用 cuDNN 8.x / 9.x 仍保留的 "legacy API"。
//  新项目推荐 cuDNN Frontend（header-only，更易用），
//  但 legacy API 更能让你看清"描述符"这个东西的本质。
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include <cudnn.h>
#include "../../../common/cuda_check.h"

#define CUDNN_CHECK(expr)                                                          \
    do {                                                                           \
        cudnnStatus_t _s = (expr);                                                 \
        if (_s != CUDNN_STATUS_SUCCESS) {                                          \
            fprintf(stderr, "\n[CUDNN ERROR] %s:%d\n  call : %s\n  msg  : %s\n",   \
                    __FILE__, __LINE__, #expr, cudnnGetErrorString(_s));           \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

int main() {
    printDeviceInfo();

    // ---------------- 问题规模 ----------------
    // N 张图，每张 C 通道，高 H 宽 W；卷积核 K 个输出通道，R x S
    const int N = 2, C = 3, H = 16, W = 16;
    const int K = 8, R = 3, S = 3;
    const int padH = 1, padW = 1;      // padding = 1 保证输出尺寸不变
    const int strideH = 1, strideW = 1;
    const int dilH = 1, dilW = 1;

    printf("\n卷积: 输入 %dx%dx%dx%d，核 %dx%dx%dx%d，pad=%d，stride=%d\n",
           N, C, H, W, K, C, R, S, padH, strideH);

    // ---------------- 准备数据 ----------------
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    const size_t xElems = (size_t)N * C * H * W;
    const size_t wElems = (size_t)K * C * R * S;
    std::vector<float> hX(xElems), hW(wElems);
    for (auto& v : hX) v = dist(rng);
    for (auto& v : hW) v = dist(rng);

    // ---------------- 1. 句柄 ----------------
    cudnnHandle_t handle = nullptr;
    CUDNN_CHECK(cudnnCreate(&handle));

    // ---------------- 2. 描述符 ----------------
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

    // NCHW 是最常见的布局，cuDNN 支持多种；这里只描述"形状"
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, C, H, W));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           K, C, R, S));
    // CROSS_CORRELATION = 深度学习里的"卷积"（不翻转核），别选成 CUDNN_CONVOLUTION
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideH, strideW,
                                                dilH, dilW, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // 输出形状让库来算，自己算很容易在 padding/stride 上出错
    int n_out = 0, c_out = 0, h_out = 0, w_out = 0;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc,
                                                      &n_out, &c_out, &h_out, &w_out));
    printf("输出形状: %dx%dx%dx%d\n", n_out, c_out, h_out, w_out);
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           n_out, c_out, h_out, w_out));

    // ---------------- 3. 分配显存 ----------------
    const size_t yElems = (size_t)n_out * c_out * h_out * w_out;
    float *dX = nullptr, *dW = nullptr, *dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, xElems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW, wElems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, yElems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), xElems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), wElems * sizeof(float), cudaMemcpyHostToDevice));

    // ---------------- 4. 挑算法 ----------------
    // 初始化一个"不可能达到"的性能值，否则比较时会误判
    cudnnConvolutionFwdAlgoPerf_t perf;
    int returned = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc,
                                                       1, &returned, &perf));
    printf("cuDNN 选择的算法 id = %d，预计耗时 %.4f ms\n",
           (int)perf.algo, perf.time);

    size_t wsBytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc,
                                                        yDesc, perf.algo, &wsBytes));
    printf("算法需要的 workspace = %zu 字节（%.2f MB）\n",
           wsBytes, wsBytes / 1024.0 / 1024.0);
    void* dWs = nullptr;
    if (wsBytes > 0) CUDA_CHECK(cudaMalloc(&dWs, wsBytes));

    // ---------------- 5. 执行 ----------------
    const float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK(cudaMemset(dY, 0, yElems * sizeof(float)));
    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, xDesc, dX,
                                        wDesc, dW, convDesc, perf.algo,
                                        dWs, wsBytes, &beta, yDesc, dY));
    CUDA_CHECK_SYNC();

    std::vector<float> hY(yElems);
    CUDA_CHECK(cudaMemcpy(hY.data(), dY, yElems * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------------- 6. 与 CPU 直接卷积对拍 ----------------
    // 3x3 卷积核 + 16x16 输入 + 2x3 通道，CPU 直接算完全够快
    std::vector<float> hRef(yElems, 0.f);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            for (int oh = 0; oh < h_out; ++oh)
                for (int ow = 0; ow < w_out; ++ow) {
                    float acc = 0.f;
                    for (int c = 0; c < C; ++c)
                        for (int r = 0; r < R; ++r)
                            for (int s = 0; s < S; ++s) {
                                int ih = oh * strideH - padH + r * dilH;
                                int iw = ow * strideW - padW + s * dilW;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;  // padding 补 0
                                float xv = hX[((size_t)n * C + c) * H * W + ih * W + iw];
                                float wv = hW[((size_t)k * C + c) * R * S + r * S + s];
                                acc += xv * wv;
                            }
                    hRef[((size_t)n * K + k) * h_out * w_out + oh * w_out + ow] = acc;
                }

    float maxAbs = 0.f;
    for (size_t i = 0; i < yElems; ++i) maxAbs = fmaxf(maxAbs, fabsf(hY[i] - hRef[i]));
    printf("与 CPU 直接卷积的最大绝对误差 = %.3e  ->  %s\n",
           maxAbs, maxAbs < 1e-4f ? "通过 ✔" : "失败 ✘");
    printf("样例输出 y[0] = %.6f，参考值 = %.6f\n", hY[0], hRef[0]);

    // ---------------- 7. 释放 ----------------
    if (dWs) CUDA_CHECK(cudaFree(dWs));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dY));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(wDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
    return EXIT_SUCCESS;
}
