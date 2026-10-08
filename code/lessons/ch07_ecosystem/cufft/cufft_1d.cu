// ============================================================
//  cufft_1d.cu —— 用 cuFFT 做一维复数 FFT，并与朴素 DFT 对拍
//
//  知识点：
//    1) cufftPlan1d / cufftExecC2C 的最小用法
//    2) cufftComplex 就是 float2，内存布局是 (re, im) 交错
//    3) 库的结果到底对不对：用 O(N^2) 的朴素 DFT 在小 N 上验证
//
//  编译：
//    nvcc -O3 -arch=native cufft_1d.cu -o cufft_1d -lcufft
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include <cufft.h>
#include <cuda_runtime.h>
#include "../../../common/cuda_check.h"

// 自己定义 pi：MSVC 下 M_PI 需要 _USE_MATH_DEFINES 才可见，
// 直接写常量最省事，也避免跨平台编译差异。
static const double kPi = 3.14159265358979323846;

// cuFFT 也有自己的状态码类型 cufftResult
#define CUFFT_CHECK(expr)                                                          \
    do {                                                                           \
        cufftResult _r = (expr);                                                   \
        if (_r != CUFFT_SUCCESS) {                                                 \
            fprintf(stderr, "\n[CUFFT ERROR] %s:%d\n  call : %s\n  code : %d\n",   \
                    __FILE__, __LINE__, #expr, (int)_r);                           \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

// ------------------------------------------------------------
// 填充输入数据：一个"双频正弦波"，方便肉眼看出频谱峰值
// ------------------------------------------------------------
__global__ void fillSignal(cufftComplex* x, int n, float f1, float f2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;  // 线程数通常多于数据量，必须做边界检查
    float t = (float)i / (float)n;
    float v = sinf(2.0f * 3.14159265f * f1 * t) + 0.5f * sinf(2.0f * 3.14159265f * f2 * t);
    x[i].x = v;  // 实部
    x[i].y = 0.f;  // 虚部：实数信号虚部为 0
}

// ------------------------------------------------------------
// CPU 朴素 DFT：O(N^2)，只用来在小 N 上验证 cuFFT
// X[k] = sum_{n} x[n] * exp(-2*pi*i*k*n/N)
// ------------------------------------------------------------
static void naiveDFT(const std::vector<cufftComplex>& in,
                     std::vector<cufftComplex>& out, int n) {
    for (int k = 0; k < n; ++k) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < n; ++j) {
            double ang = -2.0 * kPi * (double)k * (double)j / (double)n;
            double c = cos(ang), s = sin(ang);
            re += (double)in[j].x * c - (double)in[j].y * s;
            im += (double)in[j].x * s + (double)in[j].y * c;
        }
        out[k].x = (float)re;
        out[k].y = (float)im;
    }
}

int main() {
    printDeviceInfo();

    // 用小尺寸做验证：N=1024 时朴素 DFT 也只有百万次运算，瞬间完成
    const int N = 1024;
    printf("\n一维 FFT，N = %d\n\n", N);

    // ---------- 1. 建 plan ----------
    // plan 里保存了旋转因子表、分解方案等，属于"昂贵的准备工作"。
    // 正确做法是：数据和形状不变时，建一次 plan，反复 cufftExec。
    cufftHandle plan = 0;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));  // 1 = 批量个数（batch=1）

    // ---------- 2. 分配设备内存并填充 ----------
    cufftComplex* d_in = nullptr;
    cufftComplex* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(cufftComplex)));

    const int block = 256;
    const int grid = (N + block - 1) / block;
    fillSignal<<<grid, block>>>(d_in, N, 5.0f, 23.0f);
    CUDA_CHECK_KERNEL();

    // ---------- 3. 执行变换 ----------
    // 注意：cufftExecC2C 默认是异步的（和 kernel 一样），
    // 计时/取回结果前必须同步。
    CUFFT_CHECK(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));
    CUDA_CHECK_SYNC();

    // ---------- 4. 取回并验证 ----------
    std::vector<cufftComplex> hIn(N), hOut(N), hRef(N);
    CUDA_CHECK(cudaMemcpy(hIn.data(), d_in, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hOut.data(), d_out, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    naiveDFT(hIn, hRef, N);

    float maxRel = 0.f;
    // 只比较前 64 个频点，打印也一样，避免刷屏
    for (int k = 0; k < 64; ++k) {
        float dr = fabsf(hOut[k].x - hRef[k].x);
        float di = fabsf(hOut[k].y - hRef[k].y);
        float mag = fmaxf(1.f, sqrtf(hRef[k].x * hRef[k].x + hRef[k].y * hRef[k].y));
        maxRel = fmaxf(maxRel, fmaxf(dr, di) / mag);
    }
    printf("前 64 个频点与朴素 DFT 的最大相对误差 = %.3e  ->  %s\n",
           maxRel, maxRel < 1e-3f ? "通过 ✔" : "失败 ✘");

    // 找幅度最大的频点，应该正好落在 f1=5 与 f2=23 上
    int kMax = 0;
    float aMax = 0.f;
    for (int k = 0; k < N / 2; ++k) {
        float a = sqrtf(hOut[k].x * hOut[k].x + hOut[k].y * hOut[k].y);
        if (a > aMax) { aMax = a; kMax = k; }
    }
    printf("幅度最大的频点 k = %d（对应频率 %.0f），幅度 = %.1f\n",
           kMax, (float)kMax, aMax);
    printf("对照: 输入信号含 f1=5（幅度 1.0）与 f2=23（幅度 0.5），\n");
    printf("      单个正弦在 N 点 FFT 中的峰值幅度约为 N/2 * 该正弦幅度\n");

    // ---------- 5. 反向变换，应能还原原信号 ----------
    CUFFT_CHECK(cufftExecC2C(plan, d_out, d_out, CUFFT_INVERSE));  // 原地反向
    CUDA_CHECK_SYNC();
    CUDA_CHECK(cudaMemcpy(hOut.data(), d_out, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    // cuFFT 的正/反变换都不带 1/N 归一化，所以要在外面自己除
    float maxErr = 0.f;
    for (int i = 0; i < N; ++i) {
        float re = hOut[i].x / (float)N;
        maxErr = fmaxf(maxErr, fabsf(re - hIn[i].x));
    }
    printf("反变换还原信号的最大绝对误差 = %.3e（记得除以 N）\n", maxErr);

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return maxRel < 1e-3f ? EXIT_SUCCESS : EXIT_FAILURE;
}
