// ============================================================
//  gemm_api.cpp —— 同一个文件里同时提供 CPU 与 GPU 两条实现
//
//  这是"CPU 回退（CPU fallback）"最常见的工程写法：
//    * 用 nvcc 编译本文件时（定义了 __CUDACC__），走 GPU 分支
//    * 用普通 g++/MSVC 编译本文件时，走 CPU 分支
//  好处：上层代码完全不用改，接口一模一样；
//        在没有 NVIDIA 卡的机器上工程依然能编译、能跑（只是慢）。
//
//  CMake 里通过 set_source_files_properties(... LANGUAGE CUDA)
//  决定到底用哪个编译器处理这个文件。
// ============================================================

#include "gemm.h"

#include <cstdio>
#include <cstring>
#include <cmath>

// ------------------------------------------------------------
// 第一部分：CPU 实现。任何编译器都必须能编译这一段。
// ------------------------------------------------------------
void sgemmCPU(int M, int N, int K, const float* A, const float* B, float* C) {
    // 朴素三重循环：慢，但它是"正确性基准"，必须无条件可用
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

// ============================================================
// 第二部分：GPU 实现。只有 nvcc 会进来编译。
// ============================================================
#ifdef __CUDACC__

#include <cuda_runtime.h>

// 简单的共享内存分块 GEMM（详细优化见第 8 章）
// 为什么用模板参数 TILE？因为共享内存数组大小必须是编译期常量，
// 写成模板才能在编译期把 TILE*TILE 展开，避免动态分配带来的间接寻址。
template <int TILE>
__global__ void tileSgemmKernel(int M, int N, int K,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // 越界时填 0，而不是直接 return：
        // 因为块内所有线程都要参与 __syncthreads，提前退出会造成死锁
        As[ty][tx] = (row < M && t * TILE + tx < K) ? A[row * K + t * TILE + tx] : 0.f;
        Bs[ty][tx] = (col < N && t * TILE + ty < K) ? B[(t * TILE + ty) * N + col] : 0.f;
        __syncthreads();

        // 从共享内存里读 TILE 次，替代 TILE 次全局内存访问
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

void sgemmGPU(int M, int N, int K, const float* A, const float* B, float* C) {
    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (cudaMalloc(&dA, bytesA) != cudaSuccess) { fprintf(stderr, "cudaMalloc 失败\n"); return; }
    if (cudaMalloc(&dB, bytesB) != cudaSuccess) { fprintf(stderr, "cudaMalloc 失败\n"); return; }
    if (cudaMalloc(&dC, bytesC) != cudaSuccess) { fprintf(stderr, "cudaMalloc 失败\n"); return; }

    cudaMemcpy(dA, A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytesB, cudaMemcpyHostToDevice);

    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tileSgemmKernel<TILE><<<grid, block>>>(M, N, K, dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// 查询当前机器有没有可用的 NVIDIA GPU
bool gpuAvailable() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
    return count > 0;
}

const char* backendName() { return "CUDA (GPU)"; }

// ============================================================
// 第三部分：没有 CUDA 编译器时，接口仍然要存在，只是退回 CPU。
// ============================================================
#else

void sgemmGPU(int M, int N, int K, const float* A, const float* B, float* C) {
    // 静默退化会让人误以为"GPU 跑了但很慢"，所以这里必须给出提示
    fprintf(stderr, "[提示] 本二进制未编译 CUDA 支持，sgemmGPU 退回 CPU 实现\n");
    sgemmCPU(M, N, K, A, B, C);
}

bool gpuAvailable() { return false; }
const char* backendName() { return "CPU (无 CUDA 编译器)"; }

#endif  // __CUDACC__
