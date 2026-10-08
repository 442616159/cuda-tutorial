// ============================================================
//  code/lessons/ch01_indices/matrix_add.cu
//  二维矩阵加法 C = A + B，用二维线程块实现。
//
//  这是"二维索引"最直接的应用，也是后面矩阵乘法的前置练习。
//
//  编译（在 code/lessons/ch01_indices 目录下）：
//      nvcc -O3 -arch=sm_75 matrix_add.cu -o matrix_add
//  运行：
//      ./matrix_add        (Linux)
//      matrix_add.exe      (Windows)
// ============================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.h"
#include "../../common/cuda_timer.h"

// 每块 32x8 = 256 个线程。
// 为什么是 32 列宽？因为一个 warp 是 32 个线程，
// 让 warp 恰好覆盖一整行，这样同一 warp 内 32 个线程访问的是
// 矩阵同一行的 32 个连续元素 —— 完美的合并访存（第 2 章细讲）。
// 为什么是 8 行？256/32 = 8，块大小控制在 256 是通用最优区间。
#define TILE_X 32
#define TILE_Y 8

// ------------------------------------------------------------
// 二维网格 + 二维块的矩阵加法
//
// 行主序（row-major）存储：A[row][col] 的线性地址是 row * width + col。
// 所以"越界保护"要同时对 row 和 col 做检查。
// ------------------------------------------------------------
__global__ void matrixAdd2D(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int width, int height) {
    // 本线程负责的列下标：块内 x 方向铺开，再乘上块的 x 坐标
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 本线程负责的行下标：块内 y 方向铺开，再乘上块的 y 坐标
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 两个方向都要检查！只检查一个方向是新手常见的错误。
    if (row < height && col < width) {
        int idx = row * width + col;  // 展平成一维地址
        C[idx] = A[idx] + B[idx];
    }
}

// ------------------------------------------------------------
// 对照组：一维网格实现同一个矩阵加法。
// 两者功能完全一样，但线程"怎么分块"不同。
// 你可以用 -Xptxas -v 对比两者的寄存器用量，
// 再用计时对比两者在真实矩阵尺寸下的差别。
// ------------------------------------------------------------
__global__ void matrixAdd1D(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    printDeviceInfo();

    // 1024 x 1024 的矩阵，每个 4 MB，三个共 12 MB 显存。
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const size_t count = (size_t)WIDTH * HEIGHT;
    const size_t bytes = count * sizeof(float);

    std::vector<float> hA(count), hB(count), hC(count, 0.0f), hRef(count, 0.0f);
    for (int r = 0; r < HEIGHT; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            size_t idx = (size_t)r * WIDTH + c;
            hA[idx] = (float)(r + c);
            hB[idx] = (float)(r * 2 - c);
        }
    }
    for (size_t i = 0; i < count; ++i) hRef[i] = hA[i] + hB[i];

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    // ---------- 二维网格方案 ----------
    dim3 block(TILE_X, TILE_Y);                                    // 32 x 8 = 256 线程
    dim3 grid((WIDTH + TILE_X - 1) / TILE_X,                       // 列方向向上取整
              (HEIGHT + TILE_Y - 1) / TILE_Y);                     // 行方向向上取整

    printf("\n[二维网格] block=(%d,%d)=%d 线程, grid=(%d,%d)=%d 块, 共 %d 线程\n",
           block.x, block.y, block.x * block.y,
           grid.x, grid.y, grid.x * grid.y,
           block.x * block.y * grid.x * grid.y);
    printf("           矩阵 %d x %d = %zu 个元素，启动的线程比元素多 %zu 个（靠 if 保护）\n",
           WIDTH, HEIGHT, count,
           (size_t)block.x * block.y * grid.x * grid.y - count);

    double ms2D = CudaTimer::measure([&](double& ms, int) {
        CudaTimer t;
        t.start();
        matrixAdd2D<<<grid, block>>>(dA, dB, dC, WIDTH, HEIGHT);
        CUDA_CHECK_KERNEL();
        ms = t.stopMs();
    }, 10, 2);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    bool ok2D = true;
    for (size_t i = 0; i < count; ++i) {
        if (fabs(hC[i] - hRef[i]) > 1e-5f) { ok2D = false; break; }
    }
    printf("           正确性: %s,  实测中位耗时: %.4f ms,  有效带宽: %.1f GB/s\n",
           ok2D ? "通过" : "失败", ms2D,
           CudaTimer::bandwidthGBs(3.0 * bytes, ms2D));

    // ---------- 一维网格方案（同一份数据） ----------
    const int blockSize1D = 256;
    int gridSize1D = (int)((count + blockSize1D - 1) / blockSize1D);

    printf("\n[一维网格] block=%d, grid=%d, 共 %d 线程\n",
           blockSize1D, gridSize1D, blockSize1D * gridSize1D);

    std::fill(hC.begin(), hC.end(), 0.0f);
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    double ms1D = CudaTimer::measure([&](double& ms, int) {
        CudaTimer t;
        t.start();
        matrixAdd1D<<<gridSize1D, blockSize1D>>>(dA, dB, dC, (int)count);
        CUDA_CHECK_KERNEL();
        ms = t.stopMs();
    }, 10, 2);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    bool ok1D = true;
    for (size_t i = 0; i < count; ++i) {
        if (fabs(hC[i] - hRef[i]) > 1e-5f) { ok1D = false; break; }
    }
    printf("           正确性: %s,  实测中位耗时: %.4f ms,  有效带宽: %.1f GB/s\n",
           ok1D ? "通过" : "失败", ms1D,
           CudaTimer::bandwidthGBs(3.0 * bytes, ms1D));

    printf("\n结论提示：\n");
    printf("  对逐元素这种访存密集、计算极少的操作，两种映射的带宽往往很接近，\n");
    printf("  因为都已经是合并访存，瓶颈在显存带宽上。\n");
    printf("  二维网格的真正价值在于可读性，以及能自然表达二维邻域关系（如卷积、模板计算）。\n");
    printf("  具体差多少，在你的机器上实测约为多少可能不同，请以自己测到的为准。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    printf("\n全部完成。\n");
    return 0;
}
