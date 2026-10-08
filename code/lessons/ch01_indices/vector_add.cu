// ============================================================
//  code/lessons/ch01_indices/vector_add.cu
//  向量加法 C = A + B —— 本书第一个"有用"的 CUDA 程序。
//
//  这个文件里放了两个核函数，用来对比两种线程映射策略：
//      vectorAddNaive    ：一个线程处理一个元素（必须 N <= 能启动的线程数）
//      vectorAddStrided  ：Grid-Stride Loop（任意 N 都能处理）
//
//  编译（在 code/lessons/ch01_indices 目录下）：
//      nvcc -O3 -arch=sm_75 vector_add.cu -o vector_add
//  运行：
//      ./vector_add            (Linux)
//      vector_add.exe          (Windows)
// ============================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_check.h"
#include "../../common/cuda_timer.h"

// ------------------------------------------------------------
// 版本一：最朴素的写法
//   每个线程负责一个元素，索引就是"全局唯一编号"。
//
//   全局编号 = 我前面有多少个块 * 每块多少线程 + 我在本块内的编号
//            = blockIdx.x * blockDim.x + threadIdx.x
//
//   这个写法好懂，但有个硬限制：
//   能启动的总线程数有上限（网格维度 x 每块线程数），
//   而且如果为了处理 1 亿个元素而真的启动 1 亿个线程，
//   光"创建/调度线程"的开销就不可忽略了。
// ------------------------------------------------------------
__global__ void vectorAddNaive(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 我的全局编号

    // 关键保护：启动的线程数通常是块大小的整数倍，
    // 一般会多于 n。多出来的线程必须直接返回，
    // 否则就会越界写坏别人的内存（这是新手最常见的崩溃原因）。
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// ------------------------------------------------------------
// 版本二：Grid-Stride Loop（网格-步长循环）
//   让"线程数"和"数据量"彻底解耦：
//   启动一定数量的线程（比如刚好铺满 GPU），
//   然后每个线程隔一段"总线程数"的距离去处理下一个元素。
//
//   "步长"(stride) = 整个网格的线程总数 = blockDim.x * gridDim.x
//
//   为什么这样反而更快？
//   因为线程的创建是免费的（硬件本来就按 warp 调度），
//   而"启动一个巨大的网格"会让每个线程执行的循环变短、
//   导致访存延迟无法被隐藏。固定一个够大的网格 + 循环，
//   能同时得到"访存规整"和"调度开销小"两个好处。
// ------------------------------------------------------------
__global__ void vectorAddStrided(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int n) {
    // 起始位置还是那个熟悉的全局编号
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 步长 = 一次"网格跨越"能覆盖的元素个数
    int stride = blockDim.x * gridDim.x;

    // 注意：循环条件 i < n 保证了不会越界，
    // 所以这个版本不需要额外的 if 保护。
    for (; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// ------------------------------------------------------------
// CPU 参考实现：用来验证 GPU 结果对不对。
// 任何 GPU 程序都必须有一个"可信的 CPU 版本"做对拍。
// ------------------------------------------------------------
static void vectorAddCPU(const std::vector<float>& A,
                         const std::vector<float>& B,
                         std::vector<float>& C) {
    for (size_t i = 0; i < A.size(); ++i) {
        C[i] = A[i] + B[i];
    }
}

// 逐元素比较，浮点必须用容差而不是 ==
static bool verify(const std::vector<float>& got,
                   const std::vector<float>& ref,
                   float tol = 1e-5f) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (fabs(got[i] - ref[i]) > tol) {
            fprintf(stderr, "结果不一致 @ i=%zu: got=%f ref=%f\n", i, got[i], ref[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printDeviceInfo();

    // N 取 1<<20 = 1048576（约 100 万）。
    // 三个数组各 4 MB，加上拷贝也远低于任何现代显卡的显存，
    // 保证了"小白机器也能跑"。
    const int N = 1 << 20;
    const size_t bytes = (size_t)N * sizeof(float);

    // ---- 1. 主机端准备数据 ----
    std::vector<float> hA(N), hB(N), hC(N, 0.0f), hRef(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        hA[i] = 1.0f * i;
        hB[i] = 2.0f * i;
    }
    vectorAddCPU(hA, hB, hRef);

    // ---- 2. 申请设备内存 ----
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // ---- 3. 主机 → 设备 拷贝 ----
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    // ---- 4. 决定执行配置 ----
    // 每块 256 个线程：是 32 的倍数（不会有半个 warp 浪费），
    // 又小到能让一个 SM 同时挂载多个块，是通用的安全选择。
    const int blockSize = 256;
    // （N + blockSize - 1) / blockSize 是最常用的"向上取整除法"，
    // 保证 ceil(N / blockSize) 个块能覆盖全部 N 个元素。
    const int gridSizeNaive = (N + blockSize - 1) / blockSize;

    printf("\n[朴素版]   N = %d, 块大小 = %d, 网格 = %d 个块, 共启动 %d 个线程\n",
           N, blockSize, gridSizeNaive, gridSizeNaive * blockSize);
    printf("          注意：启动了 %d 个线程，但只有 %d 个真干活，浪费了 %.2f%%\n",
           gridSizeNaive * blockSize, N,
           100.0 * (gridSizeNaive * blockSize - N) / (gridSizeNaive * blockSize));

    // ---- 5. 跑朴素版并计时 ----
    double msNaive = CudaTimer::measure([&](double& ms, int) {
        CudaTimer t;
        t.start();
        vectorAddNaive<<<gridSizeNaive, blockSize>>>(dA, dB, dC, N);
        CUDA_CHECK_KERNEL();
        ms = t.stopMs();  // stopMs 内部已经同步，所以测到的是真实设备耗时
    }, 10, 2);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    bool okNaive = verify(hC, hRef);
    printf("          正确性: %s,  实测中位耗时: %.4f ms\n",
           okNaive ? "通过" : "失败", msNaive);

    // ---- 6. 跑 Grid-Stride 版并计时 ----
    // 网格大小不再由 N 决定，而是由"设备能同时容纳多少线程"决定。
    // 这里用 SM 数量 * 每 SM 最大线程数 / 块大小，作为一个合理的上界。
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int maxBlocksPerSm = prop.maxThreadsPerMultiProcessor / blockSize;
    int gridSizeStrided = prop.multiProcessorCount * maxBlocksPerSm;
    // 再和"按 N 算出来的块数"取小，避免数据量很小时启动一堆空块
    if (gridSizeStrided > gridSizeNaive) gridSizeStrided = gridSizeNaive;
    if (gridSizeStrided < 1) gridSizeStrided = 1;

    printf("\n[Grid-Stride 版] 块大小 = %d, 网格 = %d 个块 (%d SM * %d 块/SM), 共 %d 个线程\n",
           blockSize, gridSizeStrided, prop.multiProcessorCount, maxBlocksPerSm,
           gridSizeStrided * blockSize);
    printf("          每个线程大约要循环 %d 次\n",
           (N + gridSizeStrided * blockSize - 1) / (gridSizeStrided * blockSize));

    std::fill(hC.begin(), hC.end(), 0.0f);
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    double msStrided = CudaTimer::measure([&](double& ms, int) {
        CudaTimer t;
        t.start();
        vectorAddStrided<<<gridSizeStrided, blockSize>>>(dA, dB, dC, N);
        CUDA_CHECK_KERNEL();
        ms = t.stopMs();
    }, 10, 2);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    bool okStrided = verify(hC, hRef);
    printf("          正确性: %s,  实测中位耗时: %.4f ms\n",
           okStrided ? "通过" : "失败", msStrided);

    // ---- 7. 计算有效带宽 ----
    // 一次加法要读 A、读 B、写 C，共 3 * N * 4 字节。
    // "有效带宽" = 实际搬运字节数 / 耗时，单位 GB/s。
    // 拿它和显卡的理论带宽对比，就能知道离峰值还有多远。
    double bytesMoved = 3.0 * bytes;
    printf("\n[有效性指标] 本次共搬运 %.1f MB\n", bytesMoved / 1024.0 / 1024.0);
    printf("  朴素版      有效带宽: %.1f GB/s\n",
           CudaTimer::bandwidthGBs(bytesMoved, msNaive));
    printf("  Grid-Stride 有效带宽: %.1f GB/s\n",
           CudaTimer::bandwidthGBs(bytesMoved, msStrided));
    printf("  （对比一下你的显卡理论带宽，能到 60%%~85%% 就算正常；\n");
    printf("    在哪张卡上实测约为多少，你的机器可能不同。）\n");

    // ---- 8. 释放设备内存 ----
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    printf("\n全部完成。\n");
    return 0;
}
