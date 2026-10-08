// ============================================================
//  code/lessons/ch04_timing/timer/timer_demo.cu
//  第 4 章示例 5：把计时这件事做对
//
//  这个程序想让你亲眼看到三件事：
//    1. 只用一次 clock() 得到的数字有多不可信
//    2. CudaTimer（code/common/cuda_timer.h）该怎么用
//    3. 同一个 kernel 的耗时为什么还会上下浮动，以及该怎么汇报
//
//  编译： nvcc -O3 -arch=sm_70 timer_demo.cu -o timer_demo
//  运行： ./timer_demo
// ============================================================
#include "../../../common/cuda_check.h"
#include "../../../common/cuda_timer.h"
#include "../../../common/cuda_bench.h"

#include <cstdio>
#include <vector>
#include <algorithm>

#define BLOCK 256
#define N     (1 << 22)          // 4M 个 float

// y = a * x + y
__global__ void saxpyKernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    printf("=== 计时方法论演示 (N = %d) ===\n", N);

    size_t bytes = (size_t)N * sizeof(float);
    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemset(d_x, 0, bytes));
    CUDA_CHECK(cudaMemset(d_y, 0, bytes));

    int grid = (N + BLOCK - 1) / BLOCK;
    // FLOPs：每个元素 1 次乘法 + 1 次加法 = 2 次；再加索引计算，粗算 2*N
    double flops = 2.0 * N;
    // 字节：读 x + 读 y + 写 y = 3 * N * 4
    double movedBytes = 3.0 * bytes;

    auto launch = [&] { saxpyKernel<<<grid, BLOCK>>>(d_x, d_y, 2.0f, N); };

    // ------------------------------------------------------------
    // 1. 反面教材：只测一次
    // ------------------------------------------------------------
    printf("\n--- 1. 只测一次的后果 ---\n");
    {
        std::vector<double> singles;
        for (int k = 0; k < 8; ++k) {
            CudaTimer t;
            t.start();
            launch();
            singles.push_back((double)t.stopMs());
        }
        std::sort(singles.begin(), singles.end());
        printf("  连测 8 次的结果(ms)：");
        for (double v : singles) printf("%.4f ", v);
        printf("\n");
        printf("  最小 %.4f / 中位 %.4f / 最大 %.4f，最大是最小的 %.2f 倍\n",
               singles.front(), singles[singles.size() / 2], singles.back(),
               singles.back() / singles.front());
        printf("  这就是为什么「跑一次就下结论」是不可接受的。\n");
        printf("  第一次通常最慢：时钟还没升频、指令缓存还没热。\n");
    }

    // ------------------------------------------------------------
    // 2. CudaTimer 的手工用法
    // ------------------------------------------------------------
    printf("\n--- 2. CudaTimer 手工用法 ---\n");
    {
        // 先热身几次，再正式计时
        for (int k = 0; k < 3; ++k) launch();
        CUDA_CHECK(cudaDeviceSynchronize());

        CudaTimer timer;
        timer.start();
        for (int k = 0; k < 20; ++k) launch();      // 一次记 20 遍的总时间
        double total = timer.stopMs();
        printf("  20 次连续调用共 %.4f ms，平均每次 %.4f ms\n", total, total / 20.0);
        printf("  注意 timer.stopMs() 内部已经做了同步，\n");
        printf("  所以不需要再自己写 cudaDeviceSynchronize()。\n");
    }

    // ------------------------------------------------------------
    // 3. 规范测量：预热 + 多次 + 中位数
    // ------------------------------------------------------------
    printf("\n--- 3. 规范测量（预热 3 次，采样 21 次取中位数）---\n");
    double median = CudaTimer::measure(
        [&](double& ms, int /*it*/) {
            CudaTimer t;
            t.start();
            launch();
            ms = t.stopMs();
        },
        /*iters=*/21, /*warmup=*/3);

    printf("  中位数耗时 = %.4f ms\n", median);
    printf("  有效带宽   = %.1f GB/s   （搬运 %.1f MB）\n",
           CudaTimer::bandwidthGBs(movedBytes, median), movedBytes / 1048576.0);
    printf("  算力       = %.1f GFLOPS （做了 %.1f MFLOP）\n",
           CudaTimer::gflops(flops, median), flops / 1e6);
    printf("  参考：这是一个纯带宽受限的 kernel，\n");
    printf("        拿它的带宽和第 4 章 bandwidth_test 的 copy 结果对比一下，\n");
    printf("        数量级接近就说明你写对了。\n");

    // ------------------------------------------------------------
    // 4. cuda_bench.h 的 medianMs：更省事的写法
    // ------------------------------------------------------------
    printf("\n--- 4. medianMs 便捷封装 ---\n");
    {
        double ms = medianMs([&] { launch(); }, /*warmup=*/3, /*repeats=*/21);
        printf("  medianMs(...) = %.4f ms，有效带宽 %.1f GB/s\n",
               ms, gbps(movedBytes, ms));
        printf("  它和上面 CudaTimer::measure 得到的结果应当非常接近。\n");
    }

    // ------------------------------------------------------------
    // 5. 汇报性能时的三条规矩
    // ------------------------------------------------------------
    printf("\n--- 5. 汇报性能数据时的规矩 ---\n");
    printf("  1) 写清楚测的是什么：数据规模、块大小、网格大小、编译选项、显卡型号。\n");
    printf("  2) 写清楚怎么测的：预热几次、采样几次、取的是什么统计量。\n");
    printf("  3) 带上误差范围：比如「中位数 %.4f ms，8 次采样的极差在 ±5%% 内」，\n",
           median);
    printf("     而不是只报一个光秃秃的数字。\n");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    printf("\n完成。\n");
    return 0;
}
