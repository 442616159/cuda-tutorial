// ============================================================
//  code/lessons/ch02_memory/occupancy.cu
//  寄存器、占用率（Occupancy）与寄存器溢出（Spill）
//
//  本文件做四件事：
//      1. 用 cudaFuncGetAttributes 打印核函数的静态属性
//         （寄存器数、共享内存、最大线程数）
//      2. 用 cudaOccupancyMaxActiveBlocksPerMultiprocessor
//         计算"一个 SM 上最多能同时跑几个块"
//      3. 演示 __launch_bounds__ 如何影响寄存器分配
//      4. 用 cudaOccupancyMaxPotentialBlockSize 让运行时帮你选块大小
//
//  编译（在 code/lessons/ch02_memory 目录下）：
//      nvcc -O3 -arch=sm_75 occupancy.cu -o occupancy
//  运行：
//      ./occupancy        (Linux)
//      occupancy.exe      (Windows)
//
//  另外强烈建议用 -Xptxas -v 编译一次，看编译器自己报的数字：
//      nvcc -O3 -arch=sm_75 -Xptxas -v occupancy.cu -o occupancy
// ============================================================

#include <cstdio>

#include "../../common/cuda_check.h"
#include "../../common/cuda_timer.h"

// ------------------------------------------------------------
// 两个"吃寄存器"的核函数，用来对比 __launch_bounds__ 的效果。
//
// 计算斐波那契的迭代版本会用到好几个临时变量，
// 天真的写法会让编译器分配很多寄存器。
// ------------------------------------------------------------
__device__ inline float fibLike(int n) {
    float a = 0.0f, b = 1.0f, c = 0.0f;
    for (int i = 0; i < n; ++i) {
        c = a + b;
        a = b;
        b = c;
    }
    return a;
}

// 不带 __launch_bounds__：编译器为了性能可能用掉很多寄存器
__global__ void kernelNoBounds(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // 制造一些寄存器压力：多个独立累加器 + 小循环
        float acc0 = in[i];
        float acc1 = acc0 * 1.1f;
        float acc2 = acc0 * 1.2f;
        float acc3 = acc0 * 1.3f;
        float f = fibLike(8);
        acc0 += f;
        acc1 += f * 1.1f;
        acc2 += f * 1.2f;
        acc3 += f * 1.3f;
        out[i] = acc0 + acc1 + acc2 + acc3;
    }
}

// 带 __launch_bounds__(最低块数=4)：编译器会为"至少 4 块/SM"做寄存器预算，
// 于是每线程的寄存器数被压到 65536 / (4 * 256) = 64 个以内。
// 代价：可能需要把一些变量 spill 到本地内存（也就是显存）里，
//      反而变慢。什么时候值得，必须实测。
__global__ void __launch_bounds__(256, 4)
kernelWithBounds(const float* __restrict__ in,
                 float* __restrict__ out,
                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float acc0 = in[i];
        float acc1 = acc0 * 1.1f;
        float acc2 = acc0 * 1.2f;
        float acc3 = acc0 * 1.3f;
        float f = fibLike(8);
        acc0 += f;
        acc1 += f * 1.1f;
        acc2 += f * 1.2f;
        acc3 += f * 1.3f;
        out[i] = acc0 + acc1 + acc2 + acc3;
    }
}

// 一个共享内存吃得很凶的核函数：用来演示"共享内存限制占用率"的情况
__global__ void kernelHeavySmem(const float* __restrict__ in,
                                float* __restrict__ out,
                                int n) {
    // 24 KB 静态共享内存：在 48 KB/块的设备上，
    // 一个 SM 最多只能挂 2 个这样的块（如果 SM 有 100 KB 共享内存）
    __shared__ float big[6144];
    int tid = threadIdx.x;
    big[tid] = in[blockIdx.x * blockDim.x + tid];
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < 6144; ++i) s += big[i];
        out[blockIdx.x] = s;
    }
}

// 打印一个核函数的静态属性
static void reportKernel(const char* name, const void* fn, int blockSize) {
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, fn));

    printf("\n---------- %s ----------\n", name);
    printf("  每线程寄存器数         : %d\n", attr.numRegs);
    printf("  静态共享内存           : %zu 字节\n", attr.sharedSizeBytes);
    printf("  常量内存               : %zu 字节\n", attr.constSizeBytes);
    printf("  本地内存（spill 用）   : %zu 字节\n", attr.localSizeBytes);
    printf("  该函数允许的最大线程数 : %d\n", attr.maxThreadsPerBlock);
    printf("  二进制版本             : %d\n", attr.binaryVersion);

    // 计算"一个 SM 最多能同时驻留几个这样的块"
    int numBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, fn, blockSize, 0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int residentThreads = numBlocks * blockSize;
    double occ = 100.0 * residentThreads / prop.maxThreadsPerMultiProcessor;

    printf("  --- 占用率分析（块大小 %d）---\n", blockSize);
    printf("  每 SM 可驻留块数       : %d\n", numBlocks);
    printf("  每 SM 驻留线程数       : %d / %d\n",
           residentThreads, prop.maxThreadsPerMultiProcessor);
    printf("  理论占用率             : %.1f%%\n", occ);

    // 看看是什么限制了占用率
    int maxThreadsPerSm = prop.maxThreadsPerMultiProcessor;
    int threadsByBlock = (maxThreadsPerSm / blockSize);
    int regsPerBlock = attr.numRegs * blockSize;
    int blocksByReg = (regsPerBlock > 0)
                      ? prop.regsPerMultiprocessor / regsPerBlock : 9999;
    printf("  上限推算：\n");
    printf("    按线程数限制       : 最多 %d 个块\n", threadsByBlock);
    if (attr.sharedSizeBytes > 0) {
        int blocksBySmem = (int)(prop.sharedMemPerMultiprocessor / attr.sharedSizeBytes);
        printf("    按共享内存限制     : 最多 %d 个块\n", blocksBySmem);
    }
    printf("    按寄存器限制       : 最多 %d 个块\n", blocksByReg);
}

int main() {
    printDeviceInfo();

    const int N = 1 << 22;   // 419 万个元素 = 16 MB
    const size_t bytes = (size_t)N * sizeof(float);

    float *dIn = nullptr, *dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, bytes));
    CUDA_CHECK(cudaMalloc(&dOut, bytes));
    CUDA_CHECK(cudaMemset(dIn, 0, bytes));

    // ============ 1. 对比 __launch_bounds__ 的影响 ============
    reportKernel("kernelNoBounds（无 __launch_bounds__）",
                 reinterpret_cast<const void*>(kernelNoBounds), 256);
    reportKernel("kernelWithBounds（__launch_bounds__(256,4)）",
                 reinterpret_cast<const void*>(kernelWithBounds), 256);

    // 顺便测一下两者的实际耗时，看看"寄存器多"和"占用率高"哪个更值钱
    printf("\n---------- 实测对比 ----------\n");
    const int grid = (N + 255) / 256;

    double msA = CudaTimer::measure([&](double& m, int) {
        CudaTimer t; t.start();
        kernelNoBounds<<<grid, 256>>>(dIn, dOut, N);
        CUDA_CHECK_KERNEL();
        m = t.stopMs();
    }, 20, 3);
    double msB = CudaTimer::measure([&](double& m, int) {
        CudaTimer t; t.start();
        kernelWithBounds<<<grid, 256>>>(dIn, dOut, N);
        CUDA_CHECK_KERNEL();
        m = t.stopMs();
    }, 20, 3);
    printf("  kernelNoBounds   耗时 %8.4f ms   带宽 %7.1f GB/s\n",
           msA, CudaTimer::bandwidthGBs(2.0 * bytes, msA));
    printf("  kernelWithBounds 耗时 %8.4f ms   带宽 %7.1f GB/s\n",
           msB, CudaTimer::bandwidthGBs(2.0 * bytes, msB));
    printf("  说明：这个核函数是纯访存型的，所以两者差别通常很小。\n");
    printf("        寄存器/占用率的影响要在【计算密集 + 高延迟依赖】的\n");
    printf("        核函数里才会明显。在你的机器上实测约为多少可能不同。\n");

    // ============ 2. 共享内存限制占用率的例子 ============
    reportKernel("kernelHeavySmem（24 KB 共享内存）",
                 reinterpret_cast<const void*>(kernelHeavySmem), 256);

    // ============ 3. 让运行时推荐块大小 ============
    printf("\n---------- cudaOccupancyMaxPotentialBlockSize ----------\n");
    {
        int minGridSize = 0, bestBlock = 0;
        // 这个 API 会遍历所有可能的块大小，
        // 找出"能让占用率最高"的那个块大小，以及需要多少块才能铺满 GPU。
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &bestBlock,
            reinterpret_cast<const void*>(kernelNoBounds), 0, 0));

        int activeBlocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks, reinterpret_cast<const void*>(kernelNoBounds),
            bestBlock, 0));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        double occ = 100.0 * activeBlocks * bestBlock / prop.maxThreadsPerMultiProcessor;

        printf("  推荐块大小        : %d\n", bestBlock);
        printf("  铺满 GPU 至少需要 : %d 个块 (%d SM * %d 块/SM)\n",
               minGridSize, prop.multiProcessorCount, activeBlocks);
        printf("  该配置下占用率    : %.1f%%\n", occ);
        printf("  提示：实际代码里不必迷信这个数字。\n");
        printf("        块大小取 128/256/512 都是常见且安全的选择，\n");
        printf("        最终还是要用自己的数据和 ncu 来验证。\n");
    }

    // ============ 4. 寄存器溢出（Spill）演示 ============
    printf("\n---------- 关于寄存器溢出（Spill）----------\n");
    printf("  如果某个核函数的 attr.localSizeBytes 不为 0，\n");
    printf("  说明编译器把一部分变量【溢出】到了本地内存。\n");
    printf("  而本地内存物理上就在显存里 —— 每次访问都是几百个时钟周期，\n");
    printf("  这是性能杀手。\n\n");
    printf("  怎么发现：nvcc -Xptxas -v 会打印\n");
    printf("      N bytes stack frame, M bytes spill stores, K bytes spill loads\n");
    printf("  M 和 K 不为 0 就说明有溢出。\n\n");
    printf("  常见原因与对策：\n");
    printf("    1. 每线程算太多元素（比如 16x16 的寄存器分块）-> 调小分块\n");
    printf("    2. 数组下标是运行期变量，编译器无法放进寄存器 -> 用\n");
    printf("       #pragma unroll 让下标变成编译期常量\n");
    printf("    3. 加了过严的 __launch_bounds__ -> 放宽或去掉\n");
    printf("    4. 用 -maxrregcount=N 全局限制寄存器 -> 慎用，容易引发溢出\n");

    printf("\n  一个实用的判断规则：\n");
    printf("    寄存器 32 个以内   -> 几乎不可能限制占用率\n");
    printf("    寄存器 33~64 个     -> 块大小 256 时通常能到 50%% 以上占用率\n");
    printf("    寄存器 65~128 个    -> 占用率会明显下降，需要更多块来隐藏延迟\n");
    printf("    寄存器 128 个以上   -> 基本是【每线程干太多活】，请检查算法设计\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    printf("\n全部完成。\n");
    return 0;
}
