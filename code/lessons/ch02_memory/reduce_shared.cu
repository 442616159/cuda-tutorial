// ============================================================
//  code/lessons/ch02_memory/reduce_shared.cu
//  共享内存的实战：把 1677 万个 float 加起来。
//
//  三个版本，逐级消除瓶颈：
//      v1 reduceSharedStatic  ：每个线程多个元素 + 共享内存树形归约
//      v2 reduceSharedDynamic ：动态共享内存 extern __shared__
//      v3 reduceWarpShuffle   ：Volta 及以后用 __shfl_down_sync 省掉共享内存
//
//  这个例子最直接地展示了"共享内存的读写 + __syncthreads() 也会成为瓶颈"。
//
//  编译（在 code/lessons/ch02_memory 目录下）：
//      nvcc -O3 -arch=sm_75 reduce_shared.cu -o reduce_shared
//  运行：
//      ./reduce_shared        (Linux)
//      reduce_shared.exe      (Windows)
// ============================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.h"
#include "../../common/cuda_timer.h"

constexpr int BLOCK = 256;

// ------------------------------------------------------------
// v1: 静态共享内存 + 树形归约
//
// 分三步：
//   步骤 1（加载）：每个线程用 grid-stride 循环把多个元素累加到自己身上。
//                  这一步把全局内存访问量降到最低 —— 每个元素只读一次。
//   步骤 2（块内归约）：经典树形归约，每轮把规模减半，log2(256) = 8 轮。
//   步骤 3（写回）：每块写一个部分和。
//
// 关键细节：sdata 必须声明为 volatile。
//   原因是编译器有权假设"共享内存里的值不会被别人悄悄改掉"，
//   于是它可能把 sdata[tid + s] 的值缓存在寄存器里，
//   在 __syncthreads() 之后仍然读到旧值，结果就错了。
//   volatile 强制每次都真的去读共享内存。
//   （另一个办法是读完后手动 __syncwarp()，但 volatile 更直观。）
//
// __syncthreads() 放在 if 外面是安全的：
//   {tid < s} 里 s 对整个块是统一的，所以要么全块都进去、
//   要么全块都不进去，不会出现"部分线程永远等不到"的死锁。
// ------------------------------------------------------------
__global__ void reduceSharedStatic(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int n) {
    __shared__ volatile float sdata[BLOCK];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 步骤 1：本线程负责的那些元素先加到自己的寄存器里
    float sum = 0.0f;
    for (; i < n; i += stride) {
        sum += in[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // 步骤 2：树形归约。
    // 为什么从 256 一路减半到 1 要 log2(256) = 8 轮？
    // 因为每轮把参与线程数减半，这是 O(log n) 的经典套路。
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        // 每一轮都要同步，否则线程 tid 可能读到别人还没更新的值
        __syncthreads();
    }

    // 步骤 3：每块贡献一个部分和
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// ------------------------------------------------------------
// v2: 动态共享内存
//
// extern __shared__ float s[]; 声明的是"大小由启动时决定"的共享内存。
// 启动时用 <<<grid, block, sharedBytes>>> 的第三个参数指定字节数。
//
// 什么时候需要动态共享内存？
//   - 共享内存大小依赖运行期参数（比如"每块处理多少个元素"是算出来的）；
//   - 想让同一个 kernel 用不同的共享内存配置来跑；
//   - 多个数组需要复用同一段共享内存。
//
// 注意：动态共享内存只有一个"起点"。如果你需要多块不同大小的数组，
// 得自己用指针偏移手工切分：
//     extern __shared__ float smem[];
//     float* a = smem;
//     float* b = smem + aSize;
// ------------------------------------------------------------
__global__ void reduceSharedDynamic(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n) {
    // 注意这里没有第二个维度，大小在启动时才知道
    extern __shared__ volatile float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (; i < n; i += stride) {
        sum += in[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// ------------------------------------------------------------
// v3: 前 32 个元素只用 warp shuffle，不再走共享内存
//
// 观察 v1 的循环：当 s 降到 32 及以下时，参与运算的只有第 0 个 warp。
// 此时每个线程读写 sdata 都要走共享内存（几十个周期），
// 而且整个块都在等这一个 warp 慢慢做完。
//
// warp shuffle 让同一个 warp 内的线程直接交换寄存器值：
//   __shfl_down_sync(mask, value, delta)
//     -> 从"比本线程大 delta 号"的线程那里取 value
//   mask 是参与线程的掩码。0xffffffffu 表示全部 32 个线程都参与。
//
// 好处：零共享内存访问、零 __syncthreads()，延迟低得多。
// 这个技巧从 Kepler 就有，Volta 起要求显式写 mask。
// 完整的 warp 级原语会在第 3 章展开。
// ------------------------------------------------------------
__device__ float warpReduceSum(float val) {
    // 5 轮：32 -> 16 -> 8 -> 4 -> 2 -> 1
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    // 此时 lane 0（即 tid % 32 == 0 的那个线程）持有整个 warp 的和
    return val;
}

__global__ void reduceWarpShuffle(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n) {
    __shared__ float warpSums[BLOCK / 32];   // 8 个 warp，只要 8 个槽

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (; i < n; i += stride) {
        sum += in[i];
    }

    // 每个 warp 先把自己的 32 个数归约成一个
    sum = warpReduceSum(sum);

    // 只有每个 warp 的 0 号线程把结果写进共享内存
    if ((tid & 31) == 0) {
        warpSums[tid >> 5] = sum;
    }
    __syncthreads();

    // 第一个 warp 负责把 8 个 warp 的结果再归约一次
    if (tid < 32) {
        // 注意：只有 8 个有效值，其余位置要填 0，
        // 否则会把未初始化的共享内存（可能是垃圾）加进来
        float v = (tid < (BLOCK / 32)) ? warpSums[tid] : 0.0f;
        v = warpReduceSum(v);
        if (tid == 0) {
            out[blockIdx.x] = v;
        }
    }
}

int main() {
    printDeviceInfo();

    // N = 1<<24 = 16777216。所有元素都在 [0.5, 1.0) 之间，
    // 这样和大约是 N * 0.75 ≈ 1.26e7。float 的尾数只有 24 bit
    // （能精确表示约 1.7e7 以内的整数），所以单线程顺序累加
    // 会看出明显误差 —— 正好顺便讲讲精度问题。
    const int N = 1 << 24;
    const size_t bytes = (size_t)N * sizeof(float);

    std::vector<float> hIn(N);
    double exactSum = 0.0;
    for (int i = 0; i < N; ++i) {
        // 用一个整数散列来生成 [0.5, 1.0) 内的伪随机值
        float v = 0.5f + 0.5f * (float)((i * 2654435761u) % 1000) / 1000.0f;
        hIn[i] = v;
        exactSum += (double)v;
    }

    float* dIn = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, bytes));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    const int block = BLOCK;
    const int grid = 1024;   // 固定 1024 个块，靠循环覆盖全部数据
    std::vector<float> hOut(grid, 0.0f);
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, grid * sizeof(float)));

    printf("\n数组 %d 个 float = %.1f MB，元素范围 [0.5, 1.0)\n",
           N, bytes / 1024.0 / 1024.0);
    printf("CPU 用 double 累加得到的精确和 = %.6f\n", exactSum);
    printf("网格: %d 块 x %d 线程 = %d 个线程，每线程约处理 %d 个元素\n",
           grid, block, grid * block, N / (grid * block));
    printf("================================================================================\n");

    auto report = [&](const char* label, float* devOut, double ms) {
        // 把 1024 个部分和拷回来，用 double 在 CPU 上求和（保证比较公平）
        CUDA_CHECK(cudaMemcpy(hOut.data(), devOut, grid * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double sum = 0.0;
        for (int i = 0; i < grid; ++i) sum += (double)hOut[i];
        double relErr = fabs(sum - exactSum) / exactSum;
        printf("  %-26s 耗时 %7.4f ms   结果 %.4f   相对误差 %.3e   带宽 %6.1f GB/s\n",
               label, ms, sum, relErr,
               CudaTimer::bandwidthGBs((double)bytes, ms));
    };

    // ---------------- v1 ----------------
    {
        double ms = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            reduceSharedStatic<<<grid, block>>>(dIn, dOut, N);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);
        report("v1 静态共享内存", dOut, ms);
    }

    // ---------------- v2 ----------------
    {
        double ms = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            // 第三个模板参数是动态共享内存的字节数
            reduceSharedDynamic<<<grid, block, BLOCK * sizeof(float)>>>(dIn, dOut, N);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);
        report("v2 动态共享内存", dOut, ms);
    }

    // ---------------- v3 ----------------
    {
        double ms = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            reduceWarpShuffle<<<grid, block>>>(dIn, dOut, N);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);
        report("v3 warp shuffle", dOut, ms);
    }

    printf("================================================================================\n");
    printf("\n怎么解读：\n");
    printf("  1. 三个版本的瓶颈都在读那 64 MB 上，所以耗时差别通常不大。\n");
    printf("     有效带宽能到显卡理论带宽的 60%%~85%% 就算合格。\n");
    printf("  2. v3 的优势在共享内存访问与同步开销上。要看出这个优势，\n");
    printf("     需要让【读数据】的成本变小：把 N 改成 1<<18 再跑一遍，\n");
    printf("     此时 v3 相对 v1 的优势会明显得多（数据小到能留在 L2 里）。\n");
    printf("  3. 三个版本的相对误差都在 1e-6 ~ 1e-5 量级。这说明\n");
    printf("     并行归约加【分块求和】反而比【单线程顺序累加 float】更准，\n");
    printf("     因为每个线程只累加 N/(块数*线程数) 个元素，误差累积路径短。\n");
    printf("  4. 在你的机器上实测约为多少可能不同，请以自己测到的为准。\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    printf("\n全部完成。\n");
    return 0;
}
