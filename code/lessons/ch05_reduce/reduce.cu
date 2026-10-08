// ============================================================================
//  code/lessons/ch05_reduce/reduce.cu
//
//  并行归约（Reduction）四步演进 —— 把 N 个数求和
//    v1 reduceInterleaved : 交错寻址 + 分支发散 + bank conflict（最朴素）
//    v2 reduceSequential  : 顺序寻址，消除整 warp 级发散与共享内存冲突
//    v3 reduceUnrolled    : 加载阶段每线程多元素 + 归约树完全展开
//    v4 reduceShuffle     : 用 __shfl_down_sync 取代最后 5 轮共享内存归约
//
//  编译: nvcc -O3 -arch=sm_70 reduce.cu -o reduce
//  运行: ./reduce            (Linux)
//        .\reduce.exe         (Windows)
//  分析: ncu --set full ./reduce
// ============================================================================
#include "../../common/cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int   N      = 1 << 20;  // 待求和元素个数 = 1,048,576
constexpr int   BLOCK  = 256;      // 每个线程块的线程数
constexpr int   REPEAT = 30;       // 计时重复次数（取最小值，抗系统抖动）

// ---------------------------------------------------------------------------
// CudaTimer：基于 cudaEvent 的 host 端计时器
//   为什么不用 CPU 时钟：kernel 是异步启动的，CPU 侧测到的只是"提交耗时"，
//   必须用 GPU 侧的事件才能量到真实的执行时间。
// ---------------------------------------------------------------------------
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

// ===========================================================================
// v1：交错寻址（Interleaved Addressing）—— 最直观、也最慢
// ===========================================================================
__global__ void reduceInterleaved(const float* __restrict__ in,
                                  float* __restrict__ partial) {
    __shared__ float s[BLOCK];
    const int tid = threadIdx.x;

    // 每个线程先从全局内存取 1 个元素放进共享内存，之后整个归约都在片上完成。
    // 这一步是**合并访存**：相邻线程读相邻地址，一个 warp 的 32 次读合成一次事务。
    s[tid] = in[blockIdx.x * BLOCK + tid];
    __syncthreads();

    // 交错寻址：stride=1 时是 (0,1)(2,3)(4,5)…，stride=2 时是 (0,2)(4,6)…
    // 两个致命问题：
    //   1) 判断条件 tid % (2*stride) == 0 让**同一条 warp 内只有一半线程活跃**，
    //      这就是 warp 发散；而且整数取模在 GPU 上并不便宜。
    //   2) stride=1 时活跃线程访问 s[0],s[2],s[4]…（bank 0,2,4…），
    //      属于 2-way bank conflict，共享内存带宽被白白浪费一半。
    for (int stride = 1; stride < BLOCK; stride <<= 1) {
        if (tid % (2 * stride) == 0)
            s[tid] += s[tid + stride];
        __syncthreads();  // 每一轮结束都必须同步：下一轮要读别的线程刚写的值
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// ===========================================================================
// v2：顺序寻址（Sequential Addressing）
//   把"活跃线程集合"从 {0,2,4,…} 换成 {0,1,2,…,stride-1}。
//   好处：
//     * 活跃线程正好落在**整数条 warp** 上，stride >= 32 的那几轮完全没有发散；
//     * 把昂贵的取模换成了一次简单的比较；
//     * 访问 s[tid] 与 s[tid+stride] 是两条独立的 load 指令，
//       每条指令内 32 个线程访问连续地址 → 32 个不同 bank，零冲突。
//   仍然存在的开销：每一轮一次 __syncthreads；最后 5 轮只有 32 个线程干活，
//   其余 224 个线程在 barrier 上空转。
// ===========================================================================
__global__ void reduceSequential(const float* __restrict__ in,
                                 float* __restrict__ partial) {
    __shared__ float s[BLOCK];
    const int tid = threadIdx.x;

    s[tid] = in[blockIdx.x * BLOCK + tid];
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// ===========================================================================
// v3：加载阶段展开 + 归约树完全展开（Unrolling）
//   两处改动，各自回答一个"原来慢在哪"：
//     (a) 加载阶段每线程串行累加 ELEMS 个元素。
//         线程块数从 4096 降到 1024 → 需要同步的块数和总同步次数降到 1/4；
//         同时 4 条独立 load 之间形成指令级并行（ILP），能更好地掩盖访存延迟。
//         注意 base + k*BLOCK 这种"跨步 1 个 block"的写法仍保证相邻线程读相邻
//         地址，合并访存没有被破坏。
//     (b) 归约树用 #pragma unroll 完全展开：stride 变成编译期常量，
//         循环控制与判断被消除，编译器可以把 __syncthreads 调度得更紧凑。
// ===========================================================================
__global__ void reduceUnrolled(const float* __restrict__ in,
                               float* __restrict__ partial) {
    __shared__ float s[BLOCK];
    const int tid = threadIdx.x;
    constexpr int ELEMS = 4;  // 每个线程在进入归约树前先加 4 个元素

    const int base = blockIdx.x * BLOCK * ELEMS + tid;
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < ELEMS; ++k)
        sum += in[base + k * BLOCK];

    s[tid] = sum;
    __syncthreads();

    // 完全展开的归约树；stride 是编译期常量，循环开销为零
#pragma unroll
    for (int stride = BLOCK / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    // 最后 64 -> 32 这一级之后，只剩 warp 0 的 32 个线程还活跃，
    // 它们之间的数据交换不需要 __syncthreads（同一 warp 天然锁步），
    // 但 Volta 之后线程可以独立调度，所以用 __syncwarp() 显式同步，
    // 否则读到的可能是别人还没写完的值。
    if (tid < 32) {
        s[tid] += s[tid + 32];
        __syncwarp();
#pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            s[tid] += s[tid + stride];
            __syncwarp();
        }
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// ===========================================================================
// v4：Warp Shuffle 归约 —— 终极版本
//   共享内存只负责把 BLOCK 个部分和折到 32 个（一条 warp 的量），
//   剩下 5 轮直接在**寄存器之间**用 __shfl_down_sync 交换，
//   连共享内存访问和 __syncthreads 都省掉了。
//   这是 CUB / Thrust 内部的做法。
// ===========================================================================
__global__ void reduceShuffle(const float* __restrict__ in,
                              float* __restrict__ partial) {
    __shared__ float s[BLOCK];
    const int tid = threadIdx.x;
    constexpr int ELEMS = 4;

    const int base = blockIdx.x * BLOCK * ELEMS + tid;
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < ELEMS; ++k)
        sum += in[base + k * BLOCK];

    s[tid] = sum;
    __syncthreads();

    // 阶段一：共享内存归约，直到只剩 32 个数为止
    for (int stride = BLOCK / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    // 阶段二：warp 内归约，全部走寄存器和 shuffle，零共享内存、零 __syncthreads
    if (tid < 32) {
        float v = s[tid];  // 此时 s[0..31] 已经汇总了全部 BLOCK 个元素
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (tid == 0) partial[blockIdx.x] = v;
    }
}

// ===========================================================================
// 第二阶段：把各个线程块的部分和再归约成最终结果
//   用"一个块 + 块内跨步循环"，所以对输入长度没有 2 的幂要求。
// ===========================================================================
__global__ void reduceFinal(const float* __restrict__ in, int n,
                            float* __restrict__ out) {
    __shared__ float s[BLOCK];
    const int tid = threadIdx.x;

    float sum = 0.f;
    for (int i = tid; i < n; i += BLOCK) sum += in[i];
    s[tid] = sum;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *out = s[0];
}

// ===========================================================================
// 计时 + 校验宏
// ===========================================================================
#define BENCH(NAME, KERNEL, GRID)                                                 \
    do {                                                                          \
        float best = 1e30f;                                                       \
        for (int r = 0; r < REPEAT; ++r) {                                        \
            timer.start();                                                        \
            KERNEL<<<(GRID), BLOCK>>>(d_in, d_partial);                           \
            CUDA_CHECK_KERNEL();                                                  \
            reduceFinal<<<1, BLOCK>>>(d_partial, (GRID), d_out);                  \
            float ms = timer.stop();                                              \
            if (ms < best) best = ms;                                             \
        }                                                                         \
        float got = 0.f;                                                          \
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost)); \
        const double rel = std::fabs((double)got - ref) / std::fabs(ref);          \
        printf("%-24s grid=%-6d %8.4f ms  %7.1f GB/s  和=%.1f  相对误差=%.2e  %s\n",\
               (NAME), (GRID), best,                                                  \
               (double)N * sizeof(float) / ((double)best * 1e-3) / 1e9,               \
               got, rel, (rel < 1e-4 ? "[OK]" : "[FAIL]"));                           \
    } while (0)

int main() {
    printDeviceInfo();

    // ---- 1. 准备数据：用 1/(i%97+1) 这种有正有负量级差异的数据，
    //         比全 1 更能暴露浮点累加的精度问题 ----
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f / (float)((i % 97) + 1);

    // CPU 端用 double 累加作为参考值（GPU 上是 float，必然有误差）
    double ref = 0.0;
    for (int i = 0; i < N; ++i) ref += (double)h_in[i];

    float* d_in = nullptr;
    float* d_partial = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, (N / BLOCK) * sizeof(float)));  // 最多 N/BLOCK 个块
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    printf("\nCPU 参考值 (double): %.6f\n\n", ref);

    CudaTimer timer;
    // v1/v2 每线程 1 个元素；v3/v4 每线程 4 个元素（块数少 4 倍）
    BENCH("v1 interleaved", reduceInterleaved, N / BLOCK);
    BENCH("v2 sequential", reduceSequential, N / BLOCK);
    BENCH("v3 unrolled", reduceUnrolled, N / (BLOCK * 4));
    BENCH("v4 shuffle", reduceShuffle, N / (BLOCK * 4));

    printf("\n提示：在 RTX 3060 上，v1 约 0.09 ms，v4 约 0.02 ms（你的机器可能不同）。\n");
    printf("      用 ncu --set full ./reduce 可以逐版本看到 warp stall 与 bank conflict 的变化。\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
