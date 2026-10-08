// ============================================================================
//  code/lessons/ch05_numeric/numeric.cu
//
//  浮点求和的精度问题 —— 为什么"先把大数加起来"是错的
//    1) CPU 顺序求和（float）        —— 灾难：每一步都在丢低位
//    2) GPU 网格跨步 + 原子加（float）
//    3) GPU 分块求和（block reduce）+ double 累加
//    4) GPU Kahan 补偿求和（compensated summation）
//    5) GPU 全 double 累加
//
//  编译: nvcc -O3 -arch=sm_70 numeric.cu -o numeric
//  运行: ./numeric
//  注意: 本文件用了 atomicAdd(double*)，需要 Compute Capability >= 6.0
// ============================================================================
#include "../../common/cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int   N      = 1 << 24;  // 16,777,216 个 float = 64 MB
constexpr float VALUE  = 0.1f;     // 每个元素都一样，理论真值 = N * 0.1
constexpr int   BLOCK  = 256;

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

// ---------------------------------------------------------------------------
// 块内归约（warp shuffle + 共享内存），返回该块的部分和
// ---------------------------------------------------------------------------
__device__ __forceinline__ float blockReduceSum(float v) {
    __shared__ float s[32];  // 每个 warp 一个槽位
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);

    if (lane == 0) s[wid] = v;
    __syncthreads();

    const int nwarps = blockDim.x >> 5;
    if (wid == 0) {
        v = (lane < nwarps) ? s[lane] : 0.f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;  // 只有 tid==0 的返回值是有效的块总和
}

// ===========================================================================
// 版本 1：每个线程用 float 累加自己那一份，最后原子加到结果上
//   这是最"自然"的写法，也是精度最差的并行写法之一：
//   每个线程累加 N/gridDim 个数，累加器会涨到很大的值，
//   之后每加一个 0.1 都会因为 ulp 变大而被舍入掉。
// ===========================================================================
__global__ void sumFloatAtomic(const float* __restrict__ in, float* __restrict__ out, int n) {
    float sum = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += in[i];
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

// ===========================================================================
// 版本 2：分块求和 + double 累加
//   核心思想：把"一个大累加器"拆成"很多个小累加器"。
//   累加器越小，ulp 越小，每次加法的舍入误差就越小。
//   最后把块级别的部分和（数量只有几千个）用 double 汇总，
//   误差从 O(n) 级别降到 O(log n) 级别。
// ===========================================================================
__global__ void sumBlockDouble(const float* __restrict__ in, double* __restrict__ out, int n) {
    float sum = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += in[i];
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) atomicAdd(out, (double)sum);
}

// ===========================================================================
// 版本 3：Kahan 补偿求和
//   维护一个"补偿项" c，记录每次加法丢掉的那部分低位，
//   下一次加法先把它减掉，相当于把误差"捡回来"。
//   只用 float 就能把误差压到 1 个 ulp 量级，代价是每步 4 次浮点运算。
// ===========================================================================
__global__ void sumKahan(const float* __restrict__ in, double* __restrict__ out, int n) {
    float sum = 0.f;  // 累积和
    float c = 0.f;    // 补偿项（被舍入掉的那部分）
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float y = in[i] - c;      // 先把上次丢掉的部分加回去
        const float t = sum + y;        // 这个加法会丢低位
        c = (t - sum) - y;              // 精确地"找回"丢掉的部分
        sum = t;
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) atomicAdd(out, (double)sum);
}

// ===========================================================================
// 版本 4：全 double 累加（最省事、也最贵的正确做法）
//   double 有 53 位尾数，16M 次加法在这个量级上误差可以忽略。
//   代价：双精度吞吐通常只有单精度的 1/32（消费级卡）到 1/2（数据中心卡）。
// ===========================================================================
__global__ void sumDouble(const float* __restrict__ in, double* __restrict__ out, int n) {
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += (double)in[i];
    // double 的块内归约直接用共享内存做，简单起见不展开
    __shared__ double s[BLOCK];
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

int main() {
    printDeviceInfo();

    const int grid = 1024;  // 固定网格大小，让每个线程处理 N/grid/BLOCK ≈ 64 个元素
    std::vector<float> h(N, VALUE);

    // 理论真值：用 long double 算，这里 N*0.1 在 double 下几乎是精确的
    const double truth = (double)N * (double)VALUE;

    printf("\n[规模] N = %d，每个元素 = %.1ff，理论真值 = %.4f\n", N, VALUE, truth);

    // ---- CPU 顺序 float 求和 ----
    {
        float s = 0.f;
        for (int i = 0; i < N; ++i) s += h[i];
        double d = 0.0;
        for (int i = 0; i < N; ++i) d += (double)h[i];
        printf("\n[CPU] 顺序 float 求和 : %.4f   相对误差 %.3e\n", s,
               std::fabs((double)s - truth) / truth);
        printf("[CPU] 顺序 double 求和: %.4f   相对误差 %.3e\n", d,
               std::fabs(d - truth) / truth);
    }

    // ---- 准备设备内存 ----
    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    const size_t bytes = (size_t)N * sizeof(float);
    CudaTimer timer;

    // ---------- 1. float + 原子加 ----------
    {
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        float ms = 0.f;
        timer.start();
        sumFloatAtomic<<<grid, BLOCK>>>(d_in, d_out, N);
        ms = timer.stop();
        CUDA_CHECK_KERNEL();
        float got = 0.f;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("\n[GPU 1] float 原子加      : %.4f   相对误差 %.3e   (%6.4f ms, %.1f GB/s)\n",
               got, std::fabs((double)got - truth) / truth, ms, bytes / (ms * 1e-3) / 1e9);
        CUDA_CHECK(cudaFree(d_out));
    }

    // ---------- 2. 分块 + double ----------
    {
        double* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));
        timer.start();
        sumBlockDouble<<<grid, BLOCK>>>(d_in, d_out, N);
        const float ms = timer.stop();
        CUDA_CHECK_KERNEL();
        double got = 0.0;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(double), cudaMemcpyDeviceToHost));
        printf("[GPU 2] 分块 + double 累加: %.4f   相对误差 %.3e   (%6.4f ms, %.1f GB/s)\n",
               got, std::fabs(got - truth) / truth, ms, bytes / (ms * 1e-3) / 1e9);
        CUDA_CHECK(cudaFree(d_out));
    }

    // ---------- 3. Kahan ----------
    {
        double* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));
        timer.start();
        sumKahan<<<grid, BLOCK>>>(d_in, d_out, N);
        const float ms = timer.stop();
        CUDA_CHECK_KERNEL();
        double got = 0.0;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(double), cudaMemcpyDeviceToHost));
        printf("[GPU 3] Kahan 补偿求和    : %.4f   相对误差 %.3e   (%6.4f ms, %.1f GB/s)\n",
               got, std::fabs(got - truth) / truth, ms, bytes / (ms * 1e-3) / 1e9);
        CUDA_CHECK(cudaFree(d_out));
    }

    // ---------- 4. 全 double ----------
    {
        double* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));
        timer.start();
        sumDouble<<<grid, BLOCK>>>(d_in, d_out, N);
        const float ms = timer.stop();
        CUDA_CHECK_KERNEL();
        double got = 0.0;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(double), cudaMemcpyDeviceToHost));
        printf("[GPU 4] 纯 double 累加    : %.4f   相对误差 %.3e   (%6.4f ms, %.1f GB/s)\n",
               got, std::fabs(got - truth) / truth, ms, bytes / (ms * 1e-3) / 1e9);
        CUDA_CHECK(cudaFree(d_out));
    }

    printf("\n结论（这些数字会随机器变化，但量级关系是稳定的）：\n");
    printf("  * float 顺序/原子累加：误差可能在 1e-4 ~ 1e-2 量级，取决于数据规模；\n");
    printf("  * 分块求和 + double 汇总：误差立刻降到 1e-7 以下，几乎零成本；\n");
    printf("  * Kahan：只用 float 就能达到接近 double 的精度，代价是 4 倍指令数；\n");
    printf("  * 纯 double：最准，但在消费级显卡上带宽会掉到 1/2 以下。\n");
    printf("\n实用建议：默认用「float 分块累加 + double 汇总」；\n");
    printf("          只有当块内也需要极高精度时，才在块内启用 Kahan。\n");

    CUDA_CHECK(cudaFree(d_in));
    return 0;
}
