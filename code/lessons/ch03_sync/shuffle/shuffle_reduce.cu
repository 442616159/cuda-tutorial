// ============================================================
//  code/lessons/ch03_sync/shuffle/shuffle_reduce.cu
//  第 3 章 示例 5：Warp Shuffle 四个函数全家桶
//
//  演示：
//    __shfl_sync      —— 按 lane 编号取值（广播 / 任意交换）
//    __shfl_down_sync —— 从「比我大 offset 的 lane」取值，做归约
//    __shfl_up_sync   —— 从「比我小 offset 的 lane」取值，做前缀和
//    __shfl_xor_sync  —— 与 lane^offset 交换，做蝶形全归约
//  并把 shuffle 归约与「纯 shared memory 归约」做一次计时对比。
//
//  编译： nvcc -O2 -arch=sm_70 shuffle_reduce.cu -o shuffle_reduce
//  运行： ./shuffle_reduce
// ============================================================
#include "../../../common/cuda_check.h"
#include <cstdio>
#include <cmath>

#define BLOCK 256

// 前置声明：prefix-sum kernel 的实现放在文件末尾
__global__ void scanKernel(const float* in, float* out);

// 0xffffffff 表示「warp 里全部 32 个 lane 都必须参与这次通信」。
// 所有 _sync 版本的 shuffle 都要求：掩码里点名的 lane 必须真的执行到这条指令，
// 否则结果是未定义的（不是崩溃，而是悄悄算错，更可怕）。
#define FULL_MASK 0xffffffffu

// ------------------------------------------------------------
// 1) warp 内归约：__shfl_down_sync
//    每一步把「lane + offset」的值拉过来加到自己身上。
//    算完之后，只有 lane 0 拥有完整的 32 个值之和。
// ------------------------------------------------------------
__inline__ __device__ float warpReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;                 // 只有 lane 0 的结果有意义
}

// ------------------------------------------------------------
// 2) 蝶形全归约：__shfl_xor_sync
//    与 lane^offset 交换，交换方向对称，
//    所以 5 步之后 32 个 lane 全都拿到总和。
// ------------------------------------------------------------
__inline__ __device__ float warpAllReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(FULL_MASK, v, offset);
    }
    return v;                 // 每个 lane 都拿到总和
}

// ------------------------------------------------------------
// 3) warp 内前缀和（含自身）：__shfl_up_sync
//    lane i 需要 lane i-1、i-2、i-4 ... 的值，所以用 shfl_up。
//    注意：越界的 lane 会拿到自己的值，必须用 if 判断后跳过，
//    否则会把自己加了两遍。
// ------------------------------------------------------------
__inline__ __device__ float warpInclusiveScan(float v) {
    int lane = threadIdx.x & 31;
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(FULL_MASK, v, offset);
        if (lane >= offset) v += n;
    }
    return v;
}

// ------------------------------------------------------------
// 4) 广播：__shfl_sync
//    把 lane 0 的值发给整个 warp（也可以取任意源 lane）。
// ------------------------------------------------------------
__global__ void broadcastKernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[i] : 0.f;
    // 源 lane 用运行时变量也可以，这让 shuffle 能做「任意跨 lane 取值」
    float first = __shfl_sync(FULL_MASK, v, 0);
    if (i < n) out[i] = first;
}

// ------------------------------------------------------------
// 5) 块级归约：第一个 warp 收尾（shuffle + shared memory 混合）
// ------------------------------------------------------------
__global__ void reduceShuffle(const float* in, float* out, int n) {
    __shared__ float warpSums[BLOCK / 32];
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    // 网格步长循环：让有限的线程处理任意长度的数组
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        v += in[i];
    }

    v = warpReduceSum(v);                        // 每个 warp 内部先归约
    if (lane == 0) warpSums[wid] = v;            // 每个 warp 只写一个数
    __syncthreads();                             // 跨 warp 必须靠屏障 + shared

    // 第一个 warp 负责把 8 个 warp 的部分和收拢
    v = (tid < (int)(blockDim.x >> 5)) ? warpSums[tid] : 0.f;
    if (wid == 0) v = warpReduceSum(v);
    if (tid == 0) atomicAdd(out, v);             // 各块之间用原子加汇总
}

// ------------------------------------------------------------
// 6) 对照组：只用 shared memory 的传统归约（不用 shuffle）
//    用来量化「shuffle 省掉共享内存往返」到底省了多少。
// ------------------------------------------------------------
__global__ void reduceSharedOnly(const float* in, float* out, int n) {
    extern __shared__ float s[];                 // 动态共享内存，大小 = blockDim.x * 4
    int tid = threadIdx.x;

    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        v += in[i];
    }
    s[tid] = v;
    __syncthreads();

    // 交错寻址版本：每轮减半，需要 log2(blockDim.x) 次屏障
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// ------------------------------------------------------------
// 计时小工具：预热 + 多次取中位数（第 4 章会封装成 CudaTimer 类）
// ------------------------------------------------------------
static float timeKernelMs(void (*launch)(const float*, float*, int, int, int),
                          const float* d_in, float* d_out, int n, int grid, int block,
                          int repeats) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热：让时钟频率、指令缓存都进入稳态，否则第一次一定偏慢
    launch(d_in, d_out, n, grid, block);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) launch(d_in, d_out, n, grid, block);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

static void launchShuffle(const float* in, float* out, int n, int grid, int block) {
    reduceShuffle<<<grid, block>>>(in, out, n);
    CUDA_CHECK_KERNEL();
}
static void launchShared(const float* in, float* out, int n, int grid, int block) {
    reduceSharedOnly<<<grid, block, block * sizeof(float)>>>(in, out, n);
    CUDA_CHECK_KERNEL();
}

int main() {
    const int n = 1 << 22;                   // 4M 个 float
    printf("=== Warp Shuffle 全家桶演示 (n=%d) ===\n", n);

    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) h[i] = 1.0f;   // 全 1，和就等于 n

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h, n * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int grid = prop.multiProcessorCount * 4;
    printf("设备 %s，grid = %d, block = %d\n\n", prop.name, grid, BLOCK);

    // ---------- 广播 ----------
    {
        float* d_b = nullptr;
        CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
        int blocks = (n + BLOCK - 1) / BLOCK;
        broadcastKernel<<<blocks, BLOCK>>>(d_in, d_b, n);
        CUDA_CHECK_KERNEL();
        // 只把前 64 个拷回来检查
        float head[64];
        CUDA_CHECK(cudaMemcpy(head, d_b, sizeof(head), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < 64; ++i) {
            // 第 i 个元素应该等于它所在 warp 的第一个元素 = h[(i/32)*32] = 1.0f
            if (fabsf(head[i] - 1.0f) > 1e-6f) ++bad;
        }
        printf("[__shfl_sync 广播]    前 64 个元素错误数 = %d\n", bad);
        CUDA_CHECK(cudaFree(d_b));
    }

    // ---------- 前缀和 ----------
    {
        float* d_scan_in = nullptr;
        float* d_scan_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_scan_in, 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scan_out, 256 * sizeof(float)));
        float hs[256];
        for (int i = 0; i < 256; ++i) hs[i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(d_scan_in, hs, sizeof(hs), cudaMemcpyHostToDevice));

        // 每个 warp 独立做前缀和（见文件末尾 scanKernel）
        scanKernel<<<1, 256>>>(d_scan_in, d_scan_out);
        CUDA_CHECK_KERNEL();

        float hr[256];
        CUDA_CHECK(cudaMemcpy(hr, d_scan_out, sizeof(hr), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < 256; ++i) {
            float want = (float)((i % 32) + 1);   // 每个 warp 内 1,2,3,...,32
            if (fabsf(hr[i] - want) > 1e-5f) ++bad;
        }
        printf("[__shfl_up_sync 前缀和] 256 个元素错误数 = %d  (期望每个 warp 内是 1..32)\n", bad);
        CUDA_CHECK(cudaFree(d_scan_in));
        CUDA_CHECK(cudaFree(d_scan_out));
    }

    // ---------- 归约正确性 ----------
    float zero = 0.f;
    CUDA_CHECK(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));
    reduceShuffle<<<grid, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    float sum = 0.f;
    CUDA_CHECK(cudaMemcpy(&sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    float expect = (float)n;
    printf("\n[shuffle 归约] 结果 = %.1f，期望 = %.1f，相对误差 = %.3e\n",
           sum, expect, fabs((double)sum - (double)expect) / expect);

    // ---------- 计时对比 ----------
    // 注意：这里只跑 10 次，数值仅供参考，正式测量请用第 4 章的 CudaTimer
    int rep = 20;
    float tShuffle = timeKernelMs(launchShuffle, d_in, d_out, n, grid, BLOCK, rep);
    float tShared  = timeKernelMs(launchShared,  d_in, d_out, n, grid, BLOCK, rep);

    double bytes = (double)n * sizeof(float);          // 只读一遍
    printf("\n--- 块级归约：shuffle vs 纯 shared memory（各跑 %d 次取平均）---\n", rep);
    printf("  shuffle + shared 收尾 : %8.4f ms  (有效带宽 %7.1f GB/s)\n",
           tShuffle, bytes / (tShuffle * 1e-3) / 1e9);
    printf("  纯 shared memory      : %8.4f ms  (有效带宽 %7.1f GB/s)\n",
           tShared, bytes / (tShared * 1e-3) / 1e9);
    printf("  实测约为 %.2f 倍差距（你的机器可能不同；\n", tShared / tShuffle);
    printf("  这个 kernel 是带宽受限，两者常常很接近，别期待巨大差距）。\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h);
    return 0;
}

// ------------------------------------------------------------
// 供主函数调用的前缀和 kernel（放在文件末尾，避免打断阅读节奏）
// ------------------------------------------------------------
__global__ void scanKernel(const float* in, float* out) {
    int t = threadIdx.x;
    float v = in[t];
    v = warpInclusiveScan(v);       // 每个 warp 内部先做前缀和
    out[t] = v;
}
