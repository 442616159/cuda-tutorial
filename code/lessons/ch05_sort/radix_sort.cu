// ============================================================================
//  code/lessons/ch05_sort/radix_sort.cu
//
//  基数排序（Radix Sort）—— 用"稳定划分"取代"比较"
//    本文件实现的是最容易讲清楚的一种形式：**按位从低到高的稳定划分**
//    （bit-split / binary radix sort）。
//    每一趟做三件事：
//      1) 把第 b 位为 0 的元素标记出来；
//      2) 对这些标记做一次**独占前缀和**，得到每个 0 元素的新位置；
//      3) 稳定地散射到输出数组，再交换读写缓冲区。
//    关键点：每一趟都是**稳定**的，所以 32 趟之后数组按 32 位无符号数升序排列。
//
//    这也是"前缀和是并行算法的基石"的最好例证 —— 排序直接架在 scan 之上。
//
//  编译: nvcc -O3 -arch=sm_70 radix_sort.cu -o radix_sort
//  运行: ./radix_sort
// ============================================================================
#include "../../common/cuda_check.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int   BLOCK  = 256;
constexpr int   ITEMS  = 16;              // 每个线程在块级扫描里处理 16 个元素
constexpr int   TILE   = BLOCK * ITEMS;   // 每个块扫描 4096 个元素
constexpr int   N      = 1 << 16;         // 65536 个 uint32
constexpr int   NBITS  = 32;              // 32 位全排
constexpr int   REPEAT = 10;

// ===========================================================================
//  一、通用的 uint32 设备级独占扫描（三阶段）
// ===========================================================================

// ---- 阶段 1：块内 Blelloch 独占扫描 ----
__global__ void scanChunkU32(const uint32_t* __restrict__ in,
                             uint32_t* __restrict__ out,
                             uint32_t* __restrict__ chunkSum, int n) {
    __shared__ uint32_t s[TILE];
    const int tid = threadIdx.x;
    const int base = blockIdx.x * TILE;

#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int idx = base + k * BLOCK + tid;
        s[k * BLOCK + tid] = (idx < n) ? in[idx] : 0u;
    }
    __syncthreads();

    int offset = 1;
    for (int d = TILE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        for (int i = tid; i < d; i += BLOCK) {
            const int ai = offset * (2 * i + 1) - 1;
            const int bi = offset * (2 * i + 2) - 1;
            s[bi] += s[ai];
        }
        offset <<= 1;
    }

    if (tid == 0) {
        chunkSum[blockIdx.x] = s[TILE - 1];
        s[TILE - 1] = 0u;
    }

    for (int d = 1; d < TILE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        for (int i = tid; i < d; i += BLOCK) {
            const int ai = offset * (2 * i + 1) - 1;
            const int bi = offset * (2 * i + 2) - 1;
            const uint32_t t = s[ai];
            s[ai] = s[bi];
            s[bi] += t;
        }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int idx = base + k * BLOCK + tid;
        if (idx < n) out[idx] = s[k * BLOCK + tid];
    }
}

// ---- 阶段 2：对块总和做一次单块独占扫描（要求块数 <= BLOCK）----
__global__ void scanChunkSumsU32(const uint32_t* __restrict__ in,
                                 uint32_t* __restrict__ out, int n) {
    __shared__ uint32_t s[BLOCK];
    const int tid = threadIdx.x;

    s[tid] = (tid < n) ? in[tid] : 0u;
    __syncthreads();

    int offset = 1;
    for (int d = BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            const int ai = offset * (2 * tid + 1) - 1;
            const int bi = offset * (2 * tid + 2) - 1;
            s[bi] += s[ai];
        }
        offset <<= 1;
    }
    if (tid == 0) s[BLOCK - 1] = 0u;
    for (int d = 1; d < BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            const int ai = offset * (2 * tid + 1) - 1;
            const int bi = offset * (2 * tid + 2) - 1;
            const uint32_t t = s[ai];
            s[ai] = s[bi];
            s[bi] += t;
        }
    }
    __syncthreads();
    if (tid < n) out[tid] = s[tid];
}

// ---- 阶段 3：把块偏移加回块内每个元素 ----
__global__ void addChunkOffsetsU32(uint32_t* __restrict__ d,
                                   const uint32_t* __restrict__ off, int n) {
    const int base = blockIdx.x * TILE;
    const uint32_t o = off[blockIdx.x];
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int i = base + k * BLOCK + threadIdx.x;
        if (i < n) d[i] += o;
    }
}

// 把三阶段打包成一个函数，radix sort 每一趟都要调用它
static void exclusiveScanU32(const uint32_t* d_in, uint32_t* d_out,
                             uint32_t* d_chunkSum, uint32_t* d_chunkOff, int n) {
    const int numChunks = (n + TILE - 1) / TILE;
    // 本程序的 n=65536，numChunks=16 <= BLOCK，可以直接用 GPU 扫块和
    scanChunkU32<<<numChunks, BLOCK>>>(d_in, d_out, d_chunkSum, n);
    scanChunkSumsU32<<<1, BLOCK>>>(d_chunkSum, d_chunkOff, numChunks);
    addChunkOffsetsU32<<<numChunks, BLOCK>>>(d_out, d_chunkOff, n);
    CUDA_CHECK_KERNEL();
}

// ===========================================================================
//  二、按位划分（bit-split）—— 一趟基数排序
// ===========================================================================

// 谓词：第 bit 位是 0 记 1，是 1 记 0
__global__ void splitPredicate(const uint32_t* __restrict__ key,
                               uint32_t* __restrict__ pred, int n, int bit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pred[i] = (((key[i] >> bit) & 1u) == 0u) ? 1u : 0u;
}

// 稳定散射：
//   prefix[i] = 第 i 个元素之前有多少个"位为 0"的元素 → 位为 0 的元素直接落在这里
//   i - prefix[i] = 第 i 个元素之前有多少个"位为 1"的元素
//   位为 1 的元素整体排在所有 0 元素之后，起点是 totalZeros
//   因为是按原下标顺序计算的，所以划分是**稳定**的 —— 这是基数排序正确性的根基
__global__ void splitScatter(const uint32_t* __restrict__ key,
                             const uint32_t* __restrict__ prefix,
                             uint32_t totalZeros,
                             uint32_t* __restrict__ out, int n, int bit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const uint32_t z = prefix[i];
    const bool isZero = (((key[i] >> bit) & 1u) == 0u);
    const uint32_t pos = isZero ? z : (totalZeros + ((uint32_t)i - z));
    out[pos] = key[i];
}

// ---------------------------------------------------------------------------
// 一趟划分：谓词 -> 前缀和 -> 散射
// ---------------------------------------------------------------------------
static void radixPassOneBit(const uint32_t* d_key, uint32_t* d_tmp,
                            uint32_t* d_pred, uint32_t* d_prefix,
                            uint32_t* d_chunkSum, uint32_t* d_chunkOff,
                            int n, int bit) {
    const int grid = (n + BLOCK - 1) / BLOCK;
    splitPredicate<<<grid, BLOCK>>>(d_key, d_pred, n, bit);
    exclusiveScanU32(d_pred, d_prefix, d_chunkSum, d_chunkOff, n);

    // 前缀和是独占的，所以"0 的总数" = 最后一个位置的前缀 + 最后一个元素自己的谓词
    uint32_t lastPrefix = 0, lastPred = 0;
    CUDA_CHECK(cudaMemcpy(&lastPrefix, d_prefix + n - 1, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lastPred, d_pred + n - 1, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    const uint32_t totalZeros = lastPrefix + lastPred;

    splitScatter<<<grid, BLOCK>>>(d_key, d_prefix, totalZeros, d_tmp, n, bit);
    CUDA_CHECK_KERNEL();
}

// ---------------------------------------------------------------------------
// 完整基数排序：32 位从低到高各划分一趟，缓冲区交替使用
// ---------------------------------------------------------------------------
static void radixSortU32(uint32_t* d_a, uint32_t* d_b, uint32_t* d_pred,
                         uint32_t* d_prefix, uint32_t* d_chunkSum,
                         uint32_t* d_chunkOff, int n, float* t_ms) {
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));

    uint32_t* src = d_a;
    uint32_t* dst = d_b;
    for (int bit = 0; bit < NBITS; ++bit) {
        radixPassOneBit(src, dst, d_pred, d_prefix, d_chunkSum, d_chunkOff, n, bit);
        std::swap(src, dst);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(t_ms, s, e));

    // 32 是偶数，最终结果落回 d_a，调用方从 d_a 读取即可
    if (src != d_a) CUDA_CHECK(cudaMemcpy(d_a, src, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

    cudaEventDestroy(s);
    cudaEventDestroy(e);
}

int main() {
    printDeviceInfo();

    // 用固定种子生成随机 key，保证每次运行结果可复现
    std::vector<uint32_t> h(N);
    srand(20240607);
    for (int i = 0; i < N; ++i)
        h[i] = ((uint32_t)rand() << 17) ^ ((uint32_t)rand() << 3) ^ (uint32_t)rand();

    std::vector<uint32_t> h_ref = h;
    std::sort(h_ref.begin(), h_ref.end());

    // 注意：TILE 必须整除 n，否则块和数组不能一次扫完
    static_assert(N % TILE == 0, "N 必须是 TILE 的整数倍");
    static_assert(N / TILE <= BLOCK, "块和数量必须不超过 BLOCK，才能用单块扫描");

    uint32_t *d_a = nullptr, *d_b = nullptr, *d_pred = nullptr, *d_prefix = nullptr;
    uint32_t *d_chunkSum = nullptr, *d_chunkOff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_pred, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_prefix, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_chunkSum, (N / TILE) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_chunkOff, (N / TILE) * sizeof(uint32_t)));

    float best = 1e30f;
    for (int r = 0; r < REPEAT; ++r) {
        CUDA_CHECK(cudaMemcpy(d_a, h.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        float ms = 0.f;
        radixSortU32(d_a, d_b, d_pred, d_prefix, d_chunkSum, d_chunkOff, N, &ms);
        if (ms < best) best = ms;
    }

    std::vector<uint32_t> got(N);
    CUDA_CHECK(cudaMemcpy(got.data(), d_a, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (got[i] != h_ref[i]) ++bad;

    printf("\n[正确性] bit-split radix sort(%d) vs std::sort: 不匹配 = %d  %s\n",
           N, bad, bad == 0 ? "[OK]" : "[FAIL]");
    printf("[性能]   %d 个 uint32 排序: %.4f ms  (%.1f M 元素/s)\n",
           N, best, (double)N / (best * 1e-3) / 1e6);
    printf("         共 32 趟，每趟 3+2 = 5 个 kernel，总计 160 次启动\n");
    printf("         在 RTX 3060 上实测约 1.5 ms 量级，你的机器可能不同。\n");

    printf("\n提示：生产代码不会用「每趟 1 位」的写法，而是\n");
    printf("      * 每趟处理 4~8 位（16~256 个桶），把趟数从 32 降到 4~8；\n");
    printf("      * 用「每块局部直方图 + 扫描 + 块内排名」一次完成散列（onesweep）；\n");
    printf("      * 或者直接用 CUB 的 cub::DeviceRadixSort（第 7 章会讲）。\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_pred));
    CUDA_CHECK(cudaFree(d_prefix));
    CUDA_CHECK(cudaFree(d_chunkSum));
    CUDA_CHECK(cudaFree(d_chunkOff));
    return 0;
}
