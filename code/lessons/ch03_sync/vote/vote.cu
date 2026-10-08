// ============================================================
//  code/lessons/ch03_sync/vote/vote.cu
//  第 3 章 示例 6：投票函数 __ballot_sync / __any_sync / __all_sync
//
//  本程序演示：
//    1. __ballot_sync 返回 32 位掩码 —— 用它做 warp 内流压缩（compaction）
//    2. 用 __popc 把掩码变成「排名」，再用共享内存把 warp 级结果拼成块级结果
//    3. __any_sync / __all_sync 的「整 warp 一问全答」语义
//
//  编译： nvcc -O2 -arch=sm_70 vote.cu -o vote
//  运行： ./vote
// ============================================================
#include "../../../common/cuda_check.h"
#include <cstdio>

#define BLOCK 256
#define FULL_MASK 0xffffffffu

// ------------------------------------------------------------
// 块级流压缩（Stream Compaction）
//   把数组里所有 >= 0 的元素，按原顺序紧凑地搬到输出数组前面。
//   这是并行算法里非常常用的一步（过滤、去重、稀疏化都靠它）。
//
//   核心技巧：
//     __ballot_sync(pred) 拿到「本 warp 哪些 lane 合格」的掩码；
//     __popc(mask)        得到本 warp 合格元素的个数；
//     __popc(mask & ((1u<<lane)-1)) 得到「我前面有多少个合格元素」，即我的排名。
// ------------------------------------------------------------
__global__ void blockCompact(const int* in, int* out, int* blockCount, int n) {
    __shared__ int warpCount[BLOCK / 32];    // 每个 warp 合格元素个数
    __shared__ int warpOffset[BLOCK / 32];   // 每个 warp 在块内的起始偏移

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    int i    = blockIdx.x * blockDim.x + tid;

    bool pred = (i < n) && (in[i] >= 0);

    // 一次投票拿到整个 warp 的谓词位图；同时它隐含一次 warp 级同步
    unsigned mask = __ballot_sync(FULL_MASK, pred);

    // 我在本 warp 合格元素里的排名（0-based）
    int rank = __popc(mask & ((1u << lane) - 1));
    if (lane == 0) warpCount[wid] = __popc(mask);

    __syncthreads();   // warpCount 要被块内所有线程看到

    // 每块只有 8 个 warp，用「一个线程串行前缀和」最划算
    if (tid == 0) {
        int sum = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) {
            warpOffset[w] = sum;
            sum += warpCount[w];
        }
        blockCount[blockIdx.x] = sum;
    }
    __syncthreads();   // warpOffset 要被块内所有线程看到

    if (pred) {
        // 块内顺序 = warp 顺序 + warp 内排名顺序，与原始下标顺序一致
        int pos = blockIdx.x * blockDim.x + warpOffset[wid] + rank;
        out[pos] = in[i];
    }
}

// ------------------------------------------------------------
// __any_sync / __all_sync
//   它们回答的是「整个 warp 的布尔问题」：
//     __any_sync(mask, pred)  —— 掩码里有没有任何一个 lane 的 pred 为真
//     __all_sync(mask, pred)  —— 掩码里是不是所有 lane 的 pred 都为真
//   返回值是 int（0 或 1），而不是掩码。
// ------------------------------------------------------------
__global__ void anyAllKernel(const int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = (i < n) ? in[i] : 0;

    bool hasNeg  = __any_sync(FULL_MASK, v < 0);
    bool allEven = __all_sync(FULL_MASK, (v & 1) == 0);

    // 只让每个 warp 的 lane 0 写回，结果数组长度是 n/32
    if ((i & 31) == 0 && i < n) {
        out[i >> 5] = (hasNeg ? 1 : 0) | (allEven ? 2 : 0);
    }
}

int main() {
    const int n = 1 << 20;
    printf("=== 投票函数演示 (n=%d, BLOCK=%d) ===\n", n, BLOCK);

    int* h_in  = (int*)malloc(n * sizeof(int));
    // 造数据：负数占约 1/3，其余为非负
    for (int i = 0; i < n; ++i) {
        h_in[i] = (i % 3 == 0) ? -(i + 1) : (i + 1);
    }

    int *d_in = nullptr, *d_out = nullptr, *d_cnt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cnt, ((n + BLOCK - 1) / BLOCK) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice));

    // ---------- 流压缩 ----------
    int grid = (n + BLOCK - 1) / BLOCK;
    blockCompact<<<grid, BLOCK>>>(d_in, d_out, d_cnt, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_out = (int*)malloc(n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    // 计算期望结果并与 GPU 输出比对
    int bad = 0, total = 0;
    for (int i = 0; i < n; ++i) {
        if (h_in[i] >= 0) {
            if (h_out[total] != h_in[i]) ++bad;
            ++total;
        }
    }
    printf("\n[流压缩] 合格元素 %d / %d，错误元素数 = %d\n", total, n, bad);
    printf("         输出前 6 个：");
    for (int i = 0; i < 6; ++i) printf("%d ", h_out[i]);
    printf("\n         对应输入中的前 6 个非负数：");
    int shown = 0;
    for (int i = 0; i < n && shown < 6; ++i) {
        if (h_in[i] >= 0) { printf("%d ", h_in[i]); ++shown; }
    }
    printf("\n");

    // ---------- any / all ----------
    int warps = n / 32;
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flags, warps * sizeof(int)));
    anyAllKernel<<<grid, BLOCK>>>(d_in, d_flags, n);
    CUDA_CHECK_KERNEL();

    int* h_flags = (int*)malloc(warps * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_flags, d_flags, warps * sizeof(int), cudaMemcpyDeviceToHost));

    int badFlags = 0;
    for (int w = 0; w < warps; ++w) {
        int hasNeg = 0, allEven = 1;
        for (int k = 0; k < 32; ++k) {
            int v = h_in[w * 32 + k];
            if (v < 0) hasNeg = 1;
            if (v & 1) allEven = 0;
        }
        int want = (hasNeg ? 1 : 0) | (allEven ? 2 : 0);
        if (h_flags[w] != want) ++badFlags;
    }
    printf("\n[__any_sync / __all_sync] %d 个 warp，错误个数 = %d\n", warps, badFlags);
    printf("  位 0 = 本 warp 存在负数，位 1 = 本 warp 全是偶数\n");
    printf("  前 8 个 warp 的标记位：");
    for (int w = 0; w < 8; ++w) printf("%d ", h_flags[w]);
    printf("\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_cnt));
    CUDA_CHECK(cudaFree(d_flags));
    free(h_in);
    free(h_out);
    free(h_flags);
    return 0;
}
