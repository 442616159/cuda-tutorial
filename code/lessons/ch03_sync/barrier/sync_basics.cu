// ============================================================
//  code/lessons/ch03_sync/barrier/sync_basics.cu
//  第 3 章 示例 1：__syncthreads() 到底做了什么
//
//  本程序演示三件事：
//    1. 不加屏障就读别的线程写进 shared memory 的数据 —— 结果不可靠
//    2. 加上 __syncthreads() 之后结果确定正确
//    3. __syncthreads_count / __syncthreads_and / __syncthreads_or
//       这三个「顺便投票」的屏障版本
//
//  编译： nvcc -O2 -arch=sm_70 sync_basics.cu -o sync_basics
//  运行： ./sync_basics
// ============================================================
#include "../../../common/cuda_check.h"
#include <cstdio>

#define N       1024          // 总元素数
#define BLOCK   256           // 每块线程数（必须是 32 的倍数，否则最后一个 warp 会被浪费）
#define GRID    (N / BLOCK)   // 块数

// ------------------------------------------------------------
// 版本 A：错误示范 —— 写完 shared memory 后立刻读邻居的数据，中间没有屏障
// ------------------------------------------------------------
__global__ void raceNoBarrier(int* out) {
    __shared__ int s[BLOCK];      // 每块一份，块内所有线程共享这块空间
    int t = threadIdx.x;

    s[t] = t;                     // 每个线程把自己那一格写成自己的编号

    // 这里故意不加 __syncthreads()。
    // 你可能会想「隔壁线程应该早就写完了」，但 GPU 的调度单位是 warp（32 线程），
    // 不同 warp 的推进速度完全独立：warp0 可能已经跑到这里，
    // 而 warp5 连 s[t]=t 都还没执行。此刻读 s[t^1] 读到的是未初始化垃圾。
    int neighbor = s[t ^ 1];

    out[blockIdx.x * blockDim.x + t] = neighbor;
}

// ------------------------------------------------------------
// 版本 B：正确做法 —— 加一道屏障
// ------------------------------------------------------------
__global__ void correctBarrier(int* out) {
    __shared__ int s[BLOCK];
    int t = threadIdx.x;

    s[t] = t;

    // __syncthreads() 同时承担两件事，缺一不可：
    //   (1) 屏障语义：块内每个线程都必须执行到这一行，谁都不能先往下走；
    //   (2) 可见性语义：屏障之前所有线程对 shared/global 的写，
    //       在屏障之后对块内所有线程都可见（编译器也不会把它挪过屏障）。
    __syncthreads();

    out[blockIdx.x * blockDim.x + t] = s[t ^ 1];
}

// ------------------------------------------------------------
// 版本 C：带返回值的三个屏障函数
//   __syncthreads_count(pred) : 返回块内 pred 为真的线程数
//   __syncthreads_and(pred)   : 块内 pred 全为真才返回非 0
//   __syncthreads_or(pred)    : 块内 pred 有一个为真就返回非 0
//   三者都隐含一次 __syncthreads()，所以不要把它们当成「零成本查询」。
// ------------------------------------------------------------
__global__ void blockVote(const int* in, int* cnt, int* allNonNeg, int* anyZero) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[g];                       // 每个线程取一个值

    int c = __syncthreads_count(v > 0);  // 本块有多少个正数
    int a = __syncthreads_and(v >= 0);   // 本块是不是全非负
    int o = __syncthreads_or(v == 0);    // 本块有没有 0

    // 与 warp 级投票函数不同，这三个函数给出的是「块」级结果，
    // 所以只需要一个线程负责写回即可。
    if (threadIdx.x == 0) {
        cnt[blockIdx.x]       = c;
        allNonNeg[blockIdx.x] = a;
        anyZero[blockIdx.x]   = o;
    }
}

// ------------------------------------------------------------
// 工具函数：统计数组里与期望值不符的元素个数
// ------------------------------------------------------------
static int countMismatch(const int* h, int n, int (*expected)(int)) {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        if (h[i] != expected(i)) ++bad;
    }
    return bad;
}

// 期望值：位置 i 属于块内线程 t = i % BLOCK，读到的是 t^1
static int expectedNeighbor(int i) { return (i % BLOCK) ^ 1; }

int main() {
    printf("=== __syncthreads 语义演示 (N=%d, BLOCK=%d, GRID=%d) ===\n", N, BLOCK, GRID);

    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

    int h_out[N];

    // ---------- 版本 A：无屏障 ----------
    raceNoBarrier<<<GRID, BLOCK>>>(d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    int badA = countMismatch(h_out, N, expectedNeighbor);
    printf("\n[无屏障] 与期望值不符的元素数 = %d / %d\n", badA, N);
    printf("         如果这次是 0，那只是运气好，不是程序正确。\n");
    printf("         换一张卡、换一个 block 大小、加个 -O3 就可能出错。\n");

    // ---------- 版本 B：有屏障 ----------
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(int)));
    correctBarrier<<<GRID, BLOCK>>>(d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    int badB = countMismatch(h_out, N, expectedNeighbor);
    printf("\n[有屏障] 与期望值不符的元素数 = %d / %d  (应该稳定为 0)\n", badB, N);

    // ---------- 版本 C：块级投票 ----------
    int h_in[N];
    for (int i = 0; i < N; ++i) {
        // 造一组有正有负、并且恰好有一部分为 0 的数据
        h_in[i] = (i % 37 == 0) ? 0 : ((i % 3 == 0) ? -i : i);
    }
    int *d_in = nullptr, *d_cnt = nullptr, *d_and = nullptr, *d_or = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cnt, GRID * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_and, GRID * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_or, GRID * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    blockVote<<<GRID, BLOCK>>>(d_in, d_cnt, d_and, d_or);
    CUDA_CHECK_KERNEL();

    int h_cnt[GRID], h_and[GRID], h_or[GRID];
    CUDA_CHECK(cudaMemcpy(h_cnt, d_cnt, GRID * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_and, d_and, GRID * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_or,  d_or,  GRID * sizeof(int), cudaMemcpyDeviceToHost));

    printf("\n[块级投票] 每个块 256 个线程\n");
    printf("  块号 | 正数个数 | 全非负 | 含 0\n");
    printf("  -----+----------+--------+-----\n");
    for (int b = 0; b < GRID && b < 4; ++b) {
        printf("  %4d | %8d | %6s | %4s\n",
               b, h_cnt[b], h_and[b] ? "是" : "否", h_or[b] ? "是" : "否");
    }
    printf("  (只打印前 4 个块)\n");

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_cnt));
    CUDA_CHECK(cudaFree(d_and));
    CUDA_CHECK(cudaFree(d_or));

    printf("\n完成。\n");
    return 0;
}
