// ============================================================
//  code/lessons/ch03_cooperative/reduce_cg.cu
//  第 3 章示例 9：Cooperative Groups 与「网格级同步」
//
//  普通 kernel 里，block 之间谁先谁后完全不可控，因此不能直接同步。
//  Cooperative Groups 的 grid_group 提供了一次真正的全网格屏障：
//      grid.sync()
//  代价是：必须用 cudaLaunchCooperativeKernel 启动，
//         并且网格规模不能超过「所有 SM 同时能装下的块数」。
//
//  本程序演示：
//    1. 检查设备是否支持协作启动
//    2. 用 grid.sync() 一次 kernel 完成整个数组的归约
//    3. cg::tiled_partition<32>() 把 block 切成 warp 级的 tile
//
//  编译： nvcc -O2 -arch=sm_70 reduce_cg.cu -o reduce_cg
//  运行： ./reduce_cg
// ============================================================
#include "../../common/cuda_check.h"
#include "../../common/cuda_bench.h"
#include <cooperative_groups.h>
#include <cstdio>
#include <cmath>

namespace cg = cooperative_groups;

#define BLOCK 256
#define MAX_GRID 4096

// 每个块的部分和；grid.sync() 之后由 0 号块汇总
__device__ float g_partial[MAX_GRID];

// ------------------------------------------------------------
// 网格级归约：一个 kernel 搞定整件事
// ------------------------------------------------------------
__global__ void reduceCooperative(const float* in, float* out, int n) {
    extern __shared__ float s[];                 // 动态共享内存，大小 = blockDim.x * 4

    // 一次调用拿到"本 grid"的句柄；它必须在所有线程上以相同方式取得
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // ---- 第 1 步：网格步长累加 ----
    float v = 0.f;
    for (int i = bid * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        v += in[i];
    }

    // ---- 第 2 步：块内归约（block.sync() 等价于 __syncthreads()，
    //             但写明了"同步的是这个 block"这个语义） ----
    s[tid] = v;
    block.sync();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        block.sync();
    }

    // ---- 第 3 步：每块交出一个部分和 ----
    if (tid == 0) g_partial[bid] = s[0];

    // ---- 第 4 步：全网格屏障 ----
    // 这一行是本章的重点：它保证"所有块都执行完上面的写"之后，
    // 才会有任何一个线程继续往下走。没有它，0 号块可能读到未写好的 g_partial。
    grid.sync();

    // ---- 第 5 步：0 号块汇总所有部分和 ----
    if (bid == 0) {
        float sum = 0.f;
        // 让 0 号块的 256 个线程分担 gridDim.x 个部分和
        for (int b = tid; b < gridDim.x; b += blockDim.x) {
            sum += g_partial[b];
        }
        s[tid] = sum;
        block.sync();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) s[tid] += s[tid + stride];
            block.sync();
        }
        if (tid == 0) *out = s[0];
    }
    // 注意：grid.sync() 必须被 grid 内所有线程执行，
    //       所以它绝对不能出现在任何 if / for 里面。
}

// ------------------------------------------------------------
// tiled_partition：把 block 切成固定大小的 tile
//   cg::tiled_partition<32>(block) 拿到的就是"一个 warp"的抽象，
//   它自带 shfl / shfl_down / sync / thread_rank 等成员，
//   不用再手写 0xffffffff 掩码，读起来也更清楚。
// ------------------------------------------------------------
__global__ void tiledPartitionDemo(const float* in, float* out, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[i] : 0.f;

    // 与 __shfl_down_sync(FULL_MASK, v, offset) 语义完全相同
    for (int offset = tile.size() / 2; offset > 0; offset >>= 1) {
        v += tile.shfl_down(v, offset);
    }

    int laneInTile = tile.thread_rank();          // 等价于 threadIdx.x & 31
    if (laneInTile == 0) {
        out[i >> 5] = v;                          // 每个 warp 写一个结果
    }
}

int main() {
    printf("=== Cooperative Groups 网格级归约演示 ===\n");

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("设备：%s（CC %d.%d，%d 个 SM）\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // ---------- 1. 检查协作启动支持 ----------
    int coopSupported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coopSupported,
                                      cudaDevAttrCooperativeLaunch, dev));
    printf("支持 cudaLaunchCooperativeKernel : %s\n",
           coopSupported ? "是" : "否（本示例将跳过协作启动部分）");

    const int n = 1 << 22;                       // 4M 个 float
    float* h_in = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) h_in[i] = 1.0f;

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    const size_t shmem = BLOCK * sizeof(float);

    if (coopSupported) {
        // 协作启动的硬约束：网格必须能被"一次性"全部放进 GPU。
        // 用 occupancy API 问出「每个 SM 能同时容纳多少个这样的块」，
        // 再乘以 SM 数量，就是本 kernel 允许的最大网格。
        int blocksPerSm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSm, reduceCooperative, BLOCK, shmem));
        int grid = blocksPerSm * prop.multiProcessorCount;
        if (grid > MAX_GRID) grid = MAX_GRID;
        printf("每 SM 可容纳 %d 个块 -> 最大协作网格 = %d\n", blocksPerSm, grid);

        // ---------- 正确性 ----------
        float zero = 0.f;
        CUDA_CHECK(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));

        void* args[] = {(void*)&d_in, (void*)&d_out, (void*)&n};
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)reduceCooperative, dim3(grid), dim3(BLOCK), args, shmem, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        float got = 0.f;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        float expect = (float)n;
        printf("\n[grid.sync 归约] 结果 = %.1f，期望 = %.1f，相对误差 = %.3e\n",
               got, expect, fabs((double)got - expect) / expect);

        // ---------- 计时 ----------
        double ms = medianMs([&] {
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)reduceCooperative, dim3(grid), dim3(BLOCK), args, shmem, 0));
        }, 3, 20);
        printf("[grid.sync 归约] 中位数耗时 = %.4f ms，有效带宽 = %.1f GB/s\n",
               ms, gbps((double)n * sizeof(float), ms));
        printf("（对比一下：普通的两段式归约要多启动一个 kernel，\n");
        printf("  这里虽然只启动一次，但 grid.sync 本身也有开销，\n");
        printf("  谁快取决于数据规模和网格大小，请自己动手测。）\n");
    }

    // ---------- 2. tiled_partition ----------
    {
        float* d_t = nullptr;
        CUDA_CHECK(cudaMalloc(&d_t, n * sizeof(float)));
        int grid = (n + BLOCK - 1) / BLOCK;
        tiledPartitionDemo<<<grid, BLOCK>>>(d_in, d_t, n);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        float head[32];
        CUDA_CHECK(cudaMemcpy(head, d_t, sizeof(head), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < 32; ++i) {
            if (fabsf(head[i] - 32.0f) > 1e-5f) ++bad;   // 每个 warp 32 个 1.0f
        }
        printf("\n[tiled_partition<32>] 前 32 个 warp 的结果错误数 = %d\n", bad);
        printf("  每个 warp 的和应为 32.0\n");
        CUDA_CHECK(cudaFree(d_t));
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    printf("\n完成。\n");
    return 0;
}
