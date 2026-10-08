// ============================================================================
//  code/lessons/ch06_advanced/cluster/cluster_dsm.cu
//
//  线程块集群（Thread Block Cluster）与分布式共享内存（DSM）
//
//  ⚠ // 需要 Compute Capability >= 9.0（Hopper / Blackwell）
//    ⚠ 需要 CUDA Toolkit >= 11.8（cudaLaunchKernelEx 与集群启动属性）
//    编译时必须指定 -arch=sm_90
//
//  编译: nvcc -O3 -arch=sm_90 cluster_dsm.cu -o cluster_dsm
//  运行: ./cluster_dsm
//
//  集群解决了什么问题？
//    传统 CUDA 只能保证**块内**用 __syncthreads() 同步，
//    块与块之间唯一的同步手段是"结束这个 kernel，再启动下一个"。
//    集群允许最多 8 个（可移植上限；非可移植可到 16）线程块组成一个组：
//      * 组内可以 cluster.sync() 做硬件屏障；
//      * 组内每个块可以直接读写**别人的共享内存**（DSM）；
//      * 于是"块间通信"不再必须绕道全局内存，延迟低、带宽高。
//    它主要服务于"一个输出 tile 的 halo 需要来自邻居块"的场景：
//    分布式共享内存让 halo 交换不用写回显存。
// ============================================================================
#include "../../../common/cuda_check.h"

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if CUDART_VERSION >= 11080
#  define HAS_CLUSTER_API 1
#endif

namespace cg = cooperative_groups;

constexpr int CLUSTER_BLOCKS = 2;  // 每个集群 2 个块
constexpr int TPB = 256;
constexpr int N = 1 << 20;

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
// 集群内跨块归约
//   每个块先算出自己负责那段的局部和，写进自己的共享内存；
//   然后 cluster.sync() 做一次集群范围的硬件屏障；
//   最后由 rank 0 的块通过 map_shared_rank() 直接读其它块的共享内存求和。
//
//   注意两处 cluster.sync()：
//     第一处保证"别人的共享内存已经写好了"；
//     第二处（退出前）保证"别人都已经读完了，我的共享内存可以安全销毁"。
//     少了第二处，某个块提前退出会让还在读它共享内存的块读到垃圾。
// ===========================================================================
__global__ void clusterReduceKernel(const float* __restrict__ in,
                                    float* __restrict__ out, int n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cg::cluster_group cluster = cg::this_cluster();
    extern __shared__ float smem[];  // 至少 CLUSTER_BLOCKS 个 float 就够了

    const int tid = threadIdx.x;
    const unsigned rank = cluster.block_rank();   // 本块在集群内的编号

    // ---- 1. 每块算局部和（网格跨步 + warp shuffle + 共享内存）----
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        v += in[i];

    __shared__ float warpSum[32];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (lane == 0) warpSum[wid] = v;
    __syncthreads();

    const int nwarps = blockDim.x >> 5;
    if (wid == 0) {
        v = (lane < nwarps) ? warpSum[lane] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) smem[rank] = v;  // 每块占一个槽位，便于 rank 0 直接索引
    }

    // ---- 2. 集群屏障：保证所有块的 smem[rank] 都已经写好 ----
    cluster.sync();

    // ---- 3. 由 rank 0 的 0 号线程汇总 ----
    if (rank == 0 && tid == 0) {
        float total = 0.f;
        for (unsigned r = 0; r < cluster.num_blocks(); ++r) {
            // map_shared_rank 把"本块的共享内存地址"翻译成"第 r 个块的同一地址"
            const float* remote = cluster.map_shared_rank(smem, r);
            total += remote[r];
        }
        out[0] = total;
    }

    // ---- 4. 退出前再同步一次，防止别的块还在读我的共享内存 ----
    cluster.sync();
#endif
}

int main() {
    printDeviceInfo();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 9) {
        printf("\n[跳过] 本机 CC = %d.%d，线程块集群需要 CC >= 9.0。\n",
               prop.major, prop.minor);
        printf("       本文件主要用于说明概念，没有 Hopper 卡的话读完注释即可。\n");
        return 0;
    }

#ifndef HAS_CLUSTER_API
    printf("\n[跳过] CUDA Toolkit 版本过旧，需要 >= 11.8 才有 cudaLaunchKernelEx。\n");
    return 0;
#else
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f / (float)((i % 97) + 1);
    double ref = 0.0;
    for (int i = 0; i < N; ++i) ref += (double)h_in[i];

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // 网格大小必须是集群大小的整数倍：这里 64 个块 = 32 个集群
    const int numBlocks = 64;
    dim3 grid(numBlocks);
    dim3 block(TPB);
    const size_t smem = CLUSTER_BLOCKS * sizeof(float);

    // 集群是通过"启动属性"指定的，不需要在核函数上写 __cluster_dims__。
    // 两种写法二选一，不要同时用。
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = nullptr;

    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_BLOCKS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    printf("\n[配置] grid = %d 块, block = %d 线程, 集群 = %d 块/集群, "
           "共 %d 个集群\n", numBlocks, TPB, CLUSTER_BLOCKS,
           numBlocks / CLUSTER_BLOCKS);

    // 预热
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, clusterReduceKernel, d_in, d_out, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    float best = 1e30f;
    for (int r = 0; r < 20; ++r) {
        timer.start();
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, clusterReduceKernel, d_in, d_out, N));
        const float ms = timer.stop();
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float got = 0.f;
    CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    const double rel = (double)(got - ref) / ref;

    printf("[正确性] 集群归约结果 = %.3f，CPU 参考 = %.3f，相对误差 = %.2e  %s\n",
           got, ref, rel, (rel > -1e-3 && rel < 1e-3) ? "[OK]" : "[FAIL]");
    printf("[性能]   %.4f ms（含集群屏障与 DSM 读取）\n", best);

    printf("\n什么场景真正需要集群？\n");
    printf("  * 大 tile 的 stencil：halo 数据可以直接从邻居块的共享内存拿，\n");
    printf("    不用先写全局内存再读回来；\n");
    printf("  * 大矩阵乘：把 K 方向的规约拆到集群内多个块，用 DSM 交换部分和；\n");
    printf("  * 需要「块间同步一次，然后各干各的」的算法：多块协作的 scan、图遍历。\n");
    printf("注意：集群大小有上限（可移植 8，非可移植 16，需要显式设置\n");
    printf("      cudaFuncAttributeNonPortableClusterSizeAllowed），\n");
    printf("      而且它只在 Hopper 及以后才有；写成库时务必做能力检测。\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
#endif
}
