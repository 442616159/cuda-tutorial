// ============================================================================
//  code/lessons/ch05_graph/bfs.cu
//
//  图算法雏形：层次化 BFS（Level-Synchronous Breadth-First Search）
//    * 邻接表用 CSR 存储（和第 5 章 SpMV 用的是同一套数据结构）
//    * frontier（当前层顶点集合）用数组 + 原子计数器维护
//    * 用 atomicCAS 保证每个顶点只被"第一次发现它的那个线程"写一次
//
//  编译: nvcc -O3 -arch=sm_70 bfs.cu -o bfs
//  运行: ./bfs
// ============================================================================
#include "../../common/cuda_check.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <queue>
#include <cstdint>

constexpr int   REPEAT = 20;
constexpr int   NV     = 1 << 20;  // 1M 个顶点
constexpr int   MAXDEG = 8;        // 每个顶点的最大出度（4 + u%5）

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
// 生成一个有向随机图（确定性伪随机，保证可复现）
//   出度 = 4 + u % 5，落在 [4, 8]，一共约 6M 条边
// ---------------------------------------------------------------------------
static void buildGraph(int n, std::vector<int>& rowPtr, std::vector<int>& colIdx) {
    rowPtr.assign(n + 1, 0);
    colIdx.clear();
    colIdx.reserve((size_t)n * MAXDEG);

    for (int u = 0; u < n; ++u) {
        const int deg = 4 + (u % 5);
        for (int j = 0; j < deg; ++j) {
            // 简单的线性同余打散，避免明显的局部结构
            const uint32_t h = (uint32_t)u * 2654435761u + (uint32_t)j * 40503u;
            colIdx.push_back((int)(h % (uint32_t)n));
        }
        rowPtr[u + 1] = (int)colIdx.size();
    }
}

// ---------------------------------------------------------------------------
// CPU 参考 BFS
// ---------------------------------------------------------------------------
static int bfsCpu(int n, const std::vector<int>& rowPtr, const std::vector<int>& colIdx,
                  int src, std::vector<int>& dist) {
    dist.assign(n, -1);
    std::queue<int> q;
    dist[src] = 0;
    q.push(src);
    int visited = 1;
    while (!q.empty()) {
        const int u = q.front();
        q.pop();
        for (int e = rowPtr[u]; e < rowPtr[u + 1]; ++e) {
            const int v = colIdx[e];
            if (dist[v] < 0) {
                dist[v] = dist[u] + 1;
                ++visited;
                q.push(v);
            }
        }
    }
    return visited;
}

// ===========================================================================
// 核心 kernel：把当前 frontier 的所有出边"推"出去
//
//   三个并发问题，逐个解决：
//     1) 多个线程可能同时发现同一个顶点 v
//        → 用 atomicCAS(&dist[v], -1, level+1) 做"抢占式标记"，
//          只有把 -1 换成 level+1 成功的那个线程才算发现了 v。
//        （如果只写 if (dist[v] == -1) 判断，会有两个线程都通过判断、
//          都往 nextFrontier 里塞一份 v，frontier 里就出现重复顶点。）
//     2) 往 nextFrontier 里写的位置不能重复
//        → 用 atomicAdd(nextSize, 1) 原子地领一个下标。
//     3) 每个顶点只处理一次
//        → 因为它只可能出现在一层的 frontier 里，这由 (1) 保证。
//
//   注意：这个版本是"边为中心"的 push 方向。
//   当 frontier 很大而图很稠密时，push 会做大量无效的边访问；
//   当 frontier 很小而图很大时，pull（每个顶点检查自己的邻居是否在 frontier）
//   反而更划算。这就是 Gunrock / Ligra 里"方向优化 BFS"的核心思想。
// ===========================================================================
__global__ void bfsExpand(const int* __restrict__ frontier, int frontierSize,
                          const int* __restrict__ rowPtr,
                          const int* __restrict__ colIdx,
                          int* __restrict__ dist,
                          int* __restrict__ nextFrontier,
                          int* __restrict__ nextSize,
                          int level) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= frontierSize) return;

    const int u = frontier[t];
    for (int e = rowPtr[u]; e < rowPtr[u + 1]; ++e) {
        const int v = colIdx[e];
        // 只有第一个把 dist[v] 从 -1 改掉的线程才处理 v
        if (atomicCAS(&dist[v], -1, level + 1) == -1) {
            const int pos = atomicAdd(nextSize, 1);
            nextFrontier[pos] = v;
        }
    }
}

// ===========================================================================
// 优化版：先在 shared memory 里攒一批，再一次性写全局
//   直接对全局 nextSize 做 atomicAdd 的话，frontier 大时全局原子争用会很严重。
//   这里每个线程先用寄存器/共享内存缓存自己的发现结果，
//   最后每个 block 只做一次全局 atomicAdd 领一段连续区间。
//   这里用更简单的做法：每个 warp 用一个 shared 计数器，减少原子冲突。
// ===========================================================================
__global__ void bfsExpandShared(const int* __restrict__ frontier, int frontierSize,
                                const int* __restrict__ rowPtr,
                                const int* __restrict__ colIdx,
                                int* __restrict__ dist,
                                int* __restrict__ nextFrontier,
                                int* __restrict__ nextSize,
                                int level) {
    // 每个 block 一个计数器，先攒在共享内存里
    __shared__ int sCount;
    __shared__ int sBase;
    if (threadIdx.x == 0) sCount = 0;
    __syncthreads();

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (t < frontierSize);
    const int u = active ? frontier[t] : -1;

    // 把本线程发现的新顶点先记在寄存器数组里
    int local[8];
    int localLen = 0;
    if (active) {
        for (int e = rowPtr[u]; e < rowPtr[u + 1]; ++e) {
            const int v = colIdx[e];
            if (atomicCAS(&dist[v], -1, level + 1) == -1 && localLen < 8)
                local[localLen++] = v;
        }
    }

    // 一次性向块内计数器申领 localLen 个连续下标
    int myBase = 0;
    if (localLen > 0) myBase = atomicAdd(&sCount, localLen);
    __syncthreads();

    // 由 0 号线程为整块向全局计数器申领一段区间
    if (threadIdx.x == 0) sBase = atomicAdd(nextSize, sCount);
    __syncthreads();

    for (int i = 0; i < localLen; ++i)
        nextFrontier[sBase + myBase + i] = local[i];
}

// ---------------------------------------------------------------------------
// 一次完整的 BFS（返回访问到的顶点数，并填好 dist）
// ---------------------------------------------------------------------------
static float bfsGpu(int n, const int* d_rowPtr, const int* d_colIdx, int src,
                    int* d_dist, int* d_frontier, int* d_next,
                    int* d_nextSize, int useShared, int* visitedOut, int* levelsOut) {
    CudaTimer timer;
    timer.start();

    CUDA_CHECK(cudaMemset(d_dist, 0xFF, (size_t)n * sizeof(int)));  // 全部置 -1
    int one = src, zero = 0;
    CUDA_CHECK(cudaMemcpy(d_dist + src, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier, &one, sizeof(int), cudaMemcpyHostToDevice));

    int frontierSize = 1;
    int level = 0;
    while (frontierSize > 0) {
        CUDA_CHECK(cudaMemset(d_nextSize, 0, sizeof(int)));
        const int grid = (frontierSize + 255) / 256;
        if (useShared)
            bfsExpandShared<<<grid, 256>>>(d_frontier, frontierSize, d_rowPtr, d_colIdx,
                                           d_dist, d_next, d_nextSize, level);
        else
            bfsExpand<<<grid, 256>>>(d_frontier, frontierSize, d_rowPtr, d_colIdx,
                                     d_dist, d_next, d_nextSize, level);
        CUDA_CHECK_KERNEL();

        // 每层都要把下一层的规模拷回 host：这是层次化 BFS 的固有开销，
        // 也是 nsys 时间线上那些"kernel 之间的小空档"的来源
        CUDA_CHECK(cudaMemcpy(&frontierSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

        int* tmp = d_frontier; d_frontier = d_next; d_next = tmp;
        ++level;
    }

    const float ms = timer.stop();
    *levelsOut = level - 1;  // 最后一轮 frontierSize==0 不算一层

    // 统计访问到的顶点数（dist >= 0）
    std::vector<int> h_dist(n);
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
    int visited = 0;
    for (int i = 0; i < n; ++i) if (h_dist[i] >= 0) ++visited;
    *visitedOut = visited;
    return ms;
}

int main() {
    printDeviceInfo();

    std::vector<int> rowPtr, colIdx;
    buildGraph(NV, rowPtr, colIdx);
    const int nnz = (int)colIdx.size();

    // ---- CPU 参考 ----
    std::vector<int> refDist;
    const int refVisited = bfsCpu(NV, rowPtr, colIdx, 0, refDist);

    printf("\n[图规模] 顶点 %d, 边 %d (%.2f M), 平均出度 %.2f\n",
           NV, nnz, nnz / 1e6, (double)nnz / NV);

    int *d_rowPtr = nullptr, *d_colIdx = nullptr, *d_dist = nullptr;
    int *d_f1 = nullptr, *d_f2 = nullptr, *d_nextSize = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (size_t)(NV + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)NV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_f1, (size_t)NV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_f2, (size_t)NV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextSize, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), (size_t)(NV + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(), (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));

    for (int variant = 0; variant < 2; ++variant) {
        float best = 1e30f;
        int visited = 0, levels = 0;
        std::vector<int> hDist;
        for (int r = 0; r < REPEAT; ++r) {
            int v = 0, l = 0;
            const float ms = bfsGpu(NV, d_rowPtr, d_colIdx, 0, d_dist, d_f1, d_f2,
                                    d_nextSize, variant, &v, &l);
            if (ms < best) best = ms;
            visited = v;
            levels = l;
        }
        hDist.resize(NV);
        CUDA_CHECK(cudaMemcpy(hDist.data(), d_dist, (size_t)NV * sizeof(int), cudaMemcpyDeviceToHost));

        int bad = 0;
        for (int i = 0; i < NV; ++i)
            if (hDist[i] != refDist[i]) ++bad;

        printf("[结果]   %-22s 访问 %d 顶点, %d 层, 不匹配 CPU 的顶点数 = %d  %s\n",
               variant == 0 ? "bfsExpand(原子)" : "bfsExpandShared(块内攒批)",
               visited, levels, bad, bad == 0 ? "[OK]" : "[FAIL]");
        printf("[性能]   %-22s %.3f ms  (%.1f 亿边/秒)\n",
               variant == 0 ? "bfsExpand(原子)" : "bfsExpandShared(块内攒批)",
               best, (double)nnz / ((double)best * 1e-3) / 1e8);
    }

    printf("\n[对照]   CPU 单线程 BFS 访问 %d 个顶点（同一个图，根都是顶点 0）\n", refVisited);
    printf("\n提示：在 RTX 3060 上 GPU 版 BFS 大约 0.5~1.5 ms 量级，你的机器可能不同。\n");
    printf("      层次化 BFS 的性能瓶颈通常不是算力，而是：\n");
    printf("        (1) 每一层一次 cudaMemcpy 回 host（约 5~10 us 的同步开销）；\n");
    printf("        (2) 层与层之间负载极不均衡（第一层只有 1 个顶点）；\n");
    printf("        (3) 当 frontier 变得很大时，push 方向会做大量无效边访问。\n");
    printf("      解决办法分别是：单 kernel 内做多轮（用原子屏障）、frontier 压缩、\n");
    printf("      以及 Gunrock / Ligra 里的方向优化（push 与 pull 自适应切换）。\n");

    CUDA_CHECK(cudaFree(d_rowPtr)); CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_dist)); CUDA_CHECK(cudaFree(d_f1));
    CUDA_CHECK(cudaFree(d_f2)); CUDA_CHECK(cudaFree(d_nextSize));
    return 0;
}
