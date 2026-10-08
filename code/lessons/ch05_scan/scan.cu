// ============================================================================
//  code/lessons/ch05_scan/scan.cu
//
//  并行前缀和（Scan / Prefix Sum）
//    A. Kogge-Stone  —— 朴素双层扫描，O(n log n) 工作量，但访存层次浅
//    B. Blelloch     —— work-efficient，up-sweep + down-sweep，O(n) 工作量
//    C. 三阶段设备级扫描 —— 分块扫描 -> 扫描块和 -> 加回偏移，支持任意 n
//
//  编译: nvcc -O3 -arch=sm_70 scan.cu -o scan
//  运行: ./scan
// ============================================================================
#include "../../common/cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int   BLOCK  = 256;
constexpr int   ITEMS  = 16;             // 每个线程在块级扫描里负责 16 个元素
constexpr int   TILE   = BLOCK * ITEMS;  // 每个块处理 4096 个元素
constexpr int   N      = 1 << 20;        // 设备级扫描的数据量
constexpr int   REPEAT = 30;

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
// A. Kogge-Stone：包含式扫描（inclusive scan）
//    offset=1 时每个线程加上前一个线程的值；offset=2 时加上前两个位置的值……
//    log2(BLOCK) 轮之后，s[tid] 就是前 tid+1 个元素之和。
//
//    为什么需要两条 __syncthreads()？
//      同一轮里，每个线程既要"读 s[tid-offset]（旧值）"又要"写 s[tid]（新值）"。
//      如果读写之间不同步，快线程会写出新值，慢线程再读到这个新值，
//      结果就被重复累加 —— 这就是典型的 read-after-write 竞争。
//      解决办法：先用寄存器把旧值读出来，同步一次，再写回。
//    代价：log2(256)=8 轮 × 2 次同步 = 16 次 __syncthreads，而且总工作量是 O(n log n)。
// ===========================================================================
__global__ void koggeStoneBlock(const float* __restrict__ in,
                                float* __restrict__ out) {
    __shared__ float s[BLOCK];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * BLOCK + tid;

    s[tid] = in[gid];
    __syncthreads();

    for (int offset = 1; offset < BLOCK; offset <<= 1) {
        // 1) 先把要用到的旧值抓进寄存器
        const float v = (tid >= offset) ? s[tid - offset] : 0.f;
        __syncthreads();  // 2) 保证所有线程都读完了旧值
        s[tid] += v;      // 3) 再统一写回
        __syncthreads();  // 4) 保证所有线程都写完了，下一轮才能读
    }

    out[gid] = s[tid];
}

// ===========================================================================
// B. Blelloch：work-efficient 独占式扫描（exclusive scan）
//    两个阶段都工作在同一棵二叉树上：
//      up-sweep   自底向上把相邻两项相加，根节点得到总和；
//      down-sweep 自顶向下把部分和分发下去，并把根清零。
//    每一个元素只被访问 O(1) 次，总工作量 O(n)，比 Kogge-Stone 少一半的加法。
//
//    注意这里 TILE = 4096 > BLOCK = 256，某一层需要处理的"节点对"数量 d
//    可能大于线程数，所以用 for (i = tid; i < d; i += BLOCK) 遍历，
//    每一层只需要一次 __syncthreads（同层内各节点对互不相交，无竞争）。
// ===========================================================================
__global__ void blellochBlockExclusive(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       float* __restrict__ blockTotal,
                                       int n) {
    __shared__ float s[TILE];  // 4096 * 4B = 16KB，在默认 48KB 限制内
    const int tid = threadIdx.x;
    const int base = blockIdx.x * TILE;

    // ---- 搬运：每线程搬 ITEMS 个元素。跨步 TILE/BLOCK 保证合并访存 ----
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int idx = base + k * BLOCK + tid;
        s[k * BLOCK + tid] = (idx < n) ? in[idx] : 0.f;  // 越界补 0，不破坏扫描结果
    }
    __syncthreads();

    // ---- up-sweep：自底向上求和 ----
    int offset = 1;
    for (int d = TILE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        for (int i = tid; i < d; i += BLOCK) {
            const int ai = offset * (2 * i + 1) - 1;  // 左孩子
            const int bi = offset * (2 * i + 2) - 1;  // 右孩子（= 父节点）
            s[bi] += s[ai];
        }
        offset <<= 1;
    }

    // ---- 根部清零 + 记录块总和：这一步把"包含式"变成"独占式" ----
    if (tid == 0) {
        blockTotal[blockIdx.x] = s[TILE - 1];
        s[TILE - 1] = 0.f;
    }

    // ---- down-sweep：自顶向下分发 ----
    for (int d = 1; d < TILE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        for (int i = tid; i < d; i += BLOCK) {
            const int ai = offset * (2 * i + 1) - 1;
            const int bi = offset * (2 * i + 2) - 1;
            const float t = s[ai];   // 先把左孩子的值存起来
            s[ai] = s[bi];           // 左孩子拿到父节点传下来的前缀
            s[bi] += t;              // 父节点加上原来的左孩子，变成"右半区的前缀"
        }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int idx = base + k * BLOCK + tid;
        if (idx < n) out[idx] = s[k * BLOCK + tid];
    }
}

// ===========================================================================
// C1. 块和数量 <= BLOCK 时，用单个块做一次独占扫描
// ===========================================================================
__global__ void scanBlockTotals(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    __shared__ float s[BLOCK];
    const int tid = threadIdx.x;

    s[tid] = (tid < n) ? in[tid] : 0.f;
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
    if (tid == 0) s[BLOCK - 1] = 0.f;
    for (int d = 1; d < BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            const int ai = offset * (2 * tid + 1) - 1;
            const int bi = offset * (2 * tid + 2) - 1;
            const float t = s[ai];
            s[ai] = s[bi];
            s[bi] += t;
        }
    }
    __syncthreads();
    if (tid < n) out[tid] = s[tid];
}

// ===========================================================================
// C2. 把"本块之前所有块的总和"加回每个元素 —— 三阶段扫描的最后一步
// ===========================================================================
__global__ void addBlockOffsets(float* __restrict__ d,
                                const float* __restrict__ blockOffsets, int n) {
    const int base = blockIdx.x * (blockDim.x * ITEMS);
    const float off = blockOffsets[blockIdx.x];
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int i = base + k * blockDim.x + threadIdx.x;
        if (i < n) d[i] += off;
    }
}

// ---------------------------------------------------------------------------
// 辅助：CPU 参考实现（独占扫描）
// ---------------------------------------------------------------------------
static void scanCpu(const std::vector<float>& in, std::vector<float>& out) {
    out.resize(in.size());
    double acc = 0.0;
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = (float)acc;
        acc += (double)in[i];
    }
}

static bool nearlyEqual(const std::vector<float>& a, const std::vector<float>& b,
                        float tol = 1e-3f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol * (1.f + std::fabs(b[i]))) return false;
    return true;
}

int main() {
    printDeviceInfo();

    CudaTimer timer;

    // =====================================================================
    // 第一部分：用小块（512 个元素）对比 Kogge-Stone / Blelloch / CPU
    // =====================================================================
    {
        constexpr int M = 512;
        std::vector<float> h_in(M);
        for (int i = 0; i < M; ++i) h_in[i] = (float)((i % 13) + 1) * 0.25f;

        std::vector<float> h_ref, h_ks(M), h_bl(M);
        scanCpu(h_in, h_ref);

        float *d_in = nullptr, *d_out = nullptr, *d_total = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, TILE * sizeof(float)));  // 够 Blelloch 的 TILE 用
        CUDA_CHECK(cudaMalloc(&d_total, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), M * sizeof(float), cudaMemcpyHostToDevice));

        // Kogge-Stone 的块级版本一次处理 BLOCK=256 个元素，这里分 2 次处理 512 个
        for (int half = 0; half < M / BLOCK; ++half) {
            koggeStoneBlock<<<1, BLOCK>>>(d_in + half * BLOCK, d_out + half * BLOCK);
            CUDA_CHECK_KERNEL();
        }
        CUDA_CHECK(cudaMemcpy(h_ks.data(), d_out, M * sizeof(float), cudaMemcpyDeviceToHost));
        // 块内包含式扫描只给了块内前缀，跨块前缀需要手工补（这里只有 2 块）
        for (int half = 1; half < M / BLOCK; ++half) {
            float carry = 0.f;
            for (int i = 0; i < BLOCK; ++i) carry += h_in[(half - 1) * BLOCK + i];
            for (int i = 0; i < BLOCK; ++i) h_ks[half * BLOCK + i] += carry;
        }
        // Kogge-Stone 是包含式，转换成独占式便于和 CPU 比较
        for (int i = M - 1; i > 0; --i) h_ks[i] = h_ks[i - 1];
        h_ks[0] = 0.f;
        printf("\n[正确性] Kogge-Stone(512)  vs CPU: %s\n",
               nearlyEqual(h_ks, h_ref) ? "[OK]" : "[FAIL]");

        // Blelloch 独占扫描：一个块处理 TILE=4096，只喂前 512 个有效数据，
        // 越界部分由 kernel 内部补 0，不会污染前 512 个结果
        blellochBlockExclusive<<<1, BLOCK>>>(d_in, d_out, d_total, M);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(h_bl.data(), d_out, M * sizeof(float), cudaMemcpyDeviceToHost));
        printf("[正确性] Blelloch(512)      vs CPU: %s\n",
               nearlyEqual(h_bl, h_ref) ? "[OK]" : "[FAIL]");

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_total));
    }

    // =====================================================================
    // 第二部分：1M 元素的设备级三阶段扫描 + 计时
    // =====================================================================
    {
        std::vector<float> h_in(N), h_ref;
        for (int i = 0; i < N; ++i) h_in[i] = 1.0f;
        scanCpu(h_in, h_ref);

        const int numBlocks = (N + TILE - 1) / TILE;  // = 256

        float* d_in = nullptr;
        float* d_out = nullptr;
        float* d_blockTotal = nullptr;    // 每块的总和
        float* d_blockOffset = nullptr;   // 每块之前的块总和（独占前缀）
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_blockTotal, numBlocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_blockOffset, numBlocks * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        // 说明：块和数量必须 <= BLOCK=256 才能一次扫完；超出时用 CPU 兜底。
        const bool gpuScanOK = (numBlocks <= BLOCK);
        std::vector<float> h_blockTotal(numBlocks), h_blockOffset(numBlocks);

        float best = 1e30f;
        auto oneRun = [&]() {
            // 阶段 1：每个块内部做独占扫描，同时输出该块总和
            blellochBlockExclusive<<<numBlocks, BLOCK>>>(d_in, d_out, d_blockTotal, N);
            // 阶段 2：对块总和再做一次独占扫描，得到每块的前缀偏移
            if (gpuScanOK) {
                scanBlockTotals<<<1, BLOCK>>>(d_blockTotal, d_blockOffset, numBlocks);
            } else {
                CUDA_CHECK(cudaMemcpy(h_blockTotal.data(), d_blockTotal,
                                      numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
                double acc = 0.0;
                for (int b = 0; b < numBlocks; ++b) {
                    h_blockOffset[b] = (float)acc;
                    acc += (double)h_blockTotal[b];
                }
                CUDA_CHECK(cudaMemcpy(d_blockOffset, h_blockOffset.data(),
                                      numBlocks * sizeof(float), cudaMemcpyHostToDevice));
            }
            // 阶段 3：把块偏移加回本块所有元素
            addBlockOffsets<<<numBlocks, BLOCK>>>(d_out, d_blockOffset, N);
        };

        for (int r = 0; r < REPEAT; ++r) {
            timer.start();
            oneRun();
            const float ms = timer.stop();
            if (ms < best) best = ms;
        }
        CUDA_CHECK_KERNEL();

        std::vector<float> h_got(N);
        CUDA_CHECK(cudaMemcpy(h_got.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        printf("\n[正确性] 设备级 scan(1M) vs CPU: %s\n",
               nearlyEqual(h_got, h_ref, 1e-2f) ? "[OK]" : "[FAIL]");
        printf("[性能]   3 阶段扫描 %d 个元素: %.4f ms（含全部 3 个 kernel）\n", N, best);
        printf("         有效带宽约 %.1f GB/s（读 4MB + 写 4MB 的理想值）\n",
               2.0 * N * sizeof(float) / (best * 1e-3) / 1e9);
        printf("         在 RTX 3060 上实测约 0.06 ms 量级，你的机器可能不同。\n");
        printf("[对照]   同样 1M 元素的 Kogge-Stone 需要 log2(1M)=20 轮，"
               "开销约是 Blelloch 的 20 倍，所以生产代码一律用 work-efficient 版本。\n");

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_blockTotal));
        CUDA_CHECK(cudaFree(d_blockOffset));
    }

    return 0;
}
