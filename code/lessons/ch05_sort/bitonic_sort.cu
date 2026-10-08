// ============================================================================
//  code/lessons/ch05_sort/bitonic_sort.cu
//
//  双调排序（Bitonic Sort）—— GPU 上最"规整"的排序算法
//    stage 1: 每个线程块用共享内存把 1024 个元素排成升序（若干轮 bitonic 网络）
//    stage 2: 块之间用全局内存做 bitonic merge，k 从 2048 一路翻倍到 n
//
//  为什么 GPU 喜欢双调排序：
//    它的比较-交换对 (i, i^j) 是**完全规律**的，任何时刻任意线程都知道
//    自己要跟谁比较、该升序还是降序，不需要分支、不需要动态调度，
//    天然适合 SIMT。代价是工作量 O(n log^2 n)，比快排多。
//
//  编译: nvcc -O3 -arch=sm_70 bitonic_sort.cu -o bitonic_sort
//  运行: ./bitonic_sort
// ============================================================================
#include "../../common/cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int   TPB    = 1024;     // 每个块 1024 个线程
constexpr int   N      = 1 << 20;  // 排序 1M 个 float
constexpr int   REPEAT = 3;        // 双调排序较慢，重复次数少一点

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
// 核心比较器：对 (i, i^j) 这一对做比较-交换
//   * ixj = i ^ j 保证每对元素只被处理一次（只有下标较小的一方动手）
//   * 方向由 (i & k) 决定：
//       (i & k) == 0  →  这一段应该**升序**，把小的放 i
//       (i & k) != 0  →  这一段应该**降序**，把大的放 i
//     这就是双调排序"升一半、降一半"的构造方式，也是它最好记的一句话。
// ===========================================================================
__device__ __forceinline__ void compareExchange(float* a, float* b, bool ascending) {
    const float va = *a;
    const float vb = *b;
    // 用异或代替 if/else，避免分支：只有 (va > vb) == ascending 时才交换
    const bool needSwap = ((va > vb) == ascending);
    *a = needSwap ? vb : va;
    *b = needSwap ? va : vb;
}

// ===========================================================================
// stage 1：块内双调排序（共享内存）
//   每个块独立地把自己的 TPB 个元素排成升序。
//   因为 k <= TPB 且块起始下标是 TPB 的整数倍，
//   块内元素 (base + tid) & k 恰好等于 tid & k，所以直接对 tid 取位即可。
// ===========================================================================
__global__ void bitonicSortTile(float* __restrict__ d, int n) {
    extern __shared__ float s[];  // 动态共享内存，大小 = TPB * sizeof(float)
    const int tid = threadIdx.x;
    const int base = blockIdx.x * blockDim.x;

    // 这里要求 n 是 blockDim 的整数倍（本程序里 n 是 2 的幂，天然满足），
    // 所以不需要 return 掉越界线程
    s[tid] = d[base + tid];
    __syncthreads();

    for (int k = 2; k <= TPB; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (ixj > tid) {
                const bool asc = ((tid & k) == 0);  // 方向判断
                compareExchange(&s[tid], &s[ixj], asc);
            }
            // 每一层比较-交换之后都必须同步：
            // 下一层要读的可能是别的线程刚写进共享内存的值
            __syncthreads();
        }
    }

    d[base + tid] = s[tid];
}

// ===========================================================================
// stage 2：跨块的双调归并（全局内存）
//   每启动一次只做"一层"比较-交换（固定 k 和 j），
//   因为不同块之间无法在 kernel 内同步，只能靠 kernel 边界来保证可见性。
//   完整的双调排序需要 (log2 n)*(log2 n + 1)/2 次启动 —— 这是它最大的缺点，
//   生产环境要么用 radix sort，要么用 CUB 的 DeviceRadixSort。
// ===========================================================================
__global__ void bitonicStep(float* __restrict__ d, int n, int k, int j) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int ixj = i ^ j;
    if (ixj > i) {
        const bool asc = ((i & k) == 0);  // 用**全局**下标取位，这是块间阶段必须的
        compareExchange(&d[i], &d[ixj], asc);
    }
}

// ---------------------------------------------------------------------------
// 一次完整的双调排序
// ---------------------------------------------------------------------------
static void bitonicSort(float* d, int n, float* t_ms) {
    CudaTimer timer;
    timer.start();

    // 1) 块内排好序（每个块升序）
    bitonicSortTile<<<n / TPB, TPB, TPB * sizeof(float)>>>(d, n);
    CUDA_CHECK_KERNEL();

    // 2) 归并：k 从 2*TPB 开始一路翻倍
    for (int k = TPB * 2; k <= n; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1)
            bitonicStep<<<(n + 255) / 256, 256>>>(d, n, k, j);

    *t_ms = timer.stop();
}

int main() {
    printDeviceInfo();
    CudaTimer timer;

    // =====================================================================
    // 1. 小规模正确性验证：与 std::sort 对拍
    // =====================================================================
    {
        constexpr int M = 1 << 16;  // 65536
        std::vector<float> h(M);
        srand(12345);
        for (int i = 0; i < M; ++i) h[i] = (float)(rand() % 100000) * 0.001f;

        std::vector<float> h_ref = h;
        std::sort(h_ref.begin(), h_ref.end());

        float* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, M * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), M * sizeof(float), cudaMemcpyHostToDevice));

        float ms = 0.f;
        bitonicSort(d, M, &ms);

        std::vector<float> got(M);
        CUDA_CHECK(cudaMemcpy(got.data(), d, M * sizeof(float), cudaMemcpyDeviceToHost));

        int bad = 0;
        for (int i = 0; i < M; ++i)
            if (std::fabs(got[i] - h_ref[i]) > 1e-6f) ++bad;
        printf("[正确性] bitonic sort(%d) vs std::sort: 不匹配 = %d  %s  (%.3f ms)\n",
               M, bad, bad == 0 ? "[OK]" : "[FAIL]", ms);

        CUDA_CHECK(cudaFree(d));
    }

    // =====================================================================
    // 2. 1M 元素性能
    // =====================================================================
    {
        std::vector<float> h(N);
        srand(4321);
        for (int i = 0; i < N; ++i) h[i] = (float)(rand() % 1000000) * 0.001f;

        float* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, N * sizeof(float)));

        float best = 1e30f;
        for (int r = 0; r < REPEAT; ++r) {
            CUDA_CHECK(cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
            float ms = 0.f;
            bitonicSort(d, N, &ms);
            if (ms < best) best = ms;
        }

        std::vector<float> got(N);
        CUDA_CHECK(cudaMemcpy(got.data(), d, N * sizeof(float), cudaMemcpyDeviceToHost));
        const bool sorted = std::is_sorted(got.begin(), got.end());
        printf("[正确性] bitonic sort(1M) 有序: %s\n", sorted ? "[OK]" : "[FAIL]");

        // n = 2^20，块内 10 个 stage 共 1+2+...+10 = 55 层；
        // 块间 k = 2^11..2^20，共 11+12+...+20 = 155 层。
        // 总层数正好等于 log2(n)*(log2(n)+1)/2 = 20*21/2 = 210。
        printf("[性能]   bitonic sort 1M float: %.3f ms\n", best);
        printf("         排序吞吐约 %.1f M 元素/s\n", (double)N / (best * 1e-3) / 1e6);
        printf("         块内 55 层同步完成 + 块间 155 层 = 155 次 kernel 启动\n");

        CUDA_CHECK(cudaFree(d));
    }

    printf("\n提示：双调排序在 RTX 3060 上排 1M 个 float 大约 2~4 ms（你的机器可能不同），\n");
    printf("      比 std::sort 的 CPU 版本快，但远不如 CUB 的 DeviceRadixSort（约 0.15 ms）。\n");
    printf("      它的价值在于结构规整、容易理解比较网络，而不是极限性能。\n");
    return 0;
}
