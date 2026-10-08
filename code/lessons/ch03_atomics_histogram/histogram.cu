// ============================================================
//  code/lessons/ch03_atomics_histogram/histogram.cu
//  第 3 章核心实战：直方图统计
//
//  同一个问题，两种写法：
//    方案 A —— 直接对全局内存里的直方图做 atomicAdd
//    方案 B —— 每个 block 先在 shared memory 里建私有直方图，
//              最后再汇总到全局（大幅减少全局原子操作次数）
//
//  程序会验证两者的正确性，并计时对比。
//
//  编译： nvcc -O2 -arch=sm_70 histogram.cu -o histogram
//  运行： ./histogram
// ============================================================
#include "../../common/cuda_check.h"
#include "../../common/cuda_bench.h"
#include <cstdio>
#include <cstdlib>

#define BINS   256        // 灰度直方图：256 个桶
#define BLOCK  256        // 每块 256 线程

// ------------------------------------------------------------
// 方案 A：全局原子直方图
//   每个线程算出自己的 bin，然后直接对「显存里的」那个桶做 atomicAdd。
//   问题在于：256 个桶挤在 1KB 内存里，几万个线程随机砸过去，
//   同一时刻会有大量原子操作排队在同一个地址上 —— 典型的争用。
// ------------------------------------------------------------
__global__ void histogramGlobalAtomic(const unsigned char* data, int n,
                                      unsigned int* hist) {
    // 网格步长循环：让有限数量的线程处理任意长度的输入
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        // 读一次全局内存 + 一次全局原子加 —— 每次都要走 L2
        atomicAdd(&hist[data[i]], 1u);
    }
}

// ------------------------------------------------------------
// 方案 B：每块私有 shared 直方图（推荐写法）
//   共享内存里的原子操作由 SM 内的硬件单元完成，延迟远低于走 L2 的
//   全局原子；而且同一块内 256 个线程砸 256 个桶时，
//   冲突大多能在 shared memory 的 bank 内部就地化解。
//   只有最后"汇总"那一步才真正碰全局内存。
// ------------------------------------------------------------
__global__ void histogramSharedPrivate(const unsigned char* data, int n,
                                       unsigned int* hist) {
    // 静态共享内存：256 个 unsigned int = 1 KB，任何设备都放得下
    __shared__ unsigned int local[BINS];

    // ---- 第 1 步：清零私有直方图 ----
    // 256 个桶 / 256 个线程 = 一人一个，循环写法兼容任意 blockDim
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        local[b] = 0u;
    }
    // 必须同步：别的线程马上就要对这些格子做原子加，
    // 如果它还读到"清零前"的旧值，结果就全错了。
    __syncthreads();

    // ---- 第 2 步：统计，全部落在共享内存上 ----
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        atomicAdd(&local[data[i]], 1u);      // 共享内存原子加，快得多
    }

    // 必须同步：等本块所有线程都统计完毕，local 才是完整的
    __syncthreads();

    // ---- 第 3 步：把私有直方图汇总到全局 ----
    // 全局原子操作的次数从 n 次降到「块数 × 256」次，
    // 而且是同块线程打不同地址，冲突也小得多。
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        unsigned int v = local[b];
        if (v > 0) {                          // 空桶直接跳过，能再省一笔
            atomicAdd(&hist[b], v);
        }
    }
}

// ------------------------------------------------------------
// 在 CPU 上算一遍，用来对拍
// ------------------------------------------------------------
static void cpuHistogram(const unsigned char* data, int n, unsigned int* h) {
    for (int b = 0; b < BINS; ++b) h[b] = 0u;
    for (int i = 0; i < n; ++i) h[data[i]]++;
}

static int compareHist(const unsigned int* a, const unsigned int* b) {
    int bad = 0;
    for (int i = 0; i < BINS; ++i) if (a[i] != b[i]) ++bad;
    return bad;
}

int main() {
    const int n = 1 << 24;                    // 16 MB 的 unsigned char 数据
    printf("=== 直方图实战 (n = %d = %d MB, bins = %d) ===\n",
           n, n / 1024 / 1024, BINS);

    // ---------- 造数据 ----------
    // 用一个确定性的伪随机序列，避免引入 <random> 依赖，
    // 也保证每次运行结果一样，方便对拍。
    unsigned char* h_data = (unsigned char*)malloc(n);
    for (int i = 0; i < n; ++i) {
        // Knuth 乘法散列取高 8 位：分布接近均匀，但并非完全均匀
        h_data[i] = (unsigned char)(((unsigned int)i * 2654435761u) >> 24);
    }

    unsigned int* h_ref = (unsigned int*)malloc(BINS * sizeof(unsigned int));
    cpuHistogram(h_data, n, h_ref);

    // ---------- 上传到 GPU ----------
    unsigned char* d_data = nullptr;
    unsigned int*  d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice));

    // ---------- 网格大小 ----------
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int grid = prop.multiProcessorCount * 32;          // 每 SM 塞 32 个块
    int maxGrid = (n + BLOCK - 1) / BLOCK;
    if (grid > maxGrid) grid = maxGrid;
    printf("设备：%s（%d 个 SM），grid = %d, block = %d\n\n",
           prop.name, prop.multiProcessorCount, grid, BLOCK);

    unsigned int* h_hist = (unsigned int*)malloc(BINS * sizeof(unsigned int));

    // ---------- 方案 A ----------
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
    histogramGlobalAtomic<<<grid, BLOCK>>>(d_data, n, d_hist);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    int badA = compareHist(h_hist, h_ref);

    double tA = medianMs([&] {
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
        histogramGlobalAtomic<<<grid, BLOCK>>>(d_data, n, d_hist);
    }, 3, 20);

    // ---------- 方案 B ----------
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
    histogramSharedPrivate<<<grid, BLOCK>>>(d_data, n, d_hist);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    int badB = compareHist(h_hist, h_ref);

    double tB = medianMs([&] {
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
        histogramSharedPrivate<<<grid, BLOCK>>>(d_data, n, d_hist);
    }, 3, 20);

    // ---------- 结果 ----------
    printf("--- 正确性（与 CPU 对拍）---\n");
    printf("  方案 A 全局原子   : 不同的桶数 = %d / %d %s\n", badA, BINS,
           badA == 0 ? "✓" : "✗");
    printf("  方案 B 私有直方图 : 不同的桶数 = %d / %d %s\n", badB, BINS,
           badB == 0 ? "✓" : "✗");

    printf("\n--- 直方图形状（前 16 个桶）---\n");
    for (int b = 0; b < 16; ++b) {
        printf("  bin %3d | %7u | ", b, h_ref[b]);
        int bars = (int)(h_ref[b] * 40.0 / (n / BINS));
        for (int k = 0; k < bars; ++k) putchar('#');
        putchar('\n');
    }

    // 有效带宽：只读了 n 字节数据，另外还有 256 个桶的读改写（可忽略）
    double bytes = (double)n;
    printf("\n--- 性能对比（各测 20 次取中位数）---\n");
    printf("  方案 A 全局原子   : %8.4f ms   有效带宽 %7.1f GB/s\n",
           tA, gbps(bytes, tA));
    printf("  方案 B 私有直方图 : %8.4f ms   有效带宽 %7.1f GB/s\n",
           tB, gbps(bytes, tB));
    printf("  实测加速比约为 %.2f 倍（你的机器可能不同）。\n", tA / tB);
    printf("  说明：这个 kernel 主要瓶颈是「读数据」的带宽，\n");
    printf("        所以方案 B 的收益取决于争用有多严重，\n");
    printf("        在数据量小、网格大的时候差距会更明显。\n");

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    free(h_data);
    free(h_ref);
    free(h_hist);
    printf("\n完成。\n");
    return 0;
}
