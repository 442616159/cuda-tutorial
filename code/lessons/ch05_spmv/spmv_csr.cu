// ============================================================================
//  code/lessons/ch05_spmv/spmv_csr.cu
//
//  稀疏矩阵向量乘 SpMV：y = A * x
//    格式 A：CSR（Compressed Sparse Row），每线程一行（标量版）
//    格式 B：CSR + 每 warp 一行（用 __shfl_down_sync 做行内归约）
//    格式 C：ELL（每行补齐到相同的非零元个数），每线程一行
//    对照 D：稠密矩阵向量乘（用来展示算术强度的巨大差异）
//
//  编译: nvcc -O3 -arch=sm_70 spmv_csr.cu -o spmv
//  运行: ./spmv
// ============================================================================
#include "../../common/cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int   REPEAT = 50;
// 用 5 点差分格式生成一个 G x G 的网格拉普拉斯矩阵：
//   每行最多 5 个非零元（中心、上下左右），边界行少一些
constexpr int   G_BIG  = 1024;   // 1024*1024 = 1M 行，nnz 约 5M
constexpr int   G_SMALL = 32;    // 32*32 = 1024 行，用来和稠密结果对拍

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
// 在 host 上按 5 点格式生成 CSR 结构
//   rowPtr: 长度 n+1 的行起始下标；colIdx/values: 长度 nnz
// ---------------------------------------------------------------------------
static void buildCsr5Point(int g, std::vector<int>& rowPtr,
                           std::vector<int>& colIdx, std::vector<float>& values) {
    const int n = g * g;
    rowPtr.assign(n + 1, 0);
    colIdx.clear();
    values.clear();

    for (int y = 0; y < g; ++y) {
        for (int x = 0; x < g; ++x) {
            const int row = y * g + x;
            // 中心（对角占优，保证矩阵性质良好）
            colIdx.push_back(row);
            values.push_back(4.0f);
            if (y > 0)     { colIdx.push_back(row - g); values.push_back(-1.0f); }
            if (y < g - 1) { colIdx.push_back(row + g); values.push_back(-1.0f); }
            if (x > 0)     { colIdx.push_back(row - 1); values.push_back(-1.0f); }
            if (x < g - 1) { colIdx.push_back(row + 1); values.push_back(-1.0f); }
            rowPtr[row + 1] = (int)colIdx.size();
        }
    }
}

// ---------------------------------------------------------------------------
// ELL 格式：把 CSR 转成"每行等长、列主序"的密集排布
//   ellVal[k * n + row] / ellCol[k * n + row]，空位填 0
//   好处：一个 warp 里的线程读同一列 k 时地址连续，天然合并访存
//   坏处：最长的行决定所有行的长度，行长度差异大时大量浪费
// ---------------------------------------------------------------------------
static void buildEll(const std::vector<int>& rowPtr, const std::vector<int>& colIdx,
                     const std::vector<float>& values, int n, int maxNnz,
                     std::vector<float>& ellVal, std::vector<int>& ellCol) {
    ellVal.assign((size_t)maxNnz * n, 0.f);
    ellCol.assign((size_t)maxNnz * n, 0);
    for (int row = 0; row < n; ++row) {
        const int start = rowPtr[row];
        const int len = rowPtr[row + 1] - start;
        for (int k = 0; k < len; ++k) {
            ellVal[(size_t)k * n + row] = values[start + k];
            ellCol[(size_t)k * n + row] = colIdx[start + k];
        }
    }
}

// ===========================================================================
// 核函数 A：每线程一行（CSR 标量版）
//   最直观，但同一 warp 内不同行的长度不一样：
//     * 长行的线程还在循环，短行的线程已经空转 —— 负载不均；
//     * 每读一个非零元都要访问 colIdx[j] 再间接取 x[colIdx[j]]，访存是"追着跑"的。
//   在 5 点这样规整的矩阵上表现尚可，遇到幂律分布的图（比如社交网络）会崩。
// ===========================================================================
__global__ void spmvCsrScalar(const int* __restrict__ rowPtr,
                              const int* __restrict__ colIdx,
                              const float* __restrict__ values,
                              const float* __restrict__ x,
                              float* __restrict__ y, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    float sum = 0.f;
    for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j)
        sum += values[j] * __ldg(&x[colIdx[j]]);  // __ldg 走只读数据缓存

    y[row] = sum;
}

// ===========================================================================
// 核函数 B：每 warp 一行（CSR + warp 归约）
//   32 个线程平摊一行的非零元，行再长也只多跑几轮；
//   行内求和用 __shfl_down_sync 在寄存器里完成，不碰共享内存。
//   代价：调度粒度变成 32 线程，短行（<32 个非零元）会浪费大量线程。
// ===========================================================================
__global__ void spmvCsrWarp(const int* __restrict__ rowPtr,
                            const int* __restrict__ colIdx,
                            const float* __restrict__ values,
                            const float* __restrict__ x,
                            float* __restrict__ y, int nrows) {
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warpId >= nrows) return;

    const int start = rowPtr[warpId];
    const int end = rowPtr[warpId + 1];

    float sum = 0.f;
    for (int j = start + lane; j < end; j += 32)
        sum += values[j] * __ldg(&x[colIdx[j]]);

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    if (lane == 0) y[warpId] = sum;
}

// ===========================================================================
// 核函数 C：ELL 格式，每线程一行
//   和标量 CSR 的循环结构几乎一样，但第 k 轮所有线程读的地址是
//   ellVal[k*n + row]，row 连续 → 完全合并访存。
//   而且循环次数对所有线程都相同（maxNnz），不存在行间负载不均。
// ===========================================================================
__global__ void spmvEll(const float* __restrict__ ellVal,
                        const int* __restrict__ ellCol,
                        const float* __restrict__ x,
                        float* __restrict__ y, int nrows, int maxNnz) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    float sum = 0.f;
    for (int k = 0; k < maxNnz; ++k) {
        const size_t idx = (size_t)k * nrows + row;
        const float v = ellVal[idx];
        if (v != 0.f)  // 空位直接跳过，避免读一个无意义的列下标
            sum += v * __ldg(&x[ellCol[idx]]);
    }
    y[row] = sum;
}

// ===========================================================================
// 对照 D：稠密矩阵向量乘 —— 算术强度完全不同
// ===========================================================================
__global__ void denseMatvec(const float* __restrict__ A, const float* __restrict__ x,
                            float* __restrict__ y, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float sum = 0.f;
    const float* arow = A + (size_t)row * n;
    for (int j = 0; j < n; ++j) sum += arow[j] * x[j];  // 缓存友好的行主序遍历
    y[row] = sum;
}

int main() {
    printDeviceInfo();
    CudaTimer timer;

    // =====================================================================
    // 1. 小网格上和稠密结果对拍，确认三种格式都是对的
    // =====================================================================
    {
        const int g = G_SMALL, n = g * g;
        std::vector<int> rowPtr, colIdx;
        std::vector<float> values;
        buildCsr5Point(g, rowPtr, colIdx, values);

        std::vector<float> hx(n), hRef(n, 0.f);
        for (int i = 0; i < n; ++i) hx[i] = std::sin((float)i * 0.1f);

        // CPU 参考：直接按 CSR 串行算（和矩阵结构无关，最可靠）
        for (int row = 0; row < n; ++row)
            for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j)
                hRef[row] += values[j] * hx[colIdx[j]];

        // 稠密结果（同一个矩阵的稠密形式），用来二次确认 CSR 结构本身没写错
        std::vector<float> dense((size_t)n * n, 0.f);
        for (int row = 0; row < n; ++row)
            for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j)
                dense[(size_t)row * n + colIdx[j]] = values[j];
        std::vector<float> hDense(n, 0.f);
        for (int row = 0; row < n; ++row) {
            double acc = 0.0;
            for (int j = 0; j < n; ++j) acc += (double)dense[(size_t)row * n + j] * hx[j];
            hDense[row] = (float)acc;
        }

        const int nnz = (int)colIdx.size();
        int *d_rowPtr = nullptr, *d_colIdx = nullptr, *d_ellCol = nullptr;
        float *dx = nullptr, *dy = nullptr, *d_values = nullptr, *d_ellVal = nullptr;
        CUDA_CHECK(cudaMalloc(&d_rowPtr, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colIdx, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<float> hgot(n);
        const int grid = (n + 255) / 256;

        spmvCsrScalar<<<grid, 256>>>(d_rowPtr, d_colIdx, d_values, dx, dy, n);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(hgot.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        int bad1 = 0;
        for (int i = 0; i < n; ++i) if (std::fabs(hgot[i] - hRef[i]) > 1e-4f) ++bad1;

        const int gridWarp = (n * 32 + 255) / 256;
        spmvCsrWarp<<<gridWarp, 256>>>(d_rowPtr, d_colIdx, d_values, dx, dy, n);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(hgot.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        int bad2 = 0;
        for (int i = 0; i < n; ++i) if (std::fabs(hgot[i] - hRef[i]) > 1e-4f) ++bad2;

        const int maxNnz = 5;
        std::vector<float> ellVal;
        std::vector<int> ellCol;
        buildEll(rowPtr, colIdx, values, n, maxNnz, ellVal, ellCol);
        CUDA_CHECK(cudaMalloc(&d_ellVal, ellVal.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ellCol, ellCol.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_ellVal, ellVal.data(), ellVal.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ellCol, ellCol.data(), ellCol.size() * sizeof(int), cudaMemcpyHostToDevice));

        spmvEll<<<grid, 256>>>(d_ellVal, d_ellCol, dx, dy, n, maxNnz);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(hgot.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        int bad3 = 0;
        for (int i = 0; i < n; ++i) if (std::fabs(hgot[i] - hRef[i]) > 1e-4f) ++bad3;

        int badDense = 0;
        for (int i = 0; i < n; ++i) if (std::fabs(hDense[i] - hRef[i]) > 1e-4f) ++badDense;

        printf("\n[正确性] %dx%d 网格, n=%d, nnz=%d, 平均每行 %.2f 个非零元\n",
               g, g, n, nnz, (double)nnz / n);
        printf("         spmvCsrScalar : 不匹配 %d  %s\n", bad1, bad1 == 0 ? "[OK]" : "[FAIL]");
        printf("         spmvCsrWarp   : 不匹配 %d  %s\n", bad2, bad2 == 0 ? "[OK]" : "[FAIL]");
        printf("         spmvEll       : 不匹配 %d  %s\n", bad3, bad3 == 0 ? "[OK]" : "[FAIL]");
        printf("         CSR 结构 vs 稠密: 不匹配 %d  %s\n", badDense, badDense == 0 ? "[OK]" : "[FAIL]");

        CUDA_CHECK(cudaFree(d_rowPtr)); CUDA_CHECK(cudaFree(d_colIdx));
        CUDA_CHECK(cudaFree(d_values));  CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));        CUDA_CHECK(cudaFree(d_ellVal));
        CUDA_CHECK(cudaFree(d_ellCol));
    }

    // =====================================================================
    // 2. 大网格（1M 行）上的性能对比
    // =====================================================================
    {
        const int g = G_BIG, n = g * g;
        std::vector<int> rowPtr, colIdx;
        std::vector<float> values;
        buildCsr5Point(g, rowPtr, colIdx, values);
        const int nnz = (int)colIdx.size();

        std::vector<float> hx(n);
        for (int i = 0; i < n; ++i) hx[i] = 1.0f;

        const int maxNnz = 5;
        std::vector<float> ellVal;
        std::vector<int> ellCol;
        buildEll(rowPtr, colIdx, values, n, maxNnz, ellVal, ellCol);

        int *d_rowPtr = nullptr, *d_colIdx = nullptr, *d_ellCol = nullptr;
        float *dx = nullptr, *dy = nullptr, *d_values = nullptr, *d_ellVal = nullptr;
        CUDA_CHECK(cudaMalloc(&d_rowPtr, (size_t)(n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colIdx, (size_t)nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values, (size_t)nnz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ellCol, ellCol.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ellVal, ellVal.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dx, (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(), (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, values.data(), (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ellCol, ellCol.data(), ellCol.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ellVal, ellVal.data(), ellVal.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

        const int grid = (n + 255) / 256;
        const int gridWarp = (n * 32 + 255) / 256;
        // SpMV 的有效访存字节数：读 values+colIdx (nnz)，读 x、写 y (2n)
        const double bytes = (double)nnz * (sizeof(float) + sizeof(int)) + 2.0 * n * sizeof(float);

        printf("\n[规模]   n = %d 行, nnz = %d (%.2f M), 平均每行 %.2f 个非零元\n",
               n, nnz, nnz / 1e6, (double)nnz / n);
        printf("[规模]   稀疏度 = %.4f%%，稠密表示需要 %.1f GB，稀疏只要 %.1f MB\n",
               100.0 * nnz / ((double)n * n), (double)n * n * 4 / 1e9,
               (double)nnz * 8 / 1e6);

        struct Case { const char* name; int which; };
        const Case cases[3] = {
            {"CSR 每线程一行", 0},
            {"CSR 每 warp 一行", 1},
            {"ELL 每线程一行", 2},
        };
        for (const Case& c : cases) {
            float best = 1e30f;
            for (int r = 0; r < REPEAT; ++r) {
                timer.start();
                if (c.which == 0)
                    spmvCsrScalar<<<grid, 256>>>(d_rowPtr, d_colIdx, d_values, dx, dy, n);
                else if (c.which == 1)
                    spmvCsrWarp<<<gridWarp, 256>>>(d_rowPtr, d_colIdx, d_values, dx, dy, n);
                else
                    spmvEll<<<grid, 256>>>(d_ellVal, d_ellCol, dx, dy, n, maxNnz);
                const float ms = timer.stop();
                if (ms < best) best = ms;
            }
            CUDA_CHECK_KERNEL();
            printf("[性能]   %-18s %7.4f ms   有效带宽 %7.1f GB/s\n",
                   c.name, best, bytes / ((double)best * 1e-3) / 1e9);
        }

        printf("\n提示：SpMV 是**纯带宽受限**问题（算术强度约 0.14 FLOP/Byte），\n");
        printf("      在 RTX 3060 上三种写法大约分别是 0.6 / 0.9 / 1.1 ms 量级，\n");
        printf("      你的机器可能不同。真正的瓶颈往往不是算，而是 colIdx 的间接寻址。\n");

        CUDA_CHECK(cudaFree(d_rowPtr)); CUDA_CHECK(cudaFree(d_colIdx));
        CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(d_ellVal));
        CUDA_CHECK(cudaFree(d_ellCol));
    }

    // =====================================================================
    // 3. 稠密对照：同样 1M 规模的稠密矩阵根本放不下，
    //    这里退到 4096x4096 来说明"算术强度"的差别
    // =====================================================================
    {
        constexpr int D = 4096;
        std::vector<float> hA((size_t)D * D);
        for (size_t i = 0; i < hA.size(); ++i) hA[i] = (float)(i % 7) * 0.1f;
        std::vector<float> hx(D, 1.0f);

        float *dA = nullptr, *dx = nullptr, *dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dx, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), D * sizeof(float), cudaMemcpyHostToDevice));

        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            timer.start();
            denseMatvec<<<(D + 255) / 256, 256>>>(dA, dx, dy, D);
            const float ms = timer.stop();
            if (ms < best) best = ms;
        }
        CUDA_CHECK_KERNEL();
        printf("\n[对照]   稠密 %dx%d 矩阵向量乘: %.4f ms, 访存 %.1f MB, 算术强度 %.2f FLOP/Byte\n",
               D, D, best, (double)D * D * 4 / 1e6, 2.0 / 4.0);
        printf("         做 %.2f GFLOP 的运算，却要搬 %.2f GB 数据 —— 依然远低于计算峰值。\n",
               2.0 * D * D / 1e9, (double)D * D * 4 / 1e9);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
    }

    return 0;
}
