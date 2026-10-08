// ============================================================
//  code/lessons/ch04_case_study/matmul_opt.cu
//  第 4 章核心案例：优化前后对比的完整模板
//
//  三个版本的矩阵乘法 C = A * B（都是 n x n 的方阵，行主序）：
//    v1 朴素版        ：每个线程算一个元素，直接从全局内存反复读
//    v2 shared 分块版  ：16x16 的 tile 搬到共享内存，复用度高得多
//    v3 分块 + 寄存器分块：64x64 的块，每线程算 4x4，寄存器内累加
//
//  每一版都做三件事：
//    1. 用 CPU 侧的对拍公式验证正确性（O(n^2)，不用真跑三重循环）
//    2. 计时并换算成 GFLOPS
//    3. 和上一版比较加速比
//
//  这是第 4 章「优化五步闭环」的落地产物：
//    正确 -> 基线 -> 找瓶颈 -> 只改最热的那一处 -> 回归验证
//
//  编译： nvcc -O3 -arch=sm_70 matmul_opt.cu -o matmul_opt
//  运行： ./matmul_opt
// ============================================================
#include "../../common/cuda_check.h"
#include "../../common/cuda_bench.h"
#include <cstdio>
#include <cmath>

// ------------------------------------------------------------
// v1：朴素版
//   每个线程负责 C 的一个元素，内层循环要走完整整 n 次全局读。
//   A 的一行会被读 n 次，B 的一列也会被读 n 次 ——
//   这就是「数据复用不足」，也是它慢的根本原因。
// ------------------------------------------------------------
__global__ void matmulNaive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;

    float sum = 0.f;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

// ------------------------------------------------------------
// v2：shared memory 分块（TILE = 16）
//   一个 16x16 的线程块负责 C 的一个 16x16 子块。
//   每个 k 步里：
//     - A 的 16x16 子块、B 的 16x16 子块各被搬进共享内存一次；
//     - 之后 16x16=256 个线程都从共享内存里取数，
//       全局内存的访问量因此降到原来的 1/16 左右。
//   __syncthreads() 出现两次，理由不同（见 ch02 的详细解释）：
//     第一次：等数据搬完才能读；
//     第二次：等所有人都读完，才能覆盖写入下一轮数据。
// ------------------------------------------------------------
#define TILE 16
__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.f;
    for (int k0 = 0; k0 < n; k0 += TILE) {
        As[ty][tx] = A[row * n + k0 + tx];
        Bs[ty][tx] = B[(k0 + ty) * n + col];
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();     // 必须：否则下一轮写 As/Bs 会覆盖还没读完的数据
    }
    C[row * n + col] = sum;
}

// ------------------------------------------------------------
// v3：分块 + 寄存器分块
//   块级 tile 放大到 64x64，blockDim = (16,16) = 256 线程，
//   每个线程计算 4x4 个输出元素，全部在寄存器里累加。
//
//   好处：
//     1) 每个线程从共享内存读 4+4 个数，却做了 16 次乘加 —— 共享内存流量大降；
//     2) 累加器在寄存器里，不再每个 k 都读写共享内存；
//     3) 64x64 的块意味着每份数据被复用 64 次（而不是 v2 的 16 次）。
// ------------------------------------------------------------
#define BM 64           // 块级 tile 的行数
#define BN 64           // 块级 tile 的列数
#define BK 16           // k 方向的步长
#define TM 4            // 每线程负责的行数
#define TN 4            // 每线程负责的列数
// blockDim.x = BN / TN = 16，blockDim.y = BM / TM = 16

__global__ void matmulTiledReg(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int n) {
    __shared__ float As[BM][BK];      // 64 x 16
    __shared__ float Bs[BK][BN];      // 16 x 64

    int tx = threadIdx.x;             // 0..15
    int ty = threadIdx.y;             // 0..15

    int rowBase = blockIdx.y * BM + ty * TM;   // 本线程负责的起始行
    int colBase = blockIdx.x * BN + tx * TN;   // 本线程负责的起始列

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < n; k0 += BK) {
        // ---- 协作搬入 A 的 64x16 分块 ----
        // 线程 (tx,ty) 负责第 ty*TM + i 行的第 tx 列，共 TM 行
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            As[ty * TM + i][tx] = A[(rowBase + i) * n + k0 + tx];
        }
        // ---- 协作搬入 B 的 16x64 分块 ----
        // 线程 (tx,ty) 负责第 ty 行的第 tx*TN + j 列，共 TN 列
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            Bs[ty][tx * TN + j] = B[(k0 + ty) * n + colBase + j];
        }
        __syncthreads();

        // ---- 在共享内存上做 BK 次乘加，把结果留在寄存器 ----
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float aReg[TM], bReg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(rowBase + i) * n + colBase + j] = acc[i][j];
}

// ============================================================
// 验证与工具
// ============================================================

// 把 A 和 B 初始化成确定性的伪随机值
static void initMatrix(float* m, int n, unsigned int seed) {
    for (int i = 0; i < n * n; ++i) {
        unsigned int x = (unsigned int)i * 1664525u + seed;
        m[i] = (float)((x >> 8) & 0xFFFF) / 65535.0f - 0.5f;   // [-0.5, 0.5)
    }
}

// 在 CPU 上算 C 所有元素的和，用来对拍。
// 利用 sum_ij C_ij = sum_k (sum_i A_ik) * (sum_j B_kj)，
// 复杂度只有 O(n^2)，不需要真的做三重循环。
static double referenceChecksum(const float* A, const float* B, int n) {
    // 用 double 累加，避免 CPU 侧的舍入误差被误判成"GPU 算错"
    double total = 0.0;
    for (int k = 0; k < n; ++k) {
        double colSumA = 0.0;   // sum_i A[i][k]
        double rowSumB = 0.0;   // sum_j B[k][j]
        for (int i = 0; i < n; ++i) colSumA += A[i * n + k];
        for (int j = 0; j < n; ++j) rowSumB += B[k * n + j];
        total += colSumA * rowSumB;
    }
    return total;
}

int main() {
    const int n = 1024;                       // 1024 x 1024
    printf("=== 矩阵乘法优化对比 (n = %d) ===\n", n);

    const size_t bytes = (size_t)n * n * sizeof(float);
    const double flops = 2.0 * (double)n * n * n;   // 一次乘加算 2 次浮点运算

    // ---------- 主机端数据 ----------
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    initMatrix(h_A, n, 1u);
    initMatrix(h_B, n, 2u);

    double refSum = referenceChecksum(h_A, h_B, n);
    printf("CPU 对拍用的 C 元素总和 = %.4f\n\n", refSum);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // 用 float 累加 n=1024 个乘积，误差可到 1e-4 量级，
    // 而总和规模在 1e2 左右，所以相对容差取 1e-3 比较稳妥。
    auto check = [&](const char* name) -> bool {
        float* h_C = (float*)malloc(bytes);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        double s = 0.0;
        for (int i = 0; i < n * n; ++i) s += (double)h_C[i];
        double rel = fabs(s - refSum) / (fabs(refSum) + 1e-12);
        printf("  %-24s GPU 总和 = %14.4f，相对误差 = %.2e  %s\n",
               name, s, rel, rel < 1e-3 ? "✓" : "✗");
        free(h_C);
        return rel < 1e-3;
    };

    // ---------- v1 ----------
    dim3 block16(TILE, TILE);
    dim3 gridTiled(n / TILE, n / TILE);
    dim3 blockReg(BN / TN, BM / TM);              // (16, 16)
    dim3 gridReg(n / BN, n / BM);

    printf("--- 正确性 ---\n");

    matmulNaive<<<dim3((n + 31) / 32, (n + 31) / 32), dim3(32, 32)>>>(d_A, d_B, d_C, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    check("v1 naive");

    matmulTiled<<<gridTiled, block16>>>(d_A, d_B, d_C, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    check("v2 shared 16x16");

    matmulTiledReg<<<gridReg, blockReg>>>(d_A, d_B, d_C, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    check("v3 shared+寄存器");

    // ---------- 计时 ----------
    double t1 = medianMs([&] {
        matmulNaive<<<dim3((n + 31) / 32, (n + 31) / 32), dim3(32, 32)>>>(d_A, d_B, d_C, n);
    }, 2, 10);

    double t2 = medianMs([&] {
        matmulTiled<<<gridTiled, block16>>>(d_A, d_B, d_C, n);
    }, 2, 20);

    double t3 = medianMs([&] {
        matmulTiledReg<<<gridReg, blockReg>>>(d_A, d_B, d_C, n);
    }, 2, 20);

    // ---------- 汇总表 ----------
    printf("\n--- 性能对比（各测若干次取中位数）---\n");
    printf("  %-22s | %9s | %10s | %8s\n", "版本", "耗时(ms)", "GFLOPS", "加速比");
    printf("  -----------------------+-----------+------------+---------\n");
    printf("  %-22s | %9.3f | %10.1f | %7s\n", "v1 朴素", t1, gflopsOf(flops, t1), "1.00x");
    printf("  %-22s | %9.3f | %10.1f | %6.2fx\n", "v2 shared 16x16", t2,
           gflopsOf(flops, t2), t1 / t2);
    printf("  %-22s | %9.3f | %10.1f | %6.2fx\n", "v3 shared + 寄存器分块", t3,
           gflopsOf(flops, t3), t1 / t3);

    // ---------- 有效带宽视角 ----------
    // 这三个 kernel 的真实瓶颈都在访存，所以再看一眼"每毫秒搬了多少字节"
    double minBytes = 3.0 * bytes;         // 理论最小访存量：读 A + 读 B + 写 C
    printf("\n--- 按「理论最小访存量 %.1f MB」换算的等效带宽 ---\n",
           minBytes / 1024.0 / 1024.0);
    printf("  v1 : %7.1f GB/s\n", gbps(minBytes, t1));
    printf("  v2 : %7.1f GB/s\n", gbps(minBytes, t2));
    printf("  v3 : %7.1f GB/s\n", gbps(minBytes, t3));
    printf("  v1 的等效带宽会远超硬件峰值 —— 因为它根本没做到最小访存，\n");
    printf("  它实际搬运的字节数远多于 3*n^2*4。这个对比正好说明：\n");
    printf("  只看 GFLOPS 不够，还要知道「瓶颈到底是算力还是带宽」。\n");

    printf("\n以上数字在 GTX 1660 / RTX 3060 这一档显卡上，\n");
    printf("v1 大约几毫秒、v3 大约零点几毫秒，你的机器可能不同。\n");
    printf("把数字记下来，再去跑 ncu 看瓶颈，两者能互相印证。\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    return 0;
}
