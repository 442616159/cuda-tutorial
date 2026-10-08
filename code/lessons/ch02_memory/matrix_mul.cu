// ============================================================
//  code/lessons/ch02_memory/matrix_mul.cu
//  矩阵乘法 C = A * B 的三级演进：
//
//      v1 gemmNaive     朴素三重循环：每个线程算 C 的一个元素
//      v2 gemmTiled     共享内存分块（Tiling）
//      v3 gemmRegTiled  分块 + 寄存器分块（每线程 4x4 累加器）
//
//  对比这三级版本，你能亲眼看到"内存层次"到底值多少钱。
//
//  编译（在 code/lessons/ch02_memory 目录下）：
//      nvcc -O3 -arch=sm_75 -Xptxas -v matrix_mul.cu -o matrix_mul
//  运行：
//      ./matrix_mul        (Linux)
//      matrix_mul.exe      (Windows)
// ============================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_check.h"
#include "../../common/cuda_timer.h"

// ---------------- 编译期常量 ----------------
// v1 / v2 用：32x32 的块一共 1024 个线程，
// 是"每块最大线程数"的极限，也是教科书里最经典的配置。
constexpr int TILE = 32;

// v3 用：块覆盖 64x64 的输出，每次从 K 方向切 16 深。
constexpr int BM = 64;   // 块覆盖的输出行数
constexpr int BN = 64;   // 块覆盖的输出列数
constexpr int BK = 16;   // 每次从 K 方向加载的深度

// ------------------------------------------------------------
// v1: 朴素实现
//
// 每个线程算一个 C[row][col]。要对 K 做一整轮循环，
// 每次循环读 A[row*K+k] 和 B[k*N+col] 各一次。
//
// 问题在哪？
//   同一行的所有线程都要读同一个 A[row][k]；
//   同一列的所有线程都要读同一个 B[k][col]。
//   这些重复读取全都打到全局内存上，而全局内存的延迟是
//   几百个时钟周期。于是 GPU 大部分时间在等内存，不是在算。
//
// 算术强度（每读 1 字节能做多少次浮点运算）极低，是典型的
// "访存受限"（memory bound）程序。
// ------------------------------------------------------------
__global__ void gemmNaive(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            // 每一次循环迭代都要走两趟全局内存。
            // 相邻线程的 A[row*K+k] 是同一个地址（广播），
            // 但 B[k*N+col] 相邻线程是连续的 —— 这一半是好的。
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// ------------------------------------------------------------
// v2: 共享内存分块（Tiling）
//
// 核心思想：把 A 和 B 的一小块搬到共享内存，让一个块里的 1024 个
// 线程共享它。这样全局内存的访问次数从每线程 O(K) 次
// 降到每线程 O(K/TILE) 次，减少了约 TILE 倍。
//
// 流程（对 K 方向的每一块 k0）：
//   阶段 1：全块协作把 A 的 TILE x TILE 和 B 的 TILE x TILE 搬进共享内存
//   阶段 2：__syncthreads()，等所有人都搬完
//   阶段 3：每个线程在自己的 TILE 循环里累加
//   阶段 4：__syncthreads()，再做下一轮（否则会覆盖别人还在读的数据）
//
// 为什么要两次 __syncthreads()？
//   第一次是"生产者-消费者"屏障：确保共享内存写完了才读。
//   第二次是"防覆盖"屏障：确保所有人都读完了，才能写入下一轮的数据。
//   漏掉任一个都会得到难以复现的错误结果（第 3 章细讲）。
// ------------------------------------------------------------
__global__ void gemmTiled(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    // 静态共享内存：编译期就知道大小，编译器能更好地安排资源。
    // 两个 32x32 的 float 数组 = 2 * 4 KB = 8 KB，远低于 48 KB 上限。
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;   // 本线程输出的行
    int col = blockIdx.x * TILE + tx;   // 本线程输出的列

    float acc = 0.0f;

    // 沿 K 方向一块一块推进
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // ---- 阶段 1：协作加载 ----
        // 每个线程负责 As 和 Bs 各一个元素。
        // 越界时填 0：0 乘任何数都是 0，不影响结果，
        // 同时避免了在热循环里加判断。
        int aCol = k0 + tx;
        int bRow = k0 + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        // ---- 阶段 2：等大家都加载完 ----
        __syncthreads();

        // ---- 阶段 3：在共享内存上做累加 ----
        // 这次 As / Bs 的访问走的是共享内存（几十个周期），
        // 而不是全局内存（几百个周期）。
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        // ---- 阶段 4：等大家都算完，再进入下一轮 ----
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ------------------------------------------------------------
// v3: 分块 + 寄存器分块（Register Tiling）
//
// v2 已经用了共享内存，但每个线程只算 1 个输出元素，
// 于是它在内层循环里从共享内存读的每个值都只用了一次。
// 这就是"数据复用不足"。
//
// 寄存器分块让每个线程算 4x4 个输出。于是：
//   - 每个线程从共享内存读 4 + 4 = 8 个数，
//     就能完成 4*4 = 16 次乘加；
//   - 复用了 16/8 = 2 倍（用 8x8 则是 4 倍）；
//   - 累加器全在寄存器里，没有共享内存的读写开销。
//
// 这是手写 GEMM 从"能跑"到"接近 cuBLAS"的关键一步。
// ------------------------------------------------------------
__global__ void gemmRegTiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
    // 每线程负责 4x4 个输出，所以 16x16 = 256 个线程
    // 刚好覆盖 64x64 的块。(64/16) = 4 = 每线程的行数。
    constexpr int TM = 4;
    constexpr int TN = 4;

    __shared__ float As[BM][BK];   // 64 x 16
    __shared__ float Bs[BK][BN];   // 16 x 64

    // 本线程在块内的坐标
    int tx = threadIdx.x;          // 0..15
    int ty = threadIdx.y;          // 0..15
    int threadRowBase = ty * TM;   // 本线程负责的起始行（块内坐标）
    int threadColBase = tx * TN;   // 本线程负责的起始列（块内坐标）

    // 寄存器累加器：TM x TN = 16 个 float，全部在寄存器里
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // 本块在全局矩阵中的左上角坐标
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---------- 协作加载 ----------
        // As 有 BM*BK = 64*16 = 1024 个元素，Bs 有 BK*BN = 16*64 = 1024 个。
        // 256 个线程，所以每个线程负责 1024/256 = 4 个元素。
        // 用"线性编号 → 二维坐标"的方式展开，索引计算更直白，
        // 也保证了同一 warp 的 32 个线程访问的是连续地址（合并访存）。
        #pragma unroll
        for (int t = 0; t < (BM * BK) / 256; ++t) {
            int flat = t * 256 + ty * 16 + tx;   // 0..1023
            int r = flat / BK;                   // 0..63
            int c = flat % BK;                   // 0..15
            int gr = blockRow + r;               // 全局行
            int gc = k0 + c;                     // 全局列（K 方向）
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        #pragma unroll
        for (int t = 0; t < (BK * BN) / 256; ++t) {
            int flat = t * 256 + ty * 16 + tx;
            int r = flat / BN;                   // 0..15（K 方向）
            int c = flat % BN;                   // 0..63（N 方向）
            int gr = k0 + r;
            int gc = blockCol + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        // ---------- 计算：共享内存 → 寄存器 → 乘加 ----------
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 先把这一列/这一行读进寄存器（每个只读一次，然后复用）
            float aReg[TM];
            float bReg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[threadRowBase + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[k][threadColBase + j];

            // 然后做 TM*TN 次乘加，全部用寄存器里的值
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        __syncthreads();
    }

    // ---------- 写回结果 ----------
    // 写回同样按行主序、相邻线程写相邻地址，保证合并访存。
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = blockRow + threadRowBase + i;
        if (row >= M) break;   // 行越界，后面的行只会更越界，直接跳出
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int col = blockCol + threadColBase + j;
            if (col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

// ------------------------------------------------------------
// CPU 参考实现（朴素三重循环）。
// 只用来校验，跑一次就行，所以不优化。
// ------------------------------------------------------------
static void gemmCPU(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;              // 用 double 累加，减少 CPU 侧的累积误差
            for (int k = 0; k < K; ++k) {
                acc += (double)A[(size_t)i * K + k] * (double)B[(size_t)k * N + j];
            }
            C[(size_t)i * N + j] = (float)acc;
        }
    }
}

// 矩阵乘法的比较必须用相对误差：
// K 次累加会让误差随 K 增长，用固定绝对误差会误判。
static bool verifyGEMM(const std::vector<float>& got,
                       const std::vector<float>& ref,
                       int K) {
    double maxAbsErr = 0.0, maxRelErr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double e = fabs((double)got[i] - (double)ref[i]);
        maxAbsErr = std::max(maxAbsErr, e);
        double denom = std::max(1e-6, fabs((double)ref[i]));
        maxRelErr = std::max(maxRelErr, e / denom);
    }
    // 阈值随 K 放大：K=512 时 1e-3 量级的相对误差是完全正常的
    double tol = 1e-5 * K;
    printf("   最大绝对误差 = %.3e, 最大相对误差 = %.3e, 阈值 = %.3e -> %s\n",
           maxAbsErr, maxRelErr, tol, (maxRelErr < tol) ? "通过" : "失败");
    return maxRelErr < tol;
}

// 核函数指针类型别名，方便把三个 kernel 用同一段代码测试
using GemmKernel = void (*)(const float*, const float*, float*, int, int, int);

static void runCase(const char* name,
                    GemmKernel kernel,
                    dim3 grid, dim3 block,
                    const float* dA, const float* dB, float* dC,
                    const std::vector<float>& hRef,
                    std::vector<float>& hC,
                    int M, int N, int K,
                    size_t bytesC) {
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));

    double ms = CudaTimer::measure([&](double& m, int) {
        CudaTimer t;
        t.start();
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK_KERNEL();
        m = t.stopMs();
    }, 5, 2);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // 一共 2*M*N*K 次浮点运算（每个乘加算 2 flops）
    double flops = 2.0 * M * N * K;
    bool ok = verifyGEMM(hC, hRef, K);

    printf("\n----- %s -----\n", name);
    printf("   grid=(%u,%u) block=(%u,%u)  共 %u 线程\n",
           grid.x, grid.y, block.x, block.y,
           grid.x * grid.y * block.x * block.y);
    printf("   实测中位耗时: %.4f ms\n", ms);
    printf("   性能: %.1f GFLOPS\n", CudaTimer::gflops(flops, ms));
    printf("   结果: %s\n", ok ? "正确" : "错误");
}

int main() {
    printDeviceInfo();

    // 用 512x512x512 的中等规模：
    //   浮点运算量 = 2*512^3 ≈ 2.68e8 flops，够大到能看出差异；
    //   三个矩阵各 1 MB，小白机器毫无压力。
    const int M = 512, N = 512, K = 512;

    const size_t countA = (size_t)M * K;
    const size_t countB = (size_t)K * N;
    const size_t countC = (size_t)M * N;
    const size_t bytesA = countA * sizeof(float);
    const size_t bytesB = countB * sizeof(float);
    const size_t bytesC = countC * sizeof(float);

    std::vector<float> hA(countA), hB(countB), hRef(countC), hC(countC, 0.0f);
    // 填 -3..3 的小整数：方便人工核对，也避免浮点精度问题被放大
    srand(1234);
    for (size_t i = 0; i < countA; ++i) hA[i] = (float)(rand() % 7 - 3);
    for (size_t i = 0; i < countB; ++i) hB[i] = (float)(rand() % 7 - 3);

    printf("\n计算 CPU 参考结果（%d x %d x %d），这一步是串行的，会慢一点...\n", M, N, K);
    gemmCPU(hA, hB, hRef, M, N, K);
    printf("CPU 参考结果计算完成。\n");

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    // ============ v1 朴素 ============
    {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        runCase("v1 朴素 gemmNaive（无共享内存）", gemmNaive, grid, block,
                dA, dB, dC, hRef, hC, M, N, K, bytesC);
    }

    // ============ v2 共享内存分块 ============
    {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        runCase("v2 分块 gemmTiled（共享内存）", gemmTiled, grid, block,
                dA, dB, dC, hRef, hC, M, N, K, bytesC);
    }

    // ============ v3 寄存器分块 ============
    {
        // 块 16x16 = 256 线程，每线程 4x4 输出，块覆盖 64x64
        dim3 block(16, 16);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        runCase("v3 寄存器分块 gemmRegTiled", gemmRegTiled, grid, block,
                dA, dB, dC, hRef, hC, M, N, K, bytesC);
    }

    printf("\n================ 结论提示 ================\n");
    printf("1. v2 相对 v1 的加速来自减少全局内存访问，常见量级是 3~10 倍；\n");
    printf("2. v3 相对 v2 的加速来自减少共享内存访问并提高寄存器复用，\n");
    printf("   常见量级是再快 1.5~3 倍；\n");
    printf("3. 具体数字取决于显卡的显存带宽与算力之比，\n");
    printf("   在你的机器上实测约为多少可能不同，请以自己测到的为准；\n");
    printf("4. 想看细节：用 nvcc -O3 -Xptxas -v 编译本文件，\n");
    printf("   对比三个 kernel 各自的寄存器数与共享内存占用。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    printf("\n全部完成。\n");
    return 0;
}
