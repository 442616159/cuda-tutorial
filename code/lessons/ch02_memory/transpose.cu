// ============================================================
//  code/lessons/ch02_memory/transpose.cu
//  矩阵转置的三个版本，专门用来演示"合并访存"的威力：
//
//      v1 transposeNaive    直接对着读，写的时候是跨步的
//      v2 transposeShared   用共享内存把访问模式掰正
//      v3 transposePadded   共享内存 + padding，消除 bank conflict
//
//  编译（在 code/lessons/ch02_memory 目录下）：
//      nvcc -O3 -arch=sm_75 transpose.cu -o transpose
//  运行：
//      ./transpose        (Linux)
//      transpose.exe      (Windows)
// ============================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.h"
#include "../../common/cuda_timer.h"

// 32x32 的 tile，pad 之后的 stride 是 33
constexpr int BDIM = 32;
constexpr int PAD = 1;

// ------------------------------------------------------------
// v1: 朴素转置
//
//   out[col][row] = in[row][col]      （行列互换）
//
// 问题：
//   读 in 的时候，相邻线程读相邻地址 —— 好的，合并的。
//   写 out 的时候，线程 (tx=0..31) 写的是 out[0..31][blockRow]，
//   在行主序下这些地址相隔 N 个 float —— 完全散开。
//   一个 warp 的 32 次写会变成 32 个独立的内存事务，
//   有效带宽可能掉到峰值的 1/8 ~ 1/32。
//
// 这就是"读合并、写不合并"的经典反例。
// ------------------------------------------------------------
__global__ void transposeNaive(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n) {
    int col = blockIdx.x * BDIM + threadIdx.x;
    int row = blockIdx.y * BDIM + threadIdx.y;

    if (row < n && col < n) {
        // 写：out 的第 col 行、第 row 列
        out[col * n + row] = in[row * n + col];
    }
}

// ------------------------------------------------------------
// v2: 用共享内存做"中转站"
//
// 思路：把"不合并的那一半"从全局内存挪到共享内存上。
//   阶段 1（读）：线程按 (row, col) 读 in —— 全局内存上合并访存
//   阶段 2（存）：写进共享内存 tile[ty][tx]
//   阶段 3（取）：从共享内存里"横着"取 tile[tx][ty] —— 转置就完成了
//   阶段 4（写）：线程按 (col, row) 写 out —— 全局内存上合并访存
//
// 为什么这样能变快？
//   朴素版的病根是"写全局内存时一个 warp 的 32 个地址相距 n 个 float"，
//   一次写被拆成 32 个事务。现在把这个"散开"的动作搬到了共享内存里
//   （延迟几十个时钟周期，而不是几百个），
//   全局内存上的读和写都变成了连续的 128 字节。
//
// v2 还能再快吗？能。共享内存这一层仍然有 bank conflict 的隐患，
// 而且 v2 的写法本身在多数架构上并不产生严重的冲突 —— 所以先看数据，
// 再决定要不要上 v3 的 padding。
// ------------------------------------------------------------
__global__ void transposeShared(const float* __restrict__ in,
                                float* __restrict__ out,
                                int n) {
    __shared__ float tile[BDIM][BDIM];   // 32x32，无 padding

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * BDIM + tx;
    int row = blockIdx.y * BDIM + ty;

    // 阶段 1+2：合并地读全局内存，写进共享内存
    if (row < n && col < n) {
        tile[ty][tx] = in[row * n + col];
    }
    __syncthreads();   // 必须等整个 tile 填满，才能开始读它

    // 阶段 3+4：交换坐标后写回全局内存（写是合并的）
    int outRow = blockIdx.x * BDIM + ty;   // 注意这里 tx/ty 互换了角色
    int outCol = blockIdx.y * BDIM + tx;
    if (outRow < n && outCol < n) {
        out[outRow * n + outCol] = tile[tx][ty];   // ← 这里会 bank conflict
    }
}

// ------------------------------------------------------------
// v3: 共享内存 + padding（进阶版）
//
// 只改一件事：把 tile 声明成 [BDIM][BDIM + PAD]，也就是 32x33。
//
// 什么时候 padding 有用？
//   当"同一 warp 内的相邻线程访问共享内存的地址差"是 32 的倍数时，
//   它们会全部落在同一个 bank 上（32 路冲突）。
//   把行宽从 32 改成 33 之后，地址差变成 33，
//   而 33 与 32 互质，于是 32 个线程必然落在 32 个不同的 bank 上。
//
// 计算一下：tile[tx][ty] 在 33 列布局下的地址是 tx*33 + ty，
//   bank = (tx*33 + ty) % 32 = (tx + ty) % 32。
//   固定 i、让 tx 变化时，(i + tx) 覆盖了全部 32 个 bank —— 无冲突。
//
// 代价：共享内存用量从 4096 字节变成 4224 字节（多 3%），可以忽略。
// 收益：如果原本存在 bank conflict，通常能显著变快。
// 注意：**不是所有转置写法都有 bank conflict**，请先用数据说话。
// ------------------------------------------------------------
__global__ void transposePadded(const float* __restrict__ in,
                                float* __restrict__ out,
                                int n) {
    // [32][33]：多出的一列就是"错位"，专门用来打散 bank
    __shared__ float tile[BDIM][BDIM + PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * BDIM + tx;
    int row = blockIdx.y * BDIM + ty;

    if (row < n && col < n) {
        tile[ty][tx] = in[row * n + col];
    }
    __syncthreads();

    int outRow = blockIdx.x * BDIM + ty;
    int outCol = blockIdx.y * BDIM + tx;
    if (outRow < n && outCol < n) {
        out[outRow * n + outCol] = tile[tx][ty];
    }
}

// ------------------------------------------------------------
// 参照组：读和写都不合并的最坏情况
//
// 它算的东西和 v1 一样（out = in 的转置），但"列优先"地去访问：
//   in[col * n + row] 与 out[row * n + col]
// 同一个 warp 内 col 在变，这两个地址都相差 n 个 float ——
// 读不合并、写也不合并。这是访存模式的绝对下限。
// ------------------------------------------------------------
__global__ void transposeAllStrided(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n) {
    int col = blockIdx.x * BDIM + threadIdx.x;
    int row = blockIdx.y * BDIM + threadIdx.y;
    if (row < n && col < n) {
        // 读：地址随 col 跨 n 个 float（不合并）
        float v = in[col * n + row];
        // 写：地址随 col 跨 n 个 float（不合并）
        out[row * n + col] = v;
    }
}

using TransKernel = void (*)(const float*, float*, int);

static void runTranspose(const char* label,
                         TransKernel k,
                         const float* dIn, float* dOut,
                         const std::vector<float>& hRef,
                         std::vector<float>& hOut,
                         int n) {
    const size_t bytes = (size_t)n * n * sizeof(float);
    CUDA_CHECK(cudaMemset(dOut, 0, bytes));

    dim3 block(BDIM, BDIM);
    dim3 grid((n + BDIM - 1) / BDIM, (n + BDIM - 1) / BDIM);

    double ms = CudaTimer::measure([&](double& m, int) {
        CudaTimer t;
        t.start();
        k<<<grid, block>>>(dIn, dOut, n);
        CUDA_CHECK_KERNEL();
        m = t.stopMs();
    }, 10, 2);

    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < hRef.size(); ++i) {
        if (fabs(hOut[i] - hRef[i]) > 1e-5f) { ok = false; break; }
    }

    // 一次转置读 n*n 个 float、写 n*n 个 float，共 8*n*n 字节
    double bytesMoved = 8.0 * n * n;
    printf("  %-22s 耗时 %8.4f ms   有效带宽 %8.1f GB/s   结果 %s\n",
           label, ms, CudaTimer::bandwidthGBs(bytesMoved, ms),
           ok ? "正确" : "错误");
}

int main() {
    printDeviceInfo();

    // 2048x2048 = 4M 个 float = 16 MB，输入输出共 32 MB。
    const int N = 2048;
    const size_t count = (size_t)N * N;
    const size_t bytes = count * sizeof(float);

    std::vector<float> hIn(count), hRef(count), hOut(count, 0.0f);
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            hIn[(size_t)r * N + c] = (float)(r * 3 + c);   // 值唯一，便于查错
        }
    }
    // CPU 参考转置
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            hRef[(size_t)c * N + r] = hIn[(size_t)r * N + c];
        }
    }

    float *dIn = nullptr, *dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, bytes));
    CUDA_CHECK(cudaMalloc(&dOut, bytes));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    printf("\n矩阵大小 %d x %d，共搬运 %.1f MB（读+写）\n", N, N, 2.0 * bytes / 1024 / 1024);
    printf("--------------------------------------------------------------------\n");
    runTranspose("v1 朴素（写跨步）", transposeNaive, dIn, dOut, hRef, hOut, N);
    runTranspose("v2 共享内存", transposeShared, dIn, dOut, hRef, hOut, N);
    runTranspose("v3 共享内存+Padding", transposePadded, dIn, dOut, hRef, hOut, N);
    runTranspose("参照：读写都不合并", transposeAllStrided, dIn, dOut, hRef, hOut, N);
    printf("--------------------------------------------------------------------\n");

    printf("\n怎么解读这些数字：\n");
    printf("  v1 的关键问题是写不合并：一个 warp 的 32 次写会拆成多个事务。\n");
    printf("  v2 把全局内存的读和写都掰成了合并访存，通常比 v1 快好几倍。\n");
    printf("  v3 只多用一个 padding，用来消除共享内存里可能的 bank conflict。\n");
    printf("     如果你的卡上 v3 和 v2 差不多，说明这个写法本来就没有明显冲突，\n");
    printf("     这本身也是一个有价值的结论 —— 优化要用数据说话，不要迷信技巧。\n");
    printf("  在什么卡上实测约为多少，你的机器可能不同，请以自己测到的为准。\n");
    printf("\n  想验证 bank conflict：用 ncu 看 l1tex__data_bank_conflicts_pipe_lsu 指标，\n");
    printf("  或者把 v3 的 PAD 改成 0 再跑一遍对比。\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    printf("\n全部完成。\n");
    return 0;
}
