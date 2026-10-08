// ============================================================
//  code/lessons/ch02_memory/bank_conflict.cu
//  Bank Conflict（存储体冲突）的量化实验与消除方法。
//
//  共享内存被切成 32 个 bank，每 bank 宽 4 字节。
//  同一时刻，如果一个 warp 的 32 个线程访问的地址落在
//  不同 bank 上 -> 一次访存搞定（无冲突）。
//  如果多个线程落在同一个 bank 但地址不同 -> 硬件必须串行化，
//  32 路冲突就等于把共享内存访问速度除以 32。
//
//  本文件做三类实验：
//      A. 无冲突访问（stride 1，行主序，array[i]）
//      B. 冲突访问（array[i * 32]，所有线程落在同一 bank）
//      C. padding 消除冲突（[32][33] 布局的矩阵转置）
//
//  编译（在 code/lessons/ch02_memory 目录下）：
//      nvcc -O3 -arch=sm_75 bank_conflict.cu -o bank_conflict
//  运行：
//      ./bank_conflict        (Linux)
//      bank_conflict.exe      (Windows)
// ============================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.h"
#include "../../common/cuda_timer.h"

constexpr int BLOCK = 256;
constexpr int TILE = 32;
constexpr int ITER = 512;                 // 每个线程重复访问的次数，把共享内存访问放大成瓶颈
constexpr int SMEM_FLOATS = BLOCK * TILE; // 8192 个 float = 32 KB 共享内存上限内

// ------------------------------------------------------------
// A. stride = 1：无冲突
//
// 线程 t 访问 buf[t * stride]。stride = 1 时，
// 32 个线程访问 32 个连续的 4 字节字 —— 恰好填满 32 个 bank，
// 每个 bank 一个线程，一次搞定。
// ------------------------------------------------------------
__global__ void strideAccessKernel(float* __restrict__ out, int stride) {
    // 8192 * 4 = 32 KB。大于 48 KB 会在老卡上直接启动失败，这个尺寸是安全的。
    __shared__ float buf[SMEM_FLOATS];

    // 先填一个确定的值，避免读到垃圾
    for (int i = threadIdx.x; i < SMEM_FLOATS; i += blockDim.x) {
        buf[i] = (float)(i % 17);
    }
    __syncthreads();

    float acc = 0.0f;
    for (int it = 0; it < ITER; ++it) {
        // 每次访问挪一下起点。用 & 掩码代替取模，
        // 因为 SMEM_FLOATS 正好是 2 的幂（8192），二者等价且更快。
        int idx = (threadIdx.x * stride + it) & (SMEM_FLOATS - 1);
        acc += buf[idx];
    }

    // 把结果汇总到共享内存里，防止编译器把整个循环优化掉
    // （如果算出来的值没人用，优化器有权删掉它）
    __shared__ float partial[BLOCK];
    partial[threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < BLOCK; ++i) s += partial[i];
        // 写入全局内存，保证结果"被观察到"
        out[blockIdx.x] = s;
    }
}

// ------------------------------------------------------------
// B. 二维 tile 里按"列"访问：典型的 32 路冲突
//
// tile[THREAD_ROW][STRIDE]，线程 (tx, ty) 访问 tile[ty][tx]，
// 也就是说：同一个 warp 内 tx 在变、ty 不变，大家访问的是"同一行"。
//
//   STRIDE = 32：地址 = ty * 32 + tx，bank = (ty*32 + tx) % 32 = tx。
//                32 个线程的 tx 各不相同，于是落在 32 个不同 bank —— 无冲突。
//   STRIDE = 33：地址 = ty * 33 + tx，bank = (ty*33 + tx) % 32 = (ty + tx) % 32。
//                线程 0 和线程 31 会落在同一个 bank（差 31 或 33 冲突），
//                这就是 padding 反而制造出来的冲突。
//
// 所以"按行访问该用 32 列、按列访问才需要 padding"——
// 关键在于"同一 warp 内相邻线程的地址差"是否与 32 互质。
// ------------------------------------------------------------
template <int STRIDE>
__global__ void rowAccessKernel(float* __restrict__ out) {
    __shared__ float tile[TILE][STRIDE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 初始化：按行写
    tile[ty][tx] = (float)((ty * 7 + tx) % 13);
    __syncthreads();

    float acc = 0.0f;
    for (int it = 0; it < 8; ++it) {
        int dx = (tx + it) % TILE;
        #pragma unroll
        for (int ty2 = 0; ty2 < TILE; ++ty2) {
            // 固定 ty2、让 dx 变化 —— 这正是 warp 内 tx 在变的场景
            acc += tile[ty2][dx] * 0.5f;
        }
    }
    __syncthreads();

    // 把结果汇总回共享内存，避免整段被优化掉
    __shared__ float partial[TILE][TILE + 1];
    partial[ty][tx] = acc;
    __syncthreads();
    if (ty == 0 && tx == 0) {
        float s = 0.0f;
        for (int i = 0; i < TILE; ++i)
            for (int j = 0; j < TILE; ++j) s += partial[i][j];
        out[blockIdx.x] = s;
    }
}

// ------------------------------------------------------------
// C. 真实案例：转置 + 可选 padding
//
// 用模板参数控制是否 padding，同一个 kernel 编译两份，
// 这样对比最公平（除了 padding 之外一切相同）。
// ------------------------------------------------------------
template <int PADDING>
__global__ void transposeTiledKernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int n) {
    __shared__ float tile[TILE][TILE + PADDING];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    if (row < n && col < n) {
        tile[ty][tx] = in[row * n + col];      // 读：合并；写共享内存：无冲突
    }
    __syncthreads();
    if (row < n && col < n) {
        // 交换坐标：这是转置的关键。读 tile[tx][ty] 时，
        // 同一 warp 内 tx 变化 -> 地址跨 TILE 个元素。
        // PADDING=0 时 bank = ty 固定（冲突）；
        // PADDING=1 时 bank = (tx+ty)%32（无冲突）。
        out[col * n + row] = tile[tx][ty];
    }
}

int main() {
    printDeviceInfo();

    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, 4096 * sizeof(float)));

    printf("\n========== 实验 1：一维数组的不同 stride ==========\n");
    printf("每个线程做 %d 次共享内存读，网格 256 个块 x 256 线程\n", ITER);
    printf("--------------------------------------------------------------------------------\n");

    for (int stride : {1, 2, 4, 8, 16, 32}) {
        double ms = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            strideAccessKernel<<<256, BLOCK>>>(dOut, stride);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);

        // 一个块 8 个 warp，256 个块 = 2048 个 warp
        // 每个 warp 做 ITER 次共享内存读
        double accesses = 2048.0 * ITER;
        printf("  stride = %2d   耗时 %8.4f ms   (%.2f G 次共享内存访问/s)\n",
               stride, ms, accesses / (ms * 1e-3) / 1e9);
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("  规律：stride 与 32 不互质时冲突严重，stride = 32 时最慢（32 路冲突）。\n");
    printf("        stride 与 32 互质（如 1、3、5）时完全无冲突。\n");
    printf("        直觉：32 个线程访问的地址相差 stride 个字，\n");
    printf("        当 stride 与 32 互质时它们必然落在 32 个不同 bank 上。\n");

    printf("\n========== 实验 2：二维 tile 里按行访问 ==========\n");
    printf("--------------------------------------------------------------------------------\n");
    {
        dim3 block(TILE, TILE);
        double msNoPad = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            rowAccessKernel<TILE><<<64, block>>>(dOut);          // [32][32]：无冲突
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);
        double msPad = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            rowAccessKernel<TILE + 1><<<64, block>>>(dOut);      // [32][33]：人为制造冲突
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);

        printf("  [32][32] 不 padding   耗时 %8.4f ms   （按行访问，本来就无冲突）\n", msNoPad);
        printf("  [32][33] 加 padding   耗时 %8.4f ms   （反而引入冲突，对照用）\n", msPad);
        printf("  padding 带来的惩罚: %.2f x\n", msPad / msNoPad);
        printf("  结论：padding 不是万能药，要看你的 warp 内访问方向。\n");
    }

    printf("\n========== 实验 3：完整的矩阵转置（2048x2048） ==========\n");
    printf("--------------------------------------------------------------------------------\n");
    {
        const int N = 2048;
        const size_t bytes = (size_t)N * N * sizeof(float);
        std::vector<float> hIn((size_t)N * N);
        for (size_t i = 0; i < hIn.size(); ++i) hIn[i] = (float)(i % 251);

        float *dIn = nullptr, *dT = nullptr;
        CUDA_CHECK(cudaMalloc(&dIn, bytes));
        CUDA_CHECK(cudaMalloc(&dT, bytes));
        CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        double msNoPad = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            transposeTiledKernel<0><<<grid, block>>>(dIn, dT, N);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);

        double msPad = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            transposeTiledKernel<1><<<grid, block>>>(dIn, dT, N);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);

        printf("  无 padding (tile[32][32])   耗时 %8.4f ms\n", msNoPad);
        printf("  有 padding (tile[32][33])   耗时 %8.4f ms\n", msPad);
        printf("  加速比: %.2f x\n", msNoPad / msPad);
        printf("  多用的共享内存: %zu 字节 / 块（约 %.1f%%）\n",
               (size_t)TILE * sizeof(float),
               100.0 * TILE * sizeof(float) / (TILE * TILE * sizeof(float)));

        CUDA_CHECK(cudaFree(dIn));
        CUDA_CHECK(cudaFree(dT));
    }

    printf("\n--------------------------------------------------------------------------------\n");
    printf("怎么解读：\n");
    printf("  1. padding 只多用了 3%% 的共享内存，却常常能带来 1.5~3 倍的转置加速。\n");
    printf("  2. 加速比的大小取决于你的 kernel 是共享内存受限还是显存带宽受限。\n");
    printf("     当显存带宽成为瓶颈时，消除 bank conflict 的收益会被掩盖。\n");
    printf("  3. 在你的机器上实测约为多少可能不同，请以自己测到的为准。\n");
    printf("  4. 用 ncu 可以精确看到 bank conflict 次数：\n");
    printf("     ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./bank_conflict\n");

    CUDA_CHECK(cudaFree(dOut));
    printf("\n全部完成。\n");
    return 0;
}
