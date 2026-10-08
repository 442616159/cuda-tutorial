// ============================================================
//  code/lessons/ch03_atomics/atomics.cu
//  第 3 章 示例 7：原子操作全家桶
//
//  演示：
//    1. 非原子自增（错误示范） vs atomicAdd（正确）
//    2. atomicMax（顺便演示 float 取最大值的位技巧）
//    3. atomicExch（一次性开关 / 解锁）
//    4. atomicCAS：手写 double 原子的加、手写自旋锁
//    5. 争用（contention）：所有线程砸同一个地址 vs 砸不同地址
//
//  编译： nvcc -O2 -arch=sm_70 atomics.cu -o atomics
//  运行： ./atomics
// ============================================================
#include "../../common/cuda_check.h"
#include "../../common/cuda_bench.h"
#include <cstdio>
#include <cmath>

#define BLOCK 256
#define N     (1 << 20)          // 1048576 个元素

// 设备端全局变量：自旋锁与一次性开关
__device__ int g_mutex = 0;
__device__ int g_flag  = 0;

// ------------------------------------------------------------
// 1) 非原子自增：错误示范
//    *counter = *counter + 1 会被编译成「读 -> 加 -> 写」三步。
//    多个线程交错执行时，两次自增可能只留下一次的效果（丢失更新）。
//    这里把指针声明为 volatile，是为了防止编译器直接优化成 *counter += iters，
//    从而让"竞态"真实发生。
// ------------------------------------------------------------
__global__ void countNonAtomic(volatile int* counter, int iters) {
    for (int k = 0; k < iters; ++k) {
        *counter = *counter + 1;
    }
}

// ------------------------------------------------------------
// 2) 原子自增：无论多少线程、多少次，一次都不会丢
// ------------------------------------------------------------
__global__ void countAtomic(int* counter, int iters) {
    for (int k = 0; k < iters; ++k) {
        atomicAdd(counter, 1);
    }
}

// ------------------------------------------------------------
// 3) atomicMax 求最大值
//    内建 atomicMax 只支持整数类型，没有 float 版本。
//    对「非负浮点数」有一个经典技巧：IEEE-754 的位模式
//    与它作为整数比较的大小顺序是一致的，
//    所以把 float 按位重新解释成 int，再 atomicMax 即可。
//    注意：只对非负数成立（负数的最高位是符号位，会排到很大）。
// ------------------------------------------------------------
__global__ void findMaxNonNegative(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    if (v < 0.f) return;              // 本例约定输入非负，负数直接跳过
    atomicMax(reinterpret_cast<int*>(out), __float_as_int(v));
}

// ------------------------------------------------------------
// 4) atomicExch：把地址换成新值，返回旧值
//    经典用法：一次性开关（谁先到谁赢）
// ------------------------------------------------------------
__global__ void firstArrival(int* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int prev = atomicExch(&g_flag, 1);   // 把 g_flag 置 1，拿到它原来的值
    if (prev == 0) {                      // 只有"原来就是 0"的那一个线程进来
        out[0] = t;
    }
}

// ------------------------------------------------------------
// 5) 用 atomicCAS 手写 double 版本原子加
//    CUDA 自 CC 6.0（sm_60）起才对 double 提供内建 atomicAdd，
//    老架构或想理解 CAS 原理时，就这样自己转圈。
// ------------------------------------------------------------
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
    // CAS 作用在 64 位无符号整数上（double 与它同宽）
    unsigned long long* p = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *p;
    unsigned long long assumed;
    do {
        assumed = old;
        // 用 assumed 还原出当前值，加上 val，再尝试一次 CAS；
        // 如果期间被别的线程改过，CAS 会失败并返回最新值，继续重试。
        old = atomicCAS(p, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void sumDoubleCAS(const double* in, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAddDouble(out, in[i]);
}

// ------------------------------------------------------------
// 6) 自旋锁
//    上锁：反复 CAS(0 -> 1)，成功即拿到锁
//    解锁：atomicExch 置回 0
//    注意：解锁前必须 __threadfence()，保证临界区里的写先被看见，
//          否则别的线程可能拿到锁却读到旧数据。
// ------------------------------------------------------------
__global__ void lockedIncrement(volatile int* counter, int iters) {
    for (int k = 0; k < iters; ++k) {
        // ---- 上锁 ----
        while (atomicCAS(&g_mutex, 0, 1) != 0) {
            // 自旋等待。真实工程里应该插入 __nanosleep(20) 之类的退避，
            // 否则大量线程空转会把内存总线占满。
        }
        // ---- 临界区 ----
        int tmp = *counter;
        *counter = tmp + 1;
        // ---- 解锁 ----
        __threadfence();          // 确保临界区内的写对其他线程可见
        atomicExch(&g_mutex, 0);
    }
}

// ------------------------------------------------------------
// 7) 争用对比：同样次数的原子操作，打向不同地址
// ------------------------------------------------------------
__global__ void atomToOneAddress(int* counter, int iters) {
    for (int k = 0; k < iters; ++k) atomicAdd(counter, 1);
}

__global__ void atomToOwnAddress(int* counters, int iters) {
    // 每个线程打自己的地址，互不冲突（地址按 128 字节对齐分布，
    // 避免落在同一个 L2 sector / 同一个 bank）
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int* mine = counters + (i * 32);
    for (int k = 0; k < iters; ++k) atomicAdd(mine, 1);
}

int main() {
    printf("=== 原子操作演示 ===\n");

    // ---------- 1. 计数：非原子 vs 原子 ----------
    {
        const int grid = 64, block = BLOCK, iters = 200;
        int* d_cnt = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int)));
        int zero = 0;
        int expect = grid * block * iters;

        CUDA_CHECK(cudaMemcpy(d_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
        countNonAtomic<<<grid, block>>>(d_cnt, iters);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
        int got = 0;
        CUDA_CHECK(cudaMemcpy(&got, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        printf("\n[非原子自增] 期望 %d，实际 %d，丢了 %d 次更新\n",
               expect, got, expect - got);
        printf("             (具体丢多少每次都不一样，这就是「竞态」的特征)\n");

        CUDA_CHECK(cudaMemcpy(d_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
        countAtomic<<<grid, block>>>(d_cnt, iters);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&got, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[原子自增]   期望 %d，实际 %d，%s\n",
               expect, got, got == expect ? "完全一致 ✓" : "出错了 ✗");

        // 计时对比（原子版因为要串行化到同一地址，通常反而更慢）
        CUDA_CHECK(cudaMemcpy(d_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
        double tNon = medianMs([&] { countNonAtomic<<<grid, block>>>(d_cnt, iters); }, 3, 10);
        CUDA_CHECK(cudaMemcpy(d_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
        double tAtom = medianMs([&] { countAtomic<<<grid, block>>>(d_cnt, iters); }, 3, 10);
        printf("             非原子版 %.4f ms（结果错误），原子版 %.4f ms（结果正确）\n",
               tNon, tAtom);
        printf("             提示：正确性优先，原子版慢一点是值得的。\n");

        CUDA_CHECK(cudaFree(d_cnt));
    }

    // ---------- 2. atomicMax ----------
    {
        float* h = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; ++i) h[i] = (float)((i * 37) % 10000) * 0.001f;
        float* d_in = nullptr;
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h, N * sizeof(float), cudaMemcpyHostToDevice));
        float init = 0.f;
        CUDA_CHECK(cudaMemcpy(d_out, &init, sizeof(float), cudaMemcpyHostToDevice));

        findMaxNonNegative<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_out, N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        float got = 0.f;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        float want = 0.f;
        for (int i = 0; i < N; ++i) if (h[i] > want) want = h[i];
        printf("\n[atomicMax] GPU = %.6f，CPU = %.6f，%s\n",
               got, want, fabsf(got - want) < 1e-6f ? "一致 ✓" : "不一致 ✗");
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        free(h);
    }

    // ---------- 3. atomicExch ----------
    {
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
        int zero = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(g_flag, &zero, sizeof(int)));   // 复位开关
        firstArrival<<<16, BLOCK>>>(d_out);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
        int winner = -1;
        CUDA_CHECK(cudaMemcpy(&winner, d_out, sizeof(int), cudaMemcpyDeviceToHost));
        printf("\n[atomicExch] 抢到开关的全局线程号 = %d（每次运行可能不同）\n", winner);
        printf("             硬件不保证谁先到，这个例子只用来演示语义。\n");
        CUDA_CHECK(cudaFree(d_out));
    }

    // ---------- 4. double 原子加（CAS 版） ----------
    {
        const int n = 1 << 18;
        double* h = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; ++i) h[i] = 1.0;
        double* d_in = nullptr;
        double* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_in, h, n * sizeof(double), cudaMemcpyHostToDevice));
        double zero = 0.0;
        CUDA_CHECK(cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice));

        sumDoubleCAS<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_out, n);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
        double got = 0.0;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(double), cudaMemcpyDeviceToHost));
        printf("\n[atomicCAS 手写 double 原子加] 结果 = %.1f，期望 = %d，%s\n",
               got, n, fabs(got - (double)n) < 1e-6 ? "一致 ✓" : "不一致 ✗");
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        free(h);
    }

    // ---------- 5. 自旋锁 ----------
    {
        const int grid = 4, block = 32, iters = 2000;
        int* d_cnt = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int)));
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(g_mutex, &zero, sizeof(int)));  // 复位锁

        lockedIncrement<<<grid, block>>>(d_cnt, iters);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        int got = 0;
        CUDA_CHECK(cudaMemcpy(&got, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        printf("\n[自旋锁] 期望 %d，实际 %d，%s\n",
               grid * block * iters, got,
               got == grid * block * iters ? "一致 ✓" : "不一致 ✗");
        printf("         把 grid 调大你会看到它慢得离谱 —— 大量线程纯空转。\n");
        CUDA_CHECK(cudaFree(d_cnt));
    }

    // ---------- 6. 争用（contention）对比 ----------
    {
        const int grid = 128, block = BLOCK, iters = 200;
        // 一个计数器：一个地址
        int* d_one = nullptr;
        CUDA_CHECK(cudaMalloc(&d_one, sizeof(int)));
        // 每线程一个计数器，且彼此间隔 32 个 int（128 字节）避免伪共享
        int total = grid * block;
        int* d_many = nullptr;
        CUDA_CHECK(cudaMalloc(&d_many, (size_t)total * 32 * sizeof(int)));
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_one, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_many, 0, (size_t)total * 32 * sizeof(int)));

        double tSame = medianMs(
            [&] { atomToOneAddress<<<grid, block>>>(d_one, iters); }, 3, 20);
        double tOwn = medianMs(
            [&] { atomToOwnAddress<<<grid, block>>>(d_many, iters); }, 3, 20);

        printf("\n--- 原子争用对比（每线程 %d 次原子加）---\n", iters);
        printf("  全部砸同一个地址 : %8.4f ms\n", tSame);
        printf("  每线程独立地址   : %8.4f ms\n", tOwn);
        printf("  实测约为 %.1f 倍差距（你的机器可能不同）。\n", tSame / tOwn);
        printf("  原因：对同一地址的原子操作必须在 L2 里串行排队，\n");
        printf("        而不同地址可以并行在不同 bank/sector 上完成。\n");

        CUDA_CHECK(cudaFree(d_one));
        CUDA_CHECK(cudaFree(d_many));
    }

    printf("\n完成。\n");
    return 0;
}
