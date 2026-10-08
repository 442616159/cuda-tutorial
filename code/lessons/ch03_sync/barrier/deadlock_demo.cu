// ============================================================
//  code/lessons/ch03_sync/barrier/deadlock_demo.cu
//  第 3 章 示例 2：__syncthreads() 放进分支里会发生什么
//
//  默认编译出来的程序只会运行「安全版」并打印解释。
//  想亲眼看到 GPU 卡死，请这样编译：
//
//      nvcc -O2 -arch=sm_70 -DDEMO_DEADLOCK=1 deadlock_demo.cu -o deadlock_demo
//      ./deadlock_demo        # 会一直卡住，用 Ctrl+C 结束
//
//  正常编译：
//      nvcc -O2 -arch=sm_70 deadlock_demo.cu -o deadlock_demo
//      ./deadlock_demo
// ============================================================
#include "../../../common/cuda_check.h"
#include <cstdio>

#ifndef DEMO_DEADLOCK
#define DEMO_DEADLOCK 0
#endif

#define BLOCK 256

// ------------------------------------------------------------
// 错误版本：屏障写在 if 分支内部
//   warp0 ~ warp3 走到 __syncthreads()，warp4 ~ warp7 直接跳过。
//   __syncthreads() 的语义是「等块内所有线程到齐」，
//   而另一半线程永远不会到齐 —— 这就是死锁。
//   在 Volta 及以后的架构上，硬件不再保证「整个 warp 一起卡住」，
//   而是可能出现部分线程永久阻塞、kernel 永不返回的现象。
// ------------------------------------------------------------
__global__ void divergentBarrierBad(int* out) {
    int t = threadIdx.x;
    if (t < BLOCK / 2) {
        __syncthreads();          // ← 只有一半线程会执行到这里，非法！
        out[t] = 1;
    } else {
        out[t] = 2;               // 这一半线程永远不会参与屏障
    }
}

// ------------------------------------------------------------
// 正确版本 1：把屏障提到分支之外
//   屏障本身通常不需要「只让一半线程做」，
//   真正需要条件化的往往只是屏障前后的读写，而不是屏障本身。
// ------------------------------------------------------------
__global__ void hoistedBarrierGood(const int* in, int* out) {
    __shared__ int s[BLOCK];
    int t = threadIdx.x;
    int v = in[blockIdx.x * BLOCK + t];

    // 前半块线程负责把数据搬进 shared memory
    if (t < BLOCK / 2) {
        s[t] = v * 2;
    }

    // 屏障放在 if 外面 —— 无论走哪个分支，块内 256 个线程都会到达这里
    __syncthreads();

    // 后半块线程读取前半块搬进来的数据
    if (t >= BLOCK / 2) {
        out[blockIdx.x * BLOCK + t] = s[t - BLOCK / 2] + v;
    } else {
        out[blockIdx.x * BLOCK + t] = v;
    }
}

// ------------------------------------------------------------
// 正确版本 2：用「统一布尔量」代替分支
//   当每个线程的谓词不同、但大家都必须参与屏障时，
//   先算出条件（不改变控制流），屏障照常执行，之后再按条件行事。
// ------------------------------------------------------------
__global__ void predicatedBarrierGood(const int* in, int* out) {
    int t = threadIdx.x;
    int v = in[blockIdx.x * BLOCK + t];
    bool active = (v > 0);        // 把「要不要干活」写成布尔量，而不是 if

    __syncthreads();              // 所有线程无条件到达

    out[blockIdx.x * BLOCK + t] = active ? v * 10 : 0;
}

int main() {
    int* d_in = nullptr;
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, BLOCK * sizeof(int)));

    int h_in[BLOCK], h_out[BLOCK];
    for (int i = 0; i < BLOCK; ++i) h_in[i] = i + 1;   // 全正数
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));

#if DEMO_DEADLOCK
    printf("!!! 你打开了 DEMO_DEADLOCK=1，即将启动一个含非法屏障的 kernel。\n");
    printf("!!! 程序大概率不会返回，请用 Ctrl+C 结束，然后重新编译去掉该宏。\n");
    fflush(stdout);

    divergentBarrierBad<<<1, BLOCK>>>(d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_SYNC();      // ← 卡在这一行
#else
    printf("=== 屏障与分支的正确用法演示 ===\n\n");
    printf("被跳过的错误版本 divergentBarrierBad 长这样：\n");
    printf("    if (t < blockDim.x / 2) {\n");
    printf("        __syncthreads();   // 只有一半线程到集合点 -> 未定义行为\n");
    printf("        ...\n");
    printf("    } else { ... }\n");
    printf("想亲眼看到卡死，用 -DDEMO_DEADLOCK=1 重新编译。\n\n");

    // 版本 1：把屏障提到分支外
    hoistedBarrierGood<<<1, BLOCK>>>(d_in, d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    int bad1 = 0;
    for (int t = 0; t < BLOCK; ++t) {
        int want = (t >= BLOCK / 2) ? (h_in[t - BLOCK / 2] * 2 + h_in[t]) : h_in[t];
        if (h_out[t] != want) ++bad1;
    }
    printf("[提升屏障] 错误元素数 = %d / %d\n", bad1, BLOCK);

    // 版本 2：统一布尔量
    predicatedBarrierGood<<<1, BLOCK>>>(d_in, d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    int bad2 = 0;
    for (int t = 0; t < BLOCK; ++t) {
        int want = (h_in[t] > 0) ? h_in[t] * 10 : 0;
        if (h_out[t] != want) ++bad2;
    }
    printf("[谓词化条件] 错误元素数 = %d / %d\n", bad2, BLOCK);
#endif

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
