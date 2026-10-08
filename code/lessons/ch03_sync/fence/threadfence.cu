// ============================================================
//  code/lessons/ch03_sync/fence/threadfence.cu
//  第 3 章 示例 4：__threadfence() —— 让「写」在全设备范围内可见
//
//  演示一个经典模式：用「原子领号 + threadfence」实现一次性全局归约，
//  只启动一个 kernel 就能算出整个数组的和。
//
//  想看看去掉 fence 会怎样，可以这样编译：
//      nvcc -O2 -arch=sm_70 -DNO_FENCE=1 threadfence.cu -o threadfence
//      ./threadfence     # 结果可能不稳定（取决于硬件与调度）
//
//  正常编译：
//      nvcc -O2 -arch=sm_70 threadfence.cu -o threadfence
//      ./threadfence
// ============================================================
#include "../../../common/cuda_check.h"
#include <cstdio>
#include <cmath>

#ifndef NO_FENCE
#define NO_FENCE 0
#endif

#define BLOCK 256
#define MAX_GRID 1024

// 全局票据计数器：每个块干完活来这里领一个号
__device__ unsigned int g_ticket = 0;
// 每个块的部分和
__device__ float g_partial[MAX_GRID];

// ------------------------------------------------------------
// __threadfence() 的三个变体（详解见正文）：
//   __threadfence()          : 全设备范围，保证本线程之前的写对设备内所有线程可见
//   __threadfence_block()    : 只保证块内可见，比全设备版便宜
//   __threadfence_system()   : 延伸到主机与其他 GPU（页锁定内存场景）
//
// 它不阻塞、不同步任何其他线程，只约束「本线程的访存顺序与可见性」。
// ------------------------------------------------------------

// 重置票据（每次启动前调用）
__global__ void resetTicket() {
    if (blockIdx.x == 0 && threadIdx.x == 0) g_ticket = 0;
}

__global__ void reduceOnePass(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // ---- 第一步：每个线程用网格步长循环累加自己负责的元素 ----
    float v = 0.f;
    for (int i = bid * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        v += in[i];
    }

    // ---- 第二步：块内归约（shared memory 版本） ----
    s[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        // 每一轮都必须同步：下一次循环会读别人刚写的值
        __syncthreads();
    }

    // ---- 第三步：块内 0 号线程负责「交卷」 ----
    if (tid == 0) {
        g_partial[bid] = s[0];      // (1) 把自己的部分和写到全局内存

#if !NO_FENCE
        // (2) 关键的一行：
        //     g_partial[bid] = s[0] 只是一条普通的写指令，它可能还躺在
        //     写缓冲里。如果没有 fence，编译器/硬件允许把下面的 atomicInc
        //     排到这条写之前，于是「最后一个块」去读别的块的 g_partial 时，
        //     可能读到的还是旧值 0。
        //     __threadfence() 保证：本线程在它之前的所有写，都已经对
        //     全设备可见，才会继续往下走。
        __threadfence();
#endif

        // (3) 原子领号。atomicInc(&t, gridDim.x) 的语义是
        //     old = *t; *t = (old >= gridDim.x) ? 0 : old + 1; return old;
        //     所以 a 依次取到 0, 1, 2, ..., gridDim.x-1。
        unsigned int a = atomicInc(&g_ticket, gridDim.x);

        // (4) 拿到最后一个号的块，说明其他块都已经交卷了
        if (a == gridDim.x - 1) {
            float sum = 0.f;
            for (int b = 0; b < gridDim.x; ++b) {
                sum += g_partial[b];
            }
            *out = sum;
        }
    }
}

int main() {
    const int n = 1 << 20;                 // 1048576 个 float，共 4 MB
    const float value = 1.0f;              // 每个元素都是 1.0f，方便心算校验

    printf("=== __threadfence 全局归约演示 (n=%d) ===\n", n);
#if NO_FENCE
    printf("!! 本次编译去掉了 __threadfence()（-DNO_FENCE=1）\n");
#endif

    float* h_in = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) h_in[i] = value;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    // 选择网格大小：这里用 SM 数量的若干倍，保证填满机器又不越界
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int grid = prop.multiProcessorCount * 8;
    if (grid > MAX_GRID) grid = MAX_GRID;
    printf("设备 %s：%d 个 SM，使用 %d 个块\n", prop.name, prop.multiProcessorCount, grid);

    // 重复跑几次，观察结果是否稳定（不稳定就说明存在可见性问题）
    int okCount = 0;
    float last = 0.f;
    for (int rep = 0; rep < 10; ++rep) {
        resetTicket<<<1, 1>>>();
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

        reduceOnePass<<<grid, BLOCK>>>(d_in, d_out, n);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&last, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        float expect = (float)n * value;
        if (fabsf(last - expect) < 1e-3f) ++okCount;
    }

    printf("10 次运行中结果正确 %d 次，最后一次结果 = %.1f，期望 = %.1f\n",
           okCount, last, (float)n * value);

    if (okCount == 10) {
        printf("说明 fence + 原子领号的顺序约束被正确遵守。\n");
    } else {
        printf("出现了错误结果 —— 这正说明「写完就领号」这种顺序不能依赖运气，\n");
        printf("必须用 __threadfence() 明确约束。\n");
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    return 0;
}
