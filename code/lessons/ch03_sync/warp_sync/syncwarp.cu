// ============================================================
//  code/lessons/ch03_sync/warp_sync/syncwarp.cu
//  第 3 章 示例 3：__syncwarp() 与 warp 级同步
//
//  需要 Compute Capability >= 7.0（Volta 及以上），编译时请加 -arch=sm_70。
//
//  本程序演示：
//    1. warp 内通过 shared memory 传递数据时，__syncwarp() 就够了
//    2. __syncwarp(mask) 的掩码含义：只同步掩码里点名的 lane
//    3. 用 __activemask() 观察「此刻哪些 lane 还在执行」 —— 并说明为什么不要滥用它
//
//  编译： nvcc -O2 -arch=sm_70 syncwarp.cu -o syncwarp
//  运行： ./syncwarp
// ============================================================
#include "../../../common/cuda_check.h"
#include <cstdio>

#define N 256

// ------------------------------------------------------------
// 场景：warp 内做「右移一格」的数据交换。
//   lane i 想拿到 lane i-1 的值，lane 0 拿 0。
//   用 shared memory 实现时，只需要 warp 级同步：
//   因为同一个 warp 的线程是同一个调度单位，硬件层面天然接近同步，
//   但编译器优化和 Volta 的独立线程调度仍可能打乱顺序，
//   所以「同 warp 内通过 shared 传数据」的正规写法是 __syncwarp()。
// ------------------------------------------------------------
__global__ void warpShiftShared(const int* in, int* out) {
    __shared__ int s[N];
    int t = threadIdx.x;
    int lane = t & 31;

    s[t] = in[t];

    // 只需要本 warp 的 32 个 lane 对齐；用 __syncthreads() 会拖上整块，
    // 属于「大炮打蚊子」，块越大浪费越明显。
    __syncwarp();

    out[t] = (lane == 0) ? 0 : s[t - 1];
}

// ------------------------------------------------------------
// 演示 __syncwarp(mask) 的掩码语义：
//   只有掩码中置位的 lane 会参与这次同步，其余 lane 直接掠过。
//   这里让偶数 lane 之间同步，奇数 lane 之间同步，互不等待。
// ------------------------------------------------------------
__global__ void warpPartialSync(int* out) {
    int t = threadIdx.x;
    int lane = t & 31;
    unsigned evenMask = 0x55555555u;   // bit0,bit2,bit4,... 置位
    unsigned oddMask  = 0xAAAAAAAAu;   // bit1,bit3,bit5,... 置位

    int v = lane;
    if ((lane & 1) == 0) {
        __syncwarp(evenMask);          // 只有偶数 lane 会到这里
        v += 100;
    } else {
        __syncwarp(oddMask);           // 只有奇数 lane 会到这里
        v += 200;
    }
    out[t] = v;
}

// ------------------------------------------------------------
// __activemask() 的用途与陷阱
//   它返回「当前此刻仍在执行这条指令的 lane 掩码」。
//   它的结果依赖调度，不可预测，所以官方明确建议：
//   不要用 __activemask() 去猜掩码来喂给 __shfl_sync，
//   而应该在发散之前就把掩码算好、存下来。
// ------------------------------------------------------------
__global__ void activeMaskDemo(int* out) {
    int t = threadIdx.x;
    int lane = t & 31;

    // 正确做法：在控制流还没分岔时，把「本 warp 全部 lane」记下来
    unsigned full = __activemask();

    int v = lane * 3;
    if (v < 48) {
        // 错误做法（这里只是演示，不要照抄）：
        // 此刻活跃 lane 已经变少了，用 current 当掩码会漏掉不活跃的 lane，
        // 后续 __shfl_sync(current, ...) 拿到的结果就是未定义的。
        unsigned current = __activemask();
        out[t] = (int)(full == 0xffffffffu ? 1 : 0) * 1000 + (int)__popc(current);
    } else {
        out[t] = -1;
    }
}

int main() {
    printf("=== __syncwarp 演示 (需要 CC >= 7.0) ===\n");

    int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

    int h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = i * 2;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));

    // ---------- 1. warp 内右移 ----------
    warpShiftShared<<<1, N>>>(d_in, d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int t = 0; t < N; ++t) {
        int want = ((t & 31) == 0) ? 0 : h_in[t - 1];
        if (h_out[t] != want) ++bad;
    }
    printf("[warp 内右移] 错误元素数 = %d / %d\n", bad, N);

    // ---------- 2. 掩码同步 ----------
    warpPartialSync<<<1, N>>>(d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    int bad2 = 0;
    for (int t = 0; t < N; ++t) {
        int lane = t & 31;
        int want = (lane & 1) ? lane + 200 : lane + 100;
        if (h_out[t] != want) ++bad2;
    }
    printf("[掩码同步]   错误元素数 = %d / %d\n", bad2, N);
    printf("             (偶数 lane 加 100，奇数 lane 加 200，两组互不干扰)\n");

    // ---------- 3. activemask ----------
    activeMaskDemo<<<1, N>>>(d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    printf("\n[activemask] 前 8 个 lane 的输出：\n  ");
    for (int t = 0; t < 8; ++t) printf("%d ", h_out[t]);
    printf("\n  说明：每个线程打印的是 (全掩码?1000:0) + 当前活跃 lane 数。\n");
    printf("  你会看到活跃 lane 数小于 32 —— 这就是它的危险之处：\n");
    printf("  结果取决于调度，不可预测，所以不要用它来构造 shuffle 的掩码。\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
