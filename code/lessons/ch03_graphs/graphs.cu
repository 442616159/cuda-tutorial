// ============================================================
//  code/lessons/ch03_graphs/graphs.cu
//  第 3 章示例 11：CUDA Graphs —— 把一串启动"录下来"再反复重放
//
//  为什么需要它？
//    每次 kernel<<<>>> 都要走一遍完整的启动路径（参数打包、驱动校验、
//    下发命令），在 CPU 上大约几微秒。如果你的 kernel 本身只跑
//    十微秒，那启动开销就占了一大半，而且 CPU 很容易跟不上 GPU。
//    CUDA Graphs 把一整套「启动序列」预先录制成一张图，
//    之后一次 cudaGraphLaunch 就把整张图交给 GPU，启动开销大幅下降。
//
//  注意：图带来的是「启动开销」的降低，不是「计算」的加速。
//        长 kernel / 带宽受限的 kernel 用它几乎没有收益。
//
//  编译： nvcc -O2 -arch=sm_70 graphs.cu -o graphs
//  运行： ./graphs
// ============================================================
#include "../../common/cuda_check.h"
#include "../../common/cuda_bench.h"
#include <cstdio>
#include <cmath>

#define BLOCK 256
#define N     (1 << 20)          // 1048576

// y = x + b
__global__ void addKernel(float* x, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += b;
}

// y = a * x
__global__ void mulKernel(float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

// y = y + (x - 1)
__global__ void mixKernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += (y[i] - 1.0f);
}

int main() {
    printf("=== CUDA Graphs 演示 (N = %d) ===\n", N);

    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    const int grid = (N + BLOCK - 1) / BLOCK;
    // 期望结果：a=1 -> +2 -> 3 -> *1.5 -> 4.5 -> +(2-1) -> 5.5
    const float expect = 5.5f;

    // ---------- 1. 直接启动的基线 ----------
    auto directLaunch = [&] {
        addKernel<<<grid, BLOCK>>>(d_a, 2.0f, N);
        mulKernel<<<grid, BLOCK>>>(d_a, 1.5f, N);
        mixKernel<<<grid, BLOCK>>>(d_a, d_b, N);
    };

    directLaunch();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    float got = 0.f;
    CUDA_CHECK(cudaMemcpy(&got, d_a, sizeof(float), cudaMemcpyDeviceToHost));
    printf("直接启动：a[0] = %.2f（期望 %.2f）%s\n",
           got, expect, fabsf(got - expect) < 1e-5f ? "✓" : "✗");

    // ---------- 2. 把同一串启动"录"成一张图 ----------
    // 录制的本质：所有在捕获流上提交的 CUDA 操作都不会真的执行，
    // 而是被记成图里的节点。
    cudaStream_t capStream;
    CUDA_CHECK(cudaStreamCreate(&capStream));

    CUDA_CHECK(cudaStreamBeginCapture(capStream, cudaStreamCaptureModeGlobal));
    addKernel<<<grid, BLOCK, 0, capStream>>>(d_a, 2.0f, N);
    mulKernel<<<grid, BLOCK, 0, capStream>>>(d_a, 1.5f, N);
    mixKernel<<<grid, BLOCK, 0, capStream>>>(d_a, d_b, N);
    // 注意：捕获期间不要调用 CUDA_CHECK_KERNEL()/cudaDeviceSynchronize()，
    //       它们会和捕获机制冲突（同步操作在捕获中是非法的）。

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(capStream, &graph));
    CUDA_CHECK(cudaStreamDestroy(capStream));

    // ---------- 3. 实例化：把图"编译"成可执行的图对象 ----------
    // 实例化只做一次，之后可以反复 launch。
    cudaGraphExec_t graphExec = nullptr;
#if CUDART_VERSION >= 12000
    // CUDA 12 起简化为 (pGraphExec, graph, flags)
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
#else
    // CUDA 11 及更早是五参数版本
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
#endif

    // ---------- 4. 重放并校验 ----------
    // 先把 a 恢复成初值，再跑一次图
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&got, d_a, sizeof(float), cudaMemcpyDeviceToHost));
    printf("图执行  ：a[0] = %.2f（期望 %.2f）%s\n",
           got, expect, fabsf(got - expect) < 1e-5f ? "✓" : "✗");

    // ---------- 5. 计时对比 ----------
    // 为了让"启动开销"成为可观测量，这里故意用小规模网格循环很多次。
    // 单次耗时会非常小，所以我们改成测量「连续提交 100 次」的总时间。
    const int batch = 100;

    double tDirect = medianMs([&] {
        for (int k = 0; k < batch; ++k) directLaunch();
        CUDA_CHECK(cudaDeviceSynchronize());
    }, 3, 20);

    double tGraph = medianMs([&] {
        for (int k = 0; k < batch; ++k) {
            CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }, 3, 20);

    printf("\n--- 连续提交 %d 轮（每轮 3 个 kernel）---\n", batch);
    printf("  逐个启动 : %8.4f ms  (平均每轮 %.4f ms)\n", tDirect, tDirect / batch);
    printf("  图  启动 : %8.4f ms  (平均每轮 %.4f ms)\n", tGraph,  tGraph / batch);
    printf("  实测约快 %.2f 倍（你的机器可能不同）。\n", tDirect / tGraph);
    printf("  更值得看的是「每轮 nsys 时间线上 kernel 之间的空隙」：\n");
    printf("  逐个启动会有一串小空隙，图启动几乎连成一片。\n");

    // ---------- 清理 ----------
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_a);
    free(h_b);

    printf("\n完成。\n");
    return 0;
}
