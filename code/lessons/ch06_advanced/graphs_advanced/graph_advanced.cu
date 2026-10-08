// ============================================================================
//  code/lessons/ch06_advanced/graphs_advanced/graph_advanced.cu
//
//  CUDA Graphs 进阶：显式构图、条件节点（if/else）、循环节点（while）
//
//  // 基础图捕获见 ch03（流与并发）；这里讲的是 12.3 之后才有的控制流节点
//  // 条件/循环节点需要 CUDA Toolkit >= 12.3；更老的版本会自动跳过那一段。
//
//  编译: nvcc -O3 -arch=sm_70 graph_advanced.cu -o graph_advanced
//  运行: ./graph_advanced
//
//  为什么需要条件节点？
//    普通 CUDA Graph 是一张**静态**的 DAG：节点和依赖在实例化时就固定了。
//    但真实程序里到处是 "if (残差 > 阈值) 再迭代一次" 这种控制流。
//    在 12.3 之前，你只能：把整个图重新提交、或者在 host 端做分支（引入同步）。
//    条件节点把**分支判断放进了设备端**：
//      一个 kernel 用 cudaGraphSetConditional() 设置条件值，
//      图里挂着的 If / Else 两个节点根据这个值决定谁执行 —— 全程不回 host。
// ============================================================================
#include "../../../common/cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int N = 1 << 20;
constexpr int TPB = 256;
constexpr int REPEAT = 200;

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
// 三个小 kernel，用来串成一张图
//   每个 kernel 只做一次"乘一个常数"，单独看都没有意义 ——
//   我们要观察的是"启动 3 个 kernel"和"提交 1 张图"的开销差别。
// ---------------------------------------------------------------------------
__global__ void initKernel(float* d, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 1.0f;
}

__global__ void scaleKernel(float* d, int n, float a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= a;
}

// ===========================================================================
// 条件节点的设备端接口
//   cudaGraphSetConditional 由 host 端在实例化时注入 —— 你只要把它当普通函数调用。
//   ⚠ 需要 CUDA >= 12.3
// ===========================================================================
#if CUDART_VERSION >= 12030
__global__ void evalCondition(cudaGraphConditionalHandle handle, const float* data,
                              float threshold) {
    // 只要有一个线程设置就够了；这里让 block 0 的 thread 0 负责
    if (blockIdx.x == 0 && threadIdx.x == 0)
        cudaGraphSetConditional(handle, (data[0] < threshold) ? 1u : 0u);
}

__global__ void branchKernel(float* data, float value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) data[0] = value;
}

// 循环节点：每一轮把计数器加一，并重新设置条件值
//   条件值 != 0 → 继续循环；条件值 == 0 → 退出循环
__global__ void loopBodyKernel(cudaGraphConditionalHandle handle, int* counter,
                               int limit) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int c = ++counter[0];
        cudaGraphSetConditional(handle, (c < limit) ? 1u : 0u);
    }
}
#endif

int main() {
    printDeviceInfo();

    const int grid = (N + TPB - 1) / TPB;
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0, (size_t)N * sizeof(float)));

    CudaTimer timer;

    // =====================================================================
    // 第一部分：显式构图 vs 逐个启动
    // =====================================================================
    {
        // ---- 逐个 kernel 启动 ----
        float seqMs = 0.f;
        timer.start();
        for (int r = 0; r < REPEAT; ++r) {
            initKernel<<<grid, TPB>>>(d, N);
            scaleKernel<<<grid, TPB>>>(d, N, 2.0f);
            scaleKernel<<<grid, TPB>>>(d, N, 3.0f);
        }
        seqMs = timer.stop();
        CUDA_CHECK_KERNEL();

        std::vector<float> h(N, 0.f);
        CUDA_CHECK(cudaMemcpy(h.data(), d, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < N; ++i)
            if (std::fabs(h[i] - 6.0f) > 1e-6f) ++bad;
        printf("\n[逐个启动] %d 轮 x 3 个 kernel: %.3f ms（平均 %.1f us/轮）\n",
               REPEAT, seqMs, seqMs * 1000.0f / REPEAT);
        printf("[逐个启动] 结果正确性: 不匹配 %d  %s\n", bad, bad == 0 ? "[OK]" : "[FAIL]");

        // ---- 用显式构图把这三个 kernel 串起来 ----
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaGraphCreate(&graph, 0));

        cudaGraphNode_t nInit = nullptr, nS1 = nullptr, nS2 = nullptr;

        // 注意：kernelParams 里放的是**参数地址**，
        // 这些局部变量必须活到 cudaGraphAddKernelNode 返回之后、
        // 直到图被实例化完成。所以别用临时表达式。
        float a2 = 2.0f, a3 = 3.0f;
        int n = N;

        cudaKernelNodeParams kp = {};
        kp.sharedMemBytes = 0;
        kp.gridDim = dim3(grid);
        kp.blockDim = dim3(TPB);

        void* argsInit[] = {&d, &n};
        kp.func = (void*)initKernel;
        kp.kernelParams = argsInit;
        CUDA_CHECK(cudaGraphAddKernelNode(&nInit, graph, nullptr, 0, &kp));

        void* argsS1[] = {&d, &n, &a2};
        kp.func = (void*)scaleKernel;
        kp.kernelParams = argsS1;
        CUDA_CHECK(cudaGraphAddKernelNode(&nS1, graph, &nInit, 1, &kp));

        void* argsS2[] = {&d, &n, &a3};
        kp.kernelParams = argsS2;
        CUDA_CHECK(cudaGraphAddKernelNode(&nS2, graph, &nS1, 1, &kp));

        cudaGraphExec_t exec = nullptr;
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));

        // ---- 用图执行同样的工作 ----
        CUDA_CHECK(cudaMemset(d, 0, (size_t)N * sizeof(float)));
        timer.start();
        for (int r = 0; r < REPEAT; ++r) {
            CUDA_CHECK(cudaGraphLaunch(exec, 0));
        }
        const float graphMs = timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h.data(), d, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        bad = 0;
        for (int i = 0; i < N; ++i)
            if (std::fabs(h[i] - 6.0f) > 1e-6f) ++bad;

        printf("\n[Graph]    %d 轮 x 1 张图（3 个节点）: %.3f ms（平均 %.1f us/轮）\n",
               REPEAT, graphMs, graphMs * 1000.0f / REPEAT);
        printf("[Graph]    结果正确性: 不匹配 %d  %s\n", bad, bad == 0 ? "[OK]" : "[FAIL]");
        printf("[Graph]    相对逐个启动的提速约 %.2fx（kernel 越小、启动开销占比越高，收益越大）\n",
               seqMs / graphMs);
        printf("           在 RTX 3060 上这个规模的数据量下，提升通常在 1.2~2x 之间，\n");
        printf("           你的机器可能不同。数据量越大，kernel 自身耗时占比越高，收益越小。\n");

        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    // =====================================================================
    // 第二部分：条件节点（If / Else）—— 让设备端决定走哪条分支
    // =====================================================================
#if CUDART_VERSION >= 12030
    {
        printf("\n[条件节点] CUDA %d.%d >= 12.3，开始演示 If/Else 控制流\n",
               CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

        float* dval = nullptr;
        CUDA_CHECK(cudaMalloc(&dval, sizeof(float)));
        float threshold = 0.0f;

        // ---- 构造两个分支的子图 ----
        cudaGraph_t bodyTrue = nullptr, bodyFalse = nullptr;
        CUDA_CHECK(cudaGraphCreate(&bodyTrue, 0));
        CUDA_CHECK(cudaGraphCreate(&bodyFalse, 0));

        float vTrue = 111.0f, vFalse = 222.0f;
        cudaKernelNodeParams kp = {};
        kp.sharedMemBytes = 0;
        kp.gridDim = dim3(1);
        kp.blockDim = dim3(32);
        kp.func = (void*)branchKernel;

        void* argsTrue[] = {&dval, &vTrue};
        kp.kernelParams = argsTrue;
        cudaGraphNode_t nTrue = nullptr;
        CUDA_CHECK(cudaGraphAddKernelNode(&nTrue, bodyTrue, nullptr, 0, &kp));

        void* argsFalse[] = {&dval, &vFalse};
        kp.kernelParams = argsFalse;
        cudaGraphNode_t nFalse = nullptr;
        CUDA_CHECK(cudaGraphAddKernelNode(&nFalse, bodyFalse, nullptr, 0, &kp));

        // ---- 主图：条件求值节点 + If 节点 + Else 节点 ----
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaGraphCreate(&graph, 0));

        // 条件句柄：默认值 0，表示"默认走 Else 分支"
        cudaGraphConditionalHandle handle = 0;
        CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0,
                                                    cudaGraphCondAssignDefault));

        cudaKernelNodeParams ekp = {};
        ekp.sharedMemBytes = 0;
        ekp.gridDim = dim3(1);
        ekp.blockDim = dim3(32);
        ekp.func = (void*)evalCondition;
        void* argsEval[] = {&handle, &dval, &threshold};
        ekp.kernelParams = argsEval;
        cudaGraphNode_t nEval = nullptr;
        CUDA_CHECK(cudaGraphAddKernelNode(&nEval, graph, nullptr, 0, &ekp));

        cudaGraphConditionalNodeParams cp = {};
        cp.handle = handle;
        cp.size = 1;

        cp.type = cudaGraphCondTypeIf;
        cp.phGraph = bodyTrue;
        cudaGraphNode_t nIf = nullptr;
        CUDA_CHECK(cudaGraphAddConditionalNode(&nIf, graph, &nEval, 1, &cp));

        cp.type = cudaGraphCondTypeElse;
        cp.phGraph = bodyFalse;
        cudaGraphNode_t nElse = nullptr;
        CUDA_CHECK(cudaGraphAddConditionalNode(&nElse, graph, &nEval, 1, &cp));

        cudaGraphExec_t exec = nullptr;
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));

        // ---- 情形 1：把 dval 设成负数 -> 条件为真 -> 应该走 If（得到 111） ----
        {
            const float start = -5.0f;
            CUDA_CHECK(cudaMemcpy(dval, &start, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(exec, 0));
            CUDA_CHECK(cudaDeviceSynchronize());
            float got = 0.f;
            CUDA_CHECK(cudaMemcpy(&got, dval, sizeof(float), cudaMemcpyDeviceToHost));
            printf("  dval 初值 = %5.1f (< 0)  ->  走 If 分支，结果 = %.0f  %s\n",
                   start, got, (got == 111.0f) ? "[OK]" : "[FAIL]");
        }
        // ---- 情形 2：把 dval 设成正数 -> 条件为假 -> 应该走 Else（得到 222） ----
        {
            const float start = 5.0f;
            CUDA_CHECK(cudaMemcpy(dval, &start, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(exec, 0));
            CUDA_CHECK(cudaDeviceSynchronize());
            float got = 0.f;
            CUDA_CHECK(cudaMemcpy(&got, dval, sizeof(float), cudaMemcpyDeviceToHost));
            printf("  dval 初值 = %5.1f (>= 0) ->  走 Else 分支，结果 = %.0f  %s\n",
                   start, got, (got == 222.0f) ? "[OK]" : "[FAIL]");
        }
        printf("  注意：两次执行用的是**同一张已实例化的图**，host 端没有做任何分支判断，\n");
        printf("        也没有在两次执行之间做 cudaStreamSynchronize 之外的同步。\n");

        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaGraphDestroy(bodyTrue));
        CUDA_CHECK(cudaGraphDestroy(bodyFalse));
        CUDA_CHECK(cudaFree(dval));
    }

    // =====================================================================
    // 第三部分：循环节点（While）
    // =====================================================================
    {
        int* dcount = nullptr;
        CUDA_CHECK(cudaMalloc(&dcount, sizeof(int)));
        CUDA_CHECK(cudaMemset(dcount, 0, sizeof(int)));

        cudaGraph_t body = nullptr;
        CUDA_CHECK(cudaGraphCreate(&body, 0));

        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaGraphCreate(&graph, 0));

        // 循环节点同样需要一个条件句柄；默认值给 1，表示"先进循环体"
        cudaGraphConditionalHandle handle = 0;
        CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1,
                                                    cudaGraphCondAssignDefault));

        cudaKernelNodeParams kp = {};
        kp.sharedMemBytes = 0;
        kp.gridDim = dim3(1);
        kp.blockDim = dim3(32);
        kp.func = (void*)loopBodyKernel;
        int limit = 5;
        void* argsBody[] = {&handle, &dcount, &limit};
        kp.kernelParams = argsBody;
        cudaGraphNode_t nBody = nullptr;
        CUDA_CHECK(cudaGraphAddKernelNode(&nBody, body, nullptr, 0, &kp));

        cudaGraphConditionalNodeParams cp = {};
        cp.handle = handle;
        cp.size = 1;
        cp.type = cudaGraphCondTypeWhile;
        cp.phGraph = body;

        cudaGraphNode_t nWhile = nullptr;
        CUDA_CHECK(cudaGraphAddConditionalNode(&nWhile, graph, nullptr, 0, &cp));

        cudaGraphExec_t exec = nullptr;
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));

        CUDA_CHECK(cudaMemset(dcount, 0, sizeof(int)));
        CUDA_CHECK(cudaGraphLaunch(exec, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        int got = 0;
        CUDA_CHECK(cudaMemcpy(&got, dcount, sizeof(int), cudaMemcpyDeviceToHost));
        printf("\n[循环节点] While 循环体执行了 %d 次（期望 %d） %s\n",
               got, limit, got == limit ? "[OK]" : "[FAIL]");
        printf("           整个循环只提交了 1 次图，host 一次都没参与。\n");

        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaGraphDestroy(body));
        CUDA_CHECK(cudaFree(dcount));
    }
#else
    printf("\n[跳过] 条件节点与循环节点需要 CUDA >= 12.3，当前是 %d.%d。\n",
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    printf("       升级到 CUDA 12.3+ 后重新编译即可看到完整演示。\n");
#endif

    printf("\n什么时候值得用 Graph？\n");
    printf("  * 一段固定的 kernel 序列被反复执行（推理、迭代求解、数据预处理管线）；\n");
    printf("  * 单个 kernel 执行时间很短（几十微秒以内），启动开销占比高；\n");
    printf("  * 需要把「设备端控制流」也放进图里（条件节点、循环节点）。\n");
    printf("什么时候不值得？\n");
    printf("  * 每次执行的拓扑都不一样（图的实例化本身有开销，约几十微秒到毫秒）；\n");
    printf("  * kernel 本身就跑了很久，启动开销可以忽略。\n");
    printf("\n常用排错命令：\n");
    printf("  cudaGraphDebugDotPrint(graph, \"graph.dot\", 0);  // 导出成 Graphviz 图\n");
    printf("  dot -Tpng graph.dot -o graph.png                  // 直观看到节点与依赖\n");

    CUDA_CHECK(cudaFree(d));
    return 0;
}
