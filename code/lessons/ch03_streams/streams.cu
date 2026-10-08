// ============================================================
//  code/lessons/ch03_streams/streams.cu
//  第 3 章示例 10：流（Stream）、异步拷贝与多流流水线
//
//  本程序演示：
//    1. 页锁定内存（cudaHostAlloc）为什么是异步拷贝的前提
//    2. 默认流上的「串行三段式」：H2D -> kernel -> D2H
//    3. 切成 4 段放到 4 条流上，让「传输」和「计算」重叠起来
//    4. cudaEventRecord / cudaStreamWaitEvent 建立跨流依赖
//
//  编译： nvcc -O2 -arch=sm_70 streams.cu -o streams
//  运行： ./streams
// ============================================================
#include "../../common/cuda_check.h"
#include "../../common/cuda_bench.h"
#include <cstdio>
#include <cmath>

#define BLOCK     256
#define NSTREAM   4
#define SEG_N     (1 << 21)                   // 每段 2M 个 float = 8 MB
#define TOTAL_N   (SEG_N * NSTREAM)           // 总共 32 MB

// 一个"计算量刚好不算太小"的 kernel：y = a * x + b
__global__ void scaleKernel(float* x, float a, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] * a + b;
    }
}

int main() {
    printf("=== 流与重叠演示 ===\n");

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // 用属性接口而不是 cudaDeviceProp 的对应字段，跨 CUDA 版本更稳
    int dev = 0, asyncEngines = 0, canMap = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&asyncEngines, cudaDevAttrAsyncEngineCount, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&canMap, cudaDevAttrCanMapHostMemory, dev));

    printf("设备：%s\n", prop.name);
    printf("  异步引擎数 (asyncEngineCount) = %d\n", asyncEngines);
    printf("  支持映射锁页内存 (canMapHostMemory) = %s\n",
           canMap ? "是" : "否");
    if (asyncEngines == 0) {
        printf("  ⚠ 该设备没有独立拷贝引擎，H2D/D2H 与 kernel 无法真正重叠，\n");
        printf("    本示例中的加速比会接近 1，这是正常的。\n");
    }

    const size_t segBytes   = (size_t)SEG_N * sizeof(float);
    const size_t totalBytes = (size_t)TOTAL_N * sizeof(float);
    printf("\n每段 %.1f MB，共 %d 段，合计 %.1f MB\n",
           segBytes / 1024.0 / 1024.0, NSTREAM, totalBytes / 1024.0 / 1024.0);

    // ---------- 1. 页锁定内存 ----------
    // 普通 malloc 得到的是"可分页"内存，驱动无法直接让 DMA 引擎访问，
    // 于是必须先拷到一块内部暂存区，异步 API 也就被迫变成同步。
    // cudaHostAlloc 分配的是锁页内存（pinned / page-locked），
    // DMA 可以直接搬运，cudaMemcpyAsync 才真正异步。
    float* h_in  = nullptr;
    float* h_out = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&h_in,  totalBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_out, totalBytes, cudaHostAllocDefault));

    for (int i = 0; i < TOTAL_N; ++i) h_in[i] = 1.0f;
    for (int i = 0; i < TOTAL_N; ++i) h_out[i] = -1.0f;

    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, totalBytes));

    // ---------- 2. 创建流 ----------
    cudaStream_t streams[NSTREAM];
    for (int s = 0; s < NSTREAM; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    const int gridTotal = (TOTAL_N + BLOCK - 1) / BLOCK;
    const int gridSeg   = (SEG_N + BLOCK - 1) / BLOCK;

    // ---------- 3. 基线：默认流上的串行三段式 ----------
    double tSerial = medianMs([&] {
        // cudaMemcpy 是同步的：函数返回时数据一定已经在设备上了
        CUDA_CHECK(cudaMemcpy(d_buf, h_in, totalBytes, cudaMemcpyHostToDevice));
        scaleKernel<<<gridTotal, BLOCK>>>(d_buf, 2.0f, 1.0f, TOTAL_N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(h_out, d_buf, totalBytes, cudaMemcpyDeviceToHost));
    }, 2, 10);

    // 校验
    int badSerial = 0;
    for (int i = 0; i < TOTAL_N; ++i) {
        if (fabsf(h_out[i] - 3.0f) > 1e-5f) ++badSerial;
    }

    // ---------- 4. 多流流水线 ----------
    // 关键点：不同流之间的操作可以并发。
    // 流 s 里的顺序仍是 H2D -> kernel -> D2H，但流 s 的 kernel
    // 可以和流 s+1 的 H2D 同时进行，这就是"重叠"。
    double tStreams = medianMs([&] {
        for (int s = 0; s < NSTREAM; ++s) {
            float* d_seg = d_buf + (size_t)s * SEG_N;
            // 异步 H2D：立刻返回，实际传输由拷贝引擎后台完成
            CUDA_CHECK(cudaMemcpyAsync(d_seg, h_in + (size_t)s * SEG_N, segBytes,
                                       cudaMemcpyHostToDevice, streams[s]));
            // 同一个流里的 kernel 会等前面的 H2D 完成才开跑
            scaleKernel<<<gridSeg, BLOCK, 0, streams[s]>>>(d_seg, 2.0f, 1.0f, SEG_N);
            // 同一个流里的 D2H 会等 kernel 完成
            CUDA_CHECK(cudaMemcpyAsync(h_out + (size_t)s * SEG_N, d_seg, segBytes,
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
        // 异步 API 只保证"提交"，必须显式同步才知道全部干完
        for (int s = 0; s < NSTREAM; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        }
    }, 2, 10);

    // 校验
    int badStreams = 0;
    for (int i = 0; i < TOTAL_N; ++i) {
        if (fabsf(h_out[i] - 3.0f) > 1e-5f) ++badStreams;
    }

    printf("\n--- 结果 ---\n");
    printf("  默认流（串行）: %8.4f ms   错误元素 %d\n", tSerial, badSerial);
    printf("  %d 条流（流水线）: %8.4f ms   错误元素 %d\n",
           NSTREAM, tStreams, badStreams);
    printf("  实测加速比约为 %.2f 倍（你的机器可能不同）。\n", tSerial / tStreams);
    printf("  加速来自「拷贝引擎搬第 s+1 段时，SM 在算第 s 段」。\n");
    printf("  如果数据全部留在显存里（不涉及 H2D/D2H），流就没什么用了。\n");

    // ---------- 5. 跨流依赖：cudaEventRecord + cudaStreamWaitEvent ----------
    {
        cudaEvent_t ev;
        // 只用来做依赖时，加 cudaEventDisableTiming 可以略微降低开销
        CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));

        // 在 stream[0] 上排队一段工作
        scaleKernel<<<gridSeg, BLOCK, 0, streams[0]>>>(d_buf, 1.0f, 0.0f, SEG_N);
        CUDA_CHECK_KERNEL();
        // 在 stream[0] 的这个时间点打一个"路标"
        CUDA_CHECK(cudaEventRecord(ev, streams[0]));
        // 让 stream[1] 在路标之后才能继续 —— 这就是跨流同步
        CUDA_CHECK(cudaStreamWaitEvent(streams[1], ev, 0));
        // 之后 stream[1] 上的工作一定在 stream[0] 的 kernel 之后执行
        scaleKernel<<<gridSeg, BLOCK, 0, streams[1]>>>(d_buf, 1.0f, 0.0f, SEG_N);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK(cudaStreamSynchronize(streams[1]));
        printf("\n[跨流依赖] cudaEventRecord + cudaStreamWaitEvent 执行完毕。\n");
        printf("  cudaStreamWaitEvent 只阻塞「目标流上此后的工作」，\n");
        printf("  不会阻塞 CPU，这是它比 cudaStreamSynchronize 更适合做依赖的原因。\n");

        CUDA_CHECK(cudaEventDestroy(ev));
    }

    // ---------- 清理 ----------
    for (int s = 0; s < NSTREAM; ++s) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));

    printf("\n完成。\n");
    return 0;
}
