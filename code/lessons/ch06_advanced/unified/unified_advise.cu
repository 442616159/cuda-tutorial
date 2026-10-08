// ============================================================================
//  code/lessons/ch06_advanced/unified/unified_advise.cu
//
//  统一内存进阶：cudaMemAdvise 与按需页迁移
//
//  // 需要 Compute Capability >= 6.0（Pascal 起才有硬件页错误与按需迁移）
//
//  编译: nvcc -O3 -arch=sm_70 unified_advise.cu -o unified_advise
//  运行: ./unified_advise
//
//  统一内存（Unified Memory）让你写一段"看起来只有一份指针"的代码：
//      cudaMallocManaged(&p, bytes);
//      kernel<<<...>>>(p);     // GPU 直接用
//      printf("%f", p[0]);     // CPU 直接用
//  代价是：页是在**第一次被访问时**才迁移过去的，
//  每次"错在哪边访问"都会触发一次页错误 + 迁移，单次开销可达几十微秒。
//
//  cudaMemAdvise 就是提前把"这段数据的行为特征"告诉驱动，
//  让它别每次都靠页错误去猜。四条最常用的建议：
//    cudaMemAdviseSetReadMostly        只读数据，可以在多个处理器上放副本
//    cudaMemAdviseSetPreferredLocation 建议这段数据常驻在哪
//    cudaMemAdviseSetAccessedBy        允许某个设备直接访问（不迁移，走 NVLink/PCIe）
//    cudaMemAdviseUnsetXXX             撤销上面的建议
//  配合 cudaMemPrefetchAsync 可以做到"访问之前就把页搬过去"，
//  把几十微秒的页错误开销彻底消掉。
// ============================================================================
#include "../../../common/cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

constexpr int   N      = 1 << 22;  // 4M 个 float = 16 MB
constexpr int   TPB    = 256;
constexpr int   REPEAT = 10;

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

// 读 x、写 y，覆盖 16 MB 的读写，用来放大"页在哪边"的影响
__global__ void saxpy(const float* __restrict__ x, float* __restrict__ y,
                      float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + 1.0f;
}

int main() {
    printDeviceInfo();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 6) {
        printf("\n[跳过] 本机 CC = %d.%d，统一内存的按需页迁移需要 CC >= 6.0。\n",
               prop.major, prop.minor);
        return 0;
    }

    int concurrentManaged = 0, pageableAccess = 0, unifiedAddressing = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrentManaged, cudaDevAttrConcurrentManagedAccess, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&pageableAccess, cudaDevAttrPageableMemoryAccess, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&unifiedAddressing, cudaDevAttrUnifiedAddressing, 0));
    printf("\n[能力] 并发托管访问(ConcurrentManagedAccess) = %s\n",
           concurrentManaged ? "支持" : "不支持");
    printf("[能力] 可分页内存直接访问(PageableMemoryAccess) = %s\n",
           pageableAccess ? "支持（Grace Hopper 类平台）" : "不支持");
    printf("[能力] 统一寻址(UnifiedAddressing) = %s\n", unifiedAddressing ? "支持" : "不支持");

    float* x = nullptr;
    float* y = nullptr;
    const size_t bytes = (size_t)N * sizeof(float);
    CUDA_CHECK(cudaMallocManaged(&x, bytes));
    CUDA_CHECK(cudaMallocManaged(&y, bytes));

    // ---- 关键的 cudaMemAdvise：告诉驱动"x 只读，优先放 GPU 显存" ----
    // 只读标记让驱动可以在 CPU 和 GPU 两侧各留一份副本，避免来回迁移
#if CUDART_VERSION >= 13000
    // CUDA 13 起 cudaMemAdvise 的第 4 个参数由 device id 变成 cudaMemLocation
    int advDev = 0;
    CUDA_CHECK(cudaGetDevice(&advDev));
    cudaMemLocation advLoc{};
    advLoc.type = cudaMemLocationTypeDevice;
    advLoc.id   = advDev;
    CUDA_CHECK(cudaMemAdvise(x, bytes, cudaMemAdviseSetReadMostly,       advLoc));
    CUDA_CHECK(cudaMemAdvise(x, bytes, cudaMemAdviseSetPreferredLocation, advLoc));
    // y 会被反复写，标记"只能由 GPU 访问"，驱动就不必考虑 CPU 侧的副本
    CUDA_CHECK(cudaMemAdvise(y, bytes, cudaMemAdviseSetPreferredLocation, advLoc));
    CUDA_CHECK(cudaMemAdvise(y, bytes, cudaMemAdviseSetAccessedBy,       advLoc));
#else
    CUDA_CHECK(cudaMemAdvise(x, bytes, cudaMemAdviseSetReadMostly, 0));
    CUDA_CHECK(cudaMemAdvise(x, bytes, cudaMemAdviseSetPreferredLocation, 0));
    // y 会被反复写，标记"只能由 GPU 访问"，驱动就不必考虑 CPU 侧的副本
    CUDA_CHECK(cudaMemAdvise(y, bytes, cudaMemAdviseSetPreferredLocation, 0));
    CUDA_CHECK(cudaMemAdvise(y, bytes, cudaMemAdviseSetAccessedBy, 0));
#endif

    // CPU 侧初始化（这一步会让页变成"主机常驻"）
    for (int i = 0; i < N; ++i) {
        x[i] = (float)((i % 100) * 0.01);
        y[i] = 0.f;
    }

    const int grid = (N + TPB - 1) / TPB;
    CudaTimer timer;

    // =====================================================================
    // 场景 A：不做预取，让 GPU 靠页错误按需拉取
    // =====================================================================
    {
        float best = 1e30f;
        for (int r = 0; r < REPEAT; ++r) {
            timer.start();
            saxpy<<<grid, TPB>>>(x, y, 2.0f, N);
            const float ms = timer.stop();
            if (ms < best) best = ms;
        }
        CUDA_CHECK_KERNEL();
        printf("\n[A] 不预取（靠页错误迁移）: %.4f ms（取 %d 次最小值）\n", best, REPEAT);
    }

    // =====================================================================
    // 场景 B：先用 cudaMemPrefetchAsync 把 x、y 都搬到设备 0
    //   prefetch 的语义是"现在就开始搬，但不阻塞调用它的线程"，
    //   所以要在同一个流里发一个事件/同步来确保搬完。
    // =====================================================================
    {
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        float best = 1e30f;
        for (int r = 0; r < REPEAT; ++r) {
            // 每次迭代前先把数据"拉回"到设备侧（模拟真实迭代求解器的节奏）
#if CUDART_VERSION >= 13000
            // CUDA 13 起 cudaMemPrefetchAsync 的第 3 个参数由 device id 变成 cudaMemLocation，
            // 并新增了第 4 个 flags 参数
            cudaMemLocation pLoc{};
            pLoc.type = cudaMemLocationTypeDevice;
            pLoc.id   = 0;
            CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, pLoc, 0, s));
            CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, pLoc, 0, s));
#else
            CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, 0, s));
            CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, 0, s));
#endif
            CUDA_CHECK(cudaStreamSynchronize(s));
            timer.start();
            saxpy<<<grid, TPB>>>(x, y, 2.0f, N);
            const float ms = timer.stop();
            if (ms < best) best = ms;
        }
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaStreamDestroy(s));
        printf("[B] 预取到设备 (cudaMemPrefetchAsync): %.4f ms\n", best);
    }

    // ---- 校验结果 ----
    {
        // 先搬回 CPU 再读，避免在 CPU 上访问 GPU 常驻页造成额外页错误
#if CUDART_VERSION >= 13000
        // CUDA 13：目标位置改用 cudaMemLocation，搬到 CPU 用 cudaMemLocationTypeHost
        cudaMemLocation hostLoc{};
        hostLoc.type = cudaMemLocationTypeHost;
        hostLoc.id   = 0;
        CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, hostLoc, 0, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, hostLoc, 0, 0));
#else
        CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, cudaCpuDeviceId, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, cudaCpuDeviceId, 0));
#endif
        CUDA_CHECK(cudaDeviceSynchronize());
        int bad = 0;
        for (int i = 0; i < N; ++i) {
            const float expect = 2.0f * (float)((i % 100) * 0.01) + 1.0f;
            if (std::fabs(y[i] - expect) > 1e-4f) ++bad;
        }
        printf("\n[正确性] 不匹配元素 = %d  %s\n", bad, bad == 0 ? "[OK]" : "[FAIL]");
    }

    printf("\n实用经验（数字随机器变化，但结论稳定）：\n");
    printf("  * 首次访问的开销永远最大 —— 页错误 + 迁移一次通常几十微秒，\n");
    printf("    4M 个 float 相当于 4 MB 的页（2 MB 大页），一次错误就是 2 MB 的搬运；\n");
    printf("  * cudaMemAdviseSetReadMostly 对「只读、被反复读」的数据最有效，\n");
    printf("    尤其在有 NVLink 的多卡场景下，可以让每张卡各留一份副本；\n");
    printf("  * 迭代型算法（CG、Jacobi、训练循环）里，数据本来就是常驻 GPU 的，\n");
    printf("    最省事的做法是别用 Unified Memory，直接 cudaMalloc；\n");
    printf("    只有在「原型阶段想省掉手工 memcpy」时才用 Managed；\n");
    printf("  * 用 nsys 看 UVM 页错误：nsys profile --stats=true ./unified_advise，\n");
    printf("    报告里的 Unified Memory 一节会列出页错误次数和迁移字节数。\n");

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    return 0;
}
