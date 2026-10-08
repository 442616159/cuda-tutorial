// ============================================================
//  thrust_basics.cu —— Thrust：像写 STL 一样写 GPU 代码
//
//  本程序把同一件事做两遍：
//    (1) 手写 kernel（你在第 1~5 章学的）
//    (2) Thrust 一行搞定
//  然后对比两者的耗时和代码量，让你建立"什么时候用库"的直觉。
//
//  编译：
//    nvcc -O3 -arch=native --extended-lambda thrust_basics.cu -o thrust_basics
//  （Thrust 是 header-only，不需要额外 -l 参数。
//    下面用到了带 __device__ 标注的 lambda，nvcc 要求显式加 --extended-lambda）
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>   // std::accumulate，用 CPU 结果做交叉验证

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "../../../common/cuda_check.h"

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

// ------------------------------------------------------------
// 手写版本：向量求和归约（参照 ch05 的写法）
// 目标：和 thrust::reduce 做同样的工作，比谁快、比谁短
// ------------------------------------------------------------
__global__ void blockReduceSum(const float* __restrict__ in, float* __restrict__ out, int n) {
    // 动态共享内存：块大小在启动时决定
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    // 网格步长循环：让任意网格大小都能覆盖全部数据
    float local = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        local += in[i];
    }
    sdata[tid] = local;
    __syncthreads();

    // 树形归约：每一轮活跃线程减半，把访存冲突降到最低
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // 每个块的 0 号线程把块内和写到全局
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ------------------------------------------------------------
// 自定义仿函数（functor）：Thrust 的 transform 拿它做逐元素映射
// 写成 struct 是因为 CUDA 不允许把普通函数指针随意传到设备端
// ------------------------------------------------------------
struct Saxpy {
    float a;
    Saxpy(float a_) : a(a_) {}
    __host__ __device__ float operator()(float x, float y) const { return a * x + y; }
};

int main() {
    printDeviceInfo();

    const int N = 1 << 22;  // 约 420 万，足够看出带宽差异，又不会被显存卡住
    printf("\n数据规模 N = %d\n", N);

    // --------------------------------------------------------
    //  1. host_vector / device_vector：内存的两种"视图"
    // --------------------------------------------------------
    // host_vector 在 CPU 内存，device_vector 在显存。
    // 两者用法和 std::vector 几乎一样，但每次赋值/拷贝都要跨 PCIe，
    // 所以要避免在循环里反复访问 device_vector 的单个元素。
    thrust::host_vector<float> hX(N), hY(N);
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i = 0; i < N; ++i) { hX[i] = dist(rng); hY[i] = dist(rng); }

    // 构造函数直接完成一次 Host->Device 拷贝
    thrust::device_vector<float> dX = hX;
    thrust::device_vector<float> dY = hY;
    thrust::device_vector<float> dZ(N);

    // --------------------------------------------------------
    //  2. thrust::transform —— 对应 Map 模式
    // --------------------------------------------------------
    const float alpha = 2.0f;
    thrust::transform(dX.begin(), dX.end(), dY.begin(), dZ.begin(), Saxpy(alpha));
    // 抽样验证：只取回前 8 个元素对比，避免整段拷回
    thrust::host_vector<float> hZ8(dZ.begin(), dZ.begin() + 8);
    printf("\n[transform] z = %.1f*x + y，前 4 个结果：\n", alpha);
    bool mapOk = true;
    for (int i = 0; i < 4; ++i) {
        float expect = alpha * hX[i] + hY[i];
        printf("  z[%d] = %+.6f（期望 %+.6f）\n", i, hZ8[i], expect);
        if (fabsf(hZ8[i] - expect) > 1e-4f) mapOk = false;
    }
    printf("  正确性: %s\n", mapOk ? "通过 ✔" : "失败 ✘");

    // --------------------------------------------------------
    //  3. thrust::reduce —— 对应 Reduce 模式
    //  加 float 累加 420 万个元素会丢精度，所以用 double 做累加器
    //  （这在 ch05 里讲过：浮点求和顺序不同，结果会有差异）
    // --------------------------------------------------------
    thrust::device_vector<float> dXf = hX;
    CudaTimer timer;

    // 手写 kernel 版本：两趟归约（第一趟 block 求和，第二趟再归约）
    const int threads = 256;
    const int blocks = 512;
    thrust::device_vector<float> dPartial(blocks);
    thrust::device_vector<float> dTotal(1);

    blockReduceSum<<<blocks, threads, threads * sizeof(float)>>>(
        thrust::raw_pointer_cast(dXf.data()), thrust::raw_pointer_cast(dPartial.data()), N);
    CUDA_CHECK_KERNEL();
    blockReduceSum<<<1, threads, threads * sizeof(float)>>>(
        thrust::raw_pointer_cast(dPartial.data()), thrust::raw_pointer_cast(dTotal.data()), blocks);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_SYNC();

    float msHand = 0.f;
    timer.start();
    for (int r = 0; r < 20; ++r) {
        blockReduceSum<<<blocks, threads, threads * sizeof(float)>>>(
            thrust::raw_pointer_cast(dXf.data()), thrust::raw_pointer_cast(dPartial.data()), N);
        blockReduceSum<<<1, threads, threads * sizeof(float)>>>(
            thrust::raw_pointer_cast(dPartial.data()), thrust::raw_pointer_cast(dTotal.data()), blocks);
    }
    msHand = timer.stop() / 20;
    float sumHand = dTotal[0];

    timer.start();
    double sumThrust = 0.0;
    for (int r = 0; r < 20; ++r) {
        sumThrust = thrust::reduce(thrust::device, dXf.begin(), dXf.end(), 0.0, thrust::plus<double>());
    }
    float msThrust = timer.stop() / 20;

    printf("\n[reduce] 手写 kernel  : %.3f ms，结果 = %.6f\n", msHand, sumHand);
    printf("[reduce] thrust::reduce: %.3f ms，结果 = %.6f\n", msThrust, sumThrust);
    printf("  两者相对差 = %.3e（差异来自浮点求和顺序，属正常现象）\n",
           fabs((double)sumHand - sumThrust) / fmax(1.0, fabs(sumThrust)));
    printf("  参考: CPU 端 std::accumulate 结果为 %.6f\n",
           (double)std::accumulate(hX.begin(), hX.end(), 0.0f));

    // --------------------------------------------------------
    //  4. thrust::sort —— 手写双调排序/基数排序要几百行，
    //     这里一行。库的价值在"排序"这种复杂算法上最明显。
    // --------------------------------------------------------
    const int M = 1 << 22;
    thrust::host_vector<int> hKeys(M);
    thrust::host_vector<int> hVals(M);
    for (int i = 0; i < M; ++i) { hKeys[i] = (int)(rng() % 1000000); hVals[i] = i; }
    thrust::device_vector<int> dKeys = hKeys;
    thrust::device_vector<int> dVals = hVals;

    timer.start();
    // SortByKey：key 排序的同时把 value 跟着搬走，这就是"稳定重排"
    thrust::sort_by_key(dKeys.begin(), dKeys.end(), dVals.begin());
    float msSort = timer.stop();
    printf("\n[sort] 对 %d 个 int 做 sort_by_key：%.3f ms\n", M, msSort);
    printf("  前 5 个 key: ");
    for (int i = 0; i < 5; ++i) printf("%d ", dKeys[i]);
    printf("\n  是否有序: %s\n", thrust::is_sorted(dKeys.begin(), dKeys.end()) ? "是 ✔" : "否 ✘");

    // --------------------------------------------------------
    //  5. 其他常用算法：count_if / sequence / fill
    // --------------------------------------------------------
    int nNeg = (int)thrust::count_if(thrust::device, dXf.begin(), dXf.end(),
                                     [] __device__(float v) { return v < 0.f; });
    printf("\n[count_if] x 中小于 0 的元素个数 = %d（占比 %.1f%%）\n",
           nNeg, 100.0 * nNeg / N);

    thrust::device_vector<int> dSeq(10);
    thrust::sequence(dSeq.begin(), dSeq.end(), 100);  // 100,101,102,...
    printf("[sequence] ");
    for (int i = 0; i < 10; ++i) printf("%d ", dSeq[i]);
    printf("\n");

    return EXIT_SUCCESS;
}
