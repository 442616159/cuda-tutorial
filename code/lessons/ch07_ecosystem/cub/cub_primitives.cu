// ============================================================
//  cub_primitives.cu —— CUB：设备级并行原语
//
//  Thrust 给你"整个容器上的算法"，CUB 给你"块/设备级的积木"。
//  当你已经有一个自己的核函数，只想在其中插入一次 device 级归约时，
//  用 CUB 比用 Thrust 更自然、也更容易控制临时显存。
//
//  三个最常用的原语：
//    cub::DeviceReduce::Sum      —— 设备级求和/最值
//    cub::DeviceScan::InclusiveSum —— 设备级前缀和
//    cub::DeviceRadixSort::SortPairs —— 设备级基数排序（key-value）
//
//  编译：
//    nvcc -O3 -arch=native cub_primitives.cu -o cub_primitives
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>

#include <cub/cub.cuh>
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

int main() {
    printDeviceInfo();
    const int N = 1 << 22;  // 约 420 万

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::uniform_int_distribution<int> idist(0, 1 << 20);

    std::vector<float> hIn(N);
    for (auto& v : hIn) v = dist(rng);

    float* dIn = nullptr;
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    //  CUB 的使用套路（三部曲）：
    //    1) 先传 nullptr 的临时显存，问它"你需要多少字节"
    //    2) cudaMalloc 出这块临时显存
    //    3) 真正调用原语
    //  这样设计是因为 CUB 内部会按设备调整分块策略，
    //  临时空间大小只有运行时才知道。
    // --------------------------------------------------------
    void* dTemp = nullptr;
    size_t tempBytes = 0;
    CudaTimer timer;

    // ---------- 1. DeviceReduce::Sum ----------
    // 第一次调用只为了问出临时显存大小：临时指针传 nullptr，
    // CUB 会走一条"只算需求、不干活"的轻量路径。
    cub::DeviceReduce::Sum(nullptr, tempBytes, dIn, dOut, N);
    CUDA_CHECK(cudaMalloc(&dTemp, tempBytes));

    timer.start();
    cub::DeviceReduce::Sum(dTemp, tempBytes, dIn, dOut, N);
    float msReduce = timer.stop();
    float hSum = 0.f;
    CUDA_CHECK(cudaMemcpy(&hSum, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    // CPU 参考：用 double 累加避免参考值自己就不准
    double cpuSum = std::accumulate(hIn.begin(), hIn.end(), 0.0);
    printf("\n[DeviceReduce::Sum] %.3f ms\n", msReduce);
    printf("  GPU 结果 = %.6f\n", hSum);
    printf("  CPU 结果 = %.6f（double 累加）\n", cpuSum);
    printf("  相对误差 = %.3e\n", fabs(hSum - cpuSum) / fmax(1.0, fabs(cpuSum)));
    printf("  临时显存 = %zu 字节（CUB 自己算出来的）\n", tempBytes);

    // ---------- 2. DeviceScan::InclusiveSum（前缀和） ----------
    // 前缀和是很多算法的地基：压缩、radix sort 的分桶、稀疏矩阵的行偏移等
    size_t scanBytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, scanBytes, dIn, dOut, N);
    void* dScanTemp = nullptr;
    CUDA_CHECK(cudaMalloc(&dScanTemp, scanBytes));
    timer.start();
    cub::DeviceScan::InclusiveSum(dScanTemp, scanBytes, dIn, dOut, N);
    float msScan = timer.stop();

    // 验证：前缀和的最后一个元素必须等于总和
    float hScanLast = 0.f;
    CUDA_CHECK(cudaMemcpy(&hScanLast, dOut + (N - 1), sizeof(float), cudaMemcpyDeviceToHost));
    printf("\n[DeviceScan::InclusiveSum] %.3f ms\n", msScan);
    printf("  scan[N-1] = %.6f，reduce = %.6f，两者应相等\n", hScanLast, hSum);

    // ---------- 3. DeviceRadixSort::SortPairs ----------
    const int M = 1 << 22;
    std::vector<int> hKeys(M), hVals(M);
    for (int i = 0; i < M; ++i) { hKeys[i] = idist(rng); hVals[i] = i; }
    std::vector<int> hKeysCpu = hKeys;

    int *dKeys = nullptr, *dVals = nullptr, *dKeysOut = nullptr, *dValsOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dKeys, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dVals, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dKeysOut, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dValsOut, M * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dKeys, hKeys.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVals, hVals.data(), M * sizeof(int), cudaMemcpyHostToDevice));

    size_t sortBytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, dKeys, dKeysOut, dVals, dValsOut, M);
    void* dSortTemp = nullptr;
    CUDA_CHECK(cudaMalloc(&dSortTemp, sortBytes));

    timer.start();
    cub::DeviceRadixSort::SortPairs(dSortTemp, sortBytes, dKeys, dKeysOut, dVals, dValsOut, M);
    float msSort = timer.stop();
    CUDA_CHECK(cudaMemcpy(hKeys.data(), dKeysOut, M * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> hValsCheck(M);
    CUDA_CHECK(cudaMemcpy(hValsCheck.data(), dValsOut, M * sizeof(int), cudaMemcpyDeviceToHost));

    bool sorted = true;
    for (int i = 1; i < M; ++i)
        if (hKeys[i - 1] > hKeys[i]) { sorted = false; break; }
    // 校验 value 是否跟着 key 一起搬过去了：同一个 key 的 value 必须来自原数组且值一致
    bool paired = true;
    for (int i = 0; i < M && paired; ++i) {
        if (hValsCheck[i] < 0 || hValsCheck[i] >= M) paired = false;
        else if (hKeys[i] != hKeysCpu[hValsCheck[i]]) paired = false;
    }
    printf("\n[DeviceRadixSort::SortPairs] %.3f ms（%d 个 key-value）\n", msSort, M);
    printf("  key 有序: %s，value 正确配对: %s\n",
           sorted ? "是 ✔" : "否 ✘", paired ? "是 ✔" : "否 ✘");

    // 收尾：CUB 不负责释放你申请的临时显存
    CUDA_CHECK(cudaFree(dTemp));
    CUDA_CHECK(cudaFree(dScanTemp));
    CUDA_CHECK(cudaFree(dSortTemp));
    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    CUDA_CHECK(cudaFree(dKeys));
    CUDA_CHECK(cudaFree(dVals));
    CUDA_CHECK(cudaFree(dKeysOut));
    CUDA_CHECK(cudaFree(dValsOut));
    return EXIT_SUCCESS;
}
