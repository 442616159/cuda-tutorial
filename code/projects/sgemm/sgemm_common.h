#pragma once
// ============================================================
//  sgemm_common.h —— 第 8 章 SGEMM 项目公共基础设施
//
//  这里放三样东西，所有版本共用：
//    1) CudaTimer      —— 统一的 GPU 计时（预热 + 多次取中位数）
//    2) cpuSgemm       —— CPU 参考实现，用来验证正确性
//    3) benchmark/report —— 统一的性能测量与输出格式
//
//  为什么要抽出来？因为"每一版都改一遍计时和验证代码"是巨大的
//  重复劳动，而且很容易在改动中不小心把基准测试写歪，
//  导致性能对比失去意义。
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>

#include "../../common/cuda_check.h"

namespace sgemm {

// ------------------------------------------------------------
// GPU 事件计时器
// ------------------------------------------------------------
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&s_));
        CUDA_CHECK(cudaEventCreate(&e_));
    }
    ~CudaTimer() { cudaEventDestroy(s_); cudaEventDestroy(e_); }
    void start() { CUDA_CHECK(cudaEventRecord(s_, 0)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(e_, 0));
        CUDA_CHECK(cudaEventSynchronize(e_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s_, e_));
        return ms;
    }

private:
    cudaEvent_t s_{}, e_{};
};

// ------------------------------------------------------------
// 性能结果
// ------------------------------------------------------------
struct Result {
    float ms = 0.f;        // 中位数耗时
    float msMin = 0.f;     // 最小值（有些报告里更愿意用 min，因为噪声只会让它变大）
    double gflops = 0.0;
    double gbps = 0.0;     // 有效带宽：只统计"必须从全局内存读/写"的字节
};

// ------------------------------------------------------------
// 生成测试矩阵（固定种子 -> 结果可复现）
// ------------------------------------------------------------
inline void initMatrix(std::vector<float>& m, unsigned seed, float lo = -1.f, float hi = 1.f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : m) v = dist(rng);
}

// ------------------------------------------------------------
// CPU 参考实现。
// 用 i-k-j 循环顺序 + 分块，比 i-j-k 快很多：内层循环写 C 是连续的，
// 读 A[i][k] 是标量广播。注意这里只是"参考"，不追求极致速度，
// 但也不能慢到让验证本身成为瓶颈。
// ------------------------------------------------------------
inline void cpuSgemm(int M, int N, int K,
                     const float* __restrict A,
                     const float* __restrict B,
                     float* __restrict C) {
    const int MC = 64, KC = 256, NC = 64;
    std::memset(C, 0, sizeof(float) * (size_t)M * N);

    for (int k0 = 0; k0 < K; k0 += KC) {
        const int kmax = std::min(k0 + KC, K);
        // 把 A 的一个列块打包成连续缓冲，提升 CPU 缓存命中率
        for (int i0 = 0; i0 < M; i0 += MC) {
            const int imax = std::min(i0 + MC, M);
            for (int j0 = 0; j0 < N; j0 += NC) {
                const int jmax = std::min(j0 + NC, N);
                for (int i = i0; i < imax; ++i) {
                    float* cRow = C + (size_t)i * N;
                    const float* aRow = A + (size_t)i * K;
                    for (int k = k0; k < kmax; ++k) {
                        const float aik = aRow[k];
                        if (aik == 0.f) continue;   // 稀疏友好
                        const float* bRow = B + (size_t)k * N;
                        for (int j = j0; j < jmax; ++j) cRow[j] += aik * bRow[j];
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------
// 正确性验证：必须用相对误差，浮点结果永远不能直接比 ==
// ------------------------------------------------------------
inline bool verify(const std::vector<float>& ref, const std::vector<float>& got,
                   float tol = 1e-3f, bool print = true) {
    double maxAbs = 0.0, maxRel = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = std::fabs((double)ref[i] - (double)got[i]);
        double rel = d / std::fmax(1.0, std::fabs((double)ref[i]));
        if (rel > maxRel) { maxRel = rel; worst = i; }
        maxAbs = std::fmax(maxAbs, d);
    }
    if (print) {
        printf("  [验证] 最大绝对误差 = %.3e，最大相对误差 = %.3e (idx %zu)\n",
               maxAbs, maxRel, worst);
    }
    return maxRel < tol;
}

// ------------------------------------------------------------
// 基准测试：预热 + 多次 + 取中位数
//
// launch 是一个"启动一次 kernel"的闭包。把它做成参数，
// 是为了让所有版本共用同一套测量代码，测出来的数据才可比。
// ------------------------------------------------------------
inline Result benchmark(const std::function<void()>& launch,
                        int M, int N, int K,
                        int repeat = 20, int warmup = 3) {
    // 预热：第一次启动包含模块加载、上下文初始化、算法选择等一次性开销
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(repeat);
    CudaTimer timer;
    for (int i = 0; i < repeat; ++i) {
        timer.start();
        launch();
        times.push_back(timer.stop());
    }
    // 出错必须在计时结束后统一检查，否则 CUDA_CHECK 里的 exit 会丢结果
    CUDA_CHECK_SYNC();

    std::sort(times.begin(), times.end());

    Result r;
    r.ms = times[times.size() / 2];       // 中位数：抗离群点
    r.msMin = times.front();              // 最小值：最接近"纯计算"的时间
    const double flops = 2.0 * (double)M * N * K;
    r.gflops = flops / (r.ms * 1e-3) / 1e9;
    // 每个输出元素必须被写一次；A 的每一行被读 N/TN 次之类的高级分析留给 ncu，
    // 这里用"理论最小访存量"：读一遍 A、读一遍 B、写一遍 C。
    const double bytes = 4.0 * ((double)M * K + (double)K * N + (double)M * N);
    r.gbps = bytes / (r.ms * 1e-3) / 1e9;
    return r;
}

// ------------------------------------------------------------
// 统一的结果输出（表格一行）
// ------------------------------------------------------------
inline void printRow(const char* name, const Result& r,
                     double speedupVsBaseline = 0.0, double pctOfCublas = 0.0) {
    printf("| %-22s | %9.3f | %9.3f | %10.1f | %8.1f |",
           name, r.ms, r.msMin, r.gflops, r.gbps);
    if (speedupVsBaseline > 0) printf(" %8.2fx |", speedupVsBaseline);
    else                       printf(" %9s |", "-");
    if (pctOfCublas > 0) printf(" %7.1f%% |\n", pctOfCublas);
    else                 printf(" %8s |\n", "-");
    fflush(stdout);   // 崩溃时也能看到已经跑完的版本
}

inline void printTableHeader() {
    printf("\n| 版本                   |  耗时(ms) | 最小(ms)  | GFLOPS     | GB/s     | 加速比   | vs cuBLAS |\n");
    printf("|------------------------|-----------|-----------|------------|----------|----------|-----------|\n");
}

// ------------------------------------------------------------
// 显存吞吐的理论参考：把设备的峰值带宽打出来，
// 方便判断一个 kernel 是"离带宽上限很近"还是"还差得远"。
// ------------------------------------------------------------
inline void printBandwidthReference() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const int cc = prop.major * 10 + prop.minor;
    // 显存位宽 x 等效频率得到理论带宽；老设备拿不到这些字段就跳过
    if (prop.memoryClockRate > 0 && prop.memoryBusWidth > 0) {
        const double bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
        printf("设备: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
        printf("理论显存带宽 ≈ %.1f GB/s（实测通常能到 70%%~85%%）\n", bw);
    } else {
        printf("设备: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    }
    (void)cc;
}

}  // namespace sgemm
