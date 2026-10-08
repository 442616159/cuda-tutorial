#pragma once
// ============================================================
//  cuda_timer.h —— CUDA 计时公共头文件
//  所有章节示例共用。作用是：用 cudaEvent 精确测量 GPU 上的
//  耗时，并且自带"预热 + 多次运行取中位数"的规范用法，
//  避免初学者只跑一次就下结论。
// ============================================================

#include <algorithm>   // std::sort
#include <cstdio>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.h"

// 取中位数（按值接收，因为内部要排序）
static inline double medianOf(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ------------------------------------------------------------
// CudaTimer：基于 cudaEvent 的计时器
//
// 用法一（推荐，自带规范统计）：
//     double ms = CudaTimer::measure([](double& ms, int it){
//         CudaTimer t; t.start();
//         myKernel<<<g, b>>>(...);
//         CUDA_CHECK(cudaDeviceSynchronize());
//         ms = t.stop();
//     }, 5);
//
// 用法二（手工控制）：
//     CudaTimer t;
//     t.start();
//     myKernel<<<g, b>>>(...);   // 注意：kernel 启动是异步的
//     float ms = t.stopMs();     // stopMs 内部会同步，所以能测到真实耗时
//
// 为什么要用 cudaEvent 而不是 host 端 clock()？
//   因为 kernel 是"异步启动、异步执行"的，host 端计时会把
//   启动开销算进去、甚至可能测到 0。cudaEvent 记录在 GPU 的
//   时间线上，测的是真正的设备端耗时。
// ------------------------------------------------------------
class CudaTimer {
public:
    CudaTimer() {
        // cudaEventDefault 计时精度足够；不需要 cudaEventBlockingSync
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        // 析构里不抛错：即使销毁失败也不该让程序崩溃
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // 禁止拷贝：事件句柄是独占资源，拷贝会导致重复销毁
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    // 在默认流上"打一个时间戳"
    void start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }

    // 结束计时；内部做 cudaEventSynchronize，所以返回值就是真实耗时
    float stopMs() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));  // 必须等 stop 事件真的发生
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

    float stop() { return stopMs(); }

    // 规范测量：预热 warmup 次后，再跑 iters 次取中位数
    //   f 的签名是 void(double& ms, int iter)，在里面写上"要测的一段代码"
    static double measure(const std::function<void(double&, int)>& f,
                          int iters = 5, int warmup = 1) {
        std::vector<double> samples;
        for (int i = 0; i < warmup + iters; ++i) {
            double ms = 0.0;
            f(ms, i);
            if (i >= warmup) samples.push_back(ms);
        }
        return medianOf(samples);
    }

    // 顺手算一下有效带宽（GB/s）：bytes 是这次操作搬运的总字节数
    static double bandwidthGBs(double bytes, double ms) {
        if (ms <= 0.0) return 0.0;
        return bytes / (ms * 1e-3) / 1e9;
    }

    // 顺手算一下算力（GFLOPS）：flops 是浮点运算次数
    static double gflops(double flops, double ms) {
        if (ms <= 0.0) return 0.0;
        return flops / (ms * 1e-3) / 1e9;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};
