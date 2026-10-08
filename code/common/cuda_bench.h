#pragma once
// ============================================================
//  cuda_bench.h —— 第 3/4 章示例统一使用的计时与指标换算工具
//
//  它和 code/common/cuda_timer.h 是配套关系：
//    cuda_timer.h 里的 CudaTimer 类负责"测一次"，
//    本文件负责"规范地测很多次并给出可信的数字"。
//
//  提供：
//      medianMs(fn, warmup, repeats)  —— 预热 + 多次采样 + 取中位数
//      gbps(bytes, ms)                —— 有效带宽换算（GB/s）
//      gflopsOf(flops, ms)            —— 算力换算（GFLOPS）
//
//  全部是 header-only 的模板 / 内联函数，可以被多个 .cu 重复包含。
// ============================================================

#include <cuda_runtime.h>

// ------------------------------------------------------------
// 预热 warmup 次，再采样 repeats 次，返回中位数耗时（毫秒）。
// fn 里写"一次完整的待测工作"，例如：
//     double ms = medianMs([&]{ myKernel<<<g, b>>>(d_in, d_out); });
//
// 为什么要预热：第一次执行要经历时钟升频、指令/常量缓存冷启动、
//               驱动与内存分配器的懒初始化，必然偏慢。
// 为什么取中位数：偶发的长尾（别的进程抢显存、中断）会拉高平均值，
//               中位数更能代表"典型性能"。
//
// 注意：这里刻意不用 std::vector，避免在头文件里引入堆分配与
//       模板实例化开销；采样数上限 256，足够统计使用。
// ------------------------------------------------------------
template <typename Fn>
static inline double medianMs(Fn&& fn, int warmup = 3, int repeats = 20) {
    const int kMaxSamples = 256;
    if (repeats > kMaxSamples) repeats = kMaxSamples;
    if (repeats < 1) repeats = 1;

    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();     // 预热必须真的做完，否则会污染第一次采样

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double samples[kMaxSamples];
    for (int i = 0; i < repeats; ++i) {
        cudaEventRecord(start);
        fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);          // 必须同步，否则测到的是"启动耗时"
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        samples[i] = static_cast<double>(ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 插入排序取中位数：采样数很小，插入排序比快排更快也更好读
    for (int i = 1; i < repeats; ++i) {
        double key = samples[i];
        int j = i - 1;
        while (j >= 0 && samples[j] > key) {
            samples[j + 1] = samples[j];
            --j;
        }
        samples[j + 1] = key;
    }
    return samples[repeats / 2];
}

// 有效带宽（GB/s，10 进制 GB，与厂商规格一致）
static inline double gbps(double bytes, double ms) {
    if (ms <= 0.0) return 0.0;
    return bytes / (ms * 1.0e-3) / 1.0e9;
}

// 算力（GFLOPS）
static inline double gflopsOf(double flops, double ms) {
    if (ms <= 0.0) return 0.0;
    return flops / (ms * 1.0e-3) / 1.0e9;
}
