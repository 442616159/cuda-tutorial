// ============================================================
//  code/lessons/ch00_hello/device_query.cu
//  查询本机所有 CUDA 设备的详细信息。
//
//  这是官方样例 deviceQuery 的精简版，只保留初学者真正需要
//  关心的字段，并额外加了一列"这个数字意味着什么"的提示。
//
//  编译（在 code/lessons/ch00_hello 目录下）：
//      nvcc -O3 device_query.cu -o device_query
//  运行：
//      ./device_query        (Linux)
//      device_query.exe      (Windows)
//
//  注意：本程序完全不启动核函数，所以即使显卡很老也能跑起来，
//        它是你排查"环境到底装好没有"的第一道工具。
// ============================================================

#include <cstdio>

#include "../../common/cuda_check.h"

// 把计算能力换算成一个便于阅读的分数，例如 7.5 -> "7.5"
static void printComputeCapability(const cudaDeviceProp& prop) {
    printf("  计算能力 (CC)          : %d.%d\n", prop.major, prop.minor);
    printf("    说明：CC 决定了你能用哪些指令、哪些特性。\n");
    if (prop.major < 5) {
        printf("    提示：CC < 5.0 属于非常老的架构，本书大部分优化手段仍然可用，\n");
        printf("          但 warp shuffle、__ldg、统一内存等特性可能受限。\n");
    } else if (prop.major < 7) {
        printf("    提示：CC 5.x/6.x 可以完整跑通本书前 5 章的所有示例。\n");
    } else {
        printf("    提示：CC >= 7.0 支持本书提到的全部基础与进阶特性（Tensor Core 除外）。\n");
    }
}

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("检测到 %d 个 CUDA 设备。\n\n", deviceCount);

    if (deviceCount == 0) {
        printf("没有可用设备。常见原因：\n");
        printf("  1. 没有 NVIDIA 独立显卡（AMD/Intel 核显不支持 CUDA）；\n");
        printf("  2. 驱动没装好，或装的是过旧的驱动；\n");
        printf("  3. 你在 WSL / 虚拟机 / 远程桌面里，但没有做显卡直通。\n");
        return 0;
    }

    // 逐个设备查询。cudaGetDeviceCount 返回的台数包含所有可见 GPU，
    // 用 cudaGetDeviceProperties 一次拿到该设备的全部属性。
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("=================== 设备 %d ===================\n", i);
        printf("  设备名称              : %s\n", prop.name);
        printComputeCapability(prop);

        // SM（Streaming Multiprocessor，流多处理器）：GPU 的"计算核心"
        // 一个 SM 里有很多 CUDA 核心，SM 数量决定了 GPU 的并行规模
        printf("  SM 数量               : %d\n", prop.multiProcessorCount);

        printf("  全局显存大小          : %.2f GB\n",
               prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        // CUDA 13 起 cudaDeviceProp 移除了 memoryClockRate / deviceOverlap 等字段，
        // 统一改用 cudaDeviceGetAttribute 查询，跨版本更稳。
        int memClockKHz = 0, asyncEngines = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, i));
        CUDA_CHECK(cudaDeviceGetAttribute(&asyncEngines, cudaDevAttrAsyncEngineCount, i));
        printf("  显存总线宽度          : %d bit\n", prop.memoryBusWidth);
        printf("  显存频率              : %.0f MHz\n", memClockKHz / 1000.0);

        printf("  Warp 大小             : %d   <-- 永远是 32，调度单位\n", prop.warpSize);
        printf("  每块最大线程数        : %d   <-- blockDim.x*y*z 的上限\n",
               prop.maxThreadsPerBlock);
        printf("  每块最大线程维度       : (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  网格最大维度           : (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  每 SM 最大线程数      : %d\n", prop.maxThreadsPerMultiProcessor);

        printf("  每块共享内存          : %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  每 SM 共享内存        : %.1f KB\n",
               prop.sharedMemPerMultiprocessor / 1024.0);
        printf("  每 SM 寄存器数        : %d 个 32-bit 寄存器\n",
               prop.regsPerMultiprocessor);

        printf("  是否支持统一内存      : %s\n",
               prop.unifiedAddressing ? "是" : "否");
        printf("  是否支持并发拷贝      : %s\n",
               asyncEngines > 0 ? "是" : "否");
        printf("================================================\n\n");
    }

    // 顺便演示：用 cudaDeviceGetAttribute 查询单个属性，
    // 它比 cudaGetDeviceProperties 轻量，适合只关心一两个指标的场景。
    int clockRateKHz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, 0));
    printf("设备 0 的核心时钟 : %.2f GHz\n", clockRateKHz / 1e6);

    int smCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
    printf("设备 0 的 SM 数量 : %d\n", smCount);

    return 0;
}
