#pragma once
// ============================================================
//  cuda_check.h —— CUDA 错误检查公共头文件
//  所有章节示例共用。作用是：一旦 CUDA 调用失败，立刻在
//  出错的那一行报错并退出，而不是让程序莫名其妙地跑出错误结果。
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 把 cudaError_t 转成可读字符串并打印
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "\n[CUDA ERROR] %s:%d\n  call : %s\n  code : %d\n  msg  : %s\n", \
                    __FILE__, __LINE__, #expr, (int)_err,                       \
                    cudaGetErrorString(_err));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// 检查核函数启动是否成功（核函数启动本身不返回错误码）
#define CUDA_CHECK_KERNEL()                                                     \
    do {                                                                        \
        cudaError_t _err = cudaGetLastError();                                  \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "\n[CUDA KERNEL LAUNCH ERROR] %s:%d\n  msg : %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// 检查核函数执行期间是否出错（异步错误要同步后才能拿到）
#define CUDA_CHECK_SYNC()                                                       \
    do {                                                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
    } while (0)

// 打印设备信息，方便确认自己跑在哪张卡上
inline void printDeviceInfo() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("=========== 设备信息 ===========\n");
    printf("设备编号        : %d\n", dev);
    printf("设备名称        : %s\n", prop.name);
    printf("计算能力 (CC)   : %d.%d\n", prop.major, prop.minor);
    printf("SM 数量         : %d\n", prop.multiProcessorCount);
    printf("全局显存        : %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("每块共享内存    : %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("每 SM 寄存器数  : %d\n", prop.regsPerMultiprocessor);
    printf("Warp 大小       : %d\n", prop.warpSize);
    printf("每 SM 最大线程  : %d\n", prop.maxThreadsPerMultiProcessor);
    printf("================================\n");
}
