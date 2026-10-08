// ============================================================
//  code/lessons/ch00_hello/hello.cu
//  本书的第一个 CUDA 程序。
//
//  目标：让你亲眼看到 "CPU 上的代码" 与 "GPU 上的代码" 是两段
//        不同的代码，而 <<<1,1>>> 的含义就是"启动 1 个线程块，
//        每块 1 个线程"。
//
//  编译（在 code/lessons/ch00_hello 目录下）：
//      nvcc -O3 -arch=sm_75 hello.cu -o hello
//  运行：
//      ./hello        (Linux)
//      hello.exe      (Windows)
// ============================================================

#include <cstdio>

#include "../../common/cuda_check.h"

// ------------------------------------------------------------
// __global__ 表示这是一个"核函数"(kernel)：
//   - 它在 GPU（设备）上执行
//   - 它由 CPU（主机）用 <<<...>>> 语法启动
//   - 它的返回类型必须是 void（因为它的调用是异步的，返回值没地方放）
//
// 注意：核函数是"每个线程都执行一遍整个函数体"的。
//       <<<1,1>>> 表示只有 1 个线程，所以下面这句 printf 只打印一次。
//       如果写成 <<<1,4>>>，就会打印 4 次。
// ------------------------------------------------------------
__global__ void helloFromGPU() {
    // 这句 printf 是在 GPU 上执行的（设备端 printf，由 CUDA 运行时转发到终端）
    printf("Hello World from GPU!  (block %d, thread %d)\n",
           blockIdx.x, threadIdx.x);
}

int main() {
    // 1) 先确认真的有一块 NVIDIA 显卡被驱动识别到
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("没有检测到支持 CUDA 的设备，本程序无法运行。\n");
        printf("请检查：显卡是否为 NVIDIA、驱动是否安装、是否在虚拟机/远程会话中。\n");
        return 0;  // 这是"环境没准备好"，不是程序错误，所以不用 exit(1)
    }

    // 2) 打印一行主机端的问候，用来和 GPU 的输出区分开
    printf("Hello World from CPU!  (检测到 %d 个 CUDA 设备)\n", deviceCount);

    // 3) 启动核函数：<<<网格维度, 线程块维度>>>
    //    这里 gridDim = 1、blockDim = 1，也就是"总共 1 个线程"
    helloFromGPU<<<1, 1>>>();
    // 核函数启动本身不返回错误码，必须用这个宏检查启动是否成功
    CUDA_CHECK_KERNEL();

    // 4) 关键一步：同步。
    //    kernel 启动是异步的（host 发出命令后立刻往下走），
    //    而 cudaDeviceSynchronize() 会等 GPU 把所有活干完。
    //    很多初学者看到"没有输出"就是因为程序结束时 kernel 还没跑完。
    //    另外，设备端 printf 的输出也依赖这一步之后才会被刷出来。
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Done.\n");
    return 0;
}
