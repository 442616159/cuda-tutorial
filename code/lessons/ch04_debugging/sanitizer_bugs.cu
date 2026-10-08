// ============================================================
//  code/lessons/ch04_debugging/sanitizer_bugs.cu
//  第 4 章示例 4：用 compute-sanitizer 抓 bug
//
//  这个程序本身是"故意写坏"的，用命令行参数选择要演示哪种 bug：
//
//    ./sanitizer_bugs 0    # 干净版本，作为对照
//    ./sanitizer_bugs 1    # 数组越界写
//    ./sanitizer_bugs 2    # 共享内存竞态（漏掉 __syncthreads）
//    ./sanitizer_bugs 3    # 使用已释放的显存
//    ./sanitizer_bugs 4    # 读取未初始化的显存
//
//  推荐用法（在每条命令前面加上 compute-sanitizer）：
//
//    compute-sanitizer --tool memcheck    ./sanitizer_bugs 1
//    compute-sanitizer --tool racecheck   ./sanitizer_bugs 2
//    compute-sanitizer --tool initcheck   ./sanitizer_bugs 4
//    compute-sanitizer --tool synccheck   ./sanitizer_bugs 2
//    compute-sanitizer --tool memcheck --leak-check full ./sanitizer_bugs 0
//
//  不带参数直接运行时，程序会把上面的清单打印出来。
//
//  编译： nvcc -O2 -arch=sm_70 -lineinfo sanitizer_bugs.cu -o sanitizer_bugs
//        （-lineinfo 能让 sanitizer 报告里带上源码行号，强烈建议加上）
//  运行： ./sanitizer_bugs
// ============================================================
#include "../../common/cuda_check.h"
#include <cstdio>
#include <cstdlib>

#define BLOCK 256
#define N     4096

// ------------------------------------------------------------
// bug 1：数组越界写
//   kernel 里没有 `if (i < n) return;`，而网格又开多了一个块，
//   于是最后一批线程会写到 a[n] 甚至更后面。
// ------------------------------------------------------------
__global__ void oobWriteKernel(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = 1.0f;                     // ← 缺少边界检查
}

// ------------------------------------------------------------
// bug 2：共享内存竞态
//   线程 t 写 s[t]，然后读 s[(t+1)%BLOCK] —— 读的是别人写的格子，
//   但中间没有 __syncthreads()。
// ------------------------------------------------------------
__global__ void raceKernel(int* out) {
    __shared__ int s[BLOCK];
    int t = threadIdx.x;
    s[t] = t;
    // 这里必须有一句 __syncthreads();
    out[t] = s[(t + 1) % BLOCK];
}

// ------------------------------------------------------------
// bug 4：读取未初始化的显存
//   cudaMalloc 出来的内存内容是随机的，直接读就是未定义行为。
// ------------------------------------------------------------
__global__ void uninitReadKernel(const int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1;
}

// 干净版本：正确写法长这样
__global__ void goodKernel(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 1.0f;          // ← 边界检查
}

static void printUsage() {
    printf("用法: sanitizer_bugs <mode>\n");
    printf("  mode = 0  干净版本（对照）\n");
    printf("  mode = 1  数组越界写\n");
    printf("  mode = 2  共享内存竞态\n");
    printf("  mode = 3  使用已释放的显存\n");
    printf("  mode = 4  读取未初始化的显存\n");
    printf("\n建议这样跑（才能看到问题）：\n");
    printf("  compute-sanitizer --tool memcheck   ./sanitizer_bugs 1\n");
    printf("  compute-sanitizer --tool racecheck  ./sanitizer_bugs 2\n");
    printf("  compute-sanitizer --tool synccheck  ./sanitizer_bugs 2\n");
    printf("  compute-sanitizer --tool initcheck  ./sanitizer_bugs 4\n");
    printf("  compute-sanitizer --tool memcheck --leak-check full ./sanitizer_bugs 0\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 0;
    }
    int mode = atoi(argv[1]);
    printf("=== compute-sanitizer 演示，mode = %d ===\n", mode);

    float* d_f = nullptr;
    int*   d_i = nullptr;
    int*   d_o = nullptr;
    CUDA_CHECK(cudaMalloc(&d_f, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_i, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_o, N * sizeof(int)));

    const int gridExact = (N + BLOCK - 1) / BLOCK;

    switch (mode) {
        case 0:
            printf("干净版本：每个线程都有边界检查，不会有任何报告。\n");
            goodKernel<<<gridExact, BLOCK>>>(d_f, N);
            CUDA_CHECK_KERNEL();
            break;

        case 1:
            printf("越界写：故意多开一个块，最后 256 个线程写到数组外面。\n");
            // 注意这里用的是 gridExact + 1
            oobWriteKernel<<<gridExact + 1, BLOCK>>>(d_f, N);
            CUDA_CHECK_KERNEL();
            printf("（不加 sanitizer 时程序可能毫无反应地跑完 —— 这才是最危险的）\n");
            break;

        case 2:
            printf("共享内存竞态：s[t] 写完就读 s[t+1]，中间没有 __syncthreads()。\n");
            raceKernel<<<1, BLOCK>>>(d_o);
            CUDA_CHECK_KERNEL();
            break;

        case 3:
            printf("释放后使用：先 cudaFree，再拿同一个指针启动 kernel。\n");
            CUDA_CHECK(cudaFree(d_i));
            d_i = nullptr;
            // 这里用的 d_i 已经被释放了
            uninitReadKernel<<<gridExact, BLOCK>>>(d_i, d_o, N);
            CUDA_CHECK_KERNEL();
            printf("（很多平台上这一步会返回 illegal memory access）\n");
            // 注意：d_i 已经释放，下面不能再 free 它
            CUDA_CHECK(cudaFree(d_f));
            CUDA_CHECK(cudaFree(d_o));
            CUDA_CHECK_SYNC();
            return 0;

        case 4:
            printf("未初始化读：cudaMalloc 后没有写就立刻读。\n");
            uninitReadKernel<<<gridExact, BLOCK>>>(d_i, d_o, N);
            CUDA_CHECK_KERNEL();
            printf("（需要 --tool initcheck 才能稳定报出来）\n");
            break;

        default:
            printUsage();
            CUDA_CHECK(cudaFree(d_f));
            CUDA_CHECK(cudaFree(d_i));
            CUDA_CHECK(cudaFree(d_o));
            return 0;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("kernel 执行完毕。\n");
    printf("如果这是在 compute-sanitizer 之外跑的，说明错误可能被静默吞掉了，\n");
    printf("或者是运气好还没踩到 —— 请务必用 sanitizer 再跑一遍。\n");

    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_i));
    CUDA_CHECK(cudaFree(d_o));
    return 0;
}
