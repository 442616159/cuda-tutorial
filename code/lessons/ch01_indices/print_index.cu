// ============================================================
//  code/lessons/ch01_indices/print_index.cu
//  打印每个线程的 (blockIdx, threadIdx)，观察线程是怎么组织的。
//
//  这个程序很小，但它是理解线程索引体系最有效的手段：
//  把数字打出来看一遍，比读十遍文档管用。
//
//  编译（在 code/lessons/ch01_indices 目录下）：
//      nvcc -O3 -arch=sm_75 print_index.cu -o print_index
//  运行：
//      ./print_index        (Linux)
//      print_index.exe      (Windows)
// ============================================================

#include <cstdio>

#include "../../common/cuda_check.h"

// ------------------------------------------------------------
// 核函数一：打印一维索引的分解与合成
//
// gridDim  : 网格里有几个块（<<<>>第一个参数）
// blockDim : 每个块里有几个线程（<<<>>第二个参数）
// blockIdx : 我所在的块编号，范围 [0, gridDim)
// threadIdx: 我在本块内的编号，范围 [0, blockDim)
// ------------------------------------------------------------
__global__ void printLinearIndex() {
    int gi = blockIdx.x * blockDim.x + threadIdx.x;  // 全局编号

    // 打印时把"我是谁"和"我算出的全局编号"一起打出来，
    // 这样你就能亲眼看到：不同的 (blockIdx, threadIdx) 组合
    // 被唯一地映射成了一个连续的全局编号。
    printf("块 %2d/%-2d 内线程 %2d/%-2d  ->  全局编号 = %2d*%2d + %2d = %2d\n",
           blockIdx.x, gridDim.x - 1,
           threadIdx.x, blockDim.x - 1,
           blockIdx.x, blockDim.x, threadIdx.x, gi);
}

// ------------------------------------------------------------
// 核函数二：二维线程块，打印二维索引是怎么"展平"成一维的。
//
// 二维块里，(threadIdx.x, threadIdx.y) 的组合是"行主序"的：
//     展平下标 = threadIdx.y * blockDim.x + threadIdx.x
// 也就是"先横向排满一行，再换到下一行"。
// 主机的数组 A[row][col] 也是这么摆的，所以两者天然对齐。
// ------------------------------------------------------------
__global__ void print2DIndex() {
    // blockDim.x 是每行的线程数，也是二维数组的"列数"
    int flatInBlock = threadIdx.y * blockDim.x + threadIdx.x;
    // 整个网格里所有块也是按行主序排列的，所以：
    //   全局二维坐标 = (块坐标 * 块尺寸) + 块内坐标
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    printf("  (bx=%d, by=%d) (tx=%d, ty=%d)  块内展平=%2d  全局(row=%d, col=%d)\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
           flatInBlock, row, col);
}

// ------------------------------------------------------------
// 核函数三：三维线程块。三维以上 CUDA 不再支持。
// 展平公式：(((z * dim.y) + y) * dim.x) + x
//          即"先 x，再 y，最后 z"的嵌套循环顺序。
// ------------------------------------------------------------
__global__ void print3DIndex() {
    int flat = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

    // 顺便算一下全局的三维坐标
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;

    printf("  线程(3D) (tx=%d,ty=%d,tz=%d)  块内展平=%2d  全局(x=%d,y=%d,z=%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           flat, gx, gy, gz);
}

int main() {
    printDeviceInfo();

    // 问一下这台设备的网格维度上限，免得为了演示而超过硬件限制
    int maxGridX = 0, maxGridY = 0, maxGridZ = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxGridX, cudaDevAttrMaxGridDimX, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxGridY, cudaDevAttrMaxGridDimY, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxGridZ, cudaDevAttrMaxGridDimZ, 0));
    printf("本设备网格维度上限: x=%d, y=%d, z=%d\n\n", maxGridX, maxGridY, maxGridZ);

    // ================= 演示 1：一维索引 =================
    // 3 个块 × 4 个线程 = 12 个线程。
    // 故意用非 32 倍数的 4，是为了让下面打印出来的 12 行短一点、
    // 一眼能看完。真实程序里请用 128/256 这样的 32 倍数。
    printf("========== 演示 1：一维索引 (grid=3, block=4) ==========\n");
    printLinearIndex<<<3, 4>>>();
    CUDA_CHECK_KERNEL();
    // 必须同步：printf 的输出是异步的，不同步可能看不到任何东西
    CUDA_CHECK_SYNC();

    // 改一下维度，观察同一份代码在不同配置下的行为
    printf("\n========== 演示 1b：一维索引 (grid=2, block=6) ==========\n");
    printLinearIndex<<<2, 6>>>();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_SYNC();

    // ================= 演示 2：二维索引 =================
    // 二维块必须保证 x 是 32 的倍数（因为 warp 是沿着 x 方向切的），
    // 所以这里用 block(4,2) 只是为了"打印短一点"的教学目的。
    printf("\n========== 演示 2：二维索引 grid(2,2) block(4,2) ==========\n");
    print2DIndex<<<dim3(2, 2), dim3(4, 2)>>>();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_SYNC();

    // ================= 演示 3：三维索引 =================
    // z 维度通常留给"深度"或"通道"，而且 x 仍然应该是 32 的倍数。
    if (maxGridZ >= 1) {
        printf("\n========== 演示 3：三维索引 grid(1,1,2) block(2,2,2) ==========\n");
        print3DIndex<<<dim3(1, 1, 2), dim3(2, 2, 2)>>>();
        CUDA_CHECK_KERNEL();
        CUDA_CHECK_SYNC();
    }

    printf("\n观察结论：\n");
    printf("  1. printf 的顺序不是按线程编号排的，因为多线程并发写同一个缓冲。\n");
    printf("  2. 但 (blockIdx, threadIdx) 与全局编号的映射是一一对应的、连续的。\n");
    printf("  3. 二维索引的展平等价于行主序数组下标，所以两者可以放心互相换算。\n");

    return 0;
}
