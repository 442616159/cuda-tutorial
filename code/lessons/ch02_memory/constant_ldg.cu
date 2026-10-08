// ============================================================
//  code/lessons/ch02_memory/constant_ldg.cu
//  常量内存（__constant__）与只读缓存（__ldg / const __restrict__）
//
//  场景：把一个大数组按"通道"做线性变换 out[i] = a[i] * scale[c] + bias[c]。
//  通道参数数组很小（几十个 float），且所有线程都会读到同样的值。
//
//  三个版本：
//      v1 globalPerThread  ：每个线程自己从全局内存读参数（最差）
//      v2 globalSMem       ：先把参数拷进共享内存，再读（好）
//      v3 constantMem      ：参数放在常量内存里（小数组广播的最佳解）
//
//  另外附一个 __ldg 的演示：只读数据用 __ldg 可以走只读数据缓存，
//  不占用 L1 的写回通路，还可能提高命中率。
//
//  编译（在 code/lessons/ch02_memory 目录下）：
//      nvcc -O3 -arch=sm_75 constant_ldg.cu -o constant_ldg
//  运行：
//      ./constant_ldg        (Linux)
//      constant_ldg.exe      (Windows)
// ============================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.h"
#include "../../common/cuda_timer.h"

constexpr int BLOCK = 256;
constexpr int CHANNELS = 8;                 // 通道数
constexpr int MAX_CH = 64;                  // 共享内存版本的上限

// ------------------------------------------------------------
// 常量内存：__constant__ 变量在"文件作用域"声明，
// 它的生命周期是整个应用，作用域是所有核函数。
//
// 两个要点：
//   1. 它由 host 用 cudaMemcpyToSymbol 写入，不能用普通赋值。
//   2. 它的物理位置在设备内存里，但有专门的 64 KB 常量缓存。
//      当"同一个 warp 的所有线程读同一个地址"时，硬件做一次
//      广播，等价于一次访问 —— 这就是它在大数组场景下最优的原因。
//   反过来，如果 warp 内不同线程读不同地址且地址不连续，
//      常量内存会被"串行化"，比全局内存还慢。
// ------------------------------------------------------------
__constant__ float c_scale[MAX_CH];
__constant__ float c_bias[MAX_CH];

// ------------------------------------------------------------
// v1：每个线程自己从全局内存读参数
//
// 对 c = i % CHANNELS 这个模式来说，warp 内的 32 个线程
// 其实只用到 8 个不同的通道值，且地址是连续重复的。
// 现代 GPU 的 L1 缓存能吸收一部分，但仍然要走
// "计算地址 → 查 L1 → 命中判断"的完整流程。
// ------------------------------------------------------------
__global__ void scaleGlobalPerThread(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     const float* __restrict__ scale,
                                     const float* __restrict__ bias,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int ch = i % CHANNELS;
        out[i] = in[i] * scale[ch] + bias[ch];
    }
}

// ------------------------------------------------------------
// v2：先把参数搬进共享内存
//
// 块的第一个 warp 用 8 个线程读 8 个通道的值，然后 __syncthreads()。
// 之后每个线程从共享内存读参数。
//
// 这比 v1 好，但仍然有共享内存的访问开销（虽然很小），
// 而且每个块都要重复搬一次。
// ------------------------------------------------------------
__global__ void scaleGlobalSMem(const float* __restrict__ in,
                                float* __restrict__ out,
                                const float* __restrict__ scale,
                                const float* __restrict__ bias,
                                int n) {
    __shared__ float s_scale[CHANNELS];
    __shared__ float s_bias[CHANNELS];

    // 只用前 CHANNELS 个线程去读，避免重复读取
    if (threadIdx.x < CHANNELS) {
        s_scale[threadIdx.x] = scale[threadIdx.x];
        s_bias[threadIdx.x] = bias[threadIdx.x];
    }
    // 屏障：必须等参数搬完，否则别的线程会读到未初始化的共享内存
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int ch = i % CHANNELS;
        out[i] = in[i] * s_scale[ch] + s_bias[ch];
    }
}

// ------------------------------------------------------------
// v3：常量内存
//
// 代码最干净，而且当 warp 内读取的地址相同时，
// 常量缓存做一次广播，开销接近寄存器访问。
// ------------------------------------------------------------
__global__ void scaleConstantMem(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int ch = i % CHANNELS;
        // 注意：这里读的是 __constant__ 变量，不是指针参数
        out[i] = in[i] * c_scale[ch] + c_bias[ch];
    }
}

// ------------------------------------------------------------
// __ldg 演示
//
// __ldg(ptr) 表示"以只读方式加载"：数据走只读数据缓存（texture path），
// 与 L1 的写通路分离。
//
// 什么时候有用？
//   - 数据在整个 kernel 生命周期内确定不会被写；
//   - 访存有局部性，希望它在只读缓存里长时间驻留；
//   - 或者你就是想让编译器知道"这不会被改动"，
//     从而敢于做更激进的指令重排。
//
// 现代编译器（sm_35+）看到 `const T* __restrict__` 通常会自动用 LDG，
// 所以显式写 __ldg 主要是"表达意图 + 兼容老代码"。
// 注意：__ldg 要求指针是 const 且指向只读数据，否则行为未定义。
// ------------------------------------------------------------
__global__ void copyWithRestrict(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];    // 编译器看到 const __restrict__，一般会自动发 LDG
    }
}

__global__ void copyWithLdg(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // 显式要求走只读缓存
        out[i] = __ldg(in + i);
    }
}

int main() {
    printDeviceInfo();

    const int N = 1 << 24;                 // 1677 万个元素 = 64 MB
    const size_t bytes = (size_t)N * sizeof(float);

    std::vector<float> hIn(N), hOut(N, 0.0f), hRef(N, 0.0f);
    for (int i = 0; i < N; ++i) hIn[i] = (float)(i % 1000) * 0.001f;

    std::vector<float> hScale(CHANNELS), hBias(CHANNELS);
    for (int c = 0; c < CHANNELS; ++c) {
        hScale[c] = 1.0f + 0.1f * c;
        hBias[c] = 0.5f - 0.05f * c;
    }
    for (int i = 0; i < N; ++i) {
        int c = i % CHANNELS;
        hRef[i] = hIn[i] * hScale[c] + hBias[c];
    }

    float *dIn = nullptr, *dOut = nullptr, *dScale = nullptr, *dBias = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, bytes));
    CUDA_CHECK(cudaMalloc(&dOut, bytes));
    CUDA_CHECK(cudaMalloc(&dScale, CHANNELS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBias, CHANNELS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dScale, hScale.data(), CHANNELS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), CHANNELS * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- 把参数写进常量内存 ----
    // 关键 API：cudaMemcpyToSymbol 的第一个参数是"符号本身"，
    // 不是它的地址（不要加 &）。CUDA 运行时会通过符号表找到它。
    CUDA_CHECK(cudaMemcpyToSymbol(c_scale, hScale.data(), CHANNELS * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bias, hBias.data(), CHANNELS * sizeof(float)));

    const int block = BLOCK;
    const int grid = (N + block - 1) / block;

    printf("\n数组 %d 个 float = %.1f MB，通道数 %d\n", N, bytes / 1024.0 / 1024.0, CHANNELS);
    printf("--------------------------------------------------------------------------------\n");

    auto checkAndReport = [&](const char* label, float* devOut, double ms) {
        CUDA_CHECK(cudaMemcpy(hOut.data(), devOut, bytes, cudaMemcpyDeviceToHost));
        double maxErr = 0.0;
        for (int i = 0; i < N; ++i) {
            maxErr = std::max(maxErr, (double)fabs(hOut[i] - hRef[i]));
        }
        // 一次读 in、一次写 out，共 8 字节每元素
        double moved = 2.0 * bytes;
        printf("  %-26s 耗时 %7.4f ms   有效带宽 %7.1f GB/s   最大误差 %.2e  %s\n",
               label, ms, CudaTimer::bandwidthGBs(moved, ms), maxErr,
               (maxErr < 1e-5) ? "正确" : "错误");
    };

    {
        double ms = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            scaleGlobalPerThread<<<grid, block>>>(dIn, dOut, dScale, dBias, N);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);
        checkAndReport("v1 全局内存直读", dOut, ms);
    }
    {
        double ms = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            scaleGlobalSMem<<<grid, block>>>(dIn, dOut, dScale, dBias, N);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);
        checkAndReport("v2 参数进共享内存", dOut, ms);
    }
    {
        double ms = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            scaleConstantMem<<<grid, block>>>(dIn, dOut, N);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);
        checkAndReport("v3 参数进常量内存", dOut, ms);
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("__ldg 对比（就是一次朴素拷贝）：\n");
    {
        double ms = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            copyWithRestrict<<<grid, block>>>(dIn, dOut, N);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);
        checkAndReport("copy: const __restrict__", dOut, ms);
    }
    {
        double ms = CudaTimer::measure([&](double& m, int) {
            CudaTimer t; t.start();
            copyWithLdg<<<grid, block>>>(dIn, dOut, N);
            CUDA_CHECK_KERNEL();
            m = t.stopMs();
        }, 10, 2);
        checkAndReport("copy: __ldg()", dOut, ms);
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("\n怎么解读：\n");
    printf("  1. 这三个版本的瓶颈很可能都是显存带宽，因为它们都在做\n");
    printf("     【读 1 次 + 写 1 次】这种搬运算，算术强度极低。\n");
    printf("     所以 v1/v2/v3 的耗时差别可能很小 —— 这本身就是重要结论：\n");
    printf("     当程序是带宽受限时，优化参数读取方式几乎没有意义。\n");
    printf("  2. 要看出常量内存/共享内存的收益，需要让【访问参数】成为瓶颈：\n");
    printf("     把内层循环改成 for(int r=0;r<100;r++) out[i] += in[i]*scale[ch];\n");
    printf("     参数读取次数翻 100 倍，此时 v3 的优势会明显显现。\n");
    printf("  3. __ldg 与 const __restrict__ 两个版本的耗时通常非常接近，\n");
    printf("     因为现代编译器会自动为后者生成 LDG 指令。\n");
    printf("  4. 在你的机器上实测约为多少可能不同，请以自己测到的为准。\n");
    printf("  5. 想确认指令是否真的是 LDG，用：\n");
    printf("     cuobjdump -sass constant_ldg.exe | Select-String 'LDG'\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    CUDA_CHECK(cudaFree(dScale));
    CUDA_CHECK(cudaFree(dBias));
    printf("\n全部完成。\n");
    return 0;
}
