// ============================================================
//  code/lessons/ch04_perf_model/roofline.cu
//  第 4 章示例 3：Roofline（屋顶线）模型
//
//  Roofline 回答一个最根本的问题：
//      「这个 kernel 到底是被算力卡住，还是被带宽卡住？」
//
//  两个天花板：
//     算力上限：不管数据多近，每秒最多也就做这么多浮点运算  -> 一条水平线
//     带宽上限：每秒最多搬这么多字节，而每做 1 次浮点运算
//               至少要搬 1/算术强度 个字节          -> 一条斜率线
//  实际能达到的性能 = 两条线的较小值，形状像屋顶。
//
//  横轴：算术强度 AI = 浮点运算次数 / 搬运字节数（FLOP/Byte）
//  纵轴：性能（GFLOPS）
//  两条线的交点叫 ridge point（屋脊点）：
//      AI < AI_ridge  -> 带宽受限（在斜坡上）
//      AI > AI_ridge  -> 计算受限（在平顶上）
//
//  编译： nvcc -O3 -arch=sm_70 roofline.cu -o roofline
//  运行： ./roofline
// ============================================================
#include "../../common/cuda_check.h"
#include <cmath>
#include <cstdio>

// ------------------------------------------------------------
// 每 SM 的 FP32 核心数（按 Compute Capability 查表）
// 这张表随架构变化，写死是不精确的，所以下面会明确标注"估算"。
// ------------------------------------------------------------
static int fp32CoresPerSm(int major, int minor) {
    if (major == 3) return 192;                        // Kepler
    if (major == 5) return 128;                        // Maxwell
    if (major == 6) return (minor == 0) ? 64 : 128;    // Pascal：P100 是 64
    if (major == 7) return 64;                         // Volta / Turing
    if (major == 8) return (minor == 0) ? 64 : 128;    // A100 是 64，消费级是 128
    if (major == 9) return 128;                        // Hopper
    return 128;                                        // 未知架构，保守猜
}

int main() {
    printf("=== Roofline 模型演示 ===\n");

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // 用 cudaDeviceGetAttribute 而不是 cudaDeviceProp 的字段：
    // 后者的时钟相关成员在不同 CUDA 版本里被移除过，属性接口更稳定。
    int memClockKHz = 0, busWidth = 0, coreClockKHz = 0;
    // CUDA 13 起 cudaDevAttrMemoryBusWidth 改名为 cudaDevAttrGlobalMemoryBusWidth
#if CUDART_VERSION >= 13000
    const cudaDeviceAttr kBusWidthAttr = cudaDevAttrGlobalMemoryBusWidth;
#else
    const cudaDeviceAttr kBusWidthAttr = cudaDevAttrMemoryBusWidth;
#endif
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&busWidth,    kBusWidthAttr, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&coreClockKHz, cudaDevAttrClockRate, dev));

    // ---------- 两个天花板 ----------
    // 峰值带宽：显存时钟 × 位宽 / 8 × 2（DDR 每周期两次）
    double peakBW = 2.0 * memClockKHz * (busWidth / 8.0) / 1.0e6;      // GB/s

    // 峰值算力：SM 数 × 每 SM 核心数 × 时钟(Hz) × 2（FMA 算 2 次浮点）
    int coresPerSm = fp32CoresPerSm(prop.major, prop.minor);
    double peakF = (double)prop.multiProcessorCount * coresPerSm *
                   (coreClockKHz * 1.0e3) * 2.0 / 1.0e9;               // GFLOPS

    double aiRidge = peakF / peakBW;    // 屋脊点的算术强度

    printf("\n设备：%s（CC %d.%d，%d 个 SM）\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("  显存时钟 / 位宽        : %.0f MHz / %d bit\n",
           memClockKHz / 1000.0, busWidth);
    printf("  峰值带宽（估算）       : %.1f GB/s\n", peakBW);
    printf("  每 SM FP32 核心数      : %d（查表估算）\n", coresPerSm);
    printf("  核心时钟               : %.0f MHz\n", coreClockKHz / 1000.0);
    printf("  峰值 FP32 算力（估算） : %.1f GFLOPS\n", peakF);
    printf("  屋脊点 AI_ridge        : %.2f FLOP/Byte\n", aiRidge);

    // ---------- ASCII 屋顶线图 ----------
    printf("\n--- Roofline（纵轴 GFLOPS，横轴算术强度，对数刻度）---\n");

    const double aiMin = 0.0625;   // 2^-4
    const double aiMax = 1024.0;   // 2^10
    const int rows = 16;
    const int cols = 72;

    for (int r = rows; r >= 0; --r) {
        double yLevel = peakF * r / rows;
        printf("%8.0f |", yLevel);
        for (int c = 0; c < cols; ++c) {
            // 横轴按对数均匀取样
            double t = (double)c / (cols - 1);
            double ai = aiMin * std::pow(aiMax / aiMin, t);
            double y = std::fmin(peakF, ai * peakBW);
            putchar(y >= yLevel ? '#' : ' ');
        }
        putchar('\n');
    }
    // 横轴刻度
    printf("         +");
    for (int c = 0; c < cols; ++c) putchar('-');
    printf("\n          ");
    printf("0.0625   0.25      1        4       16       64      256    1024\n");
    printf("          算术强度 AI（FLOP/Byte，对数刻度）\n");
    printf("  斜坡上的点 = 带宽受限；平顶上的点 = 计算受限；\n");
    printf("  交点 R 大约在 AI = %.2f 处。\n", aiRidge);

    // ---------- 用几个典型 kernel 举例 ----------
    // 算术强度 = FLOPs / Bytes
    struct Case {
        const char* name;
        double ai;
        const char* note;
    };
    const Case cases[] = {
        {"向量加 c=a+b      ", 1.0 / 12.0, "读 2 写 1，AI 极低，铁定带宽受限"},
        {"二维 5 点 stencil ", 6.0 / 20.0, "每点 5 次乘加、读 4 邻居写 1（未做 tiling）"},
        {"矩阵乘 n=1024     ", 2.0 * 1024 / (3.0 * 4), "2n^3 FLOP / 3n^2*4 Byte，会跑到平顶上"},
        {"矩阵乘 n=64       ", 2.0 * 64 / (3.0 * 4), "n 太小时反而带宽受限"},
        {"归约求和           ", 1.0 / 4.0, "读 N 写 1，纯带宽受限"},
    };

    printf("\n--- 几个典型 kernel 的处境 ---\n");
    printf("  %-20s | %10s | %12s | %s\n", "kernel", "AI", "屋顶上限", "判断");
    printf("  ---------------------+------------+--------------+----------\n");
    for (const Case& c : cases) {
        double ceiling = std::fmin(peakF, c.ai * peakBW);
        const char* verdict = (c.ai < aiRidge) ? "带宽受限" : "计算受限";
        printf("  %-20s | %10.3f | %9.1f GF | %s（%s）\n",
               c.name, c.ai, ceiling, verdict, c.note);
    }

    printf("\n怎么用这张图指导优化：\n");
    printf("  1. 先算出你 kernel 的 AI，看它落在斜坡还是平顶。\n");
    printf("  2. 在斜坡上（带宽受限）：优化方向是「减少搬运字节」——\n");
    printf("     合并访存、复用数据（tiling）、提高有用字节占比、换更小的数据类型。\n");
    printf("  3. 在平顶上（计算受限）：优化方向是「减少运算或提高指令效率」——\n");
    printf("     用 FMA、减少冗余计算、上 Tensor Core、降低精度。\n");
    printf("  4. 但真正判断瓶颈必须回到实测：用第 4 章的 ncu\n");
    printf("     看 Memory Throughput 和 Compute (SM) Throughput 谁更接近 100%%。\n");
    printf("  5. 上面所有峰值都是「估算」：真实硬件还有各种上限，\n");
    printf("     实测能碰到理论值的 70%%~90%% 就非常好了。\n");

    return 0;
}
