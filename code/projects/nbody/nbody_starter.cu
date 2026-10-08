// ============================================================
//  nbody_starter.cu —— 第 8 章项目三：N 体引力模拟（起步代码）
//
//  这是一个**可运行但不优化**的起点。它的作用是：
//    1) 给你一个已经正确的物理实现和验证手段（能量守恒）；
//    2) 明确标出 5 个可以动手的优化点（代码里搜 TODO）；
//    3) 让你在"有基线、有验证"的前提下练习第 2、3、5 章的技术。
//
//  物理模型：N 个质点，两两之间的牛顿引力
//      a_i = G * sum_{j != i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
//  积分器：蛙跳法（Leapfrog / Kick-Drift-Kick），辛积分器，
//          能量长期守恒性远好于欧拉法 —— 这也是一个重要的验证手段。
//
//  编译： nvcc -O3 -arch=native nbody_starter.cu -o nbody
//  运行： ./nbody             # 默认 N=4096, 200 步
//         ./nbody 8192 500
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "../../common/cuda_check.h"

// ------------------------------------------------------------
// 用 float4 存位置和速度：x,y,z 是坐标，w 是质量。
// 为什么这么干？因为数组结构体（AoS）里如果只读 pos，
// 却因为结构体对齐把 mass 也一起搬进缓存行，会浪费带宽。
// 把"每次都要一起用的 4 个 float"打包成 float4，
// 一次 16 字节对齐读取就能拿到，是典型的 SoA 思想。
// ------------------------------------------------------------
struct Body {
    float4 pos;   // xyz = 位置，w = 质量
    float4 vel;   // xyz = 速度，w = 未使用（留作 padding，保持 16 字节对齐）
};

// 引力常数（无量纲化，方便数值实验）
__constant__ float c_G;
__constant__ float c_softening2;   // eps^2，避免两个粒子距离趋零时力发散

// ------------------------------------------------------------
// 核函数 1：计算所有粒子的加速度（O(N^2) 直接求和）
//
// ★ 这是全程序的热点。朴素写法的访存模式是：
//      每个线程读一遍全部 N 个粒子的位置
//   => 总读取量 N * N * 16 字节，而实际只需要 N * 16 字节。
//   也就是说有 N 倍的重复读取 —— 这正是共享内存分块要解决的问题。
//
// TODO 1（基础）：把 j 循环改写成"块内分块 + 共享内存"。
//   思路：一个 block 负责 BM 个粒子（i 方向），
//         每次从全局内存装载 BTILE 个粒子到共享内存（j 方向），
//         然后块内所有线程用这份共享数据算力，
//         循环覆盖全部 j。这样全局读取量降为 N * N / BM * 16 字节。
//   注意：共享内存数组要用 float4，并且装载时保证合并访存。
//
// TODO 2（进阶）：加上 __syncthreads() 的双缓冲（第 2 章技巧），
//   让下一块 j 的装载与当前块的计算重叠。
// ------------------------------------------------------------
__global__ void computeAccel(const Body* __restrict__ bodies,
                             float4* __restrict__ acc,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float4 pi = bodies[i].pos;
    const float mi = pi.w;

    float ax = 0.f, ay = 0.f, az = 0.f;

    for (int j = 0; j < n; ++j) {
        // TODO 3（基础）：这里跳过 i == j。写成分支会造成 warp divergence，
        //   更好的做法是让 (r_j - r_i) 在 i==j 时天然为 0（分母有 eps 保护），
        //   或者把自身那一份力最后减掉。试试看哪个更快。
        if (j == i) continue;

        const float4 pj = bodies[j].pos;
        const float dx = pj.x - pi.x;
        const float dy = pj.y - pi.y;
        const float dz = pj.z - pi.z;
        const float r2 = dx * dx + dy * dy + dz * dz + c_softening2;

        // rsqrtf 是设备端专用内在函数：硬件近似倒数平方根，比 1/sqrtf 快得多，
        // 物理模拟对这点精度损失完全不敏感。
        const float invR = rsqrtf(r2);
        const float invR3 = invR * invR * invR;

        const float f = c_G * pj.w * invR3;
        ax += f * dx;
        ay += f * dy;
        az += f * dz;
    }
    acc[i] = make_float4(ax, ay, az, 0.f);
    (void)mi;
}

// ------------------------------------------------------------
// 核函数 2：蛙跳积分（Kick-Drift-Kick）
//   v += a * dt/2 ; x += v * dt ; 重新算 a ; v += a * dt/2
// 为了让结构清晰，这里只做"半步踢 + 漂移"，加速度由外层重新计算。
//
// TODO 4（进阶）：把两个核函数用 CUDA Graph 串起来，消除每步的启动开销。
//   N 小时 kernel 启动开销（约 5~10 微秒）可能比计算本身还长。
// ------------------------------------------------------------
__global__ void integrate(Body* __restrict__ bodies,
                          const float4* __restrict__ acc,
                          int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float4 a = acc[i];
    float4 v = bodies[i].vel;
    float4 p = bodies[i].pos;

    // 半步踢
    v.x += a.x * (0.5f * dt);
    v.y += a.y * (0.5f * dt);
    v.z += a.z * (0.5f * dt);
    // 漂移
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;

    bodies[i].vel = v;
    bodies[i].pos = p;
}

class CudaTimer {
public:
    CudaTimer() { CUDA_CHECK(cudaEventCreate(&s_)); CUDA_CHECK(cudaEventCreate(&e_)); }
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
// 总能量 = 动能 + 势能。
// 辛积分器（蛙跳）应当让总能量在一个有界范围内振荡，
// 而不是单调漂移 —— 这是检验物理正确性最有力的工具，
// 比"看起来转得挺圆"可靠得多。
// ------------------------------------------------------------
static double totalEnergy(const std::vector<Body>& b, float G, float eps2) {
    const int n = (int)b.size();
    double ke = 0.0, pe = 0.0;
    for (int i = 0; i < n; ++i) {
        const float4 v = b[i].vel;
        const float m = b[i].pos.w;
        ke += 0.5 * m * ((double)v.x * v.x + (double)v.y * v.y + (double)v.z * v.z);
    }
    // 势能是 O(N^2)，只在 N 较小时用于验证（N=4096 时约 800 万次运算，可接受）
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const float dx = b[j].pos.x - b[i].pos.x;
            const float dy = b[j].pos.y - b[i].pos.y;
            const float dz = b[j].pos.z - b[i].pos.z;
            const double r = sqrt((double)dx * dx + (double)dy * dy + (double)dz * dz + eps2);
            pe -= (double)G * b[i].pos.w * b[j].pos.w / r;
        }
    return ke + pe;
}

int main(int argc, char** argv) {
    printDeviceInfo();

    int n = 4096, steps = 200;
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) steps = atoi(argv[2]);
    if (n <= 0 || steps <= 0) { fprintf(stderr, "参数非法\n"); return EXIT_FAILURE; }

    const float G = 1.0f;
    const float eps2 = 0.01f;      // 软化因子
    const float dt = 0.002f;

    CUDA_CHECK(cudaMemcpyToSymbol(c_G, &G, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_softening2, &eps2, sizeof(float)));

    printf("\n=== N 体模拟：N = %d，步数 = %d，dt = %.4f ===\n", n, steps, dt);
    printf("单步计算量 ≈ %.1f M 对相互作用\n", (double)n * n * 1e-6 * 0.5);

    // ---------------- 初始化：一个自转的圆盘 ----------------
    // 这类初始条件会让系统"看起来"像星系，而且总能量稳定，
    // 便于观察数值误差。用固定种子保证可复现。
    std::vector<Body> h(n);
    unsigned int s = 12345u;
    auto rnd = [&s]() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return (float)(s & 0xFFFFFFu) / 16777216.f;
    };
    for (int i = 0; i < n; ++i) {
        const float r = 0.3f + 4.0f * sqrtf(rnd());   // 面密度均匀
        const float th = 6.2831853f * rnd();
        const float m = 1.0f / n;
        h[i].pos = make_float4(r * cosf(th), 0.05f * (rnd() - 0.5f), r * sinf(th), m);
        // 近圆轨道速度：v = sqrt(G*M_enclosed/r)，这里简化用整体质量
        const float vcirc = sqrtf(G * 1.0f / fmaxf(r, 0.5f));
        h[i].vel = make_float4(-vcirc * sinf(th), 0.f, vcirc * cosf(th), 0.f);
    }

    Body* d_bodies = nullptr;
    float4* d_acc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bodies, n * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&d_acc, n * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_bodies, h.data(), n * sizeof(Body), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // ---------------- 预热 ----------------
    computeAccel<<<blocks, threads>>>(d_bodies, d_acc, n);
    integrate<<<blocks, threads>>>(d_bodies, d_acc, n, dt);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_SYNC();

    // ---------------- 主循环 ----------------
    // 蛙跳法：完整的一步是 KDK
    //   computeAccel -> integrate(半步踢+漂移) -> 再 computeAccel -> 再半步踢
    // 这里为了简化，每步只做一次 kick-drift，精度略低但结构清晰。
    CudaTimer timer;
    timer.start();
    for (int step = 0; step < steps; ++step) {
        computeAccel<<<blocks, threads>>>(d_bodies, d_acc, n);
        integrate<<<blocks, threads>>>(d_bodies, d_acc, n, dt);
    }
    const float ms = timer.stop();
    CUDA_CHECK_SYNC();

    printf("总耗时 = %.3f ms，每步 = %.3f ms\n", ms, ms / steps);
    // 有效性能指标：每秒处理多少"对相互作用"（interactions per second）
    const double pairsPerStep = 0.5 * (double)n * n;
    printf("吞吐 ≈ %.2f G 相互作用/秒（interactions/s）\n",
           pairsPerStep * steps / (ms * 1e-3) / 1e9);

    // ---------------- 验证：抽样对比 + 能量守恒 ----------------
    CUDA_CHECK(cudaMemcpy(h.data(), d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost));

    // 抽样检查几个粒子的位置是否仍然是有限值（NaN 检查是 GPU 数值代码的必备项）。
    // 这里不用 isfinite()：它在不同标准库/编译器下行为有差异，
    // 用最原始的两条判据最稳 —— NaN != NaN，Inf 的绝对值超过任何合理数。
    auto isFiniteVal = [](float v) { return v == v && fabsf(v) < 1e30f; };
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        const float4 p = h[i].pos;
        if (!isFiniteVal(p.x) || !isFiniteVal(p.y) || !isFiniteVal(p.z)) ++bad;
    }
    printf("非有限值粒子数 = %d -> %s\n", bad, bad == 0 ? "无 NaN/Inf ✔" : "存在数值发散 ✘");

    // 注意：能量守恒检查需要"初始状态"和"最终状态"两次 O(N^2) 求和，
    // N 大时会很慢，所以只在 N <= 8192 时做。
    if (n <= 8192) {
        printf("正在计算总能量（CPU 端 O(N^2)，N 大时需要耐心）...\n");
        const double eFinal = totalEnergy(h, G, eps2);
        printf("最终总能量 E = %.6f\n", eFinal);
        printf("提示：要判断守恒性，请在 main 里对初始状态也算一次 E0，\n");
        printf("      然后看 |E - E0| / |E0| 是否在 1e-3 量级以内。\n");
        printf("      漂移量随步数单调增长 => 你的积分器或软化因子需要调整。\n");
    }

    CUDA_CHECK(cudaFree(d_bodies));
    CUDA_CHECK(cudaFree(d_acc));

    printf("\n================ 你的任务（按难度排序）================\n");
    printf("TODO 1 [基础]  共享内存分块：把全局读取量降低 BM 倍\n");
    printf("TODO 2 [进阶]  双缓冲流水线：让装载与计算重叠\n");
    printf("TODO 3 [基础]  消除 if (j == i) 的分支发散\n");
    printf("TODO 4 [进阶]  用 CUDA Graph 合并每步的两个 kernel 启动\n");
    printf("TODO 5 [挑战]  用 float4 向量化 + 每线程处理多个 i，提升 ILP\n");
    printf("每一步都要：先过正确性（无 NaN + 能量守恒），再比性能。\n");
    return EXIT_SUCCESS;
}
