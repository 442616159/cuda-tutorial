// ============================================================
//  main.cpp —— 只依赖 gemm.h 上层驱动
//
//  位置：code/projects/cmake_skeleton/src/main.cpp
//
//  这个文件刻意不 include 任何 CUDA 头文件：
//  它在"有 GPU"和"没 GPU"的机器上用的是同一份二进制源码。
//
//  编译由 CMake 负责，见上两级目录的 CMakeLists.txt。
// ============================================================

#include "gemm.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

// CPU 端计时用 std::chrono 就够了：这里测的是毫秒级的东西
static double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static bool verify(const std::vector<float>& ref, const std::vector<float>& got,
                   size_t n, float tol = 1e-3f) {
    float maxRel = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(ref[i] - got[i]);
        maxRel = std::fmax(maxRel, d / std::fmax(1.f, std::fabs(ref[i])));
    }
    printf("    最大相对误差 = %.3e\n", maxRel);
    return maxRel < tol;
}

int main() {
    printf("后端: %s\n", backendName());

    // 尺寸刻意取小一点：CPU 参考实现是 O(N^3)，太大就要等很久
    const int M = 256, N = 256, K = 256;
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    std::vector<float> Cpu((size_t)M * N), Cgpu((size_t)M * N);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    printf("规模: C(%d x %d) = A(%d x %d) * B(%d x %d)\n\n", M, N, M, K, K, N);

    // ---------- CPU ----------
    double t0 = nowMs();
    sgemmCPU(M, N, K, A.data(), B.data(), Cpu.data());
    double tCpu = nowMs() - t0;
    printf("[CPU] %.3f ms  (%.1f MFLOPS)\n", tCpu,
           2.0 * M * N * K / (tCpu * 1e-3) / 1e6);

    // ---------- GPU ----------
    t0 = nowMs();
    sgemmGPU(M, N, K, A.data(), B.data(), Cgpu.data());
    double tGpu = nowMs() - t0;
    printf("[GPU] %.3f ms  (%.1f MFLOPS)\n", tGpu,
           2.0 * M * N * K / (tGpu * 1e-3) / 1e6);

    // ---------- 正确性 ----------
    printf("\n正确性检查（GPU vs CPU）:\n");
    bool ok = verify(Cpu, Cgpu, Cpu.size());
    printf("  结果: %s\n", ok ? "通过 ✔" : "失败 ✘");
    if (tGpu > 0) printf("\n加速比 = %.2fx（含显存拷贝；矩阵太小，拷贝占了大头）\n", tCpu / tGpu);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
