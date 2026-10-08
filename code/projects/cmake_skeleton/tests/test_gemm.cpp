// ============================================================
//  test_gemm.cpp —— 不依赖任何测试框架的最小单元测试
//
//  位置：code/projects/cmake_skeleton/tests/test_gemm.cpp
//
//  真实项目里可以用 GoogleTest/Catch2，但测试的核心思想是同一个：
//    1) 固定随机种子，让失败可复现
//    2) 覆盖边界（M/N/K 不是分块大小的整数倍、K=1、1x1）
//    3) 用相对误差而不是 == 判断浮点结果
//    4) 返回非零退出码，让 CI 能自动发现问题
//
//  通过 CMake 的 add_test 注册后，运行 ctest 即可执行。
// ============================================================

#include "gemm.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

static int g_failed = 0;

static void checkShape(int M, int N, int K) {
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    std::vector<float> C1((size_t)M * N), C2((size_t)M * N, 0.f);

    // 固定种子 -> 同一组输入永远复现同一个失败
    std::mt19937 rng(M * 1000003u + N * 1009u + K);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    sgemmCPU(M, N, K, A.data(), B.data(), C1.data());
    sgemmGPU(M, N, K, A.data(), B.data(), C2.data());

    float maxRel = 0.f;
    for (size_t i = 0; i < C1.size(); ++i)
        maxRel = std::fmax(maxRel, std::fabs(C1[i] - C2[i]) / std::fmax(1.f, std::fabs(C1[i])));

    const char* mark = maxRel < 1e-3f ? "PASS" : "FAIL";
    if (maxRel >= 1e-3f) ++g_failed;
    printf("[%s] M=%d N=%d K=%d  最大相对误差 = %.3e\n", mark, M, N, K, maxRel);
}

int main() {
    printf("=== gemm 单元测试 ===\n");
    checkShape(1, 1, 1);        // 最小规模：最容易暴露索引写死的问题
    checkShape(16, 16, 16);     // 正好一个 tile
    checkShape(17, 31, 33);     // 非整除：验证边界处理
    checkShape(64, 64, 1);      // K=1 退化成外积
    checkShape(100, 128, 96);   // 不规则形状
    printf("=== %s ===\n", g_failed == 0 ? "全部通过" : "存在失败用例");
    return g_failed == 0 ? 0 : 1;  // 退出码是 CI 判断成败的依据
}
