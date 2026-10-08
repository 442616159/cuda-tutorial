#pragma once
// ============================================================
//  gemm.h —— 工程的唯一对外接口
//
//  工程化的第一条纪律：接口与实现分离。
//  上层代码（main.cpp、测试）只 include 这个头，永远不 include
//  cuda_runtime.h。这样即使把 GPU 实现整体换成别的后端，
//  上层也不用改一行。
// ============================================================

// 行主序 C(M x N) = A(M x K) * B(K x N)
// 三个实现必须满足完全相同的语义，否则做性能对比就没有意义。
void sgemmCPU(int M, int N, int K, const float* A, const float* B, float* C);
void sgemmGPU(int M, int N, int K, const float* A, const float* B, float* C);

// 运行期能力查询：让上层能决定要不要调用 GPU 版本
bool gpuAvailable();
const char* backendName();
