// ============================================================
//  v2_global_tile.cu —— SGEMM 版本 2：寄存器分块（thread tiling）
//
//  优化思路（回答 STYLE_GUIDE 要求的三个问题）：
//    1) 原来慢在哪？v1 里每算 1 个输出元素要读 K 次 A 和 K 次 B，
//       访存/计算比是 8 字节 : 2 FLOP。
//    2) 为什么这一版会快？让每个线程一次算 TS x TS 个输出。
//       读 1 个 A 值可以被 TS 个输出复用，读 1 个 B 值同理，
//       于是访存/计算比降为原来的 1/TS。
//       算 TS=4 时，从 8 字节/2FLOP 降到 2 字节/2FLOP，理论提升 4 倍。
//    3) 怎么证明？和 v1 在同一台机器、同一组数据上跑，比 GFLOPS。
//
//  注意：这一版数据**仍然从全局内存读**，只是读的次数少了。
//  下一版（v3）才引入共享内存，解决"不同线程重复读同一块数据"的问题。
//
//  编译： nvcc -O3 -arch=native v2_global_tile.cu -o sgemm_v2
// ============================================================

#include "sgemm_common.h"

// 每个线程计算的输出块边长。4 是一个经验起点：
// 再大寄存器压力会明显上升，再小复用收益不足。
constexpr int TS = 4;

// ------------------------------------------------------------
// 线程块 16x16 = 256 线程，每线程 TS x TS = 4x4，
// 因此一个 block 负责 64x64 的输出区域。
// ------------------------------------------------------------
__global__ void sgemmGlobalTile(int M, int N, int K,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C) {
    const int rowStart = (blockIdx.y * blockDim.y + threadIdx.y) * TS;
    const int colStart = (blockIdx.x * blockDim.x + threadIdx.x) * TS;
    if (rowStart >= M || colStart >= N) return;

    // 累加器放在寄存器里：整个 K 循环期间只有它被反复读写，
    // 寄存器访问是零延迟的，这是"用寄存器换访存"的核心。
    float acc[TS][TS];
#pragma unroll
    for (int i = 0; i < TS; ++i)
#pragma unroll
        for (int j = 0; j < TS; ++j) acc[i][j] = 0.f;

    for (int k = 0; k < K; ++k) {
        // 先把这一轮 k 需要的 A 列 和 B 行 一次性读进寄存器数组
        float aReg[TS], bReg[TS];
#pragma unroll
        for (int i = 0; i < TS; ++i)
            aReg[i] = A[(size_t)(rowStart + i) * K + k];
#pragma unroll
        for (int j = 0; j < TS; ++j)
            bReg[j] = B[(size_t)k * N + colStart + j];

        // 每个读进来的值都被用了 TS 次
#pragma unroll
        for (int i = 0; i < TS; ++i)
#pragma unroll
            for (int j = 0; j < TS; ++j) acc[i][j] += aReg[i] * bReg[j];
    }

    // 统一写回。写 C 的次数没变，但读的次数大大减少了。
#pragma unroll
    for (int i = 0; i < TS; ++i)
#pragma unroll
        for (int j = 0; j < TS; ++j) {
            const int r = rowStart + i, c = colStart + j;
            if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
        }
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    sgemm::printBandwidthReference();
    printf("\n=== v2 全局内存分块（每线程 %dx%d）: %d x %d x %d ===\n", TS, TS, M, N, K);

    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    std::vector<float> hC((size_t)M * N, 0.f), hRef((size_t)M * N, 0.f);
    sgemm::initMatrix(hA, 1);
    sgemm::initMatrix(hB, 2);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((N + block.x * TS - 1) / (block.x * TS),
                    (M + block.y * TS - 1) / (block.y * TS));

    auto launch = [&]() {
        sgemmGlobalTile<<<grid, block>>>(M, N, K, dA, dB, dC);
        CUDA_CHECK_KERNEL();
    };

    launch();
    CUDA_CHECK_SYNC();
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    printf("正在计算 CPU 参考结果...\n");
    sgemm::cpuSgemm(M, N, K, hA.data(), hB.data(), hRef.data());
    const bool ok = sgemm::verify(hRef, hC);
    printf("正确性: %s\n", ok ? "通过 ✔" : "失败 ✘");

    const sgemm::Result r = sgemm::benchmark(launch, M, N, K);
    sgemm::printTableHeader();
    sgemm::printRow("v2 寄存器分块 4x4", r);

    printf("\n结论：相比 v1 应有数倍提升，但仍未达到峰值。原因有两个：\n");
    printf("  1) B 的读取模式是 [k][colStart..colStart+3]，相邻线程跨了 4 个 float，\n");
    printf("     一个 warp 的访存被拆成更多内存事务，实际带宽利用率不高；\n");
    printf("  2) A 的同一行数据被块内 16 个不同线程各读了一遍，\n");
    printf("     这是纯粹的重复劳动 —— 共享内存就是为它准备的。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
