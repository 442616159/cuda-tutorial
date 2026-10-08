// ============================================================
//  v1_naive.cu —— SGEMM 版本 1：朴素三重循环
//
//  这一版的目的不是快，而是**建立一个绝对正确的基线**。
//  后面每一版都要和它对比，确认"优化没有算错"。
//
//  问题：C(M x N) = A(M x K) * B(K x N)，全部行主序。
//
//  编译：
//    nvcc -O3 -arch=native v1_naive.cu -o sgemm_v1
//  运行：
//    ./sgemm_v1            # 默认 1024^3
//    ./sgemm_v1 2048 2048 2048
// ============================================================

#include "sgemm_common.h"

// ------------------------------------------------------------
// 每个线程算 C 的一个元素。
// 慢在哪：算 C[i][j] 需要读 A 的第 i 行（K 个 float）和 B 的第 j 列（K 个 float），
// 也就是说一次浮点运算要配两次全局内存读取。
// 全局内存延迟几百个时钟周期，这个比例下 SM 绝大部分时间在等数据。
// ------------------------------------------------------------
__global__ void sgemmNaive(int M, int N, int K,
                           const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 网格比数据大时线程会越界，必须挡住
    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        // A[row][k] 和 B[k][col]：同一个 warp 内 col 连续，
        // 所以 B 的读取是合并的，A 的读取是"广播"（所有线程同地址）。
        // 但这两次读取的比例仍然是 2:1，远高于硬件能承受的水平。
        acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    }
    C[(size_t)row * N + col] = acc;
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    sgemm::printBandwidthReference();
    printf("\n=== v1 朴素版：C(%d x %d) = A(%d x %d) * B(%d x %d) ===\n", M, N, M, K, K, N);

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

    // 16x16 的线程块：每个 warp 覆盖 2 行 x 16 列，B 的访存勉强算合并
    const dim3 block(16, 16);
    const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    auto launch = [&]() {
        sgemmNaive<<<grid, block>>>(M, N, K, dA, dB, dC);
        CUDA_CHECK_KERNEL();
    };

    // ---------- 正确性优先 ----------
    launch();
    CUDA_CHECK_SYNC();
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    printf("正在计算 CPU 参考结果（可能稍慢，请稍等）...\n");
    sgemm::cpuSgemm(M, N, K, hA.data(), hB.data(), hRef.data());
    const bool ok = sgemm::verify(hRef, hC);
    printf("正确性: %s\n", ok ? "通过 ✔" : "失败 ✘");

    // ---------- 性能 ----------
    const sgemm::Result r = sgemm::benchmark(launch, M, N, K);
    sgemm::printTableHeader();
    sgemm::printRow("v1 朴素", r);

    printf("\n结论：v1 的 GFLOPS 远低于硬件峰值，瓶颈是访存/计算比太高。\n");
    printf("      每做 1 次乘加（2 FLOP）就要读 8 字节全局内存。\n");
    printf("      下一步（v2）的思路：让一个线程多算几个输出元素，\n");
    printf("      把读进来的数据在寄存器里重复利用。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
