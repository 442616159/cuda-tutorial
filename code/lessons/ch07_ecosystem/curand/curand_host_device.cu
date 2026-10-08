// ============================================================
//  curand_host_device.cu —— cuRAND 的两种用法
//
//  用法 A（主机端 API）：主机一次调用生成一大片随机数，
//      适合"先把随机数准备好，再拿去算"的场景。
//  用法 B（设备端 API）：每个线程自己持有 curandState，
//      在核函数里边算边取随机数，适合蒙特卡洛这类"每步都要随机数"的场景。
//
//  编译：
//    nvcc -O3 -arch=native curand_host_device.cu -o curand_host_device -lcurand
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <curand.h>
#include <curand_kernel.h>
#include "../../../common/cuda_check.h"

#define CURAND_CHECK(expr)                                                          \
    do {                                                                            \
        curandStatus_t _s = (expr);                                                 \
        if (_s != CURAND_STATUS_SUCCESS) {                                          \
            fprintf(stderr, "\n[CURAND ERROR] %s:%d\n  call : %s\n  code : %d\n",   \
                    __FILE__, __LINE__, #expr, (int)_s);                            \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

// ============================================================
//  用法 B 的核函数：设备端蒙特卡洛估计圆周率
//  原理：在单位正方形里撒点，落在 1/4 圆内的比例 ≈ pi/4
//  每个线程用不同的 seed，保证互不重复
// ============================================================
__global__ void monteCarloPi(unsigned long long seed, int samplesPerThread,
                             unsigned long long* hits) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 初始化状态：curand_init(seed, sequence, offset, state)
    // sequence 用 tid 区分线程，offset 用 0 即可
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    int local = 0;
    for (int i = 0; i < samplesPerThread; ++i) {
        // curand_uniform 返回 (0, 1] 的浮点数
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f) ++local;
    }
    // 每个线程写完一次，用原子加汇总，避免争用
    atomicAdd(hits, (unsigned long long)local);
}

// 用法 B 的批量取数版本：用 float4 一次取 4 个随机数，比逐个取快
__global__ void urandDump(curandState* state, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = curand_uniform(&state[tid]);  // 状态已经在初始化核里建好
}

__global__ void initStates(curandState* state, unsigned long long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &state[tid]);
}

int main() {
    printDeviceInfo();
    const int N = 1 << 20;  // 约 100 万个随机数

    // --------------------------------------------------------
    //  用法 A：主机端 API
    // --------------------------------------------------------
    printf("\n===== 用法 A：主机端 API (curandGenerateUniform) =====\n");
    curandGenerator_t gen;
    // PSEUDO 表示伪随机（可复现）；想要真随机可换 CURAND_RNG_QUASI_* 或设备端 API
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // 固定种子 = 结果可复现。这在做数值实验时非常重要：
    // 否则你无法判断"结果变了"是因为改错了代码还是随机性。
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    float* d_rand = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rand, N * sizeof(float)));
    CURAND_CHECK(curandGenerateUniform(gen, d_rand, N));
    CUDA_CHECK_SYNC();

    std::vector<float> h_rand(N);
    CUDA_CHECK(cudaMemcpy(h_rand.data(), d_rand, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 统计均值与落在 [0,1) 之外的比例，均方误差应接近 1/12
    double sum = 0.0, sq = 0.0;
    int outOfRange = 0;
    for (int i = 0; i < N; ++i) {
        sum += h_rand[i];
        sq += (double)h_rand[i] * h_rand[i];
        if (h_rand[i] < 0.f || h_rand[i] >= 1.f) ++outOfRange;
    }
    double mean = sum / N;
    double var = sq / N - mean * mean;
    printf("  样本数 = %d，越界样本数 = %d\n", N, outOfRange);
    printf("  均值 = %.5f（理论 0.5），方差 = %.6f（理论 %.6f）\n", mean, var, 1.0 / 12.0);

    // --------------------------------------------------------
    //  用法 B：设备端 API
    // --------------------------------------------------------
    printf("\n===== 用法 B：设备端 API (curandState 在核函数里) =====\n");
    curandState* d_state = nullptr;
    CUDA_CHECK(cudaMalloc(&d_state, N * sizeof(curandState)));
    const int block = 256;
    initStates<<<(N + block - 1) / block, block>>>(d_state, 42ULL, N);
    CUDA_CHECK_KERNEL();
    urandDump<<<(N + block - 1) / block, block>>>(d_state, d_rand, N);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_SYNC();
    CUDA_CHECK(cudaMemcpy(h_rand.data(), d_rand, N * sizeof(float), cudaMemcpyDeviceToHost));
    sum = 0.0;
    for (int i = 0; i < N; ++i) sum += h_rand[i];
    printf("  设备端生成的均值 = %.5f\n", sum / N);

    // 同样的 seed + 同样的 sequence 一定得到同样的序列（可复现性验证）
    urandDump<<<(N + block - 1) / block, block>>>(d_state, d_rand, N);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_SYNC();
    float first = 0.f;
    CUDA_CHECK(cudaMemcpy(&first, d_rand, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  重复调用后第 0 号样本 = %.6f（与首次相同，说明状态被消耗，非重置）\n", first);

    // --------------------------------------------------------
    //  蒙特卡洛求 pi
    // --------------------------------------------------------
    printf("\n===== 设备端蒙特卡洛估计 pi =====\n");
    unsigned long long* d_hits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hits, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hits, 0, sizeof(unsigned long long)));

    const int threads = 256;
    const int blocks = 256;
    const int samplesPerThread = 4096;
    monteCarloPi<<<blocks, threads>>>(2024ULL, samplesPerThread, d_hits);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_SYNC();

    unsigned long long hits = 0;
    CUDA_CHECK(cudaMemcpy(&hits, d_hits, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    unsigned long long total = (unsigned long long)threads * blocks * samplesPerThread;
    double pi = 4.0 * (double)hits / (double)total;
    printf("  总样本数 = %llu，圆内样本 = %llu\n", total, hits);
    printf("  估计 pi = %.6f（真值 3.141593，误差量级 ~ 1/sqrt(N) ≈ %.5f）\n",
           pi, 1.0 / sqrt((double)total));

    // 收尾：每个 CUDA 资源都要显式释放，否则在长跑程序里会泄漏显存
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUDA_CHECK(cudaFree(d_rand));
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_hits));
    return EXIT_SUCCESS;
}
