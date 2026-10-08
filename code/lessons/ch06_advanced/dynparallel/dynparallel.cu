// ============================================================================
//  code/lessons/ch06_advanced/dynparallel/dynparallel.cu
//
//  动态并行（Dynamic Parallelism）：在设备端启动 kernel
//
//  ⚠ 四个条件，缺一不可：
//     1) 需要 Compute Capability >= 3.5
//     2) 必须用 -rdc=true（可重定位设备代码）编译，否则设备端 <<<>>> 链接不上
//     3) 目标架构必须 **低于 sm_90**：CUDA 13 起，设备端的 cudaDeviceSynchronize()
//        只在 __CUDA_ARCH__ < 900 时才提供（见 cuda_device_runtime_api.h 里的
//        #if (__CUDA_ARCH__ < 900) 保护）。在 sm_90 / sm_100 / sm_120 上会报
//        "calling a __host__ function from a __global__ function is not allowed"。
//        Hopper / Blackwell 上请改用 CDP2（cudaGridDependencySynchronize 等）。
//     4) **Windows x64 上还必须定义 CUDA_FORCE_CDP1_IF_SUPPORTED**：
//        那段声明的完整条件是
//          #if (__CUDA_ARCH__ < 900) && (defined(CUDA_FORCE_CDP1_IF_SUPPORTED) \
//                                        || (defined(_WIN32) && !defined(_WIN64)))
//        Windows x64 同时定义了 _WIN32 和 _WIN64，所以只有显式定义这个宏，
//        才拿得到设备端的声明。
//
//  编译（Windows x64）:
//    nvcc -O3 -arch=sm_80 -rdc=true -DCUDA_FORCE_CDP1_IF_SUPPORTED dynparallel.cu -o dynparallel
//    ⚠ 不要用 -arch=native：RTX 40/50 系会解析成 sm_89 / sm_120，其中 sm_120 会编译失败。
//  编译（Linux）:
//    nvcc -O3 -arch=sm_80 -rdc=true dynparallel.cu -o dynparallel
//  运行: ./dynparallel
//
//  本文件演示三件事：
//     A. 最基础的设备端启动 + 设备端同步
//     B. 用递归把大数组求和拆成子问题（自适应细分）
//     C. 设备端 malloc/free（同样需要 -rdc=true）
// ============================================================================
#include "../../../common/cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// 设备端错误检查宏
//   注意：不能在设备代码里直接用 cuda_check.h 的 CUDA_CHECK，
//   它内部会调用 fprintf/exit，语义上更适合 host。
//   设备端我们打印到 kernel 的 printf 缓冲区即可。
// ---------------------------------------------------------------------------
#define DEV_CHECK(msg)                                                             \
    do {                                                                           \
        cudaError_t _e = cudaGetLastError();                                       \
        if (_e != cudaSuccess)                                                     \
            printf("[device %s] %s\n", (msg), cudaGetErrorString(_e));             \
    } while (0)

// ===========================================================================
// A. 最基础的动态并行
//    childKernel 往一个全局计数器里累加自己的线程编号
//    parentKernel 在设备端启动 childKernel，并在设备端等它跑完
// ===========================================================================
__global__ void childKernel(int* counter, int tag) {
    // 每个子 kernel 只让 thread 0 干一次活，便于观察启动次数
    if (threadIdx.x == 0) {
        atomicAdd(counter, tag);
    }
}

__global__ void parentKernel(int* counter) {
    // 只有 block 0 的 thread 0 负责发起子 kernel，
    // 否则每个线程都启动一次会指数级放大
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int t = 1; t <= 4; ++t) {
            childKernel<<<1, 32>>>(counter, t);
            DEV_CHECK("childKernel launch");
        }
        // 设备端必须自己同步，否则拿到的 counter 可能是子 kernel 还没写完的旧值
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) printf("[parent] sync failed: %s\n", cudaGetErrorString(e));
    }
    __syncthreads();  // 让块内其它线程跟着一起等
}

// ===========================================================================
// B. 递归归约：用设备端启动把大数组一分为二
//    这是动态并行最典型的用法 —— 分治算法的深度不固定，
//    在 host 端很难预先展开成一串 kernel 启动。
// ===========================================================================
constexpr int  LEAF_SIZE = 1024;  // 递归到 <= 1024 个元素就不再往下拆
constexpr int  LEAF_TPB  = 256;

// 叶子节点：一个小块把 n <= 1024 个元素求和，结果写进 *out
__global__ void leafSumKernel(const float* in, float* out, int n) {
    __shared__ float s[LEAF_TPB];
    float v = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += in[i];
    s[threadIdx.x] = v;
    __syncthreads();
    for (int stride = LEAF_TPB / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = s[0];
}

// 递归节点：把数组拆成两半，各启动一个子 kernel（就是递归调用自己）
__global__ void recursiveSumKernel(const float* in, float* out, int n) {
    if (n <= LEAF_SIZE) {
        // 到了叶子规模，交给轻量的叶子 kernel（避免继续递归带来大量启动开销）
        if (threadIdx.x == 0) {
            leafSumKernel<<<1, LEAF_TPB>>>(in, out, n);
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) printf("[recursion] leaf launch failed: %s\n",
                                         cudaGetErrorString(e));
        }
        return;
    }

    const int half = n / 2;
    const int rest = n - half;

    // out[0] 放左半和，out[1] 放右半和，最后相加写回 out[0]
    if (threadIdx.x == 0) {
        recursiveSumKernel<<<1, 128>>>(in, out + 1, half);
        recursiveSumKernel<<<1, 128>>>(in + half, out + 2, rest);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("[recursion] child launch failed: %s\n", cudaGetErrorString(e));
            return;
        }
        out[0] = out[1] + out[2];
    }
}

// ===========================================================================
// C. 设备端 malloc/free（同样需要 -rdc=true）
//    子 kernel 可以动态分配显存，用来做"每个子问题需要可变大小暂存区"的场景。
//    注意：设备端 cudaMalloc 有配额限制（默认约 8 MB），大量分配要小心。
// ===========================================================================
__global__ void deviceMallocDemo(int* result) {
    if (threadIdx.x == 0) {
        float* buf = nullptr;
        cudaError_t e = cudaMalloc((void**)&buf, 1024 * sizeof(float));
        if (e != cudaSuccess) {
            printf("[deviceMalloc] cudaMalloc failed: %s\n", cudaGetErrorString(e));
            return;
        }
        for (int i = 0; i < 1024; ++i) buf[i] = (float)i;
        float s = 0.f;
        for (int i = 0; i < 1024; ++i) s += buf[i];
        cudaFree(buf);
        *result = (int)s;  // 0+1+...+1023 = 523776
    }
}

int main() {
    printDeviceInfo();

    const int ccMajor = [] {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
        return p.major;
    }();
    if (ccMajor < 3) {
        printf("\n[跳过] 本机 CC = %d.x，动态并行需要 CC >= 3.5。\n", ccMajor);
        return 0;
    }

    // =====================================================================
    // A. 基础演示
    // =====================================================================
    {
        int* d_counter = nullptr;
        CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

        parentKernel<<<1, 32>>>(d_counter);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        int got = 0;
        CUDA_CHECK(cudaMemcpy(&got, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
        // 1+2+3+4 = 10
        printf("\n[A] 动态并行基础：父 kernel 启动了 4 个子 kernel，"
               "counter = %d  %s\n", got, got == 10 ? "[OK]" : "[FAIL]");
        CUDA_CHECK(cudaFree(d_counter));
    }

    // =====================================================================
    // B. 递归归约
    // =====================================================================
    {
        constexpr int N = 1 << 20;  // 1M
        std::vector<float> h(N);
        for (int i = 0; i < N; ++i) h[i] = 1.0f / (float)((i % 61) + 1);
        double ref = 0.0;
        for (int i = 0; i < N; ++i) ref += (double)h[i];

        float* d_in = nullptr;
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        // 递归过程需要额外的输出槽位，多留一些余量
        CUDA_CHECK(cudaMalloc(&d_out, 4096 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, 4096 * sizeof(float)));

        recursiveSumKernel<<<1, 128>>>(d_in, d_out, N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        float got = 0.f;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        const double rel = std::fabs((double)got - ref) / std::fabs(ref);
        printf("[B] 递归归约(1M 个元素)：和 = %.3f, CPU 参考 = %.3f, 相对误差 = %.2e  %s\n",
               got, ref, rel, rel < 1e-3 ? "[OK]" : "[FAIL]");
        printf("    递归深度 = log2(%d / %d) = %d 层，每层的每个节点都会启动 2 个子 kernel\n",
               N, LEAF_SIZE, 10);
        printf("    启动次数是 O(N/LEAF_SIZE) 量级 —— 这就是动态并行最大的成本来源。\n");

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    // =====================================================================
    // C. 设备端 malloc
    // =====================================================================
    {
        int* d_result = nullptr;
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(int)));

        deviceMallocDemo<<<1, 32>>>(d_result);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        int got = 0;
        CUDA_CHECK(cudaMemcpy(&got, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[C] 设备端 malloc/free：0+1+...+1023 = %d  %s\n",
               got, got == 523776 ? "[OK]" : "[FAIL]");
        CUDA_CHECK(cudaFree(d_result));
    }

    printf("\n什么时候值得用动态并行？\n");
    printf("  ✔ 递归/分治深度在编译期未知（BVH 构建、自适应网格细分、快速多极子）；\n");
    printf("  ✔ 想省掉 host 与 device 之间的往返同步（例如自适应步长的迭代）；\n");
    printf("  ✔ 子问题规模差异极大，只有运行时才知道要不要继续拆。\n");
    printf("什么时候千万别用？\n");
    printf("  ✘ 形状规则的负载 —— 一个普通的 grid-stride kernel 快得多；\n");
    printf("  ✘ 深度固定的分治 —— 用多次 host 启动或 CUDA Graphs 更划算。\n");
    printf("\n实测经验（你的机器可能不同）：\n");
    printf("  每次设备端 kernel 启动的开销大约是 host 端启动的 2~5 倍；\n");
    printf("  递归归约通常比单 kernel 分块归约慢 3~10 倍，它的价值在于表达能力，\n");
    printf("  不在于绝对性能。用 nsys 看时间线时，你会看到大量细碎的 kernel。\n");

    return 0;
}
