// ============================================================================
//  code/lessons/ch05_cg/cg.cu
//
//  迭代求解器入门：用共轭梯度法（Conjugate Gradient, CG）解二维泊松方程
//      -Δu = f   在 [0,1]^2 上，u = 0 在边界上（Dirichlet 边界）
//  离散成 5 点差分格式后得到一个大型稀疏对称正定线性方程组 A x = b。
//
//  这是第 5 章的综合收口：CG 的每一步都由你已经学过的"原语"拼出来
//      SpMV（5 点 stencil，可以用矩阵无关的写法）
//      dot（归约，见 ch05_reduce）
//      axpy / axpby（Map，逐元素）
//      一个 host 端标量控制流（alpha、beta 的计算）
//
//  编译: nvcc -O3 -arch=sm_70 cg.cu -o cg
//  运行: ./cg
// ============================================================================
#include "../../common/cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int   G       = 1024;          // 网格 G x G，未知量 n = G*G = 1M
constexpr int   BLOCK   = 256;
constexpr int   MAXITER = 1500;
constexpr float TOL     = 1e-4f;         // 相对残差收敛阈值

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

// ===========================================================================
// A. 矩阵无关的 SpMV：y = A * x，A 是 G x G 网格上的 5 点差分算子
//    对角占优，边界外一律取 0（对应零边界条件）
//    不存任何非零元，完全靠下标算出来，省下几十 MB 显存和读取带宽。
// ===========================================================================
__global__ void stencil5(const float* __restrict__ x, float* __restrict__ y, int g) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= g || row >= g) return;

    const int i = row * g + col;
    const float c = x[i];
    const float l = (col > 0)     ? x[i - 1] : 0.f;
    const float r = (col < g - 1) ? x[i + 1] : 0.f;
    const float u = (row > 0)     ? x[i - g] : 0.f;
    const float d = (row < g - 1) ? x[i + g] : 0.f;

    // 4*c - 四邻居 = 标准的二阶中心差分（-Δh 乘以 h^2 后的形式）
    y[i] = 4.f * c - (l + r + u + d);
}

// ===========================================================================
// B. 点积：a·b，用网格跨步 + 块内 shuffle 归约 + 一次原子加
//    用 double 的原子加来汇总，避免几百个块的部分和再丢精度
// ===========================================================================
__global__ void dotPartial(const float* __restrict__ a, const float* __restrict__ b,
                           double* __restrict__ out, int n) {
    float sum = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += a[i] * b[i];

    // ---- 块内归约：先 warp 内 shuffle，再跨 warp 共享内存 ----
    __shared__ float s[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) s[wid] = sum;
    __syncthreads();
    const int nwarps = blockDim.x >> 5;
    if (wid == 0) {
        sum = (lane < nwarps) ? s[lane] : 0.f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) atomicAdd(out, (double)sum);
    }
}

// ===========================================================================
// C. axpy：y = a * x + y（CG 里用来更新 x）
// ===========================================================================
__global__ void axpy(float* __restrict__ y, const float* __restrict__ x,
                     float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

// ===========================================================================
// D. axpby：p = r + b * p（CG 里用来更新搜索方向）
// ===========================================================================
__global__ void axpby(float* __restrict__ p, const float* __restrict__ r,
                      float b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i] + b * p[i];
}

int main() {
    printDeviceInfo();

    const int n = G * G;
    const size_t bytes = (size_t)n * sizeof(float);

    // ---- 右端项 b = 1（对应均匀热源），初值 x = 0 ----
    std::vector<float> hb(n, 1.0f), hx(n, 0.0f);

    float *dx = nullptr, *db = nullptr, *dr = nullptr, *dp = nullptr, *dAp = nullptr;
    double* dDot = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dr, bytes));
    CUDA_CHECK(cudaMalloc(&dp, bytes));
    CUDA_CHECK(cudaMalloc(&dAp, bytes));
    CUDA_CHECK(cudaMalloc(&dDot, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 8);
    dim3 grid((G + 31) / 32, (G + 7) / 8);
    const int grid1D = (n + BLOCK - 1) / BLOCK;

    // ---- 计算 b 的 2-范数，用于相对残差判据 ----
    auto dot = [&](const float* a, const float* b) -> double {
        CUDA_CHECK(cudaMemset(dDot, 0, sizeof(double)));
        dotPartial<<<512, BLOCK>>>(a, b, dDot, n);
        CUDA_CHECK_KERNEL();
        double v = 0.0;
        CUDA_CHECK(cudaMemcpy(&v, dDot, sizeof(double), cudaMemcpyDeviceToHost));
        return v;
    };

    const double normB = std::sqrt(dot(db, db));
    printf("\n[问题] %dx%d 二维泊松方程, 未知量 %d, 非零元 %d\n",
           G, G, n, 5 * n - 4 * G);
    printf("[问题] 条件数约 %.3e, CG 的迭代次数大致按 sqrt(cond) 增长\n",
           (double)G * G / 2.0);
    printf("[问题] ||b|| = %.4f\n", normB);

    CudaTimer timer;
    timer.start();

    // =====================================================================
    // CG 主循环（每一步的数学都在注释里）
    //   r = b - A x        ← 初始残差
    //   p = r              ← 初始搜索方向
    //   rsold = r·r
    //   loop:
    //     Ap        = A p
    //     alpha     = rsold / (p·Ap)      ← 沿 p 方向的最优步长
    //     x        += alpha * p
    //     r        -= alpha * Ap
    //     rsnew     = r·r
    //     if sqrt(rsnew)/||b|| < tol: break
    //     p         = r + (rsnew/rsold) * p   ← 让 p 与之前所有方向 A-正交
    //     rsold     = rsnew
    // =====================================================================
    // ---- 初始残差 r = b - A x ----
    // 现有的一维逐元素 kernel 都是 y += a*x 的形式，所以先让 r = b，
    // 再用 a = -1 把 A x 减掉。这样只需要一个 axpy，不必再写新 kernel。
    stencil5<<<grid, block>>>(dx, dAp, G);                            // dAp = A x
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(dr, db, bytes, cudaMemcpyDeviceToDevice));  // r = b
    axpy<<<grid1D, BLOCK>>>(dr, dAp, -1.0f, n);                       // r = b - A x
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(dp, dr, bytes, cudaMemcpyDeviceToDevice));  // p = r

    double rsold = dot(dr, dr);
    const double rs0 = rsold;
    int iter = 0;
    for (; iter < MAXITER; ++iter) {
        stencil5<<<grid, block>>>(dp, dAp, G);          // Ap = A p
        const double pAp = dot(dp, dAp);
        const double alpha = rsold / pAp;

        axpy<<<grid1D, BLOCK>>>(dx, dp, (float)alpha, n);      // x += alpha*p
        axpy<<<grid1D, BLOCK>>>(dr, dAp, (float)-alpha, n);    // r -= alpha*Ap

        const double rsnew = dot(dr, dr);
        if (std::sqrt(rsnew / rs0) < TOL) {
            rsold = rsnew;
            ++iter;
            break;
        }
        const double beta = rsnew / rsold;
        axpby<<<grid1D, BLOCK>>>(dp, dr, (float)beta, n);      // p = r + beta*p
        rsold = rsnew;
    }
    const float ms = timer.stop();
    CUDA_CHECK_KERNEL();

    // ---- 独立验证：重新算一遍真实残差 ||b - A x|| / ||b|| ----
    CUDA_CHECK(cudaMemcpy(dr, db, bytes, cudaMemcpyDeviceToDevice));
    stencil5<<<grid, block>>>(dx, dAp, G);
    CUDA_CHECK_KERNEL();
    axpy<<<grid1D, BLOCK>>>(dr, dAp, -1.0f, n);
    const double trueRes = std::sqrt(dot(dr, dr)) / normB;

    // ---- 和解析解对比：-Δu = 1 在单位正方形上的解在中心处约为 0.0736 ----
    std::vector<float> hxout(n);
    CUDA_CHECK(cudaMemcpy(hxout.data(), dx, bytes, cudaMemcpyDeviceToHost));
    const float center = hxout[(G / 2) * G + (G / 2)];
    const float cornerNear = hxout[G + 1];  // 靠近角点，应该接近 0

    printf("\n[结果] 收敛于第 %d 次迭代（上限 %d）\n", iter, MAXITER);
    printf("[结果] 迭代过程中判据 sqrt(r·r)/sqrt(r0·r0) = %.3e\n", std::sqrt(rsold / rs0));
    printf("[结果] 独立重算的真实相对残差 ||b-Ax||/||b|| = %.3e\n", trueRes);
    printf("[结果] 解在网格中心 u(0.5,0.5) = %.5f  （解析解约 0.0736）\n", center);
    printf("[结果] 解在近角点 u(h,h)     = %.5f  （应该接近 0）\n", cornerNear);
    printf("\n[性能] 总耗时 %.2f ms，相当于每次迭代 %.4f ms\n",
           ms, ms / (float)(iter > 0 ? iter : 1));
    printf("[性能] 每次迭代做 1 次 SpMV + 2 次 dot + 3 次 axpy/axpby，"
           "共约 %d 个 kernel\n", iter * 6 + 3);
    printf("\n提示：在 RTX 3060 上求解 1024x1024 网格的泊松方程，"
           "约 500~700 次迭代、几十毫秒量级，\n");
    printf("      你的机器可能不同。性能瓶颈几乎全在每次迭代的访存上：\n");
    printf("      * 每个 kernel 都要把 1M 个 float（4 MB）完整读写一遍；\n");
    printf("      * 三次 dot/axpy 完全可以融合成一个 kernel 减少往返；\n");
    printf("      * 进一步的加速手段是预条件（Jacobi / 多重网格预条件 CG），"
           "能显著减少迭代次数。\n");

    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dr)); CUDA_CHECK(cudaFree(dp));
    CUDA_CHECK(cudaFree(dAp)); CUDA_CHECK(cudaFree(dDot));
    return 0;
}
