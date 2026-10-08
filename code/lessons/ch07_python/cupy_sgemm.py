"""
cupy_sgemm.py —— 用 CuPy 在 Python 里直接调 GPU

CuPy 的定位：把 NumPy 的 API 原封不动搬到 GPU 上。
凡是 `numpy.xxx`，基本都有 `cupy.xxx` 对应，学习成本几乎为零。

安装：
    pip install cupy-cuda12x        # 按自己的 CUDA 版本选包名
    # 或从源码编译：pip install cupy

运行：
    python cupy_sgemm.py

注意：CuPy 里的矩阵乘法最终走的就是 cuBLAS，
所以它和你在 C++ 里手写的 kernel 比，一定是"库赢"。
理解这一点，你就明白了第 7 章最核心的判断：
**通用算子优先用库，只有库覆盖不到的地方才自己写 kernel。**
"""

import time

try:
    import cupy as cp
except ImportError:
    raise SystemExit("未安装 CuPy。请执行: pip install cupy-cuda12x")

import numpy as np


def bench(fn, repeat=10, warmup=3):
    """带预热和多次取中位数的基准测试。

    为什么不能只测一次？
      1) GPU 首次调用要做上下文初始化、加载 kernel 模块，特别慢；
      2) 单次测量受系统噪声影响大。
    所以标准做法是：预热若干次 -> 跑多次 -> 取中位数（比均值更抗离群点）。
    """
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()  # 关键：异步队列必须先同步，否则测的是"下发时间"

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()  # 每次测量都要等 GPU 真正算完
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]  # 中位数


def main():
    print("=== CuPy 基础 ===")
    print("CuPy 版本:", cp.__version__)
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    print("设备名称 :", props["name"].decode())
    print("计算能力 :", f"{props['major']}.{props['minor']}")
    print("显存总量 :", f"{dev.mem_info[1] / 1024**3:.2f} GB")

    # ---------- 1. 数组创建与搬运 ----------
    a_cpu = np.random.rand(4, 4).astype(np.float32)
    a_gpu = cp.asarray(a_cpu)          # H2D
    b_gpu = cp.random.rand(4, 4, dtype=np.float32)  # 直接在 GPU 上生成
    c_gpu = a_gpu + b_gpu              # 逐元素运算，全程不落 CPU
    print("\n[逐元素] a+b 结果类型:", type(c_gpu).__name__)
    np.testing.assert_allclose(cp.asnumpy(c_gpu), a_cpu + cp.asnumpy(b_gpu), rtol=1e-5)
    print("  与 NumPy 结果一致 ✔")

    # ---------- 2. GEMM：CuPy 底层就是 cuBLAS ----------
    print("\n=== GEMM 性能对比（CuPy 走 cuBLAS） ===")
    print(f"{'尺寸':>12} {'CuPy(GPU) ms':>14} {'NumPy(CPU) ms':>15} {'GFLOPS(GPU)':>12}")
    for n in (512, 1024, 2048):
        A = cp.random.rand(n, n, dtype=cp.float32)
        B = cp.random.rand(n, n, dtype=cp.float32)
        flops = 2.0 * n ** 3

        t_gpu = bench(lambda: cp.matmul(A, B), repeat=10, warmup=2)
        # CPU 侧只用小尺寸，避免等太久
        An, Bn = cp.asnumpy(A), cp.asnumpy(B)
        t0 = time.perf_counter()
        np.matmul(An, Bn)
        t_cpu = time.perf_counter() - t0

        print(f"{n:>12} {t_gpu * 1e3:>14.3f} {t_cpu * 1e3:>15.3f} "
              f"{flops / t_gpu / 1e9:>12.1f}")
    print("（你的机器数据会不同；重点是看 GPU 随规模增长后的相对优势）")

    # ---------- 3. 用 RawKernel 写自定义 kernel ----------
    # CuPy 允许直接写 CUDA C 源码，适合"库没有、又不想开 C++ 工程"的场景
    print("\n=== 用 cupy.RawKernel 写一个自定义 kernel ===")
    saxpy_src = r"""
    extern "C" __global__
    void saxpy_kernel(const float* __restrict__ x, float* __restrict__ y,
                      float a, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a * x[i] + y[i];   // 边界检查不能省
    }
    """
    saxpy = cp.RawKernel(saxpy_src, "saxpy_kernel")
    n = 1 << 22
    x = cp.random.rand(n, dtype=cp.float32)
    y = cp.random.rand(n, dtype=cp.float32)
    a_val = 2.5
    # 先在 CPU 上用 NumPy 算一份参考值（注意：这一步只是为了验证，不计入性能）
    y_ref = a_val * cp.asnumpy(x) + cp.asnumpy(y)
    block = 256
    grid = (n + block - 1) // block
    saxpy((grid,), (block,), (x, y, np.float32(a_val), np.int32(n)))
    cp.testing.assert_allclose(cp.asnumpy(y), y_ref, rtol=1e-5, atol=1e-5)
    print("  RawKernel SAXPY 结果与 NumPy 一致 ✔")

    # ---------- 4. 什么时候不该用 CuPy ----------
    print("\n=== 一个重要的反例：小数组上的 GPU 反而更慢 ===")
    small = cp.random.rand(16, dtype=cp.float32)
    t_gpu_small = bench(lambda: small * 2.0, repeat=200, warmup=20)
    sn = cp.asnumpy(small)
    t0 = time.perf_counter()
    for _ in range(200):
        sn * 2.0
    t_cpu_small = (time.perf_counter() - t0) / 200
    print(f"  16 个元素的乘法：GPU 每次 ~{t_gpu_small * 1e6:.1f} us，"
          f"CPU 每次 ~{t_cpu_small * 1e6:.1f} us")
    print("  结论：kernel 启动开销（微秒级）远大于计算本身，小任务别上 GPU。")


if __name__ == "__main__":
    main()
