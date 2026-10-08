"""
numba_saxpy.py —— 用 Numba 的 @cuda.jit 在 Python 里写 CUDA kernel

Numba 的定位：在 Python 里直接写 CUDA C 风格的核函数，
不用装 nvcc、不用写 C++ 工程，改一行立刻能跑。

安装：
    pip install numba
运行：
    python numba_saxpy.py

它和 CuPy 的区别：
  * CuPy  : 提供"GPU 版的 NumPy 函数"，你调用现成的算子
  * Numba : 提供"写 kernel 的能力"，你自己定义每个线程干什么
  两者可以混用（共享显存时用 cuda.as_cuda_array / cupy.asarray 互转）。
"""

import math
import time

import numpy as np
from numba import cuda


@cuda.jit
def saxpy_kernel(x, y, a, n):
    """y[i] = a * x[i] + y[i]

    这个函数体就是标准的 CUDA C 核函数，唯一区别是没有类型声明。
    Numba 会在第一次调用时把它编译成机器码（JIT），
    所以**第一次调用会慢**，测性能一定要先预热。
    """
    i = cuda.grid(1)          # 等价于 blockIdx.x * blockDim.x + threadIdx.x
    if i < n:                 # 边界检查：线程数通常多于元素数
        y[i] = a * x[i] + y[i]


@cuda.jit
def reduce_sum_kernel(x, out):
    """块内归约求和：演示共享内存 + 同步在 Numba 里怎么写"""
    # 和 CUDA C 里一样：动态共享内存需要在启动时指定大小
    sdata = cuda.shared.array(shape=0, dtype=np.float32)
    tid = cuda.threadIdx.x
    # cuda.grid(1) 会自动按 grid 步长循环吗？不会，需要自己写
    acc = 0.0
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < x.shape[0]:
        acc += x[i]
        i += stride
    sdata[tid] = acc
    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s //= 2
    if tid == 0:
        out[cuda.blockIdx.x] = sdata[0]


def kernel_timer(fn, repeat=50, warmup=5):
    """Numba 的计时套路：必须用 cuda.synchronize()

    因为 kernel 是异步启动的，不 sync 的话你测的是"CPU 下发命令的时间"，
    会得到 0.00x 毫秒这种假数据。
    """
    for _ in range(warmup):
        fn()
    cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


def main():
    print("=== Numba CUDA 环境 ===")
    if not cuda.is_available():
        raise SystemExit("没有检测到可用的 CUDA 设备")
    dev = cuda.get_current_device()
    print("设备名称 :", dev.name)
    print("计算能力 :", dev.compute_capability)
    print("SM 数量  :", dev.MULTIPROCESSOR_COUNT)

    # ---------- SAXPY ----------
    n = 1 << 22
    rng = np.random.default_rng(0)
    x = rng.random(n, dtype=np.float32)
    y = rng.random(n, dtype=np.float32)
    a = np.float32(2.5)

    # to_device / copy_to_device：显式把数据搬到显存。
    # 不显式搬的话 numba 会自动做，但你无法控制何时搬、搬几次。
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)

    threads = 256
    blocks = (n + threads - 1) // threads

    # 第一次调用触发 JIT 编译，会明显慢
    t0 = time.perf_counter()
    saxpy_kernel[blocks, threads](d_x, d_y, a, n)
    cuda.synchronize()
    print(f"\n首次调用（含 JIT 编译）耗时 {time.perf_counter() - t0:.3f} s")

    ms = kernel_timer(lambda: saxpy_kernel[blocks, threads](d_x, d_y, a, n)) * 1e3
    gpu_result = d_y.copy_to_host()
    expect = a * x + y
    ok = np.allclose(gpu_result, expect, rtol=1e-5, atol=1e-6)
    print(f"SAXPY: {ms:.4f} ms，正确性: {'通过 ✔' if ok else '失败 ✘'}")

    # 有效带宽：saxpy 读写共 3 * 4 字节/元素
    bytes_moved = 3 * n * 4
    print(f"  有效带宽 = {bytes_moved / (ms * 1e-3) / 1e9:.1f} GB/s "
          f"（SAXPY 是纯带宽受限，理论上限就是你的 HBM 带宽）")

    # ---------- 归约 ----------
    blocks_r = 512
    d_out = cuda.device_array(blocks_r, dtype=np.float32)
    # 共享内存大小必须写成动态共享内存的参数
    reduce_sum_kernel[blocks_r, threads, 0, threads * 4](d_x, d_out)
    cuda.synchronize()
    partial = d_out.copy_to_host()
    gpu_sum = float(partial.sum())
    cpu_sum = float(x.astype(np.float64).sum())
    print(f"\n归约求和: GPU = {gpu_sum:.4f}，CPU(double) = {cpu_sum:.4f}，"
          f"相对误差 = {abs(gpu_sum - cpu_sum) / cpu_sum:.2e}")

    # ---------- 与 CuPy 互操作 ----------
    print("\n=== 与其它库互操作 ===")
    try:
        import cupy as cp
        from numba import cuda as nbcuda
        # 关键：通过 __cuda_array_interface__ 协议零拷贝共享显存
        gpu_arr = cp.arange(8, dtype=cp.float32)
        nb_view = nbcuda.as_cuda_array(gpu_arr)
        back = cp.asarray(nb_view)
        print("  CuPy <-> Numba 零拷贝共享显存:", cp.asnumpy(back))
    except ImportError:
        print("  未安装 CuPy，跳过互操作演示")

    print("\n提示：Numba 适合快速原型；如果 kernel 成为性能瓶颈，")
    print("      再把它翻译成 .cu 用 nvcc 编译，通常还能再快一些。")
    print(f"（参考：math 模块也可用，例如 sqrt(2) = {math.sqrt(2):.6f}）")


if __name__ == "__main__":
    main()
