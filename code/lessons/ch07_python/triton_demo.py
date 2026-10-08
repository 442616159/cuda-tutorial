"""
triton_demo.py —— Triton：用 Python 写"块级"kernel

Triton 的定位（了解即可，不要求掌握）：
  * 你写的是"一个程序处理一个数据块（block）"的逻辑，
    而不是"一个线程处理一个元素"的逻辑；
  * 循环、掩码、向量化交给编译器自动做；
  * 在矩阵乘、LayerNorm、Softmax 这类算子上，
    手写 Triton 常常能达到接近 cuBLAS/cuDNN 的性能。

和 CUDA C 的心智模型对照：
    CUDA C  : 线程 -> 元素       （你需要自己管 threadIdx、共享内存、同步）
    Triton  : 程序 -> 数据块      （你只管块内的向量化运算）

安装：
    pip install triton
运行：
    python triton_demo.py        # 需要 NVIDIA GPU + Linux/Windows
"""

import time

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as e:
    raise SystemExit(f"缺少依赖: {e}. 请执行 pip install torch triton")


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    """向量加法：注意这里没有 threadIdx，只有"我负责哪一段"。

    pid  = 这是第几个程序实例（对应 CUDA 里的 block）
    offs = 这个实例负责的一整段下标（一个向量，长度 BLOCK_SIZE）
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements          # 掩码等价于 CUDA 里的 if (i < n)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    """按行做 softmax —— 这是 Triton 最擅长的模式之一：
    一行数据一次读进寄存器/共享内存，全程不做多次全局访存。
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = in_ptr + row_idx * in_row_stride
    row = tl.load(row_start + col_offsets, mask=mask, other=-float("inf"))
    # 数值稳定性：先减去最大值，避免 exp 溢出（和 CPU 上的写法完全一样）
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out_ptr + row_idx * in_row_stride + col_offsets,
             numerator / denominator, mask=mask)


def bench(fn, repeat=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


def main():
    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 CUDA 设备")
    print("=== Triton 演示 ===")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Triton 版本:", triton.__version__)

    # ---------- 1. 向量加法 ----------
    n = 1 << 22
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)   # cdiv = 向上取整除法
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    print("\n[add] 与 torch 结果一致:",
          torch.allclose(out, x + y, atol=1e-5))

    t_triton = bench(lambda: add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE))
    t_torch = bench(lambda: x + y)
    print(f"  Triton: {t_triton * 1e3:.3f} ms   PyTorch: {t_torch * 1e3:.3f} ms")
    print("  （纯访存算子两者通常很接近，因为都撞到带宽上限）")

    # ---------- 2. 按行 softmax ----------
    rows, cols = 4096, 1024
    mat = torch.randn(rows, cols, device="cuda")
    out2 = torch.empty_like(mat)
    softmax_grid = (rows,)
    softmax_kernel[softmax_grid](out2, mat, mat.stride(0), cols,
                                 BLOCK_SIZE=triton.next_power_of_2(cols))
    torch.cuda.synchronize()
    ref = torch.softmax(mat, dim=1)
    print("\n[softmax] 与 torch.softmax 最大误差:",
          f"{(out2 - ref).abs().max().item():.3e}")

    t_tri = bench(lambda: softmax_kernel[softmax_grid](
        out2, mat, mat.stride(0), cols, BLOCK_SIZE=triton.next_power_of_2(cols)))
    t_ref = bench(lambda: torch.softmax(mat, dim=1))
    print(f"  Triton: {t_tri * 1e3:.3f} ms   PyTorch: {t_ref * 1e3:.3f} ms")

    # ---------- 3. 和 CUDA C 的关系 ----------
    print("\n结论：Triton 把「线程映射/共享内存/同步」这三件最费脑的事"
          "交给了编译器，")
    print("      代价是你失去了对硬件细节的完全控制。")
    print("      先学 CUDA C 打好基础，再决定要不要用 Triton 提效率。")


if __name__ == "__main__":
    main()
