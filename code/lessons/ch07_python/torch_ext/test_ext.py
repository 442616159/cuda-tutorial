"""
test_ext.py —— 构建完成后，验证自定义 CUDA 扩展

运行：
    python setup.py build_ext --inplace
    python test_ext.py
"""

import time

import torch

try:
    import vector_add_ext
except ImportError:
    raise SystemExit(
        "扩展未构建。请先执行：\n"
        "    python setup.py build_ext --inplace\n"
        "或  python setup.py install"
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 CUDA 设备")

    print("=== PyTorch CUDA 扩展测试 ===")
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 版本  :", torch.version.cuda)
    print("设备       :", torch.cuda.get_device_name(0))

    # ---------- 1. 正确性 ----------
    n = 1 << 22
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")

    c_ext = vector_add_ext.vector_add(a, b)
    c_ref = a + b                       # PyTorch 自带实现作为参考

    max_err = (c_ext - c_ref).abs().max().item()
    print(f"\n[正确性] 与 torch 加法最大误差 = {max_err:.3e}"
          f"  ->  {'通过 ✔' if max_err < 1e-5 else '失败 ✘'}")

    # ---------- 2. 非连续输入（这是最容易出 bug 的场景） ----------
    a2 = torch.randn(n * 2, device="cuda")[::2]      # 步长视图，不连续
    b2 = torch.randn(n * 2, device="cuda")[::2]
    assert not a2.is_contiguous()
    c2 = vector_add_ext.vector_add(a2, b2)
    err2 = (c2 - (a2 + b2)).abs().max().item()
    print(f"[非连续输入] 最大误差 = {err2:.3e}"
          f"  ->  {'通过 ✔' if err2 < 1e-5 else '失败 ✘'}")

    # ---------- 3. 与 TorchScript（静态图）结合 ----------
    # 自定义算子可以直接被 torch.jit.script 调用，便于部署
    scripted = torch.jit.script(lambda x, y: vector_add_ext.vector_add(x, y))
    c3 = scripted(a, b)
    print("[TorchScript] 调用成功:", torch.allclose(c3, c_ref, atol=1e-5))

    # ---------- 4. 性能对比 ----------
    def bench(fn, repeat=100, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()     # 必须同步，否则测的是下发时间
        return (time.perf_counter() - t0) / repeat

    t_ext = bench(lambda: vector_add_ext.vector_add(a, b))
    t_torch = bench(lambda: a + b)
    print(f"\n[性能] 自定义扩展 {t_ext * 1e6:.1f} us    "
          f"torch 原生 {t_torch * 1e6:.1f} us")
    print(f"  有效带宽: 扩展 {3 * n * 4 / t_ext / 1e9:.1f} GB/s，"
          f"torch {3 * n * 4 / t_torch / 1e9:.1f} GB/s")
    print("  说明: 纯访存算子上自己写的 kernel 通常和 PyTorch 打平，"
          "因为都受限于显存带宽，而不是实现水平。")

    # ---------- 5. 错误处理演示 ----------
    print("\n[错误处理] 传入 CPU 张量时应当抛出异常：")
    try:
        vector_add_ext.vector_add(torch.randn(4), torch.randn(4))
        print("  未抛出异常 ✘（说明 TORCH_CHECK 没生效）")
    except RuntimeError as e:
        print("  已捕获异常 ✔:", str(e).splitlines()[0])

    print("\n在 GPU 上跑 nsys/ncu 时，记得给扩展源码加 -lineinfo，")
    print("否则性能报告里只会显示汇编，看不到对应的源码行。")


if __name__ == "__main__":
    main()
