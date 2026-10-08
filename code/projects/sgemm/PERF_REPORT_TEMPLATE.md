# 性能报告模板

> 复制本文件为 `PERF_REPORT.md` 并逐项填写。
> 报告的价值在于**可复现**和**可解释**，不在于数字漂亮。

## 1. 测试环境

| 项目 | 值 |
|---|---|
| GPU 型号 | （例：NVIDIA GeForce RTX 3060） |
| 计算能力 (CC) | （`nvidia-smi --query-gpu=compute_cap --format=csv`） |
| SM 数量 | |
| 显存 / 位宽 | |
| 理论显存带宽 | GB/s |
| CPU | |
| 驱动版本 | （`nvidia-smi` 左上角） |
| CUDA Toolkit | （`nvcc --version`） |
| 编译命令 | 例：`nvcc -O3 -arch=sm_86 -Xptxas -v v7.cu` |
| 操作系统 | |

> 换任何一项都要重新测，不能把不同环境的数据混在一张表里。

## 2. 正确性验证

| 版本 | 测试形状 | 最大相对误差 | 阈值 | 结论 |
|---|---|---|---|---|
| v1 | 1024³ | | 1e-3 | |
| v2 | 1024³ | | 1e-3 | |
| v3 | 1024³ | | 1e-3 | |
| v4 | 1024³ | | 1e-3 | |
| v5 | 1024³ | | 1e-3 | |
| v6 | 1024³ | | 1e-3 | |
| v7 | 1024³ | | 1e-3 | |
| cuBLAS | 1024³ | | 1e-3 | |
| 边界用例 | 1000x1016x1008 | | 1e-3 | |

**正确性必须排在性能前面**：如果一个版本更快但误差超标，它不算优化成果。

## 3. 性能汇总（测试形状 1024 x 1024 x 1024，重复 20 次取中位数）

| 版本 | 耗时 (ms) | 最小值 (ms) | GFLOPS | 有效带宽 (GB/s) | 相对 v1 加速比 | 相对 cuBLAS |
|---|---|---|---|---|---|---|
| v1 朴素 | | | | | 1.00x | |
| v2 寄存器分块 4x4 | | | | | | |
| v3 共享内存分块 | | | | | | |
| v4 共享+线程分块 | | | | | | |
| v5 float4 向量化 | | | | | | |
| v6 双缓冲流水线 | | | | | | |
| v7 8x8 寄存器分块 | | | | | | |
| cuBLAS | | | | | | 100% |

GFLOPS = 2·M·N·K / (耗时秒 × 1e9)

## 4. 不同形状下的表现（说明结论的适用范围）

| 形状 | v1 | v4 | v7 | cuBLAS |
|---|---|---|---|---|
| 512³ | | | | |
| 1024³ | | | | |
| 2048³ | | | | |
| 4096³ | | | | |
| 1024x1024x64（K 很小） | | | | |

> K 很小的形状能暴露流水线序言/尾声开销和启动开销，是很好的反例测试。

## 5. 逐步优化收益拆解

| 步骤 | 相对上一步提升 | 主要原因 | 证据 |
|---|---|---|---|
| v1 → v2 | | 访存/计算比降低 4 倍 | ncu: Memory Throughput 下降 |
| v2 → v3 | | 全局访存减少 TILE 倍 | ncu: DRAM 读事务减少 |
| v3 → v4 | | 每次共享读服务 4 倍乘加 | ncu: Shared Throughput 下降 |
| v4 → v5 | | 指令数降为 1/4 | ncu: Inst Executed 下降 |
| v5 → v6 | | 全局延迟被计算掩盖 | ncu: Long Scoreboard 占比下降 |
| v6 → v7 | | 数据复用率再提升 | ncu: Compute Throughput 接近 100% |

## 6. Profile 数据

粘贴 `ncu` 的关键指标表（至少包含 4 个版本）：

```
ncu --set full --launch-count 1 ./sgemm_v7 1024 1024 1024
```

需要保留的指标：
- Duration
- Compute (SM) Throughput / Memory Throughput
- Achieved Occupancy
- Registers Per Thread / Shared Memory Configuration
- Warp Stall Reasons（Top 3）
- L1/TEX Hit Rate、L2 Hit Rate

截图或文本贴到 `profile/` 目录下，本文件里只保留关键表格。

## 7. 结论与不足

- 最终版本达到 cuBLAS 的 ____%，原因是 ____。
- 与 cuBLAS 的差距主要来自：____（例如寄存器双缓冲、swizzle 去除 bank conflict、
  更细的线程块形状调优、针对特定形状的算法选择）。
- 尚未解决的问题：____。
- 如果再做完一轮，我会优先尝试：____。
