# CUDA 从零到实战 · 完整教程

> 一套面向**零 CUDA 经验**开发者的系统教程。
> 只要你会 C/C++，就能从装环境一路学到写出接近 cuBLAS 性能的矩阵乘法。
>
> 全部示例代码都在 `code/` 下，**每个都是完整可编译的**，复制粘贴即可运行。

---

## 快速开始：先把项目拿到本地

想把教程拉到自己电脑上学习、练习、提交笔记？见 **[如何在你的电脑上用 SourceTree 获取本项目](SourceTree使用指南.md)** —— 从安装 Git、配置 SSH 免密，到克隆、提交、推送的完整步骤，以及常见问题排查。

只想用命令行，一行就够：

```bash
git clone git@github.com:442616159/cuda-tutorial.git
```

---

## 这套教程怎么用

1. **先读** [docs/CUDA学习步骤与教程大纲.md](docs/CUDA学习步骤与教程大纲.md) —— 了解全貌和节奏。
2. **按顺序读 `docs/` 里的章节** —— 章节之间是递进关系，跳读会踩坑。
3. **每章的代码都要亲手敲一遍** —— 看懂 ≠ 会写，CUDA 尤其如此。
4. **每章末尾的练习必须做** —— 那是真正的学习发生的地方。
5. **从第 4 章开始，养成写 profile 的习惯** —— 不测量的优化都是猜测。

---

## 章节目录

| 章 | 标题 | 核心内容 | 预计用时 | 难度 |
|---|---|---|---|---|
| 0 | [环境搭建与第一个程序](docs/ch00_环境搭建与第一个程序.md) | 装驱动/Toolkit、nvcc 编译原理、hello CUDA、设备查询 | 3～5 天 | ⭐ |
| 1 | [编程模型与线程索引](docs/ch01_编程模型与线程索引.md) | grid/block/thread、四个内建变量、Grid-Stride Loop、warp | 1 周 | ⭐⭐ |
| 2 | [内存层次与访存优化](docs/ch02_内存层次与访存优化.md) | 内存金字塔、合并访存、shared memory、bank conflict、分块矩阵乘 | 1.5 周 | ⭐⭐⭐ |
| 3 | [同步通信与原子操作](docs/ch03_同步通信与原子操作.md) | `__syncthreads`、warp shuffle、原子操作、Cooperative Groups、流 | 1 周 | ⭐⭐⭐ |
| 4 | [性能分析与优化方法论](docs/ch04_性能分析与优化方法论.md) | 带宽/GFLOPS/Roofline、Nsight Compute & Systems、编译选项、PTX/SASS | 1 周 | ⭐⭐⭐ |
| 5 | [核心算法模式实战](docs/ch05_核心算法模式实战.md) | 归约、前缀和、Stencil、排序、SpMV、BFS | 2 周 | ⭐⭐⭐⭐ |
| 6 | [CUDA 高级特性](docs/ch06_CUDA高级特性.md) | 动态并行、Tensor Core/WMMA、`cp.async`、线程块集群、多 GPU | 1.5 周 | ⭐⭐⭐⭐⭐ |
| 7 | [生态库与工程化](docs/ch07_生态库与工程化.md) | cuBLAS/cuFFT/cuRAND/Thrust/CUB、Python 衔接、CMake 工程 | 1 周 | ⭐⭐⭐ |
| 8 | [综合项目实战](docs/ch08_综合项目实战.md) | 高性能 SGEMM 渐进优化、光线追踪、N 体模拟、交付标准 | 2 周 | ⭐⭐⭐⭐⭐ |

**总计约 11～12 周**（每天 1.5～2 小时）。

---

## 学习路线图

```
                    ┌─────────────────────────────────────────┐
                    │  第 0 章  环境搭建                        │
                    │  能编译能运行，会查设备信息               │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │  第 1 章  线程索引                        │
                    │  理解"我是谁，我该处理哪个数据"           │
                    └────────────────┬────────────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────┐
        │  第 2 章  内存层次 ★ 全书第一个分水岭                     │
        │  合并访存 + shared memory 分块，性能提升 10 倍以上         │
        └────────────────────────────┬────────────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
   ┌──────────▼──────────┐                     ┌────────────▼────────────┐
   │  第 3 章  同步与原子  │                     │  第 4 章  性能分析       │
   │  块内协作、warp 通信  │                     │  ncu/nsys，用数据说话   │
   └──────────┬──────────┘                     └────────────┬────────────┘
              └──────────────────────┬──────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │  第 5 章  算法模式                        │
                    │  归约/扫描/排序/Stencil/SpMV 通用套路     │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │  第 6 章  高级特性                        │
                    │  Tensor Core、异步流水线、集群            │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │  第 7 章  生态与工程化                    │
                    │  会用库、能建工程、能接 Python            │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │  第 8 章  综合项目 ★ 结业                 │
                    │  从朴素 SGEMM 优化到接近 cuBLAS           │
                    └─────────────────────────────────────────┘
```

---

## 目录结构

```
CUDALearning/
├── README.md                          ← 你正在读的文件（唯一入口）
├── CONTRIBUTING.md                    ← 教程编写规范（维护者用）
├── docs/                              ← 全部文档
│   ├── CUDA学习步骤与教程大纲.md        ← 学习路线总览
│   ├── ch00_环境搭建与第一个程序.md
│   ├── ch01_编程模型与线程索引.md
│   ├── ch02_内存层次与访存优化.md
│   ├── ch03_同步通信与原子操作.md
│   ├── ch04_性能分析与优化方法论.md
│   ├── ch05_核心算法模式实战.md
│   ├── ch06_CUDA高级特性.md
│   ├── ch07_生态库与工程化.md
│   └── ch08_综合项目实战.md
├── code/
│   ├── Makefile                       ← 一键编译全部示例
│   ├── CMakeLists.txt                 ← 统一 CMake 构建入口
│   ├── common/                        ← 公共头文件（所有章节共用）
│   │   ├── cuda_check.h               ← 错误检查宏
│   │   ├── cuda_timer.h               ← CudaTimer 计时类（第 0~2 章）
│   │   └── cuda_bench.h               ← medianMs/gbps/gflopsOf（第 3~8 章）
│   ├── lessons/                       ← 教学示例（按章节组织）
│   │   ├── ch00_hello/                ← 第一个程序 + 设备查询
│   │   ├── ch01_indices/              ← 索引、向量加法、矩阵加法
│   │   ├── ch02_memory/               ← 访存优化与矩阵乘法三级演进（9 个文件）
│   │   ├── ch03_sync/                 ← 同步原语（barrier/warp_sync/fence/shuffle/vote）
│   │   ├── ch03_atomics/              ← 原子操作
│   │   ├── ch03_atomics_histogram/    ← 直方图实战
│   │   ├── ch03_cooperative/          ← Cooperative Groups
│   │   ├── ch03_streams/              ← 流与异步流水线
│   │   ├── ch03_graphs/               ← CUDA Graphs 基础
│   │   ├── ch04_timing/               ← 计时与带宽测量
│   │   ├── ch04_perf_model/           ← Roofline 模型
│   │   ├── ch04_occupancy/            ← 占用率分析
│   │   ├── ch04_case_study/           ← 优化前后对比案例
│   │   ├── ch04_debugging/            ← compute-sanitizer 实战
│   │   ├── ch05_reduce/ … ch05_cg/    ← 归约、扫描、Stencil、排序、SpMV、BFS、精度、CG
│   │   ├── ch06_advanced/             ← 动态并行、WMMA、cp.async、集群、VMM、P2P
│   │   ├── ch07_ecosystem/            ← cuBLAS/cuFFT/cuRAND/Thrust/CUB/cuSPARSE/cuDNN
│   │   └── ch07_python/               ← CuPy / Numba / Triton / PyTorch 扩展
│   └── projects/                      ← 完整项目（可独立交付）
│       ├── sgemm/                     ← 七级渐进 GEMM + cuBLAS 参照 + 性能报告模板
│       ├── raytracer/                 ← 并行光线追踪器
│       ├── nbody/                     ← N 体模拟（含 5 个 TODO 的分步指引）
│       └── cmake_skeleton/            ← 生产级 CMake 工程骨架（含 ctest）
└── tools/                             ← 环境体检与维护脚本
    ├── check_env.sh / check_env.ps1   ← 一键环境体检
    ├── verify_all.ps1                 ← 批量编译校验所有示例
    ├── fix_links.py                   ← 文档链接检查/修复
    ├── cleanup.sh / cleanup.ps1       ← 目录重组残留清理（已完成使命）
    └── reorganize.js                  ← 目录结构迁移工具（已完成使命）
```

---

## 快速开始

### 0. 一键环境体检（推荐先跑这个）

```bash
# Linux / macOS / WSL / Git Bash
bash tools/check_env.sh
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy Bypass -File tools/check_env.ps1
```

脚本会依次检查：NVIDIA 驱动 → CUDA Toolkit → 主机编译器 → 构建工具 →
性能分析工具 → **实机编译并运行一个 CUDA 程序**，最后给出结论。
如果有任何一项 FAIL，先按提示修好再开始第 0 章。

### 1. 检查你的机器

```bash
nvidia-smi          # 看到显卡型号和驱动版本就说明驱动 OK
nvcc --version      # 看到 release 版本号说明 Toolkit 装好了
```

### 2. 编译第一个示例

```bash
cd code/lessons/ch00_hello
nvcc -O3 hello.cu -o hello
./hello
```

预期输出类似：

```
Hello World from CPU!  (检测到 1 个 CUDA 设备)
Hello World from GPU!  (block 0, thread 0)
Done.
```

> 如果只看到第一行和 `Done.`，说明设备端输出没刷出来——检查是否漏了
> `cudaDeviceSynchronize()`（详见第 0 章）。

### 3. 看一眼你的显卡参数

```bash
cd code/lessons/ch00_hello
nvcc -O3 device_query.cu -o device_query
./device_query
```

会打印设备名称、计算能力、SM 数量、显存大小等，第 1 章之后的所有性能数据
都要结合这些参数来解读。

### 4. 一键编译所有示例

```bash
cd code
make            # 编译 code/ 下所有示例，产物在 code/bin/
make list       # 先看看会编译哪些
make ch02       # 只编译第 2 章的示例
make clean      # 清理
```

可执行文件名 = 「相对路径压平后加下划线」，例如
`lessons/ch04_case_study/matmul_opt.cu` → `bin/lessons_ch04_case_study_matmul_opt`。
不确定名字时先跑 `make list`。

Windows 用户如果没有 `make`，每个示例目录里都给出了独立的 `nvcc` 命令，直接复制即可。

### 5. 批量校验所有示例能否编译

```powershell
# Windows：需要已安装 nvcc
powershell -ExecutionPolicy Bypass -File tools/verify_all.ps1
```

它会逐个编译 `code/` 下所有 `.cu` 文件，最后报告「通过 / 失败 / 跳过」清单
（自动为动态并行、集群、VMM 等示例补上特殊参数）。

> 注意：第 6 章的 `cluster_dsm.cu` 需要 Hopper（CC 9.0）才能编译，
> `cpasync_gemm.cu` 需要 Ampere（CC 8.0），如果你的卡不支持，
> 这两个示例编译失败是**预期行为**，直接跳过即可，不影响其余章节。

---

## 学习建议（血泪经验）

| 建议 | 说明 |
|---|---|
| **别只看不写** | CUDA 的坑 90% 只有亲手踩过才会记住 |
| **每章都建基线** | 任何优化前先记录原始耗时，否则你不知道自己在进步还是退步 |
| **从第 4 章起必开 profiler** | "我觉得这里慢" 和 "ncu 显示这里停顿 60%" 是两回事 |
| **先正确再快速** | 一个跑得快但结果是错的 kernel 价值为零 |
| **善用 `CUDA_CHECK`** | 早期忽略错误码，后期会花十倍时间 debug |
| **数值结果必须验证** | GPU 结果一定要和 CPU 参考实现对比，写容差判断 |
| **别急着上高级特性** | 第 2 章的内存优化能解决 80% 的性能问题 |

---

## 前置知识自检

在开始第 0 章前，请确认你能回答这些问题（不能的话先补一下）：

- [ ] C 语言里指针和数组的关系，`a[i]` 等价于什么？
- [ ] 什么是内存对齐？为什么结构体大小可能比成员加起来大？
- [ ] 栈内存和堆内存有什么区别？局部变量放在哪里？
- [ ] 编译一个 C 程序分成哪几个阶段？（预处理/编译/汇编/链接）
- [ ] 延迟（latency）和吞吐量（throughput）的区别是什么？

---

## 环境要求

### 一、软硬件清单

| 项目 | 最低要求 | 推荐 |
|---|---|---|
| NVIDIA 显卡 | Compute Capability ≥ 3.5 | ≥ 8.0（Ampere 及以后） |
| 显卡驱动 | 满足所装 CUDA Toolkit 的最低版本要求 | 最新版 |
| CUDA Toolkit | 11.0（老卡）／12.8+（RTX 50 系） | 12.6 或 13.x，见下一节 |
| 主机编译器 | Linux: g++ 7+　／　Windows: MSVC 2019+ | VS 2022（CUDA 13 建议） |
| 操作系统 | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| 内存 | 8 GB | 16 GB+ |

> **编译不需要显卡，运行才需要。** nvcc 完全在 CPU 上工作，所以没有 N 卡也能装 Toolkit、把代码编出来（只是跑不起来）。

### 二、CUDA 版本怎么选：看显卡，不是看喜好

同一份源码可以在多个 CUDA 版本上编译（源码里对 API 变更做了 `#if CUDART_VERSION` 版本兼容），但**版本常常是被显卡"逼"出来的**：

| 你的显卡 | 可用的 CUDA 版本 | 说明 |
|---|---|---|
| Maxwell / Pascal（GTX 9/10 系，sm_50~61） | **≤ 12.x** | CUDA 13 已不支持（13.0 最低只到 `compute_75`） |
| Volta（Titan V，sm_70） | **≤ 12.x** | 同上 |
| Turing（RTX 20 系，sm_75） | 10 以上都行 | 12.x / 13.x 都可以 |
| Ampere（RTX 30 系，sm_80/86） | 11 以上 | 12.x / 13.x 都可以 |
| Ada（RTX 40 系，sm_89） | 11.8 以上 | 12.x / 13.x 都可以 |
| Hopper（H100，sm_90） | 11.8 以上 | 12.x / 13.x 都可以 |
| **Blackwell（RTX 50 系，sm_100/120）** | **12.8 以上** | CUDA 11 完全不认这代卡 |

> **实测环境**：CUDA 13.0 + RTX 5060（CC 12.0）+ VS 2022 + Windows 11，
> `code/` 下 66 个示例**全部编译通过**（1 个需要 PyTorch 的除外）。
> CUDA 12.x 及更早版本走的是源码里的兼容分支，逻辑上等价于原实现，但**未在实机上逐一验证**。

### 三、各示例的硬件门槛

绝大多数示例对显卡没有特殊要求，只有第 6 章和 sgemm 的一个版本例外：

| 示例 | 最低 Compute Capability | 原因 |
|---|---|---|
| 其余全部示例 | ≥ 3.5（任何在役 N 卡均可） | — |
| `ch06_advanced/wmma/wmma_gemm.cu` | **≥ 7.0** | Tensor Core 从 Volta 起才有 |
| `ch06_advanced/cpasync/cpasync_gemm.cu` | **≥ 8.0** | `cp.async` 是 Ampere 引入的 |
| `projects/sgemm/v6_double_buffer.cu` | **≥ 8.0** | 同上 |
| `ch06_advanced/cluster/cluster_dsm.cu` | **≥ 9.0（Hopper）** | 线程块集群只有 H100 那代支持 |
| `ch06_advanced/dynparallel/dynparallel.cu` | **< 9.0**（反向限制） | CUDA 13 起设备端同步在 sm_90+ 被移除；Windows x64 还需 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` |
| `ch06_advanced/multigpu/p2p.cu` | 需要**两张**显卡 | 编译不需要，运行需要 |
| `ch07_python/*` | Python + torch / cupy / numba / triton | 与 CUDA 版本无关，是缺包问题 |

### 四、换一台电脑时的检查清单

```bash
nvidia-smi               # ① 显卡型号 + 驱动版本 + 计算能力(CC)
nvcc --version           # ② Toolkit 版本
nvcc --list-gpu-arch     # ③ 这个 Toolkit 支持哪些架构
```

拿到 CC 之后，编译任意示例（**不需要任何额外参数**）：

```bash
nvcc -O3 -std=c++17 -arch=sm_XX 文件名.cu -o 文件名
```

`sm_XX` 就填第 ① 步看到的 CC：`8.6` → `-arch=sm_86`，`12.0` → `-arch=sm_120`。

**两个容易踩的坑：**

1. `-arch=native` 很方便，**但 `dynparallel.cu` 不能用**——RTX 40/50 系会被探测成 sm_89 / sm_120，而该示例要求 CC < 9.0，必须手写 `-arch=sm_80`。
2. **新增源文件请保存为「UTF-8 with BOM」**：当本机 ANSI 代码页是 GBK 时，没有 BOM 的中文源码会被编译器错解，产生一大堆 `expected a declaration`、`missing closing quote` 之类的**假语法错误**。详细说明见 `tools/verify_all.ps1` 顶部注释。

---

## 章节编号对照

本教程正文文件在 `docs/` 目录下，与大纲阶段的对应关系：

| 大纲阶段 | 教程章节 |
|---|---|
| 第 0 阶段：环境搭建与前置知识 | 第 0 章 |
| 第 1 阶段：CUDA 编程模型入门 | 第 1 章 |
| 第 2 阶段：线程组织与内存层次 | 第 2 章 |
| 第 3 阶段：同步、通信与原子操作 | 第 3 章 |
| 第 4 阶段：性能分析与优化方法论 | 第 4 章 |
| 第 5 阶段：核心算法模式实战 | 第 5 章 |
| 第 6 阶段：CUDA 高级特性 | 第 6 章 |
| 第 7 阶段：生态、库与工程化 | 第 7 章 |
| 第 8 阶段：综合项目与结业 | 第 8 章 |
