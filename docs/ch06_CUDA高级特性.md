# 第 6 章 CUDA 高级特性

> **本章目标**：掌握只有现代 GPU 才具备的八项能力——动态并行、Tensor Core（WMMA）、`cp.async` 异步流水线、线程块集群与分布式共享内存、统一内存进阶、虚拟内存管理（VMM）、多 GPU 与 P2P、CUDA Graphs 的条件/循环节点。本章风格和第 5 章不同：**一个特性配一个完整可编译的示例**，每个示例都明确标注它需要的 **Compute Capability（CC）**、编译参数，以及「卡不支持时会发生什么、怎么优雅跳过」。
>
> **前置章节**：[第 1 章](ch01_编程模型与线程索引.md)（线程索引、网格步长循环）、[第 2 章](ch02_内存层次与访存优化.md)（共享内存、分块、double buffering）、[第 3 章](ch03_同步通信与原子操作.md)（流、CUDA Graphs 基础）、[第 4 章](ch04_性能分析与优化方法论.md)（ncu / nsys、编译选项）、[第 5 章](ch05_核心算法模式实战.md)（归约、stencil、SGEMM 分块）。
>
> **预计耗时**：1.5 周（每天 1.5～2 小时）。8 个示例不要求都在你的机器上跑通——**很多特性的门槛是硬件**——但每个示例的代码都必须读懂，它们是现代 GPU 编程绕不开的词汇表。

---

## 0. 本章导读

### 0.1 学完这一章你能做什么

前五章你已经能把一个问题「归类 → 映射线程 → 摆好数据 → 优化访存 → 用 ncu 验证」，这套功夫在任何一张 NVIDIA 显卡上都能用。但从 Volta（2017）开始，GPU 每两年就多出一批**不是「更快」而是「不一样」**的硬件能力：

- 你写的 GEMM 怎么优化都追不上 cuBLAS，因为你压根没用**张量核心**；
- 分块矩阵乘里「加载到共享内存」这一步，在 Ampere 上有一条**绕过寄存器**的近路；
- stencil 的 halo 数据明明就在隔壁块的共享内存里，你却在绕道显存；
- 递归深度在运行时才知道，host 端根本展开不出一串固定的 kernel 启动；
- 显存碎片化严重，`cudaMalloc` 给不出你要的连续大块；
- 一段推理管线有 40 个 kernel，每个只跑 10 微秒，**启动开销比计算时间还长**。

学完本章，你将能够：

1. **在设备端启动 kernel**（动态并行），用递归/分治表达运行时才知道形状的负载，并说清它的成本上限在哪。
2. **用 `nvcuda::wmma` 写 FP16 矩阵乘**：看懂 `fragment` 的每个模板参数、`load_matrix_sync` / `mma_sync` / `store_matrix_sync` 三步，知道 TF32 / BF16 / INT8 / FP8 各需要哪一代架构。
3. **用 `cp.async` 搭多阶段软件流水线**，说清它与第 2 章 double buffering 的区别，以及 stage 数怎么选。
4. **用线程块集群 + 分布式共享内存（DSM）** 让块与块直接交换数据，不再绕道全局内存。
5. **用 `cudaMemAdvise` / `cudaMemPrefetchAsync` 驯服统一内存**，知道它什么时候快、什么时候慢得离谱。
6. **用 VMM API 把「虚拟地址」和「物理显存」解耦**，理解大模型框架为什么要自己管显存。
7. **写出多卡程序**：查 P2P 能力矩阵、开启 peer access、用 `cudaMemcpyPeer` 直达，并知道集合通信该交给 NCCL。
8. **给 CUDA Graph 加上条件节点和循环节点**，把「if 残差还大就再迭代一次」这种设备端控制流也放进图里。
9. **对任何高级特性做能力检测**：用 `cudaGetDeviceProperties` / `cudaDeviceGetAttribute` 判 CC，不支持时打印一句人话然后**优雅退出**，而不是崩在 `cudaErrorNotSupported`。

### 0.2 一句话主线

> **高级特性不是「更快的技巧」，而是「新的可能性」——它们把原本必须由 host 编排、必须绕道显存、必须拆成多个 kernel 的事情，变成了单个 kernel 内部、设备侧、硬件直接支持的操作。**

代价是**兼容性**。本章每个示例都遵循同一条流程：

```text
查 CC / 查属性  →  不支持就打印原因并 return 0  →  支持才跑  →  跑完和 CPU 对拍
```

### 0.3 特性 × 硬件能力速查表（本章最重要的表）

| 特性 | 最低 CC | 额外硬性要求 | 不支持时的表现 |
|---|---|---|---|
| **动态并行**（设备端 `<<<>>>`） | **3.5** | 必须 `-rdc=true`；设备端 `malloc` 还看堆配额 | 不加 `-rdc` 编译直接报错；加了但卡不支持则启动报 `cudaErrorNotSupported` 或静默不执行 |
| **WMMA（FP16）** | **7.0**（Volta） | `-arch=sm_70`+，`#include <mma.h>` | 编译期失败：`fragment` 无法实例化 / `wmma` 未定义 |
| **WMMA（INT8）** | **7.2**（Turing） | `-arch=sm_72`+ | 同上 |
| **WMMA（TF32 / BF16 / FP64）** | **8.0**（Ampere） | `precision::tf32` / `__nv_bfloat16` | 编译期失败 |
| **WMMA（FP8）** | **8.9**（Ada）/ **9.0**（Hopper） | `-arch=sm_89` / `sm_90` | 编译期失败 |
| **`cp.async`** | **8.0**（Ampere） | `#include <cuda_pipeline.h>` | **能编过、结果也对，但退化成同步 ld/st，拿不到任何加速** |
| **`cuda::pipeline`（libcu++）** | 8.0 才有真异步 | `<cuda/pipeline>`，C++17 | 低架构退化为同步路径 |
| **线程块集群 / DSM** | **9.0**（Hopper） | Toolkit **≥ 11.8**（`cudaLaunchKernelEx`）；`-arch=sm_90` | 旧 Toolkit 编不过；新 Toolkit + 旧卡启动返回 `cudaErrorInvalidValue` / `cudaErrorNotSupported` |
| **`cudaMallocManaged`** | 2.0（基本托管） | — | 老架构不支持 GPU 超配（oversubscription） |
| **按需页迁移 / `cudaMemPrefetchAsync`** | **6.0**（Pascal） | 需 `cudaDevAttrConcurrentManagedAccess` | 预取返回 `cudaErrorInvalidValue`；退化成一次性整段迁移 |
| **`cudaMemAdvise`** | **6.0** | `SetAccessedBy` 需并发托管访问 | 调用失败，或建议被驱动忽略（它只是「提示」） |
| **VMM（`cuMemCreate` / `cuMemMap`）** | **6.0** + 驱动 **≥ 440** | `CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED`；链接 `-lcuda` | 属性查询返回 0；`cuMemCreate` 返回 `CUDA_ERROR_NOT_SUPPORTED` |
| **多 GPU P2P** | ≥ 2 张卡，CC ≥ 2.0 + 64 位进程 + UVA | 主板/PCIe 拓扑，NVLink 或同 PCIe 交换机 | `cudaDeviceCanAccessPeer` 返回 **0**（不是报错），必须走主机中转 |
| **CUDA Graphs（基础）** | 3.5 | Toolkit ≥ 10.0 | `cudaStreamBeginCapture` 返回错误 |
| **Graph 条件 / 循环节点** | 见下 | **Toolkit ≥ 12.3** | 用 `CUDART_VERSION >= 12030` 编译期裁剪；实例化失败返回 `cudaErrorNotSupported` |

> **关于条件/循环节点**：它的**主要门槛是 Toolkit 版本**（12.3 才引入这套 API），示例用 `#if CUDART_VERSION >= 12030` 做编译期保护。此外设备侧条件求值依赖设备端图启动机制，**在很老的架构上实例化可能返回 `cudaErrorNotSupported`**。习惯是：**凡是返回 `cudaError_t` 的 Graph API 都检查返回值**。参见 [NVIDIA graphConditionalNodes 样例](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/graphConditionalNodes/README.md)。

**怎么查自己的卡**——本章所有示例都用了同一个套路：

```cpp
cudaDeviceProp prop{};
CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
if (prop.major < 8) {
    printf("[跳过] 本机 CC = %d.%d，本特性需要 CC >= 8.0。\n", prop.major, prop.minor);
    return 0;   // 优雅退出：不是崩溃，也不是跑出错误结果
}
```

`prop.major` 是主版本号（7、8、9），`prop.minor` 是次版本号（0、5、6、9），`sm_86` 对应 `major=8, minor=6`。**注意 3.5 这个门槛没法用 `major < 3` 精确表达**（`sm_30` 也是 `major=3`），所以 `dynparallel.cu` 用的是 `ccMajor < 3` 并在注释里说明真实要求是 3.5——这是教程为了简洁做的取舍，生产代码请写 `(prop.major < 3 || (prop.major == 3 && prop.minor < 5))`。

### 0.4 本章示例代码一览

| 目录 | 文件 | 讲什么 | 关键 CC | 特殊编译参数 |
|---|---|---|---|---|
| `code/lessons/ch06_advanced/dynparallel/` | `dynparallel.cu` | 设备端启动、递归归约、设备端 `malloc` | ≥ 3.5 | `-rdc=true` |
| `code/lessons/ch06_advanced/wmma/` | `wmma_gemm.cu` | FP16 输入 + FP32 累加的 WMMA 矩阵乘 | ≥ 7.0 | `-arch=sm_70`+ |
| `code/lessons/ch06_advanced/cpasync/` | `cpasync_gemm.cu` | `cp.async` 双缓冲流水线 GEMM | ≥ 8.0（低架构退化） | `-arch=sm_80` |
| `code/lessons/ch06_advanced/cluster/` | `cluster_dsm.cu` | 集群内跨块归约、`map_shared_rank` | ≥ 9.0 | `-arch=sm_90` |
| `code/lessons/ch06_advanced/unified/` | `unified_advise.cu` | `cudaMemAdvise` 四条建议 + 预取对比 | ≥ 6.0 | `-arch=sm_70` |
| `code/lessons/ch06_advanced/vmm/` | `vmm_alloc.cu` | VMM 四步：预留 / 创建 / 映射 / 授权 | ≥ 6.0 + 驱动 440 | `-lcuda` |
| `code/lessons/ch06_advanced/multigpu/` | `p2p.cu` | P2P 能力矩阵、`cudaMemcpyPeer`、远端 kernel | 需 ≥ 2 张卡 | — |
| `code/lessons/ch06_advanced/graphs_advanced/` | `graph_advanced.cu` | 显式构图、If/Else 条件节点、While 循环节点 | Toolkit ≥ 12.3 | `-arch=sm_70` |

### 0.5 学习建议

1. **先看表，再看码**。0.3 节那张表决定你的机器能跑哪几个示例。跑不了的**不要跳过阅读**——很多生产代码在你的开发机上跑不了，在部署机上跑得了。
2. **每个示例都先编译、再运行、最后用 `ncu` / `nsys` 看一眼**。高级特性的收益往往藏在时间线里。
3. **对拍是底线**。这 8 个示例全都自带 CPU 参考实现或边界校验，一个都不能省。
4. **不要为了用而用**。动态并行、集群、VMM 都有明确的适用场景，用错地方会比朴素写法更慢、更复杂。

---

## 1. 概念讲解

### 1.1 动态并行：在设备端启动 kernel

**先打个比方**。你是装修公司的老板（host），传统做法是你把所有工序排好：「1 号楼先刷墙，2 号楼再刷墙……」。工人（block）干完手里的活就下班，**不能自己去雇人**。动态并行就是给了现场工头一个权力：**他可以当场决定再招一批临时工**（启动子 kernel），甚至根据现场情况决定要不要继续往下拆。

**严谨定义**：动态并行（Dynamic Parallelism）是指在 `__global__` 函数内部**直接使用 `<<<>>>` 启动另一个 kernel**，由设备端 CUDA Runtime 在 GPU 上完成调度，**不需要 host 参与**。CUDA 5.0 引入，硬件要求 **CC ≥ 3.5**。

为什么必须在设备端启动？

1. **递归**：BVH 构建、快速多极子（FMM）、自适应网格细分（AMR）的递归深度**编译期未知**，host 端展开不出一串固定启动语句。
2. **自适应细分**：某区域误差还大就再细一层，判断只有运行时才知道，而每次「判断 → 回 host → 再启动」都要几十微秒往返。
3. **省掉 host 往返**：迭代型算法里「残差还大就再来一轮」若放在 host 判断，每轮都要 `cudaMemcpy` + 同步。

**代价（必须记住）**：

- 设备端启动一个 kernel 的开销**大约是 host 端启动的 2～5 倍**（要经过设备侧运行时、维护待启动队列）。
- 它**占用显存并为每个 launch 分配资源**，默认待启动队列深度有限（`cudaLimitDevRuntimePendingLaunchCount`），队列满时**要么阻塞、要么丢弃启动**，这是动态并行最经典的死锁来源。
- `nsys` 时间线上会出现**大量细碎 kernel**，调度开销会吃掉一切收益。

**两个硬性条件**（缺一不可）：

```text
① 目标架构 CC ≥ 3.5
② 编译时必须加 -rdc=true（Relocatable Device Code，可重定位设备代码）
```

`-rdc=true` 为什么必须加？因为设备端 `<<<>>>` 需要把「子 kernel 的设备函数符号」延迟到**链接期**才确定，这要求设备代码不能像默认那样「每个 `.cu` 单独编成完整 cubin」，而要像普通 C++ 一样分「编译 → 链接」两步；链接完成后还要把设备端运行时库 `cudadevrt` 链进去。

```bash
nvcc -O3 -arch=sm_70 -rdc=true dynparallel.cu -o dynparallel
# 现代 nvcc 在 -rdc=true 时会自动链接 cudadevrt；
# 若报 undefined reference to '__cudaRegisterLinkedBinary...'，手动补上：
nvcc -O3 -arch=sm_70 -rdc=true dynparallel.cu -o dynparallel -lcudadevrt
```

**设备端 `cudaDeviceSynchronize()` 的行为与 host 端不同**（高频面试题）：

| | host 端 | 设备端 |
|---|---|---|
| 等待范围 | 该设备上**所有流**的所有工作 | **当前线程**派生的子 kernel（含后代） |
| 是否是全局屏障 | 是（对整个设备） | **否**——不保证别的 block 也到了这里 |
| 代价 | 一次 host-device 同步 | 设备侧等待，不切回 host |

正因为它**不是块间屏障**，`parentKernel` 里那个 `__syncthreads()` 才是必要的：发起子 kernel 的只有 `block 0` 的 `thread 0`，块里其它线程得等它把 `cudaDeviceSynchronize()` 做完才能安全读结果。

**适用 / 不适用场景**：

| ✔ 值得用 | ✘ 千万别用 |
|---|---|
| 递归/分治深度编译期未知 | 形状规则的负载（普通 grid-stride kernel 快得多） |
| 子问题规模差异极大，只有运行时才知道要不要继续拆 | 深度固定的分治（多次 host 启动或 CUDA Graphs 更划算） |
| 想省掉 host 往返（自适应步长迭代） | 追求极限性能的计算密集 kernel |

**经验结论**：递归归约通常比「单 kernel 分块归约」**慢 3～10 倍**（实测随机器不同）。它的价值在**表达能力**，不在绝对性能。

### 1.2 张量核心（Tensor Core）与 WMMA

**先打个比方**。普通 CUDA Core 做矩阵乘，像砌墙工人一次搬一块砖（一次 FMA = 一次乘加）。张量核心像一台**预制板浇筑机**：你给它一整排砖（16×16 的小矩阵），它一次浇出一整块楼板（16×16×16 的乘加）。同样是算乘加，单位时间搬进去的砖数差一个数量级。在 RTX 3060 这一级硬件上，FP32 的 CUDA Core 吞吐是十几 TFLOPS 量级，而 **FP16 Tensor Core 吞吐可达 40～100+ TFLOPS**（随型号与是否稀疏而变，本教程不承诺具体数字）。

**严谨定义**：**张量核心（Tensor Core）** 是一组专门执行**小矩阵乘加**（`D = A × B + C`）的硬件单元，自 Volta（CC 7.0）起进入 NVIDIA GPU。**WMMA（Warp Matrix Multiply Accumulate）** 是 CUDA 暴露给程序员的 C++ API，位于 `nvcuda::wmma` 命名空间，头文件 `<mma.h>`。

WMMA 有三个必须记住的特点：

1. **它是 warp 级操作**：一次 `mma_sync` 由**一整个 warp 的 32 个线程协作**完成，不是「每个线程算自己那一格」。第 1 章你熟悉的「线程 ↔ 元素」映射在这里不成立。
2. **数据布局是硬件细节**：累加器分散在 32 个线程的寄存器里，哪个线程拿哪几个元素是**不透明的**，你不该、也不需要知道。
3. **`fragment` 就是这个「分布式寄存器集合」的抽象**。所有操作都围绕它展开：

```text
fragment 声明（只占寄存器，不访问内存）
        ↓
load_matrix_sync   —— 把一块数据从内存/共享内存搬进 fragment
        ↓
mma_sync           —— 一次算完 D = A×B + C
        ↓
store_matrix_sync  —— 把结果写回内存
```

**`fragment` 的模板参数逐个解释**：

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
//              ─────┬──────  ─┬─ ─┬─ ─┬─  ───┬──  ────────┬─────────
//                   │        │   │   │      │              │
//  ① 用途           │        │   │   │      │              └─ ⑥ 内存布局
//  ② M（行数）      │        │   │   │      └─ ⑤ 元素类型
//  ③ N（列数）──────┘        │   │   └─ ④ K（内积维度）
//  ④ K（内积维度）──────────┘   │
```

| 参数 | 含义 | 说明 |
|---|---|---|
| ① 用途 | `matrix_a` / `matrix_b` / `accumulator` | 三种 fragment 的**寄存器布局不同**，不能混用 |
| ② M | 输出行数 | 2 的幂，`16` 最常见 |
| ③ N | 输出列数 | 同上 |
| ④ K | 内积维度 | 一次 `mma_sync` 只能吃掉 16 个 K，所以 K 方向要循环 |
| ⑤ 元素类型 | `__half` / `__nv_bfloat16` / `float` / `int8_t` / `precision::tf32` | 输入和累加器**可以不同** |
| ⑥ 布局 | `row_major` / `col_major` | 只在 A/B 上有意义；累加器在 `store` 时用 `mem_row_major` 指定 |

**leading dimension（`ldm`）是什么？** 就是**「下一行（或下一列）的第一个元素，距离当前行首有多少个元素」**。对行主序矩阵，`ldm` = 一行有多少个元素；但如果你只取大矩阵里的一个子块，`ldm` 依然是**大矩阵的整行长度**，而不是子块宽度。这是最常见的错误来源：

```cpp
// A 是 M×K 的行主序矩阵，取第 warpM 个 16×16 子块、从第 k 列开始
wmma::load_matrix_sync(a_frag, a + (size_t)warpM * WMMA_M * K + k, K);
//                                                                  ↑ 这里是 K，不是 WMMA_K
```

**精度组合表**（决定你的代码能在哪张卡上编过）：

| 输入类型 | 累加类型 | 最低 CC | 备注 |
|---|---|---|---|
| FP16 | FP16 | 7.0 | 精度最差，很少用 |
| **FP16** | **FP32** | **7.0** | **最经典的混合精度组合**，本示例用的就是它 |
| INT8 | INT32 | 7.2 | 量化推理 |
| TF32 | FP32 | 8.0 | 输入仍以 FP32 存储，硬件内部截成 19 位 |
| BF16 | FP32 | 8.0 | 动态范围同 FP32，精度低于 FP16 |
| FP64 | FP64 | 8.0 | 科学计算 |
| FP8 (E4M3/E5M2) | FP32 | 8.9 / 9.0 | 大模型训练的新宠 |

**为什么累加器一定要用 FP32？** FP16 只有 10 位尾数，K 一大（512、1024）累加结果就会严重失真。**输入用低精度省带宽 / 换吞吐，累加用高精度保数值稳定**——这就是「混合精度」这个词的全部含义。

**它和 cuBLAS 的关系**：WMMA 是**硬件接口**（你自己管分块、流水线、寄存器分块），cuBLAS 是**库**（内部用 Tensor Core + autotuning）。同样硬件上 cuBLAS 通常比手写 WMMA 快 **3～10 倍**，差距来自三件事：**共享内存分块**（减少全局访存）、**`cp.async` 多阶段流水线**（访存与计算重叠）、**每线程多个累加器**（寄存器分块，提升 ILP）。

本章 `wmma_gemm.cu` 为把 WMMA 本身讲清楚，**故意**从全局内存直接 `load_matrix_sync` 进 fragment，所以性能很朴素。真正优化版 = 第 5 章的分块 GEMM + 1.3 节 `cp.async` + WMMA，第 8 章的 SGEMM 项目会走完这条路线。

**与混合精度训练（PyTorch AMP）的联系**：`torch.cuda.amp.autocast` 做的事，本质就是「把线性层/卷积派给 FP16 Tensor Core，把 LayerNorm / softmax / 损失留在 FP32，并对梯度做 loss scaling」。你在 CUDA 里手写 FP16 输入 + FP32 累加 + FP32 权重更新，就是同一个思路的手工版。

### 1.3 `cp.async` 异步拷贝与多阶段软件流水线

**先打个比方**。传统的「把货从仓库搬到车间储物柜」要求**必须经过你的手**：`global --ld--> register --st--> shared`。数据要先放进你手里的箱子（寄存器），再放进储物柜（共享内存）。两个问题：**占手**（你要等数据到手才能干别的，LSU 也得等）、**多一道工序**（两条指令占两个发射槽）。

`cp.async` 相当于**给仓库和储物柜之间装了传送带**：`global --cp.async--> shared`，数据**完全不经过寄存器**。按下按钮（发起拷贝）你就可以去干别的，需要时再回头确认货到没到。

**严谨定义**：`cp.async` 是 Ampere（**CC ≥ 8.0**）引入的异步拷贝指令，把数据从全局内存搬到共享内存，**绕过寄存器**。CUDA C++ 通过 `<cuda_pipeline.h>` 暴露三个原语：

| 原语 | 作用 |
|---|---|
| `__pipeline_memcpy_async(dst, src, size)` | 发起一次异步拷贝（`size` 只能是 **4 / 8 / 16 字节**） |
| `__pipeline_commit()` | 把「上一组发起的拷贝」打包成一个**提交组（commit group）** |
| `__pipeline_wait_prior(N)` | 等到「只剩 N 个提交组未完成」。`wait_prior(0)` = 等全部完成，`wait_prior(1)` = 只等「当前这块」就绪 |

**`commit` / `wait` 的语义是这段代码的灵魂**：

```text
发起 S0 的拷贝 → commit()   ← 提交组 #1
发起 S1 的拷贝 → commit()   ← 提交组 #2
wait_prior(1)               ← 保证 #1 完成（S0 就绪），#2 还在后台搬
计算 S0
wait_prior(0)               ← 全部完成
```

**多阶段软件流水线的 stage 数怎么选**：

| stage 数 | 共享内存占用 | 效果 |
|---|---|---|
| 1（等于关掉流水线） | 最小 | 每次都要「先等拷贝完，再算」，完全串行 |
| **2（双缓冲）** | 2× | 计算当前块的同时搬下一块，**性价比最高**，本示例用它 |
| 3～4 | 3～4× | 能藏住更长的访存延迟，但共享内存压力大、占用率下降 |
| 更多 | 更多 | 收益递减，通常 3～4 阶段后到平台期 |

**怎么选**：先看共享内存预算（每块上限 48 KB 起步，可配置到 100 KB+），再看**单块计算时间能不能盖住单块拷贝时间**。如果计算很轻（比如 BK 很小），加 stage 也没用。用 `ncu` 看 Warp Stall 里 `Long Scoreboard` 的比例，是判断「流水线够不够深」的正确方法。

**和第 2 章 double buffering 的区别**：

| | 第 2 章的手工 double buffering | `cp.async` 双缓冲 |
|---|---|---|
| 数据路径 | global → **寄存器** → shared | global → shared（**绕过寄存器**） |
| 指令数 | 每元素 ld + st 两条 | 一条 `cp.async` 指令 |
| 寄存器压力 | 需要额外的中间寄存器 | 不占寄存器 |
| 重叠方式 | 先把下一块 ld 到寄存器，算完再 st 进另一块 shared | commit group + `wait_prior` 分组等待 |
| 最低架构 | 所有架构 | 8.0；低架构**退化为同步** |

**两个关键提醒**：

1. 在 CC < 8.0 的卡上，`__pipeline_memcpy_async` 会**退化成普通的同步 ld/st**：程序**能编过、结果也对**，只是**拿不到任何加速**。所以 `cpasync_gemm.cu` 在 CC < 8.0 时只打印提示、不退出——这段代码仍然有意义（正确），只是没收益。
2. **`__pipeline_wait_prior` 只保证「本线程」发起的拷贝完成了**，而计算阶段要读**别的线程**搬进来的数据，所以 `wait_prior` 之后**必须再加一次 `__syncthreads()`**。

**libcu++ 的等价写法**（更安全、能和 CUDA Graph 一起用）：

```cpp
#include <cuda/pipeline>
cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline();
pipe.producer_acquire();
cuda::memcpy_async(group, dst, src, size, pipe);
pipe.producer_commit();
// ...
pipe.consumer_wait();  /* ... 计算 ... */  pipe.consumer_release();
```

它和 `__pipeline_*` 系列是**同一套底层的两种封装**。

### 1.4 线程块集群（Thread Block Cluster）与分布式共享内存

**先打个比方**。传统 CUDA 里每个线程块是一个**独立办公室**：内部随便交流（`__syncthreads()`），但隔壁办公室的门是锁着的——想传文件只能放到一楼大厅（全局内存）让对方来取；而且两个办公室**必须「交完班才能换人」**：块间唯一的同步手段就是「结束这个 kernel，再启动下一个」。

线程块集群（**CC ≥ 9.0**，Hopper）把这个模式改了：**最多 8 个（可移植上限）办公室被安排在同一层楼，中间的墙打通了**。于是：组内可以做硬件屏障 `cluster.sync()`，不用结束 kernel；组内可以直接读写别人的共享内存，这就是**分布式共享内存（DSM, Distributed Shared Memory）**；块间通信不再必须绕道全局内存。

**严谨定义**：**线程块集群**是在**同一个 GPC** 上协同调度的一组线程块。关键 API：

| API | 作用 |
|---|---|
| `cg::this_cluster()` | 拿到当前集群的 `cluster_group` 句柄（`#include <cooperative_groups.h>`） |
| `cluster.block_rank()` | 本块在集群内的编号（0 起） |
| `cluster.num_blocks()` | 集群里有几个块 |
| `cluster.sync()` | **集群范围的硬件屏障**（所有块的所有线程都到齐） |
| `cluster.map_shared_rank(ptr, r)` | 把「**本块**共享内存里的某个地址」翻译成「**第 r 个块**的同一个地址」 |

**集群怎么指定**：两种写法**二选一，不要同时用**。

```cpp
// 写法 A：核函数上的编译期属性（适合固定集群大小）
__global__ void __cluster_dims__(2, 1, 1) myKernel(...);

// 写法 B：启动属性（更灵活，本示例用它）
cudaLaunchAttribute attrs[1] = {};
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim.x = 2; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
```

**两条铁律**：

1. **`gridDim` 必须是 `clusterDim` 的整数倍**（示例是 64 块 / 每集群 2 块 = 32 个集群），否则启动失败。
2. **退出前必须再 `cluster.sync()` 一次**。某个块提前退出后，它占用的共享内存可能被回收，还在 `map_shared_rank` 读它的块就会读到垃圾。这是集群最经典的 bug。

**集群大小上限**：**可移植上限 8**；若卡/驱动支持可到 **16**，但必须显式设置 `cudaFuncAttributeNonPortableClusterSizeAllowed`，并用 `cudaOccupancyMaxActiveClusters` 确认放得下。

**什么时候真正需要集群？** 大 tile 的 stencil（halo 直接从邻居块的共享内存拿）；大矩阵乘（把 K 方向规约拆到集群内多个块，用 DSM 交换部分和）；需要「块间同步一次，然后各干各的」的算法（多块协作的 scan、图遍历的层次屏障）。写库时务必做能力检测——集群**只在 Hopper 及以后才有**。

### 1.5 统一内存进阶：`cudaMemAdvise` 与按需页迁移

**先打个比方**。统一内存像**一个「看起来只有一份」的文件柜**：`cudaMallocManaged(&p, bytes)` 之后，CPU 和 GPU 用的是同一个指针 `p`。但柜里的纸有**物理位置**——要么在主机内存，要么在显存。默认行为是「**谁碰它，就搬到谁那边**」，靠**页错误（page fault）**触发。这就像每次要用文件都得叫搬家公司搬整个柜子：**搬家的固定开销（几十微秒）比读文件还贵**。

`cudaMemPrefetchAsync` 是「**提前叫好搬家公司**」：我马上要用，你现在就搬。`cudaMemAdvise` 是「**提前告诉搬家公司这份文件的使用习惯**」，让它别每次都靠页错误去猜。

**严谨定义**：**按需页迁移（on-demand page migration）** 指 GPU 访问不在本地的托管内存页时触发**硬件页错误**，驱动把该页迁移过来。这一机制需要 **CC ≥ 6.0**（Pascal）以及 `cudaDevAttrConcurrentManagedAccess = 1`。四条最常用的建议：

| 建议 | 含义 | 什么时候用 |
|---|---|---|
| `cudaMemAdviseSetReadMostly` | 这段数据**只读**，允许驱动在多个处理器上**各留一份副本** | 只读、被反复读的数据；多卡场景收益最大 |
| `cudaMemAdviseSetPreferredLocation` | 建议这段数据**常驻**在某个设备（或 CPU） | 数据本来就属于某张卡，避免来回迁移 |
| `cudaMemAdviseSetAccessedBy` | 允许某个设备**直接访问**（走 NVLink / PCIe），**不迁移** | 偶尔访问、不值得整段搬的数据 |
| `cudaMemAdviseUnsetXXX` | 撤销上面的建议 | 数据用途变了 |

`cudaMemPrefetchAsync(ptr, bytes, dstDevice, stream)` 的语义是「**现在就开始搬，但不阻塞调用它的线程**」，所以要在同一个流里同步才能保证搬完；`dstDevice` 传 `cudaCpuDeviceId` 就是搬回主机。

**典型性能陷阱（值得抄在笔记本上）**：

1. **首次访问永远最慢**。页错误 + 迁移通常几十微秒，而且是**按页**触发的（2 MB 大页就是一次搬 2 MB）。
2. **CPU 和 GPU 交替访问同一块托管内存 = 灾难**，每次换边都是一次迁移。对策：`cudaMemAdvise` + 预取，或者干脆别用托管内存。
3. **迭代型算法（CG、Jacobi、训练循环）里数据本来就常驻 GPU**，最省事的做法是**别用 Unified Memory，直接 `cudaMalloc`**。托管内存的价值在**原型阶段省掉手工 memcpy**，不在最终性能。
4. **在 CPU 上访问 GPU 常驻的托管页，同样触发页错误**。读结果前先 `cudaMemPrefetchAsync(..., cudaCpuDeviceId, ...)`。
5. **用 `nsys` 看页错误**：`nsys profile --stats=true ./unified_advise`，报告里的 Unified Memory 一节会列出页错误次数和迁移字节数。

### 1.6 虚拟内存管理（VMM）：把地址和物理内存解耦

**先打个比方**。`cudaMalloc` 像**买现房**：你看中 40 GB 显存，就得一次性找到 40 GB **物理上连续**的空间，而且**地址和房子一次性绑死**。VMM 像**买地皮 + 自己盖楼**：先在地图上圈一块地（`cuMemAddressReserve`，此时还没有物理内存），再分批买建材（`cuMemCreate`，拿到 handle），把建材盖到地皮上（`cuMemMap`），最后办通行证（`cuMemSetAccess`）。

**严谨定义**：**VMM（Virtual Memory Management）API** 是 CUDA Driver API 提供的一组显存管理接口，把「**虚拟地址预留**」和「**物理分配**」彻底解耦。要求 **CC ≥ 6.0** 且驱动 **≥ 440**，并先用 `cuDeviceGetAttribute(..., CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev)` 查询支持性。

**为什么大模型框架（PyTorch caching allocator、vLLM 的 PagedAttention、TensorRT-LLM）要自己用它？**

- **可增长的数据结构**：先预留一大段虚拟地址，之后一块块把物理内存映射进去；KV Cache 按需增长时不用重新分配和搬移。
- **碎片整理**：让**两段物理上不连续**的显存，在虚拟地址上看起来连续（外部库给你两块碎片，但你的 kernel 只接受一个连续指针）。本章示例演示的正是这个。
- **别名映射（aliasing）**：把**同一块物理内存**映射到多个虚拟地址，实现「一份数据、多个视图」的零拷贝。
- **跨进程 / 跨设备共享**：配合 `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle` 以及 CUDA IPC。
- **绕开运行时分配器**：不受 `cudaMalloc` 的缓存与碎片化策略影响。

**四步流程与释放顺序**：

```text
创建：cuMemAddressReserve → cuMemCreate → cuMemMap → cuMemSetAccess
释放：cuMemUnmap        → cuMemRelease → cuMemAddressFree
            （必须先解映射，再释放物理，最后归还虚拟地址）
```

**分配粒度**：物理分配必须是**粒度（granularity）的整数倍**，用 `cuMemGetAllocationGranularity(..., CU_MEM_ALLOC_GRANULARITY_MINIMUM)` 查询，现代 GPU 通常是 **2 MB**。

**代价与风险**：这段 API **完全绕过运行时 API 的分配器和缓存**。一旦忘记 `cuMemRelease`，那块显存会**一直占着直到进程退出**——`cudaFree` 管不了它，`nvidia-smi` 看起来像显存泄漏。

### 1.7 多 GPU 与点对点（P2P）访问

**先打个比方**。两张显卡是**两栋楼**。楼 A 的工人要用楼 B 的货，本来只有一条路：**搬到楼下、装车、开过马路、卸货**（`GPU0 → 固定内存 → CPU → 固定内存 → GPU1`，两次 PCIe 往返）。P2P 就是在两栋楼之间**修了一座天桥**：`GPU0 → PCIe/NVLink → GPU1`，一次直达；而且开了 `cudaDeviceEnablePeerAccess` 之后，**kernel 可以直接解引用另一张卡的指针**。

**严谨定义与三步流程**：

```text
① cudaGetDeviceCount(&n)                          有几个设备
② cudaDeviceCanAccessPeer(&ok, i, j)               i 能否直接访问 j（返回 0/1，不是报错！）
③ cudaSetDevice(i); cudaDeviceEnablePeerAccess(j, 0);   开启
   → 之后 cudaMemcpyPeer(dst, j, src, i, bytes) 走 P2P 通道
   → 在 i 上启动的 kernel 可以直接读写 j 的指针
```

**前提条件**：64 位进程、统一寻址（UVA，CC ≥ 2.0）、两张卡通过 **NVLink 或同一 PCIe 交换机**相连。任何一条不满足，`cudaDeviceCanAccessPeer` 就返回 **0**——**它不报错，只是告诉你「不行」**。这是初学者最常困惑的一点。

**多卡编程的常见坑**：

| 坑 | 说明 | 对策 |
|---|---|---|
| `cudaDeviceCanAccessPeer` 返回 0 | 主板/PCIe 拓扑不支持（比如两卡挂在不同 CPU 的 PCIe 根上） | 用 `nvidia-smi topo -m` 看拓扑；退回主机中转 |
| 忘了 `cudaSetDevice` | 在错误设备上分配/启动，得到 `cudaErrorInvalidDevicePointer` | 每个 `cudaMalloc` / kernel 启动前明确设备 |
| 开启 P2P 后 UVA 行为变化 | 某些平台上开启 peer access 会影响统一寻址行为 | 开启前后都测一遍，不要假设行为不变 |
| 多进程多卡时传指针 | 跨进程**不能**传指针 | 用 CUDA IPC：`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`，共享的是**句柄** |
| 自己写 AllReduce | ring/tree 算法优化复杂，还容易死锁 | **用 NCCL** |

**NCCL 的角色**：单机多卡之上就是多机多卡，业界标准是 **NCCL**。它提供 `ncclAllReduce` / `ncclAllGather` / `ncclReduceScatter` / `ncclBroadcast` / `ncclSend` / `ncclRecv`，内部**自动选择传输层**（NVLink / PCIe P2P / 共享内存 / InfiniBand、RoCE），并在算法层面做 ring / tree 优化。[第 7 章](ch07_生态库与工程化.md) 会给出可运行的例子。

### 1.8 CUDA Graphs 进阶：条件节点与循环节点

**先打个比方**。[第 3 章](ch03_同步通信与原子操作.md) 的 CUDA Graph 是「**把一串下单录成一份标准作业流程，之后一键重放**」。但那份流程图是**静态的**：节点和依赖在实例化时就钉死了，图上**画不了菱形（if）和回路（while）**。真实程序到处是控制流：`while (residual > tol) step();` ——迭代多少次只有运行时才知道。在 CUDA 12.3 之前你只有两条路：**把整张图重新提交**，或者**在 host 端做分支**（那就引入同步，图省下的启动开销又还回去了）。

**条件节点把分支判断放进了设备端**：

```text
[ evalCondition kernel ]  ──>  设置条件句柄的值（0 或 1，全在 GPU 上）
            │
            ├──> [ If  节点 ]  条件 != 0 时执行（phGraph = bodyTrue）
            └──> [ Else节点 ]  条件 == 0 时执行（phGraph = bodyFalse）
```

**严谨定义与关键 API**（主要门槛是 **Toolkit ≥ 12.3**）：

| API | 作用 |
|---|---|
| `cudaGraphConditionalHandleCreate(&h, graph, defaultVal, flags)` | 在主图里创建**条件句柄**并给默认值 |
| `cudaGraphAddKernelNode(&n, graph, deps, nDeps, &params)` | 普通 kernel 节点（这里是「求值」节点） |
| `cudaGraphAddConditionalNode(&n, graph, deps, nDeps, &cp)` | 添加**条件节点**；`cp.type` = `cudaGraphCondTypeIf` / `...Else` / `...While`，`cp.phGraph` 指向**体图** |
| `cudaGraphAddChildGraphNode(...)` | 把**一整张子图**当成一个节点嵌进主图（复用图结构） |
| 设备端 `cudaGraphSetConditional(handle, value)` | kernel 里设置条件值；`!= 0` 继续/走 If，`== 0` 退出/走 Else |

**三个必须记住的语义细节**：

1. **If 和 Else 是两个节点，共享同一个 handle**，驱动保证只有一个会执行。
2. **循环节点（`cudaGraphCondTypeWhile`）先判断还是先执行，取决于 handle 的初始值**。示例给的是 `1`，意思是「**先进循环体**」；循环体 kernel 每轮结束时重新 `cudaGraphSetConditional`，**条件值变成 0 就退出**。
3. **条件求值必须由设备端完成**。这就是为什么示例里要有一个 `evalCondition` kernel——host 端不参与分支判断，整段控制流**只用一次 `cudaGraphLaunch` 提交**。

**和基础 Graphs 的递进关系**：

| | [第 3 章](ch03_同步通信与原子操作.md) 的基础 Graphs | 本章的进阶 Graphs |
|---|---|---|
| 拓扑 | 静态 DAG | DAG + 条件 + 循环（**含环**） |
| 分支在哪决定 | host（要么重提交，要么同步） | **设备端**（`cudaGraphSetConditional`） |
| 构图方式 | 主要靠**流捕获** | **显式构图**（`cudaGraphAddKernelNode` / `cudaGraphAddConditionalNode`） |
| 省下的是什么 | kernel 之间的启动间隙 | 启动间隙 **+ host 往返 + 重提交开销** |

**显式构图的三个坑**：

1. **`kernelParams` 里放的是「参数地址」**，那些变量必须**活到图被实例化完成**。所以别写临时表达式，要写具名局部变量。
2. **图实例化本身有开销**（几十微秒到毫秒级）。「每次执行拓扑都不一样」的场景**不该用图**。
3. **调试图用 `cudaGraphDebugDotPrint(graph, "graph.dot", 0)` 导出 Graphviz，再用 `dot -Tpng` 看**，这是排查「谁依赖谁」最快的办法。

---

## 2. 代码实战

本节把 8 个示例的**关键代码原样贴出**（与磁盘上的文件一致；为控制篇幅，省略了与知识点无关的计时类、数据初始化与对拍片段）。每个文件都能独立编译运行。

### 2.1 动态并行：`code/lessons/ch06_advanced/dynparallel/dynparallel.cu`

文件头把两个硬性条件写在最显眼处：

```cpp
//  ch06_advanced/dynparallel/dynparallel.cu
//  动态并行（Dynamic Parallelism）：在设备端启动 kernel
//
//  ⚠ 两个硬性条件，缺一不可：
//     1) // 需要 Compute Capability >= 3.5
//     2) 必须用 -rdc=true（可重定位设备代码）编译，否则设备端 <<<>>> 链接不上
//
//  编译: nvcc -O3 -arch=sm_70 -rdc=true dynparallel.cu -o dynparallel
```

设备端不能复用 `cuda_check.h` 的 `CUDA_CHECK`（它内部会 `fprintf` + `exit`），所以文件自己定义了一个只打印的宏：

```cpp
// 设备端错误检查宏：不能在设备代码里直接用 cuda_check.h 的 CUDA_CHECK，
// 它内部会调用 fprintf/exit，语义上更适合 host。设备端打印到 printf 缓冲区即可。
#define DEV_CHECK(msg)                                                             \
    do {                                                                           \
        cudaError_t _e = cudaGetLastError();                                       \
        if (_e != cudaSuccess)                                                     \
            printf("[device %s] %s\n", (msg), cudaGetErrorString(_e));             \
    } while (0)
```

**A. 最基础的设备端启动**：

```cpp
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
```

**B. 递归归约（动态并行最典型的用法）**：

```cpp
constexpr int  LEAF_SIZE = 1024;  // 递归到 <= 1024 个元素就不再往下拆
constexpr int  LEAF_TPB  = 256;
```

树叶节点 `leafSumKernel` 就是一个普通的块内归约（每线程网格步长累加 → 写共享内存 → 树形归约），与[第 5 章](ch05_核心算法模式实战.md)的归约完全一样，这里不重复贴。下面是递归节点：

```cpp
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
```

**C. 设备端 `malloc` / `free`（同样需要 `-rdc=true`）**：`deviceMallocDemo` 在 `threadIdx.x == 0` 里 `cudaMalloc` 出 1024 个 float、填成 `0..1023`、求和后 `cudaFree`，结果 `523776`。注意**设备端 `cudaMalloc` 有配额限制（默认约 8 MB）**，大量分配要小心。

**host 端同样是「优雅降级」**：查一次 `cudaGetDeviceProperties`，`ccMajor < 3` 就打印 `[跳过] 本机 CC = %d.x，动态并行需要 CC >= 3.5。` 然后 `return 0`（**退出码 0，是优雅跳过而不是失败**）。

### 2.2 Tensor Core / WMMA：`code/lessons/ch06_advanced/wmma/wmma_gemm.cu`

```cpp
//  ⚠ // 需要 Compute Capability >= 7.0（Volta / Turing / Ampere / Ada / Hopper）
//    编译时必须指定 >= sm_70，否则 <mma.h> 里的 fragment 无法实例化。
//  编译: nvcc -O3 -arch=sm_70 wmma_gemm.cu -o wmma_gemm
#include "../common/cuda_check.h"
#include <cuda_fp16.h>
#include <mma.h>

// 一次 WMMA 操作的形状：16 x 16 x 16
//   注意：这是 **warp 级别** 的形状，不是块级别，更不是线程级别。
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int M = 512, N = 512, K = 512;   // A: MxK, B: KxN, C: MxN
```

**核函数（全文）**——三步走：`load_matrix_sync` / `mma_sync` / `store_matrix_sync`：

```cpp
//   块组织：blockDim = (32, 4)
//     threadIdx.x 恰好覆盖一个 warp 的 32 条 lane  -> 一个 warp 一个输出 tile
//     threadIdx.y 是"每个块里有几个 warp"
//   为什么 K 方向要循环：一次 mma_sync 只能吃掉 16 个 K。
//   生产代码会先把 A/B 的 tile 用 cp.async 搬进共享内存
//   （见 ch06_advanced/cpasync/cpasync_gemm.cu）。这里先把 WMMA 本身讲清楚。
__global__ void wmmaGemmKernel(const __half* __restrict__ a,
                               const __half* __restrict__ b,
                               float* __restrict__ c) {
    // blockIdx.x 决定"第几个 16 行块"，每个 warp 独占一个 tile
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // 1) 声明 fragment：这一步只是在寄存器里预留空间，不访问内存
    //    matrix_a 行主序；matrix_b 列主序 —— 后者搭配"行主序存储的 B 数组"
    //    使用，等价于读取 B 的转置，这是官方样例的标准写法。
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    //    累加器用 float：即使输入是 FP16，累加也在 FP32 里做。
    //    否则 K 一大精度就崩 —— 这就是"混合精度"这个词的由来。
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 2) 累加器清零（fill_fragment 也是 per-warp 协作的）
    wmma::fill_fragment(c_frag, 0.0f);

    // 3) 沿 K 方向循环，每次吃 16 个
    for (int k = 0; k < K; k += WMMA_K) {
        // load_matrix_sync：把一块数据从内存（或共享内存）搬进 fragment
        //   第三个参数 ldm 是行/列跨度（leading dimension）
        wmma::load_matrix_sync(a_frag, a + (size_t)warpM * WMMA_M * K + k, K);
        wmma::load_matrix_sync(b_frag, b + (size_t)k * N + warpN * WMMA_N, N);

        // mma_sync：一条指令完成 16x16x16 的乘加 D = A*B + C
        //   四个 fragment 的线程分布必须一致 —— WMMA 保证了这一点
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 4) 把累加器写回全局内存
    wmma::store_matrix_sync(c + (size_t)warpM * WMMA_M * N + warpN * WMMA_N,
                            c_frag, N, wmma::mem_row_major);
}
```

**host 端的块配置**是 `dim3 block(32, 4); dim3 grid(M / WMMA_M, N / (WMMA_N * (int)block.y));`，即 grid.x 覆盖 M/16 个行块、grid.y 覆盖 N/(16×block.y) 个列块。误差衡量用**最大绝对误差 + 相对 L1 量级**（`relL1 = maxAbsErr * M * N / (sumAbs + 1e-12)`），因为 FP16 输入 + FP32 累加的结果里很多元素非常接近 0，逐元素相对误差会失去意义。

### 2.3 异步拷贝：`code/lessons/ch06_advanced/cpasync/cpasync_gemm.cu`

```cpp
//  ⚠ // 需要 Compute Capability >= 8.0（Ampere / Ada / Hopper）
//    在更低的架构上 __pipeline_memcpy_async 会退化成普通的同步 ld/st，
//    程序仍然能编过、结果也对，但拿不到任何加速。
//  编译: nvcc -O3 -arch=sm_80 cpasync_gemm.cu -o cpasync_gemm
#include <cuda_pipeline.h>  // __pipeline_memcpy_async / __pipeline_commit / __pipeline_wait_prior

constexpr int BM = 64, BN = 64, BK = 16;   // 分块参数（都是 16 的倍数，方便和 WMMA 对接）
constexpr int TPB = 256;                   // 256 线程 = 16x16，每线程算 4x4
constexpr int STAGES = 2;                  // 双缓冲 = 2 阶段流水线
constexpr int TM = BM / 16, TN = BN / 16;  // 每线程负责的行/列数 = 4
```

**tile 加载函数** `loadATileAsync` / `loadBTileAsync` 反复用下面这段循环，**索引设计保住了合并访存**：

```cpp
#pragma unroll
    for (int i = 0; i < BM * BK / TPB; ++i) {
        const int idx = tid + i * TPB;
        const int r = idx / BK;   // tile 内的行
        const int c = idx % BK;   // tile 内的列
        // 全局地址：A[by*BM + r][k0 + c]
        // 一个 warp 内 tid 连续 -> A 的地址连续 -> 合并访存没有被破坏
        __pipeline_memcpy_async(&As[r][c],
                                &A[(size_t)(by * BM + r) * Kdim + (k0 + c)],
                                sizeof(float));   // cp.async 只支持 4/8/16 字节
    }
```

**双缓冲流水线本体**：

```cpp
//   时间轴（S=共享内存缓冲下标）：
//     k=0:  发起 S0 的 cp.async ── 等待 S0 ── 计算 S0 ── 同时发起 S1
//     k=1:  等待 S1 ── 计算 S1 ── 同时发起 S0(k=2)
//   关键：**发起下一块的拷贝** 和 **计算当前块** 是重叠的。
__global__ void gemmCpAsync(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C) {
    // STAGES 个缓冲：As[s] 是 BM x BK，Bs[s] 是 BK x BN
    __shared__ float As[STAGES][BM][BK];
    __shared__ float Bs[STAGES][BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid % 16;   // 线程在 16x16 网格里的列
    const int ty = tid / 16;   // 行
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // 每线程 4x4 的寄存器累加器 —— 寄存器分块，提升算术强度与 ILP
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    // ---- 预取第 0 块 ----
    loadATileAsync(As[0], A, K, by, 0, tid);
    loadBTileAsync(Bs[0], B, N, bx, 0, tid);
    __pipeline_commit();  // 把上面所有 cp.async 打包成一个"提交组"

    int stage = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int next = k0 + BK;
        if (next < K) {
            // 先把下一块发出去（此刻还没等它，线程立刻继续往下走）
            loadATileAsync(As[stage ^ 1], A, K, by, next, tid);
            loadBTileAsync(Bs[stage ^ 1], B, N, bx, next, tid);
            __pipeline_commit();
            // 允许 1 个提交组未完成 —— 也就是只等"当前这一块"就绪，
            // 刚发出去的那一块继续在后台搬
            __pipeline_wait_prior(1);
        } else {
            // 最后一轮没有新的提交组，等全部完成
            __pipeline_wait_prior(0);
        }

        // __pipeline_wait_prior 只保证**本线程**发起的拷贝完成了，
        // 但计算阶段要读别的线程搬进来的数据，所以还需要一次块级同步
        __syncthreads();

        // ---- 计算当前块 ----
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float aReg[TM], bReg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[stage][ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[stage][kk][tx * TN + j];
            // 外积形式：TM*TN 次 FMA 只需要 TM+TN 次共享内存读
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += aReg[i] * bReg[j];
        }

        // 计算完成前不能让下一轮覆盖这块缓冲
        __syncthreads();
        stage ^= 1;
    }
```

**host 端的提示语义**——注意 CC < 8.0 时**不退出，只提示** `[提示] 本机 CC = %d.%d < 8.0：cp.async 会退化成同步拷贝，程序仍可运行，但看不到流水线带来的收益。`

### 2.4 线程块集群与 DSM：`code/lessons/ch06_advanced/cluster/cluster_dsm.cu`

```cpp
//  ⚠ // 需要 Compute Capability >= 9.0（Hopper / Blackwell）
//    ⚠ 需要 CUDA Toolkit >= 11.8（cudaLaunchKernelEx 与集群启动属性）
//    编译时必须指定 -arch=sm_90
//  编译: nvcc -O3 -arch=sm_90 cluster_dsm.cu -o cluster_dsm
#include <cooperative_groups.h>

#if CUDART_VERSION >= 11080
#  define HAS_CLUSTER_API 1
#endif
namespace cg = cooperative_groups;
constexpr int CLUSTER_BLOCKS = 2;  // 每个集群 2 个块
```

**核函数（全文）**——注意两处 `cluster.sync()` 的用意完全不同：

```cpp
//   每个块先算出自己负责那段的局部和，写进自己的共享内存；
//   然后 cluster.sync() 做一次集群范围的硬件屏障；
//   最后由 rank 0 的块通过 map_shared_rank() 直接读其它块的共享内存求和。
//   注意两处 cluster.sync()：
//     第一处保证"别人的共享内存已经写好了"；
//     第二处（退出前）保证"别人都已经读完了，我的共享内存可以安全销毁"。
__global__ void clusterReduceKernel(const float* __restrict__ in,
                                    float* __restrict__ out, int n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cg::cluster_group cluster = cg::this_cluster();
    extern __shared__ float smem[];  // 至少 CLUSTER_BLOCKS 个 float 就够了

    const int tid = threadIdx.x;
    const unsigned rank = cluster.block_rank();   // 本块在集群内的编号

    // ---- 1. 每块算局部和（网格跨步 + warp shuffle + 共享内存）----
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        v += in[i];

    __shared__ float warpSum[32];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (lane == 0) warpSum[wid] = v;
    __syncthreads();

    const int nwarps = blockDim.x >> 5;
    if (wid == 0) {
        v = (lane < nwarps) ? warpSum[lane] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) smem[rank] = v;  // 每块占一个槽位，便于 rank 0 直接索引
    }

    // ---- 2. 集群屏障：保证所有块的 smem[rank] 都已经写好 ----
    cluster.sync();

    // ---- 3. 由 rank 0 的 0 号线程汇总 ----
    if (rank == 0 && tid == 0) {
        float total = 0.f;
        for (unsigned r = 0; r < cluster.num_blocks(); ++r) {
            // map_shared_rank 把"本块的共享内存地址"翻译成"第 r 个块的同一地址"
            const float* remote = cluster.map_shared_rank(smem, r);
            total += remote[r];
        }
        out[0] = total;
    }

    // ---- 4. 退出前再同步一次，防止别的块还在读我的共享内存 ----
    cluster.sync();
#endif
}
```

**host 端：两级能力检测 + 集群启动属性**。先 `if (prop.major < 9)` 打印「需要 CC >= 9.0」后 `return 0`；再用 `#ifndef HAS_CLUSTER_API` 兜住 Toolkit < 11.8 的情况。真正启动集群靠的是**启动属性**：

```cpp
    // 集群是通过"启动属性"指定的，不需要在核函数上写 __cluster_dims__。
    // 两种写法二选一，不要同时用。
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = nullptr;

    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_BLOCKS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
```

### 2.5 统一内存进阶：`code/lessons/ch06_advanced/unified/unified_advise.cu`

```cpp
//  // 需要 Compute Capability >= 6.0（Pascal 起才有硬件页错误与按需迁移）
//  编译: nvcc -O3 -arch=sm_70 unified_advise.cu -o unified_advise
//
//  cudaMemAdvise 就是提前把"这段数据的行为特征"告诉驱动，
//  让它别每次都靠页错误去猜。四条最常用的建议：
//    cudaMemAdviseSetReadMostly        只读数据，可以在多个处理器上放副本
//    cudaMemAdviseSetPreferredLocation 建议这段数据常驻在哪
//    cudaMemAdviseSetAccessedBy        允许某个设备直接访问（不迁移，走 NVLink/PCIe）
//    cudaMemAdviseUnsetXXX             撤销上面的建议
constexpr int   N      = 1 << 22;  // 4M 个 float = 16 MB

// 读 x、写 y，覆盖 16 MB 的读写，用来放大"页在哪边"的影响
__global__ void saxpy(const float* __restrict__ x, float* __restrict__ y,
                      float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + 1.0f;
}
```

**三个关键属性 + `cudaMemAdvise`**：

```cpp
    int concurrentManaged = 0, pageableAccess = 0, unifiedAddressing = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrentManaged, cudaDevAttrConcurrentManagedAccess, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&pageableAccess, cudaDevAttrPageableMemoryAccess, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&unifiedAddressing, cudaDevAttrUnifiedAddressing, 0));
```

```cpp
    // ---- 关键的 cudaMemAdvise：告诉驱动"x 只读，优先放 GPU 显存" ----
    // 只读标记让驱动可以在 CPU 和 GPU 两侧各留一份副本，避免来回迁移
    CUDA_CHECK(cudaMemAdvise(x, bytes, cudaMemAdviseSetReadMostly, 0));
    CUDA_CHECK(cudaMemAdvise(x, bytes, cudaMemAdviseSetPreferredLocation, 0));
    // y 会被反复写，标记"只能由 GPU 访问"，驱动就不必考虑 CPU 侧的副本
    CUDA_CHECK(cudaMemAdvise(y, bytes, cudaMemAdviseSetPreferredLocation, 0));
    CUDA_CHECK(cudaMemAdvise(y, bytes, cudaMemAdviseSetAccessedBy, 0));
```

**两个场景的对比**——A 靠页错误，B 先预取：

```cpp
    // 场景 A：不做预取，让 GPU 靠页错误按需拉取
    {
        float best = 1e30f;
        for (int r = 0; r < REPEAT; ++r) {
            timer.start();
            saxpy<<<grid, TPB>>>(x, y, 2.0f, N);
            const float ms = timer.stop();
            if (ms < best) best = ms;
        }
        CUDA_CHECK_KERNEL();
        printf("\n[A] 不预取（靠页错误迁移）: %.4f ms（取 %d 次最小值）\n", best, REPEAT);
    }

    // 场景 B：先用 cudaMemPrefetchAsync 把 x、y 都搬到设备 0
    //   prefetch 的语义是"现在就开始搬，但不阻塞调用它的线程"，
    //   所以要在同一个流里发一个事件/同步来确保搬完。
    {
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        float best = 1e30f;
        for (int r = 0; r < REPEAT; ++r) {
            // 每次迭代前先把数据"拉回"到设备侧（模拟真实迭代求解器的节奏）
            CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, 0, s));
            CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, 0, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            timer.start();
            saxpy<<<grid, TPB>>>(x, y, 2.0f, N);
            const float ms = timer.stop();
            if (ms < best) best = ms;
        }
```

### 2.6 虚拟内存管理：`code/lessons/ch06_advanced/vmm/vmm_alloc.cu`

```cpp
//  // 需要 Compute Capability >= 6.0 且驱动/设备支持虚拟地址管理
//  编译: nvcc -O3 -arch=sm_70 vmm_alloc.cu -o vmm_alloc -lcuda
//        （Windows 上写 -lcuda 等价于链接 cuda.lib）
#include <cuda.h>

// 驱动 API 的错误检查宏：多一步"把 CUresult 转成字符串"
#define CU_CHECK(expr)                                                          \
    do {                                                                        \
        CUresult _r = (expr);                                                   \
        if (_r != CUDA_SUCCESS) {                                               \
            const char* _s = nullptr;                                           \
            cuGetErrorString(_r, &_s);                                          \
            fprintf(stderr, "\n[DRIVER ERROR] %s:%d\n  call : %s\n  msg  : %s\n", \
                    __FILE__, __LINE__, #expr, _s ? _s : "(unknown)");          \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
```

**初始化驱动 API、能力检测、查询粒度**：

```cpp
    // ---- 0. 初始化驱动 API 并确保有一个当前上下文 ----
    CU_CHECK(cuInit(0));
    // 用运行时 API 触发主上下文创建；之后驱动 API 的调用都作用在这个上下文上。
    // 混用 runtime 和 driver API 时，这是最省事的做法。
    CUDA_CHECK(cudaFree(nullptr));

    CUdevice dev = 0;
    CU_CHECK(cuDeviceGet(&dev, 0));

    int vmmSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(&vmmSupported,
                                  CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                                  dev));
    if (!vmmSupported) {
        printf("\n[跳过] 本设备不支持虚拟地址管理（需要 CC >= 6.0 且驱动 >= 440）。\n");
        return 0;
    }

    // ---- 1. 查询最小分配粒度（通常是 2 MB）----
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;      // 显存必须用 PINNED 类型
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    size_t granularity = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM));
```

**核心四步 + 释放顺序**：

```cpp
    // ---- 2. 预留虚拟地址（此时还没有物理内存）----
    CUdeviceptr va = 0;
    CU_CHECK(cuMemAddressReserve(&va, totalBytes, 0 /*对齐要求*/, 0 /*建议地址*/, 0));

    // ---- 3. 创建两块独立的物理分配，映射到虚拟地址的两个偏移上 ----
    CUmemGenericAllocationHandle handles[2] = {};
    for (int i = 0; i < 2; ++i) {
        CU_CHECK(cuMemCreate(&handles[i], segBytes, &prop, 0));
        CU_CHECK(cuMemMap(va + i * segBytes, segBytes, 0, handles[i], 0));
    }

    // ---- 4. 授予访问权限（不设置的话 kernel 一碰就崩）----
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(va, totalBytes, &access, 1));

    // ---- 5. 像用普通指针一样用它 ----
    float* d_ptr = reinterpret_cast<float*>(va);
```

```cpp
    // ---- 6. 释放：顺序必须和创建时相反 ----
    CU_CHECK(cuMemUnmap(va, segBytes));            // 先解映射
    CU_CHECK(cuMemUnmap(va + segBytes, segBytes));
    CU_CHECK(cuMemRelease(handles[0]));            // 再释放物理分配
    CU_CHECK(cuMemRelease(handles[1]));
    CU_CHECK(cuMemAddressFree(va, totalBytes));    // 最后归还虚拟地址
```

**两条校验**：全段取值正确（前半 1.0、后半 2.0），以及**跨越两块物理分配的边界必须是突变**：`host[halfFloats - 1] == 1.0f && host[halfFloats] == 2.0f`。

### 2.7 多 GPU P2P：`code/lessons/ch06_advanced/multigpu/p2p.cu`

```cpp
//  // 需要至少 2 张支持 CUDA 的显卡；单卡机器上程序会打印说明后正常退出
//  两张卡之间搬数据本来有两条路：
//    A. 走主机：GPU0 -> 固定内存 -> CPU -> 固定内存 -> GPU1（两次 PCIe 往返）
//    B. 走 P2P：GPU0 -> PCIe/NVLink -> GPU1（一次直达）
//  开了 cudaDeviceEnablePeerAccess 之后，kernel 还能**直接解引用**另一张卡的指针。
constexpr size_t N = 1 << 24;  // 64 MB 的 float 数据

__global__ void scale(float* p, float a, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] *= a;
}
```

**枚举设备 + 单卡优雅退出 + 打印 P2P 能力矩阵**。枚举时打印 `unifiedAddressing`（P2P 依赖 UVA）；`nDev < 2` 就打印「只有 1 张卡，无法演示 P2P」并附上判定与开启流程后 `return 0`：

```cpp
    printf("\n[拓扑] cudaDeviceCanAccessPeer 能力矩阵（行=源，列=目的）\n     ");
    for (int j = 0; j < nDev; ++j) printf("  to%d", j);
    printf("\n");
    for (int i = 0; i < nDev; ++i) {
        printf("from%d", i);
        for (int j = 0; j < nDev; ++j) {
            if (i == j) { printf("    -"); continue; }
            int ok = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&ok, i, j));
            printf("   %s", ok ? "Y" : "n");
        }
        printf("\n");
    }

    int can01 = 0, can10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    const bool p2pOK = (can01 != 0) && (can10 != 0);
```

**开启双向 P2P + 两条搬运路径的对比**：

```cpp
    if (p2pOK) {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
        printf("\n[开启] 已开启 dev0 <-> dev1 的双向 P2P 访问\n");
    } else {
        printf("\n[提示] 本机两张卡之间不支持 P2P，下面只演示走主机的搬运路径。\n");
    }
```

```cpp
    // 路径 A：经主机中转（D2H + H2D）
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(staging.data(), d0, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(d1, staging.data(), bytes, cudaMemcpyHostToDevice));
    // 路径 B：P2P 直达
    CUDA_CHECK(cudaMemcpyPeer(d1, 1, d0, 0, bytes));
```

**在 dev0 上跑 kernel 直接改 dev1 的显存**：

```cpp
        // 这个 kernel 在 dev0 上运行，但读写的是 dev1 的指针。
        // 前提是 dev0 已经通过 cudaDeviceEnablePeerAccess(1) 打开了到 dev1 的访问。
        CUDA_CHECK(cudaSetDevice(0));
        scale<<<grid, 256>>>(d1, 2.0f, N);  // d1 是 dev1 的指针！
```

### 2.8 CUDA Graphs 进阶：`code/lessons/ch06_advanced/graphs_advanced/graph_advanced.cu`

```cpp
//  // 基础图捕获见 ch03（流与并发）；这里讲的是 12.3 之后才有的控制流节点
//  编译: nvcc -O3 -arch=sm_70 graph_advanced.cu -o graph_advanced
//
//  为什么需要条件节点？
//    普通 CUDA Graph 是一张**静态**的 DAG：节点和依赖在实例化时就固定了。
//    条件节点把**分支判断放进了设备端**：
//      一个 kernel 用 cudaGraphSetConditional() 设置条件值，
//      图里挂着的 If / Else 两个节点根据这个值决定谁执行 —— 全程不回 host。

// 三个小 kernel，用来串成一张图；我们要观察的是"启动 3 个 kernel"和
// "提交 1 张图"的开销差别。
__global__ void initKernel(float* d, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 1.0f;
}

__global__ void scaleKernel(float* d, int n, float a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= a;
}
```

**设备端控制流的三个 kernel**（`#if CUDART_VERSION >= 12030` 做编译期保护）：

```cpp
//   cudaGraphSetConditional 由 host 端在实例化时注入 —— 你只要把它当普通函数调用。
//   ⚠ 需要 CUDA >= 12.3
#if CUDART_VERSION >= 12030
__global__ void evalCondition(cudaGraphConditionalHandle handle, const float* data,
                              float threshold) {
    // 只要有一个线程设置就够了；这里让 block 0 的 thread 0 负责
    if (blockIdx.x == 0 && threadIdx.x == 0)
        cudaGraphSetConditional(handle, (data[0] < threshold) ? 1u : 0u);
}

__global__ void branchKernel(float* data, float value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) data[0] = value;
}

// 循环节点：每一轮把计数器加一，并重新设置条件值
//   条件值 != 0 → 继续循环；条件值 == 0 → 退出循环
__global__ void loopBodyKernel(cudaGraphConditionalHandle handle, int* counter,
                               int limit) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int c = ++counter[0];
        cudaGraphSetConditional(handle, (c < limit) ? 1u : 0u);
    }
}
#endif
```

**显式构图**——关键点是 `kernelParams` 里放的是**参数地址**，那些变量必须活到图被实例化完成：

```cpp
        // 注意：kernelParams 里放的是**参数地址**，
        // 这些局部变量必须活到 cudaGraphAddKernelNode 返回之后、
        // 直到图被实例化完成。所以别用临时表达式。
        float a2 = 2.0f, a3 = 3.0f;
        int n = N;

        cudaKernelNodeParams kp = {};
        kp.sharedMemBytes = 0;
        kp.gridDim = dim3(grid);
        kp.blockDim = dim3(TPB);

        void* argsInit[] = {&d, &n};
        kp.func = (void*)initKernel;
        kp.kernelParams = argsInit;
        CUDA_CHECK(cudaGraphAddKernelNode(&nInit, graph, nullptr, 0, &kp));

        void* argsS1[] = {&d, &n, &a2};
        kp.func = (void*)scaleKernel;
        kp.kernelParams = argsS1;
        CUDA_CHECK(cudaGraphAddKernelNode(&nS1, graph, &nInit, 1, &kp));
```

**条件节点（If / Else）的构造**：

```cpp
        // ---- 主图：条件求值节点 + If 节点 + Else 节点 ----
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaGraphCreate(&graph, 0));

        // 条件句柄：默认值 0，表示"默认走 Else 分支"
        cudaGraphConditionalHandle handle = 0;
        CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0,
                                                    cudaGraphCondAssignDefault));

        cudaKernelNodeParams ekp = {};
        ekp.sharedMemBytes = 0;
        ekp.gridDim = dim3(1);
        ekp.blockDim = dim3(32);
        ekp.func = (void*)evalCondition;
        void* argsEval[] = {&handle, &dval, &threshold};
        ekp.kernelParams = argsEval;
        cudaGraphNode_t nEval = nullptr;
        CUDA_CHECK(cudaGraphAddKernelNode(&nEval, graph, nullptr, 0, &ekp));

        cudaGraphConditionalNodeParams cp = {};
        cp.handle = handle;
        cp.size = 1;

        cp.type = cudaGraphCondTypeIf;
        cp.phGraph = bodyTrue;
        cudaGraphNode_t nIf = nullptr;
        CUDA_CHECK(cudaGraphAddConditionalNode(&nIf, graph, &nEval, 1, &cp));

        cp.type = cudaGraphCondTypeElse;
        cp.phGraph = bodyFalse;
        cudaGraphNode_t nElse = nullptr;
        CUDA_CHECK(cudaGraphAddConditionalNode(&nElse, graph, &nEval, 1, &cp));
```

**循环节点（While）**——注意初始值给 `1` 表示「先进循环体」：

```cpp
        // 循环节点同样需要一个条件句柄；默认值给 1，表示"先进循环体"
        cudaGraphConditionalHandle handle = 0;
        CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1,
                                                    cudaGraphCondAssignDefault));
```

```cpp
        cudaGraphConditionalNodeParams cp = {};
        cp.handle = handle;
        cp.size = 1;
        cp.type = cudaGraphCondTypeWhile;
        cp.phGraph = body;

        cudaGraphNode_t nWhile = nullptr;
        CUDA_CHECK(cudaGraphAddConditionalNode(&nWhile, graph, nullptr, 0, &cp));
```

---

## 3. 代码逐行解析

这一节按**功能块**解释「为什么这么写」。请对照第 2 节的代码阅读。

### 3.1 动态并行：为什么是「谁启动、谁同步」

**① 为什么 `DEV_CHECK` 不能复用 `CUDA_CHECK`？** `CUDA_CHECK` 会调用 `fprintf` 然后 **`exit(EXIT_FAILURE)`**。设备代码里既不能 `exit`，也不该终止整个进程——一个子 kernel 启动失败，父 kernel 还可能有别的活要干。所以设备端只用 `printf`（输出进 kernel 的 printf 缓冲区，host 同步后刷出）**记录**错误，不做控制流终止。

**② 为什么只有 `block 0` 的 `thread 0` 发起子 kernel？** 因为**每个线程都能独立发起 kernel**。若 32 个线程各发一遍，你就得到 32×4 = 128 次启动而不是 4 次；在递归场景里这会导致**启动次数指数爆炸**，直接把待启动队列塞满。经验法则：**发起子 kernel 的线程数要刻意压到最少**。

**③ 设备端 `cudaDeviceSynchronize()` 之后为什么还要 `__syncthreads()`？**

```cpp
cudaError_t e = cudaDeviceSynchronize();
if (e != cudaSuccess) printf("[parent] sync failed: %s\n", cudaGetErrorString(e));
}
__syncthreads();  // 让块内其它线程跟着一起等
```

`cudaDeviceSynchronize()` 只保证「**这个线程**发起的子 kernel 完成了」。块里其它 31 个线程没被它拦住，可能立刻冲到下面去读 `counter`。**`__syncthreads()` 才是块级栅栏**——这是「设备端同步不是块间屏障」这条语义的直接后果。

**④ 递归为什么要分「递归节点」和「叶子节点」两层？** 如果一路递归到 `n == 1`，启动次数是 O(n)，1M 个元素就是上百万次设备端启动——**每次都是 host 启动 2～5 倍的代价**，程序会慢到无法运行。所以叶子阈值 `LEAF_SIZE = 1024`：到这个小规模就不再拆，用一个普通的块内归约（`leafSumKernel`）算完；启动次数降到 **O(N / LEAF_SIZE)** 量级（1M / 1024 = 1024 个叶子加上约同量级的内部节点）。**阈值怎么选，本质上是「启动开销」与「并行粒度」的权衡**：太小 → 启动爆炸；太大 → 并行度不足、负载不均。

**⑤ 输出槽位的设计**：

```cpp
recursiveSumKernel<<<1, 128>>>(in, out + 1, half);
recursiveSumKernel<<<1, 128>>>(in + half, out + 2, rest);
cudaDeviceSynchronize();
out[0] = out[1] + out[2];
```

父节点给左子节点 `out + 1`、右子节点 `out + 2`，子结果不会互相覆盖；父节点**同步完成后**才写 `out[0]`。**如果没先同步就写 `out[0]`，你会读到子 kernel 还没写完的垃圾**——而且因为 GPU 调度是异步的，这个 bug **时对时错**，是最难查的一类。host 端把 `d_out` 分配成 `4096 * sizeof(float)` 是刻意**多留余量**，避免为了「算准容量」反复调试。

**⑥ 设备端 `cudaMalloc` 的配额限制**：设备端 `malloc` / `free` 走的是**设备运行时堆（device runtime heap）**，默认大约 **8 MB**，分配会串行化、代价远高于 host 端。可以用 `cudaDeviceSetLimit(cudaLimitMallocHeapSize, bytes)` 调整上限，但**更好的做法是根本不在设备端分配**：host 端一次性分配好、把指针传进来，让子 kernel 各用各的切片。

### 3.2 WMMA：三个 fragment 与三步走

**① 为什么 `matrix_b` 用 `col_major`？** 这是最容易绕晕但必须想通的地方。数学上要算 `C = A × B`，其中 B 是 K×N 的**行主序**数组（和 `b + k * N + n` 的寻址一致）。而 WMMA 的 `matrix_b` 片段期望的是「按列主序解释的一块 K×N 矩阵」。`load_matrix_sync` 在 `col_major` 下的语义是：**「我要读的是这块矩阵的转置」**。于是「把行主序的 B 按 col_major 读进来 ≡ 读进了 Bᵀ」，配上 `row_major` 的 A，恰好给出官方样例的标准写法。**实践建议**：不要试图在脑子里模拟硬件布局，**照抄官方样例的 `row_major × col_major` 组合**，然后用 CPU 对拍验证。这也是本示例带完整 CPU 参考实现的原因。

**② `fragment` 声明为什么「不访问内存」？** `fragment` 就是一个**编译期确定大小的寄存器数组**。声明它只是分配寄存器；`fill_fragment` 把它清零。这两步都不碰内存，真正的数据流动只有两个口：`load_matrix_sync`（进）和 `store_matrix_sync`（出）。

**③ `ldm` 为什么是 `K` 和 `N`，而不是 `WMMA_K` / `WMMA_N`？** 因为我们要取的是**大矩阵 A（M×K）里的一个 16×16 子块**。子块的第 1 行和第 2 行之间，在内存里隔了 **K 个元素**（整行长度），不是 16 个。`ldm` 描述的是「**在源矩阵里怎么走到下一行/列**」，和子块宽度无关。`store_matrix_sync` 的第四参数是结果的存储布局 `wmma::mem_row_major`，因为累加器 fragment 本身没有布局属性——布局要在这里指定。

**④ 线程/块的映射为什么是 `(32, 4)`？**

```cpp
const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
const int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
```

`blockDim = (32, 4)`：`threadIdx.x` 恰好覆盖**一个 warp 的 32 条 lane**，`threadIdx.y` 是「块里有几个 warp」。这样 `(blockIdx.x * 32 + threadIdx.x) / 32` 把「同一 warp 的 32 个线程」映射成**同一个 `warpM`**——它们会一起去处理同一个 16×16 输出 tile，这正是 WMMA 的要求。换个 `blockDim.x` 也行，但必须保证同一 warp 内所有线程算出相同的 `warpM`/`warpN`，否则 warp 内会 `load` 到不同位置，`mma_sync` 的结果完全错误。

**⑤ 为什么 K 方向必须循环？** 一次 `mma_sync` 的内积深度只有 `WMMA_K = 16`（硬件决定）。K = 512 需要 32 次迭代，每次两组 `load_matrix_sync` + 一次 `mma_sync`。**这 32 次全局内存读取就是本示例的性能天花板**——生产代码会先把 A/B 的 tile 用 `cp.async` 搬进共享内存，再从共享内存 `load_matrix_sync`（共享内存的 ldm 用 tile 宽度），全局访存量能降一个数量级。这就把 3.2 和 3.3 连接起来了。

**⑥ 误差指标为什么不用逐元素相对误差？** 因为 C 的元素是「~512 个正负随机数的乘积之和」，**很多元素会非常接近 0**。此时逐元素相对误差会因为分母趋零而爆炸，完全失去意义。正确做法是**最大绝对误差** + **相对 L1 量级**：`relL1 = maxAbsErr * M * N / (sumAbs + 1e-12)`；分母加 `1e-12` 防止除零。

### 3.3 `cp.async`：为什么顺序是「先发下一块、再等当前块」

**① 为什么拷贝函数的索引长这样？**

```cpp
const int idx = tid + i * TPB;
const int r = idx / BK;   // tile 内的行
const int c = idx % BK;   // tile 内的列
```

`idx = tid + i * TPB` 保证了**在一个 warp 内、对固定的 `i`，`tid` 是连续的**。由于 `BK = 16`，连续的 32 个 `tid` 覆盖「2 行 × 16 列」，映射回全局是 `A[(by*BM + r) * Kdim + k0 + c]`——`c` 连续 → **全局地址连续** → 合并访存成立。**任何异步拷贝优化的第一前提仍是合并访存**（[第 2 章](ch02_内存层次与访存优化.md)）。

**② `sizeof(float)` = 4 字节，为什么不用 16？** `cp.async` **只支持 4 / 8 / 16 字节**。用 16 字节（`float4`）能把指令数降到 1/4，这是生产代码的标准做法，但前提是：全局地址 **16 字节对齐**（数组起始 + 每行长度都是 4 的倍数，`BK = 16` 满足）、共享内存目标也 16 字节对齐（`As[s][r][c]` 中 `c` 是 4 的倍数）。本示例为把流水线结构讲清楚用了 4 字节粒度，练习 2 会让你改成 16 字节。

**③ 流水线时间轴——整段代码的核心**：

```cpp
    // 预取第 0 块
    loadATileAsync(As[0], A, K, by, 0, tid);
    loadBTileAsync(Bs[0], B, N, bx, 0, tid);
    __pipeline_commit();                  // 提交组 #0
    int stage = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int next = k0 + BK;
        if (next < K) {
            loadATileAsync(As[stage ^ 1], A, K, by, next, tid);   // 发起下一块（不等待）
            loadBTileAsync(Bs[stage ^ 1], B, N, bx, next, tid);
            __pipeline_commit();          // 提交组 #1
            __pipeline_wait_prior(1);     // 只等到 #0 完成
        } else {
            __pipeline_wait_prior(0);     // 最后一轮等全部
        }
        __syncthreads();
        /* 计算 As[stage] / Bs[stage] */
        __syncthreads();
        stage ^= 1;
    }
```

逐点解释：

- **预取第 0 块**：不在循环外先把第 0 块发出去，第一次迭代就无法与计算重叠。
- **先发下一块、再 `wait_prior(1)`**：顺序**绝对不能反**。如果先 `wait_prior(0)`（等全部完成）再发下一块，就退化成串行——`cp.async` 的好处全部消失。这是「双缓冲」与「伪双缓冲」的分水岭。
- **`stage ^ 1`**：在 0 和 1 之间翻转，这就是双缓冲。若写 `STAGES = 3`，要改成 `(stage + 1) % STAGES`，并把 `wait_prior(1)` 改成 `wait_prior(STAGES - 1)`。
- **最后一个 `__syncthreads()`**：保证**所有线程都算完了**当前缓冲，下一轮才没人一边算、另一边把新数据覆盖进去。少这一个，你会得到**时对时错**的结果。

**④ `__pipeline_wait_prior` 之后为什么还要 `__syncthreads()`？** 因为 `wait_prior` 是**线程私有**的语义（它等的是「本线程提交的那些组」），而计算阶段每个线程读的 `As[stage][ty*TM + i][kk]` 是**别的线程**搬进来的。没有 `__syncthreads()`，你就是在读还没写完的共享内存——**这是一个只在某些调度下才复现的竞态**，用 `compute-sanitizer --tool racecheck` 才容易抓。

**⑤ 为什么用 `acc[TM][TN]` 的寄存器分块？** 这是**外积形式**：读 `TM` 个 A、`TN` 个 B，做 `TM × TN` 次 FMA，**共享内存读取次数从 TM×TN 降到 TM+TN**（4×4 时 16 次降到 8 次）。这就是「算术强度」和「指令级并行（ILP）」的来源，也是第 5 章、第 8 章反复出现的手法。**`cp.async` 负责把数据搬得快，寄存器分块负责把数据用得好——两者不能互相替代。**

### 3.4 集群与 DSM：`map_shared_rank` 的三个要点

**① 编译期保护为什么有两层？**

```cpp
#if CUDART_VERSION >= 11080
#  define HAS_CLUSTER_API 1
#endif
```

```cpp
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cg::cluster_group cluster = cg::this_cluster();
```

- 外层 `CUDART_VERSION` 是**主机侧编译期**判断：Toolkit < 11.8 根本没有 `cudaLaunchKernelEx` 和集群启动属性，主机代码编不过。
- 内层 `__CUDA_ARCH__` 是**设备侧编译期**判断：`nvcc` 会为 `-arch` 指定的每个架构各编译一遍设备代码，低架构那一遍不能出现 `this_cluster()`。

**两层都要有**——只写 `__CUDA_ARCH__` 会在旧 Toolkit 上编不过，只写 `CUDART_VERSION` 会在 `-arch=sm_70` 时报「cluster API 不可用」。这是写可移植 GPU 库的标准姿势。

**② `map_shared_rank` 到底做了什么？**

```cpp
const float* remote = cluster.map_shared_rank(smem, r);
total += remote[r];
```

`smem` 是**动态共享内存的起始地址**（用 `extern __shared__` 声明）。`map_shared_rank(smem, r)` 返回「**第 r 个块的 `smem` 起始地址**」在当前上下文里的一个可用地址；之后做普通下标访问，硬件就把请求路由到第 r 个块的共享内存。两个要点：

- **地址是「相对的」**：你必须传「本块共享内存里的某个地址」，映射结果才是「第 r 个块的同一个地址」。不同块里 `smem` 的虚拟地址通常相同，但**不要依赖这一点**——永远用 `map_shared_rank` 转换。
- **每块写自己的槽位**：`smem[rank] = v`，这样任何块都能通过 `remote[r]` 精确读到第 r 块的值。示例选 `CLUSTER_BLOCKS = 2`，两块刚好各占一个槽且不冲突。

**③ 为什么退出前必须再 `cluster.sync()`？** 块结束（exit）时，它占用的共享内存会被释放给后续的块使用。而 rank 0 正在用 `map_shared_rank` 读其它块的共享内存——**如果别的块已经退出、它们的共享内存被重新分配，rank 0 读到的是别人的数据**。第二个 `cluster.sync()` 保证「**所有块都读完了，才可以退**」。少了它，程序**大多数时候是对的**——所以更难查。

**④ `gridDim` 必须是集群大小的整数倍**：64 / 2 = 32 个集群，整除；写 65 个块启动会直接失败。这是驱动层在**启动属性**上做的校验，运行时无法自动修补。生产代码里通常用 `grid = clusters * clusterSize` 显式构造。

### 3.5 统一内存进阶：能力检测与建议的正确用法

**① 为什么查三个属性而不是只查 CC？** 因为**能力比架构号更具体**：

- `ConcurrentManagedAccess`：能否在 GPU kernel 执行的同时让 CPU 访问托管内存（页错误与按需迁移的前提）；
- `PageableMemoryAccess`：能否直接访问**可分页**主机内存（Grace Hopper 这类统一内存平台才有）；
- `UnifiedAddressing`：CPU 和 GPU 是否共享同一套虚拟地址（UVA）。

这三个属性决定的是**行为**，而不是「能不能跑」。写库时按属性分支，比按 `CC >= 6` 分支更可靠。

**② `SetReadMostly` 和 `SetPreferredLocation` 能叠用吗？** 能，而且**经常应该叠用**。前者管**副本策略**（只读 → 可在多处留副本，多卡场景收益最大），后者管**首选位置**（优先放在设备 0）。两者不冲突。

**③ `y` 为什么要同时设 `PreferredLocation` 和 `AccessedBy`？** 因为 `y` 是**反复被写**的输出数组：`PreferredLocation` 让它常驻设备 0（写的时候不用先迁页），`AccessedBy` 允许设备 0 直接访问（需要时不必整段搬）。**关键：`cudaMemAdvise` 只是「建议」，不是保证**——驱动有权忽略它。所以它**不能替代正确性检查**，只能改善性能。

**④ 场景 A / B 的对比设计为什么有意义，又有什么陷阱？** 注意 B 里**每次迭代都重新预取**，而且计时**不包含预取与同步**：

```cpp
CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, 0, s));
CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, 0, s));
CUDA_CHECK(cudaStreamSynchronize(s));
timer.start();
saxpy<<<grid, TPB>>>(x, y, 2.0f, N);
```

这模拟的是「迭代型算法每轮开始前保证数据在设备侧」的真实节奏。**这里有一个必须诚实说明的陷阱**：由于 B 的计时只包住 kernel、把预取开销排除在外，**B 看起来永远比 A 快**。但若把预取时间也算进去，在「数据本来就在设备侧」的场景里预取几乎不花时间（没有页要搬）；而在「数据刚从 CPU 端改过」的场景里，预取的成本就是页迁移的成本——**A 和 B 只是把这个成本记在了不同的账上**。**做基准测试时，一定要说清楚「计时区间的边界」**，否则你测的其实是两个不同的东西。这是[第 4 章](ch04_性能分析与优化方法论.md)反复强调的原则。

### 3.6 VMM：四步流程的每一步为什么存在

**① 为什么必须先 `cuInit(0)` 并建立上下文？**

```cpp
CU_CHECK(cuInit(0));
CUDA_CHECK(cudaFree(nullptr));   // 触发主上下文创建
```

驱动 API 是**上下文（context）相关**的。`cuMemCreate` 要知道「为哪个设备、在哪个上下文里」分配；而运行时 API 会**自动创建并管理主上下文**。混用两套 API 时，最省事的做法就是先用一句 `cudaFree(nullptr)`（一个无害的运行时调用）把上下文创建出来。

**② `CUmemAllocationProp` 每一项的含义**：`type = CU_MEM_ALLOCATION_TYPE_PINNED` 表示「固定物理位置」（显存），`SHARED` 才是主机侧共享内存；`location.type / id` 指明这块分配属于哪个设备；`requestedHandleTypes` 是是否需要可导出的句柄类型（跨进程共享时才需要，这里为 `NONE`）。**同一份 `prop` 既用来查询粒度、又用来创建分配**——两者必须一致，否则 `cuMemCreate` 会失败。

**③ `cuMemAddressReserve` 的四个参数**：

```cpp
CU_CHECK(cuMemAddressReserve(&va, totalBytes, 0 /*对齐要求*/, 0 /*建议地址*/, 0));
```

要预留多少字节、对齐要求（0 = 用默认粒度对齐）、建议的虚拟地址（0 = 让驱动自己挑）、标志位。**这一步只花虚拟地址空间，不花一个字节显存**——这就是它能做「按需增长的显存池」的原因。

**④ 为什么两块物理分配能拼成一段连续地址？** 两次 `cuMemCreate` 是**两次完全独立的物理分配**，在显存里可能相隔很远；但两次 `cuMemMap` 把它们映射到**相邻的虚拟地址区间**上。于是对 kernel 来说，`reinterpret_cast<float*>(va)` 就是一个长度 `2 * segBytes` 的连续数组。**边界校验就是为了证明这件事**：`host[halfFloats - 1] == 1.0f && host[halfFloats] == 2.0f`——如果映射错位，边界处不会是「1.0 → 2.0」的突变。**永远用数据证明你的内存布局是对的**，不要靠「应该没问题」。

**⑤ 为什么 `cuMemSetAccess` 不能省？** `cuMemMap` 只建立了「虚拟地址 ↔ 物理 handle」的映射关系，**并没有授予任何设备访问权限**。不调 `cuMemSetAccess`，kernel 一碰这块地址就是非法访问。

**⑥ 释放顺序为什么必须反着来？** 先 `cuMemUnmap`（**没解映射就释放物理 handle 是非法的**），再 `cuMemRelease`（引用计数减一，归还物理显存），最后 `cuMemAddressFree`（归还虚拟地址空间）。**注意释放粒度**：示例里 `cuMemUnmap(va, segBytes)` 和 `cuMemUnmap(va + segBytes, segBytes)` 是**分开解映射的**，因为它们是两次独立的 `cuMemMap`——**释放必须与创建一一对应，粒度也要一致**。

### 3.7 多 GPU：P2P 的判定、开启与三条路径

**① 为什么 device 枚举要打印 `unifiedAddressing`？** 因为 **P2P 访问依赖 UVA**。如果某张卡/某个进程不是 UVA（比如 32 位进程），P2P 就不可能开启。**先看这个标志，再看能力矩阵**，能省掉很多困惑。

**② `cudaDeviceCanAccessPeer` 返回 0 不是错误**：

```cpp
int ok = 0;
CUDA_CHECK(cudaDeviceCanAccessPeer(&ok, i, j));
printf("   %s", ok ? "Y" : "n");
```

它的返回码永远是 `cudaSuccess`（除非参数非法），**结果写在 `ok` 里**：1 = 可以，0 = 不可以。**不要用 `CUDA_CHECK` 判断「能不能 P2P」**，那是判断「调用成不成功」。

**③ 为什么每张卡都要单独 `cudaSetDevice`？** `cudaDeviceEnablePeerAccess(peer, flags)` 的语义是「**让当前设备能够访问 peer 设备**」。所以「双向 P2P」需要**两次调用**，每次调用前都要把当前设备切对；第二参数 `flags` 目前必须为 0。

**④ 两条搬运路径为什么带宽差不多、耗时差很多？** 路径 A（D2H + H2D）搬了 **2 × bytes**，而且 **host 内存是瓶颈**（普通可分页内存要走 bounce buffer，所以示例打印的是 `2.0 * bytes / ms` 的「等效带宽」）；路径 B 只搬 **1 × bytes**，且全程不经 host。所以即使**单次传输的带宽相近**，B 的总耗时也大约是 A 的一半——示例里打印的 `提速约 %.2fx` 就是这个意思。**注意这个倍数在不同拓扑下差别很大**（NVLink 上会远高于 PCIe），不要把示例数字当普适结论。

**⑤ 远端指针为什么能直接用？** 在 UVA 下，`d1` 这个指针在**整个进程范围内唯一标识一块显存**，与「当前设备」无关。`cudaDeviceEnablePeerAccess(1)` 打开了「从 dev0 访问 dev1」的通路后，dev0 上运行的 kernel 就能直接读写 `d1`。**但这种访问的延迟远高于本地显存**（要走 PCIe 或 NVLink），适合「偶尔访问、数据量不大」的场景。**大规模数据传输仍应用 `cudaMemcpyPeer`**——拷贝引擎（copy engine）能跑满链路，而 kernel 里的零散远端访问会互相干扰。

**⑥ NCCL 为什么不能自己写？** 集合通信要处理的复杂度包括：**传输层选择**（NVLink、PCIe P2P、共享内存、InfiniBand / RoCE）、**算法选择**（ring 带宽最优 / tree 延迟最优，还有递归倍增、分段 ring）、**拓扑感知**（多级交换机、NUMA 亲和性、多 rail 网络）、**启动与容错**（跨进程/跨机的 rendezvous、超时）。自己写一个能用的 AllReduce 很快，但**写一个在任意拓扑下都比 NCCL 快的**几乎不可能。这也是[第 7 章](ch07_生态库与工程化.md)存在的意义。

### 3.8 Graphs 进阶：条件句柄的三个语义细节

**① `cudaGraphConditionalHandleCreate` 的第三个参数是「默认值」**：

```cpp
// If/Else 场景：默认值 0，表示"默认走 Else 分支"
CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));
// While 场景：默认值 1，表示"先进循环体"
CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));
```

默认值决定「**求值 kernel 还没跑之前**，图按哪条路走」。对 If/Else 来说条件求值节点**先于**分支节点执行，所以默认值通常无关紧要（求值会覆盖它）；但对 While 来说，**初始值决定要不要进入循环体**——给 0 就意味着「一次都不执行」。`cudaGraphCondAssignDefault` 这个 flag 表示「**允许设备端覆盖默认值**」。

**② 求值 kernel 为什么只需要一个线程？**

```cpp
if (blockIdx.x == 0 && threadIdx.x == 0)
    cudaGraphSetConditional(handle, (data[0] < threshold) ? 1u : 0u);
```

因为**条件句柄是整张图共享的一个标量**，多个线程同时写它会产生**竞态**（最后谁赢不确定）。所以约定：**只让一个线程写**。如果条件真的需要并行归约（比如「残差范数 > 阈值」），正确做法是**先用一个 kernel 把归约结果写到一个标量，再用另一个只有一个线程的 kernel 去 `cudaGraphSetConditional`**——或者让那个 kernel 在 `__syncthreads()` 之后由 0 号线程写。

**③ If 和 Else 是两个节点，共享一个 handle**：

```cpp
cp.type = cudaGraphCondTypeIf;      cp.phGraph = bodyTrue;   → nIf
cp.type = cudaGraphCondTypeElse;    cp.phGraph = bodyFalse;  → nElse
CUDA_CHECK(cudaGraphAddConditionalNode(&nIf,   graph, &nEval, 1, &cp));
CUDA_CHECK(cudaGraphAddConditionalNode(&nElse, graph, &nEval, 1, &cp));
```

**注意示例里两次调用的 `cp` 是同一个结构体，只改了 `type` 和 `phGraph`**。这之所以安全，是因为 `cudaGraphAddConditionalNode` 会**拷贝**参数结构体的内容；但 `phGraph` 指向的 `bodyTrue` / `bodyFalse` 两张子图**必须活到主图被销毁为止**——这就是示例最后要 `cudaGraphDestroy(bodyTrue)` 和 `bodyFalse` 的原因。

**④ 循环节点：`cp.size` 和「谁负责退出」**：`size = 1` 表示条件句柄是**标量**（一个 `unsigned`）而不是数组。**退出循环的责任在循环体 kernel 里**：

```cpp
const int c = ++counter[0];
cudaGraphSetConditional(handle, (c < limit) ? 1u : 0u);
```

条件值 != 0 → 继续；== 0 → 退出。**如果循环体永远不把条件设成 0，这张图就会永远跑下去**——GPU 一直忙，host 的 `cudaDeviceSynchronize` 不返回。所以**必须给循环设硬上限**（这里的 `limit` 就是硬上限）。

**⑤ 和流捕获（第 3 章）的关系**：显式构图和流捕获**可以混用**——用 `cudaStreamBeginCapture` 录下线性部分，用显式 API 添加控制流节点；但**不要在有条件节点的图上再做流捕获**，捕获假设的是静态拓扑。调试图的万能工具是 `cudaGraphDebugDotPrint(graph, "graph.dot", 0)` 加 `dot -Tpng graph.dot -o graph.png`。

---

## 4. 编译与运行

### 4.1 环境确认

```bash
nvcc --version
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap --format=csv   # 查计算能力
nvidia-smi topo -m                                     # 多卡机器：看 PCIe / NVLink 拓扑
```

### 4.2 每个示例的确切编译命令

**本章有 4 个示例需要特殊参数，缺一个都编不过或跑不出效果。**

```bash
# ① 动态并行 —— 必须 -rdc=true
cd code/lessons/ch06_advanced/dynparallel
nvcc -O3 -arch=sm_70 -rdc=true dynparallel.cu -o dynparallel
# 若报 undefined reference to '__cudaRegisterLinkedBinary...'，补 -lcudadevrt：
nvcc -O3 -arch=sm_70 -rdc=true dynparallel.cu -o dynparallel -lcudadevrt

# ② WMMA —— 必须 >= sm_70（有 Ampere 卡建议 sm_80）
cd ../wmma
nvcc -O3 -arch=sm_70 wmma_gemm.cu -o wmma_gemm

# ③ cp.async —— sm_80 才有异步；低架构能编过但无收益
cd ../cpasync
nvcc -O3 -arch=sm_80 cpasync_gemm.cu -o cpasync_gemm

# ④ 线程块集群 —— 需要 Hopper，必须 sm_90
cd ../cluster
nvcc -O3 -arch=sm_90 cluster_dsm.cu -o cluster_dsm

# ⑤ 统一内存进阶
cd ../unified
nvcc -O3 -arch=sm_70 unified_advise.cu -o unified_advise

# ⑥ VMM —— 必须链接 CUDA 驱动库（Windows 上 -lcuda 等价于 cuda.lib）
cd ../vmm
nvcc -O3 -arch=sm_70 vmm_alloc.cu -o vmm_alloc -lcuda

# ⑦ 多 GPU P2P
cd ../multigpu
nvcc -O3 -arch=sm_70 p2p.cu -o p2p

# ⑧ CUDA Graphs 进阶 —— 条件节点需要 Toolkit >= 12.3
cd ../graphs_advanced
nvcc -O3 -arch=sm_70 graph_advanced.cu -o graph_advanced
```

Windows PowerShell 风格（注意 `.exe` 和反斜杠）：

```powershell
cd code\lessons\ch06_advanced\dynparallel
nvcc -O3 -arch=sm_70 -rdc=true dynparallel.cu -o dynparallel.exe
.\dynparallel.exe
```

**一键编译**（`code/Makefile` 已经为这 8 个示例配好了特殊参数）：

```bash
cd code
make list | grep ch06        # 先看看会用到什么参数
make ch06                    # 只编译第 6 章的示例
```

Makefile 里的特殊处理：

```make
	case "$*" in \
	  *dynparallel*)  EXTRA="-rdc=true" ;; \
	  *cluster_dsm*)  EXTRA="-arch=sm_90" ;; \
	  *vmm_alloc*)    EXTRA="-lcuda" ;; \
	  *)              EXTRA="" ;; \
	esac; \
```

想确认寄存器/共享内存用量，加 `-Xptxas -v`：

```bash
nvcc -O3 -arch=sm_80 -Xptxas -v cpasync_gemm.cu -o cpasync_gemm
```

### 4.3 动态并行：预期输出

```bash
./dynparallel
```

```text
=========== 设备信息 ===========
设备编号        : 0
设备名称        : NVIDIA GeForce RTX 3060
计算能力 (CC)   : 8.6
SM 数量         : 28
================================

[A] 动态并行基础：父 kernel 启动了 4 个子 kernel，counter = 10  [OK]
[B] 递归归约(1M 个元素)：和 = 76853.156, CPU 参考 = 76853.203, 相对误差 = 6.10e-07  [OK]
    递归深度 = log2(1048576 / 1024) = 10 层，每层的每个节点都会启动 2 个子 kernel
    启动次数是 O(N/LEAF_SIZE) 量级 —— 这就是动态并行最大的成本来源。
[C] 设备端 malloc/free：0+1+...+1023 = 523776  [OK]
```

> **你的数值一定不同**（`[B]` 的确切数字取决于求和顺序和浮点舍入）。要关注的是：**三个 `[OK]`**，以及「相对误差在 1e-6 量级」——递归会改变求和顺序，浮点加法不满足结合律，误差比串行求和大是正常的（见[第 5 章](ch05_核心算法模式实战.md)的精度讨论）。
>
> **在 CC < 3.5 的机器上**，你会看到 `[跳过] 本机 CC = 3.x，动态并行需要 CC >= 3.5。` 然后程序正常退出——**退出码是 0，是「优雅跳过」而不是「失败」**。

### 4.4 WMMA：预期输出

```bash
./wmma_gemm
```

```text
[配置] block = (32, 4)，grid = (32, 8)，共 1024 个 warp

[校验] 正在用 CPU(double) 计算参考结果，请稍候……

[正确性] 最大绝对误差 = 3.0518e-05，相对 L1 误差量级 = 1.842e-06  [OK]
[性能]   512x512x512 FP16->FP32 GEMM: 0.0894 ms
         算力 = 3.00 TFLOPS (FP16 输入 + FP32 累加)

[对照]   同样规模在 CPU 上单线程约需 0.5 秒（约 0.0003 TFLOPS）；
         这个朴素 WMMA 版本在 RTX 3060 上约 2~5 TFLOPS，你的机器可能不同；
         同样硬件上 cuBLAS 能到 20+ TFLOPS —— 差距来自三件事：
           1) 共享内存分块（减少全局访存）；
           2) cp.async 多阶段流水线（把访存和计算重叠起来）；
           3) 每线程多个累加器（寄存器分块，提升指令级并行）。
```

> **这里的 `3.00 TFLOPS` 是「一个朴素 WMMA 实现」的水平，不是硬件上限。** 你的机器可能是 1.5，也可能是 8——**不要拿这个数字和别人比**，要和你自己的「不优化版本」比。
>
> **在 CC < 7.0 的机器上**，你会看到 `[跳过] 本机 CC = 6.1，WMMA 需要 CC >= 7.0。`。而在**编译期**，如果你写 `-arch=sm_60`，`nvcc` 会直接报错（`fragment` 无法实例化），**连可执行文件都生成不出来**——这就是 0.3 节的表里把编译期/运行期表现分开写的原因。

**用 ncu 看张量管线的利用率**（本节最有价值的命令）：

```bash
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active ./wmma_gemm
```

**这个数低于 50% 时，瓶颈通常就是那两次 `load_matrix_sync`**——张量核心在等数据。这正是 2.3 节 `cp.async` 要解决的问题。

### 4.5 cp.async：预期输出

```bash
./cpasync_gemm
```

```text
[提示] 本机 CC = 8.6，支持 cp.async 异步拷贝。
[配置] block = 256 线程，grid = (16, 16)，每线程 4x4 累加器，2 阶段流水线
[校验] 抽查前 64 行的相对误差（完整跑 CPU 参考太慢）……

[正确性] 前 64 行最大相对误差 = 3.12e-06  [OK]
[性能]   1024x1024x1024 FP32 GEMM (cp.async 双缓冲): 1.4023 ms
         算力 = 1.53 TFLOPS (FP32)

[实验]   把 STAGES 改成 1（等于关掉流水线，每次都要先等拷贝完再算），
         重新编译跑一次，你会看到性能明显下降 —— 这就是「重叠」的价值。
         在 RTX 3060 上，双缓冲相对单缓冲通常有 20%~50% 的提升，
         你的机器可能不同。
```

> **做「关掉流水线」实验时有个坑**：`STAGES = 1` 会让 `As[stage ^ 1]` 变成 `As[1]`，**数组越界**。所以要把 `As[stage ^ 1]` 改成 `As[stage]`，并把 `__pipeline_wait_prior(1)` 改成 `__pipeline_wait_prior(0)`，再测到的才是「无重叠」的真实代价。
>
> **在 CC < 8.0 的机器上**，程序**照常运行、结果照常正确**，只是会先打印一行 `[提示] 本机 CC = 7.5 < 8.0：cp.async 会退化成同步拷贝`。这是本章唯一一个「不支持也能跑」的特性——也正因为如此，**你可能会在旧卡上误以为 cp.async 没用**，其实它只是没生效。

### 4.6 线程块集群：预期输出

有 Hopper 卡（CC ≥ 9.0）时：

```text
[配置] grid = 64 块, block = 256 线程, 集群 = 2 块/集群, 共 32 个集群
[正确性] 集群归约结果 = 84212.750，CPU 参考 = 84212.797，相对误差 = -5.58e-07  [OK]
[性能]   0.0412 ms（含集群屏障与 DSM 读取）
```

没有 Hopper 卡时：

```text
[跳过] 本机 CC = 7.5，线程块集群需要 CC >= 9.0。
       本文件主要用于说明概念，没有 Hopper 卡的话读完注释即可。
```

Toolkit 太旧（< 11.8）时：

```text
[跳过] CUDA Toolkit 版本过旧，需要 >= 11.8 才有 cudaLaunchKernelEx。
```

> **注意 `-arch=sm_90` 的编译行为**：即使你的机器没有 Hopper，**只要 Toolkit 支持 sm_90，这段代码就能编出来**（只是跑不了）。这正是「可移植库」需要的性质：一份源码能编成多个架构的 cubin，运行时挑能跑的那个。**如果 `gridDim` 不是集群大小的整数倍**，`cudaLaunchKernelEx` 会立刻返回错误（`CUDA_CHECK` 会打印并退出）。

### 4.7 统一内存进阶：预期输出

```bash
./unified_advise
```

```text
[能力] 并发托管访问(ConcurrentManagedAccess) = 支持
[能力] 可分页内存直接访问(PageableMemoryAccess) = 不支持
[能力] 统一寻址(UnifiedAddressing) = 支持

[A] 不预取（靠页错误迁移）: 0.2031 ms（取 10 次最小值）
[B] 预取到设备 (cudaMemPrefetchAsync): 0.1712 ms

[正确性] 不匹配元素 = 0  [OK]
```

> **怎么读这两个数字**：A 和 B **第一次**迭代都会因页迁移而慢，而「取 10 次最小值」会把第一次排除掉。在数据已经驻留设备侧之后，A 和 B 的 kernel 耗时其实接近——**这正是本示例想让你发现的事实：`cudaMemPrefetchAsync` 不是让你的 kernel 变快，而是让「页迁移」这件事发生在你希望它发生的时候**。
>
> 想看页错误的真实规模：`nsys profile --stats=true ./unified_advise`，报告里的 Unified Memory 一节会列出**页错误次数**和**迁移字节数**，你会看到第一次迭代贡献了绝大部分页错误。
>
> **在 CC < 6.0 的机器上**会打印 `[跳过] 本机 CC = 5.2，统一内存的按需页迁移需要 CC >= 6.0。` 后正常退出。另外注意：**在 `ConcurrentManagedAccess = 不支持` 的设备上**，`cudaMemAdvise` 和预取的效果会大打折扣，程序仍会跑，但 A/B 两个数字会几乎一样——**这本身就是一个值得记录的结果**。

### 4.8 VMM：预期输出

```bash
./vmm_alloc
```

```text
[粒度] 本设备的最小 VMM 分配粒度 = 2.00 MB
[地址] 预留虚拟地址 0x7f9c00000000，长度 4.00 MB
[映射] 已把 2 块独立的物理分配映射到 0x7f9c00000000 和 0x7f9c00200000

[结果] 元素总数 = 1048576，不匹配 = 0  [OK]
[结果] 跨物理边界的取值 (1.0 -> 2.0) [OK] 说明两块不连续的物理内存在虚拟地址上确实拼成了连续区间
[清理] 已按 cuMemUnmap -> cuMemRelease -> cuMemAddressFree 的顺序释放
```

> **虚拟地址的具体数值每次运行都不同**（由驱动分配），这是正常的。
>
> **在驱动过旧或 CC < 6.0 的设备上**，程序会打印 `[跳过] 本设备不支持虚拟地址管理（需要 CC >= 6.0 且驱动 >= 440）。` 后退出——注意这段判断用的是**驱动 API** 的 `cuDeviceGetAttribute`，因为它查的是一个驱动层属性。
>
> **内存泄漏自查**：这段代码绕过了运行时分配器，所以 `nvidia-smi` 看到的显存占用**只能靠 `cuMemRelease` 归还**。把释放那几行注释掉再跑一次，会看到显存一直占着到进程退出——这是理解「为什么框架要自己管显存」的最好方式。

### 4.9 多 GPU P2P：预期输出

双卡且支持 P2P 时：

```text
[设备] 检测到 2 个 CUDA 设备
  dev 0: NVIDIA GeForce RTX 3090       CC 8.6  24.0 GB  统一寻址
  dev 1: NVIDIA GeForce RTX 3090       CC 8.6  24.0 GB  统一寻址

[拓扑] cudaDeviceCanAccessPeer 能力矩阵（行=源，列=目的）
       to0  to1
from0    -   Y
from1    Y   -

[开启] 已开启 dev0 <-> dev1 的双向 P2P 访问

[搬运] 经主机中转 (D2H + H2D): 11.842 ms, 有效带宽 11.35 GB/s
[搬运] P2P 直达 (cudaMemcpyPeer)  : 5.921 ms, 有效带宽 11.35 GB/s
       提速约 2.00x

[正确性] dev1 上不匹配元素 = 0  [OK]
[远端 kernel] 在 dev0 上运行 kernel 修改 dev1 的显存: 不匹配 = 0  [OK]
```

单卡机器上：

```text
[设备] 检测到 1 个 CUDA 设备
  dev 0: NVIDIA GeForce RTX 3060       CC 8.6  12.00 GB  统一寻址

[跳过] 只有 1 张卡，无法演示 P2P。
```

> **最常见的输出是「能力矩阵里全是 n」**。这不代表程序有问题，而是**主板的 PCIe 拓扑不支持 P2P**。用 `nvidia-smi topo -m` 确认：如果两张卡挂在不同的 PCIe 根（比如跨 CPU socket），P2P 通常就不可用。此时程序会打印 `[提示] 本机两张卡之间不支持 P2P，下面只演示走主机的搬运路径。` 然后**把走主机的路径跑完并验证**——这也是本节想让你看到的东西。

### 4.10 CUDA Graphs 进阶：预期输出

Toolkit ≥ 12.3 时：

```text
[逐个启动] 200 轮 x 3 个 kernel: 26.481 ms（平均 132.4 us/轮）
[逐个启动] 结果正确性: 不匹配 0  [OK]

[Graph]    200 轮 x 1 张图（3 个节点）: 18.902 ms（平均 94.5 us/轮）
[Graph]    结果正确性: 不匹配 0  [OK]
[Graph]    相对逐个启动的提速约 1.40x（kernel 越小、启动开销占比越高，收益越大）
           在 RTX 3060 上这个规模的数据量下，提升通常在 1.2~2x 之间，
           你的机器可能不同。数据量越大，kernel 自身耗时占比越高，收益越小。

[条件节点] CUDA 12.3 >= 12.3，开始演示 If/Else 控制流
  dval 初值 =  -5.0 (< 0)  ->  走 If 分支，结果 = 111  [OK]
  dval 初值 =   5.0 (>= 0) ->  走 Else 分支，结果 = 222  [OK]
  注意：两次执行用的是**同一张已实例化的图**，host 端没有做任何分支判断，
        也没有在两次执行之间做 cudaStreamSynchronize 之外的同步。

[循环节点] While 循环体执行了 5 次（期望 5） [OK]
           整个循环只提交了 1 次图，host 一次都没参与。
```

Toolkit < 12.3 时，条件/循环那两段会被**编译期裁掉**：

```text
[跳过] 条件节点与循环节点需要 CUDA >= 12.3，当前是 12.0。
       升级到 CUDA 12.3+ 后重新编译即可看到完整演示。
```

> **关于 `1.40x` 这个数字**：Graph 省的是**每轮的启动开销和 kernel 间隙**，不是计算时间。所以：**kernel 越短收益越大**（这里的 kernel 只处理 1M 个 float，约 40 微秒）；**数据量越大收益越小**；**图的实例化开销**（几十微秒～毫秒）在 200 轮里被摊薄了——如果只执行 1 次，Graph 反而更慢。
>
> **调试图**：把示例里那行注释掉的 `cudaGraphDebugDotPrint(graph, "graph.dot", 0);` 打开，然后 `dot -Tpng graph.dot -o graph.png`，你会**亲眼看到** If 和 Else 两个菱形分支挂在求值节点下面。

---

## 5. 动手练习

### 练习 1 [基础] 给每个示例加上「你的机器支持吗」自检输出

给 8 个示例各写一个 `canRun*()` 函数，用 `cudaGetDeviceProperties` / `cudaDeviceGetAttribute` / `cuDeviceGetAttribute` 判断支持性，输出一张本机能力表：

```text
[能力自检] 设备: NVIDIA GeForce RTX 3060 (CC 8.6)
  ✔ 动态并行            (CC >= 3.5)
  ✔ WMMA FP16           (CC >= 7.0)
  ✘ WMMA TF32           (需要 CC >= 8.0)     ← 这一行是故意写错的，请修正判断条件
  ✔ cp.async            (CC >= 8.0)
  ✘ 线程块集群          (需要 CC >= 9.0)
  ✔ 统一内存按需迁移     (CC >= 6.0 且 ConcurrentManagedAccess)
  ✔ VMM                 (驱动 >= 440)
  ✘ P2P                 (只有 1 张卡)
  ✔ CUDA Graphs 条件节点 (CUDART_VERSION >= 12030)
```

**要点**：这张表在真实项目里非常有用——**上线前跑一遍，比上线后崩在客户机器上强**。注意 `8.6 >= 8.0`，所以 TF32 那一行应当是可用的，请修正判断条件。

### 练习 2 [基础] 把 `cp.async` 的拷贝粒度从 4 字节改成 16 字节

**做法**：把 tile 的加载改成按 `float4` 索引，用 `reinterpret_cast<float4*>` 同时处理全局与共享内存两侧的地址转换，并把 `sizeof(float)` 改成 `sizeof(float4)`。

**要求**：

1. tile 的元素个数必须能被 4 整除（`BM*BK`、`BK*BN` 都满足）；
2. **共享内存目标也要 16 字节对齐**——`As[s][r][c]` 里 `c` 必须是 4 的倍数，`BK = 16` 满足；
3. 编译加 `-Xptxas -v` 看寄存器和指令数变化；
4. 对比改前改后的耗时。**在 RTX 3060 上，指令数降到 1/4、耗时通常会有可观察的改善，但幅度取决于你的 kernel 是不是被 LSU 发射槽限制的**——用 `ncu` 看 `lsu__inst_executed` 和 `smsp__inst_executed` 确认。

### 练习 3 [进阶] 把 `STAGES` 做成模板参数，测出你的最佳流水线深度

现在的 `STAGES = 2` 写死了，而且 `stage ^ 1` 只支持 2 阶段。改成：

```cpp
template <int STAGES>
__global__ void gemmCpAsyncN(...) {
    __shared__ float As[STAGES][BM][BK];
    __shared__ float Bs[STAGES][BK][BN];
    ...
    const int next = k0 + BK;
    if (next < K) {
        loadATileAsync(As[(stage + 1) % STAGES], ...);
        loadBTileAsync(Bs[(stage + 1) % STAGES], ...);
        __pipeline_commit();
        __pipeline_wait_prior(STAGES - 1);   // 允许 STAGES-1 个提交组未完成
    } else {
        __pipeline_wait_prior(0);
    }
    ...
    stage = (stage + 1) % STAGES;
}
```

**要求**：

1. 测 `STAGES = 1, 2, 3, 4, 5`，画一张「stage 数 ↔ 耗时」的表；
2. **注意共享内存上限**：`STAGES = 5` 时 `5 * (64*16 + 16*64) * 4 B = 40 KB`，还在默认 48 KB 以内；`STAGES = 6` 就超了，需要 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` 并改用动态共享内存；
3. 解释你看到的曲线：**为什么到某个 stage 之后收益停止增长？** （提示：看 `ncu` 里 Warp Stall Reasons 中 `Long Scoreboard` 的比例，下降到某个值之后瓶颈就换人了。）

### 练习 4 [进阶] 用 WMMA + `cp.async` 把两个示例合成一个

本章最重要的练习：把 `wmma_gemm.cu` 的 `load_matrix_sync` 从**全局内存**换成**共享内存**，而共享内存由 `cp.async` 填充。

**做法**：

1. 每个块用 `cp.async` 把 A 的 64×16 tile、B 的 16×64 tile 搬进共享内存（就是 `cpasync_gemm.cu` 的加载逻辑）；
2. 用 `wmma::load_matrix_sync(a_frag, &As[s][warpRow * 16][0], BK)` 从**共享内存**加载——注意这里的 `ldm` 变成了 **tile 的宽度 `BK`**，而不是全局矩阵的 `K`；
3. 保留双缓冲流水线；
4. 和 `wmma_gemm.cu` 的朴素版对比 TFLOPS。

**要点**：**共享内存的 `ldm` 是 tile 宽度**，这是最容易搞错的地方；输入类型要统一（要么都用 `float` + TF32，需要 CC ≥ 8.0，要么都用 `__half`）；目标是在你的卡上把 TFLOPS 提高 2 倍以上。**达不到也没关系——把每一步的 `ncu` 数据记下来，比达到数字更有价值。**

### 练习 5 [挑战] 用集群实现一个「邻居 halo 交换」的 stencil

把[第 5 章](ch05_核心算法模式实战.md)的二维 Jacobi stencil 改成集群版本：每块负责一个 tile，**halo 数据直接从邻居块的共享内存里读**，不再写回全局内存。

**做法**：

1. 集群大小设为 `clusterDim.x = 2`（两个相邻的块组成一个集群）；
2. 每块把**内部区域**算完写进自己的共享内存；
3. `cluster.sync()`；
4. 每块用 `cluster.map_shared_rank(smem, neighborRank)` 读邻居的边界列，补齐自己缺失的那一圈 halo；
5. `cluster.sync()` 之后才退出。

**要点**：只有**方向上的邻居**能用 DSM 拿到，斜角邻居需要两跳，所以要退一步设计分块方式；**两次 `cluster.sync()` 都不能少**，第二次尤其容易忘；量一下相比第 5 章「全局内存 halo」版本的改善。**如果没有 Hopper 卡，就写一份设计文档 + 伪代码**，说清每一步的数据流——这也是合格的交付物。

### 练习 6 [挑战] 把「按需页迁移」的成本算清楚

`unified_advise.cu` 的 A/B 对比有一个「记账口径」的陷阱（见 3.5 节）。请写一个新实验把它讲清楚：

**要求**：

1. 设计三组计时：**T1** 只测 kernel（预热充分、数据已驻留）；**T2** 测「预取 + 同步 + kernel」的**总时间**；**T3** 测「CPU 改过数据 → kernel → CPU 读回」的**完整往返**时间（真实应用里最容易踩的场景）；
2. 用 `cudaEvent` 分别记录，打印三组数据；
3. 用 `nsys profile --stats=true` 拿到页错误次数和迁移字节数，验证你的解释；
4. **写一段结论**：什么场景下统一内存是「免费的方便」，什么场景下它是「几十倍的性能陷阱」？

**提示**：让 `x` 的大小远大于 L2（RTX 3060 的 L2 是 3 MB，示例的 16 MB 已经够了）。试试把 `N` 改成 `1 << 26`（256 MB），观察页错误次数的变化。

---

## 6. 常见错误与排查

| 错误信息 / 现象 | 原因 | 解决办法 |
|---|---|---|
| `calling a __global__ function from a __global__ function is only allowed ... with relocatable device code` | 设备端启动了 kernel，但没加 `-rdc=true` | 加 `-rdc=true` 重新编译 |
| `undefined reference to '__cudaRegisterLinkedBinary_...'` | 用了 `-rdc=true` 但没链接设备运行时库 | 补 `-lcudadevrt`；避免混用不同版本的工具链 |
| 设备端 kernel **静默不执行**，结果全错 | 待启动队列满了（`cudaLimitDevRuntimePendingLaunchCount` 默认很小） | 减少设备端启动次数（加大 `LEAF_SIZE`）；或 `cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, n)` |
| 递归 kernel 卡死 / `cudaDeviceSynchronize` 不返回 | 递归没有终止条件，或子 kernel 互相等待 | 检查递归边界；设备端同步只在「本线程发起」的范围内，不能当块间屏障 |
| 设备端 `cudaMalloc` 返回 `cudaErrorMemoryAllocation` | 设备运行时堆配额用尽（默认约 8 MB） | 改成 host 端预分配；或 `cudaDeviceSetLimit(cudaLimitMallocHeapSize, bytes)` |
| `identifier "wmma" is undefined` / `fragment` 无法实例化 | `-arch` 低于 `sm_70`，或没 `#include <mma.h>` | 改成 `-arch=sm_70` 及以上；补头文件 |
| WMMA 结果**完全错乱** | 同一个 warp 内线程算出了不同的 `warpM`/`warpN` | `blockDim.x` 必须是 32 的倍数，`warpM` 用 `threadIdx.x / warpSize` 计算 |
| WMMA 结果**只有一部分对** | `ldm` 写成了子块宽度（`WMMA_K` / `WMMA_N`），而不是整个矩阵的行长（`K` / `N`） | `ldm` 是「源矩阵里下一行的距离」，不是子块宽度 |
| WMMA 精度很差（相对误差 > 1e-2） | 累加器用了 `__half` 而不是 `float` | 累加器一律用 `float`，这就是混合精度的核心 |
| `cp.async` 版本耗时和同步版本**一模一样** | CC < 8.0，`__pipeline_memcpy_async` 退化成了同步 ld/st | 换 Ampere 及以上；先在 host 端检查 `prop.major >= 8` |
| `cp.async` 结果**时对时错** | `__pipeline_wait_prior` 之后漏了 `__syncthreads()` | 加回块级同步；用 `compute-sanitizer --tool racecheck` 定位 |
| `cp.async` 结果**全错** | 「先发下一块、再 `wait_prior(1)`」写反成了「先 `wait_prior(0)` 再发」 | 顺序必须是：发起下一块 → `commit` → `wait_prior(STAGES-1)` |
| `'__pipeline_memcpy_async' was not declared` | 没 `#include <cuda_pipeline.h>` | 补上头文件 |
| 集群启动报 `cudaErrorInvalidValue` | `gridDim` 不是 `clusterDim` 的整数倍 | `grid = clusters * clusterSize`，显式构造 |
| 集群 kernel 结果**偶发读到垃圾** | 退出前漏了第二次 `cluster.sync()` | 补上；这是集群最经典的 bug |
| 集群代码报 `cudaLaunchKernelEx` 未定义 | Toolkit < 11.8 | 升级 Toolkit；或用 `CUDART_VERSION` 做编译期保护 |
| `cudaMemAdvise` 返回 `cudaErrorInvalidValue` | 设备不支持并发托管访问（`cudaDevAttrConcurrentManagedAccess = 0`），或 `SetAccessedBy` 用在不支持的设备上 | 先查属性再调用；不支持就跳过这块优化（自动退化为默认行为） |
| 统一内存程序**第一轮特别慢**，之后正常 | 页错误 + 页迁移（首次访问） | 这是设计使然；用 `cudaMemPrefetchAsync` 把它提前到你不介意的时候 |
| 统一内存程序**每一轮都慢** | CPU 和 GPU 交替访问同一块托管内存，每轮都在迁移 | 用 `cudaMemAdvise` + 预取；或者干脆改用 `cudaMalloc` |
| `cuMemCreate` 返回 `CUDA_ERROR_NOT_SUPPORTED` | 驱动 < 440，或设备不支持 VMM | 查 `CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED`；升级驱动 |
| VMM 代码 kernel 一碰就崩 | 漏了 `cuMemSetAccess`，或 `cuMemMap` 的偏移/长度不是粒度整数倍 | 补 `cuMemSetAccess`；用 `cuMemGetAllocationGranularity` 查粒度并对齐 |
| VMM 显存「泄漏」到进程退出 | 忘了 `cuMemRelease`（`cudaFree` 管不了 VMM 分配） | 严格按 `cuMemUnmap → cuMemRelease → cuMemAddressFree` 反序释放 |
| 编译 VMM 代码报 `undefined reference to 'cuMemCreate'` | 没链接 CUDA 驱动库 | 加 `-lcuda`（Windows 上等价于 `cuda.lib`） |
| 多卡程序报 `cudaErrorInvalidDevicePointer` | 忘了 `cudaSetDevice`，在错误的设备上分配/启动 | 每个 `cudaMalloc` / kernel 启动前明确设备；把设备号写进日志 |
| `cudaDeviceEnablePeerAccess` 返回 `cudaErrorPeerAccessAlreadyEnabled` | 同一对设备重复开启 | 该错误码可直接忽略，或用 flag 记录已开启状态 |
| `cudaDeviceCanAccessPeer` 返回 0（不是报错） | 主板/PCIe 拓扑不支持，或不是 64 位进程，或没有 UVA | `nvidia-smi topo -m` 看拓扑；退回主机中转路径；**不要把它当错误处理** |
| 跨进程传指针导致崩溃 | 跨进程不能共享指针 | 用 CUDA IPC：`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` |
| `cudaGraphAddConditionalNode` 未定义 | Toolkit < 12.3 | 用 `#if CUDART_VERSION >= 12030` 做编译期保护 |
| 条件/循环节点实例化失败 | 设备或驱动不支持（返回 `cudaErrorNotSupported`） | 检查每次 Graph API 的返回值；不支持就退回「多 kernel + host 分支」 |
| 用循环节点时 GPU **一直忙，host 卡死不返回** | 循环体 kernel 永远不把条件设成 0 | 在循环体里加硬上限（计数器 + `limit`）；先用小 `limit` 调试 |
| 显式构图的 kernel 参数变成垃圾值 | `kernelParams` 指向的局部变量在实例化前就出了作用域 | 用具名局部变量（不要用临时表达式），保证它们活到 `cudaGraphInstantiate*` 返回 |
| Graph 结果正确但**比逐个启动还慢** | 只执行了一两次，图实例化的开销没被摊薄 | 只对「反复执行的固定序列」用图；否则老实用逐个启动 |
| 集群 / WMMA 代码在**别人的机器上跑不起来**（你的机器能跑） | 你的机器 CC 高，编译时写死了 `-arch=sm_90` 等 | 做运行时能力检测 + 优雅降级；或编译多个架构（`-gencode arch=compute_70,code=sm_70 -gencode ...`） |

---

## 7. 本章小结

### 7.1 要点清单

1. **高级特性的共同点是「可能性」而不是「速度」**：它们把原本必须由 host 编排、必须绕道显存、必须拆成多个 kernel 的事情，变成了设备侧、硬件直接支持的操作。
2. **每个特性都有硬性门槛，而且「编不过」和「跑不对」是两回事**：
   - **编译期失败**：WMMA 用 `-arch=sm_60`、集群用旧 Toolkit、动态并行不加 `-rdc=true`；
   - **运行期失败**：集群在非 Hopper 卡上启动失败、VMM 在旧驱动上返回 `CUDA_ERROR_NOT_SUPPORTED`；
   - **静默退化**：`cp.async` 在 CC < 8.0 上退化为同步拷贝（**能跑、正确、但没收益**）——最容易被误判的一种。
3. **动态并行的核心约束是「启动开销」**：设备端启动是 host 端的 2～5 倍，所以必须用**叶子阈值**把启动次数从 O(n) 压到 O(n / LEAF_SIZE)。它的价值在表达力（递归、自适应细分），不在绝对性能。
4. **设备端 `cudaDeviceSynchronize()` 不是块间屏障**：它只等待「当前线程」派生的子 kernel。要块内一致，还得 `__syncthreads()`。
5. **WMMA 是 warp 级操作，`fragment` 是「不透明的分布式寄存器集合」**：一个 warp 一次算 16×16×16。`ldm` 是**源矩阵的行长**，不是子块宽度——这是最常见的错误。
6. **混合精度的全部含义就是「低精度输入 + 高精度累加」**（FP16 输入 / FP32 累加，最低 CC 7.0）。PyTorch 的 AMP 是这件事的库化版本。
7. **`cp.async` 的价值在于「绕过寄存器 + 可分组等待」**：`commit` 分组、`wait_prior(N)` 只等需要的那一批。**顺序必须是「先发下一块、再等当前块」**，反了就退化。
8. **`cp.async` 必须配合 `__syncthreads()`**：`wait_prior` 只保证本线程的拷贝完成，计算阶段却要读别的线程搬进来的数据。
9. **线程块集群（CC ≥ 9.0）把「块间同步」和「块间共享内存」都变成了硬件能力**。三条铁律：`gridDim` 整除 `clusterDim`、`map_shared_rank` 做地址转换、**退出前必须再 `cluster.sync()` 一次**。
10. **统一内存的默认行为是「谁碰它搬到谁那」**，代价是页错误（几十微秒级）。`cudaMemAdvise` 是「告诉驱动使用习惯」，`cudaMemPrefetchAsync` 是「提前叫搬家公司」——两者都只是**优化提示**，不改变正确性；迭代型算法的数据本来就常驻 GPU，最省事的做法是**别用托管内存**。
11. **VMM 把「虚拟地址」和「物理分配」解耦**，四步是 `cuMemAddressReserve → cuMemCreate → cuMemMap → cuMemSetAccess`，释放必须**反序**。大模型框架用它做显存池、碎片拼接、别名映射和跨进程共享。
12. **P2P 的前提是拓扑**：`cudaDeviceCanAccessPeer` 返回 0 **不是错误**。开启后 kernel 可以直接解引用对端指针，但**大规模搬运仍应用 `cudaMemcpyPeer`**。集合通信不要自己写——用 NCCL。
13. **CUDA Graph 的条件/循环节点（Toolkit ≥ 12.3）把分支判断搬到了设备端**：`cudaGraphSetConditional` 写条件值，If/Else/While 节点按值执行。**循环体必须自己负责把条件设成 0，并且要有硬上限。**
14. **能力检测 + 优雅降级是本章的第一公民技能**：`cudaGetDeviceProperties` / `cudaDeviceGetAttribute` / `cuDeviceGetAttribute`，不支持就打印一句人话然后 `return 0`。

### 7.2 自检问题（附答案要点）

**Q1：为什么动态并行必须加 `-rdc=true`？**
> 因为设备端 `<<<>>>` 需要「子 kernel 的设备函数符号」在**链接期**才能确定，这要求设备代码像普通 C++ 一样走「编译 → 链接」两步（可重定位设备代码）。默认模式下每个 `.cu` 单独编成完整 cubin，符号在编译期就定死了，链接不上。`-rdc=true` 之后还需要设备端运行时库 `cudadevrt`。

**Q2：设备端的 `cudaDeviceSynchronize()` 和 host 端的有什么本质区别？**
> host 端等的是「该设备上所有流的所有工作」，是一次完整 host-device 同步；**设备端等的只是「当前线程派生的子 kernel（含后代）」**，它**不是块间屏障**，也不切回 host。所以在 `parentKernel` 里，发起子 kernel 的线程同步完，还得靠 `__syncthreads()` 让块里其它线程继续。

**Q3：WMMA 里 `load_matrix_sync` 的第三个参数 `ldm` 是什么？为什么不是子块的宽度？**
> `ldm`（leading dimension）是「源矩阵里，下一行（行主序）或下一列（列主序）距离当前起始位置有多少个元素」。我们取的是大矩阵里的一个 16×16 子块，子块的第 1 行和第 2 行之间隔的是**整个大矩阵的行长**（比如 K 或 N），而不是 16。写成 16 会读到错误的数据。

**Q4：为什么 FP16 输入的矩阵乘，累加器一定要用 `float`？**
> FP16 只有 10 位尾数，表示范围也窄。K 一大（512、1024），部分和不断累加，FP16 的舍入误差迅速累积，结果严重失真。用 FP32 累加几乎不损失速度（张量核心内部本来就是 FP32 累加），这就是「混合精度」的全部含义。

**Q5：`cp.async` 流水线里，为什么必须「先发起下一块，再 `wait_prior`」？**
> `wait_prior(N)` 的语义是「等到只剩 N 个提交组未完成」。如果先 `wait_prior(0)`（等全部完成）再发起下一块，等待期间根本没有拷贝在后台进行——计算与搬运**无法重叠**，双缓冲退化成串行。正确顺序是：发起下一块 → `commit` → `wait_prior(1)` → 计算当前块。

**Q6：`__pipeline_wait_prior` 之后为什么还要 `__syncthreads()`？**
> 因为 `wait_prior` 只保证**本线程**发起的那些拷贝完成了，而计算阶段每个线程读的共享内存数据是**别的线程**搬进来的。没有块级同步，你就是在读尚未写完的共享内存——这是一个只在某些调度下才复现的竞态，要用 `compute-sanitizer --tool racecheck` 才容易抓。

**Q7：集群 kernel 里为什么要有两次 `cluster.sync()`？**
> 第一次保证「别人的共享内存已经写好了，我可以安全地 `map_shared_rank` 去读」。第二次（退出前）保证「别人都已经读完了，我的共享内存才可以被回收」。少了第二次，提前退出的块的共享内存可能被重新分配，还在读它的块会读到垃圾。

**Q8：统一内存的 `cudaMemAdvise` 是「保证」还是「建议」？**
> 是**建议**。驱动有权忽略它。所以它只影响性能，不能改变程序语义，也**不能替代正确性检查**。另外，某些建议（如 `SetAccessedBy`）在不支持并发托管访问的设备上会直接返回错误。

**Q9：VMM 的 `cuMemMap` 和 `cuMemSetAccess` 各自负责什么？**
> `cuMemMap` 建立「虚拟地址区间 ↔ 物理 handle」的映射关系；`cuMemSetAccess` 授予「哪些设备可以访问这段区间、以什么权限」。**只做 `cuMemMap` 不做 `cuMemSetAccess`，kernel 一访问就是非法访问。**

**Q10：`cudaDeviceCanAccessPeer` 返回 0 表示什么？**
> 表示「设备 i 不能直接访问设备 j」，**这是合法的查询结果，不是错误**。函数本身返回 `cudaSuccess`。常见原因是 PCIe 拓扑不支持（两卡挂在不同的 PCIe 根）、不是 64 位进程、或没有 UVA。

**Q11：为什么 CUDA Graph 的条件/循环节点要求把「求值」放在一个 kernel 里？**
> 因为分支判断必须发生在**设备端**，而能写条件值的只有 `cudaGraphSetConditional`，它只能在设备代码里调用。所以图的模式是：一个求值 kernel 设置条件句柄 → If/Else 或 While 节点按值执行。这样整段控制流**只用一次 `cudaGraphLaunch` 提交，host 全程不参与**。

**Q12：为什么本章的示例几乎都在开头就 `return 0`？**
> 因为高级特性的门槛是**硬件**，不是代码能力。在不支持的机器上，正确的做法是**打印一句人话 + 干净退出（退出码 0）**，而不是让用户面对一个 `cudaErrorNotSupported` 的堆栈或者一个静默的错误结果。**能力检测 + 优雅降级**是写可移植 GPU 代码的基本素养——你写的库不可能只在自己的机器上跑。

---

## 8. 下一章预告

这一章你拿到的是**现代 GPU 的工具箱**：动态并行、张量核心、异步流水线、块间共享内存、显存自管理、多卡通信、设备端控制流。它们都很强大，也都很难写对。

但在真实工程里，你**几乎不会**从零手写这些。因为：

- 想要张量核心的极致性能？**cuBLAS / CUTLASS** 已经替你调好了分块、流水线、寄存器分块和 autotuning；
- 想要排序、扫描、归约、直方图？**CUB** 的 `DeviceRadixSort` / `DeviceScan` / `BlockReduce` 比手写版本快且更稳；
- 想要 STL 风格的并行算法？**Thrust** 的 `sort` / `reduce` / `transform` 一行搞定；
- 想要 FFT / 稀疏矩阵 / 随机数？**cuFFT / cuSPARSE / cuRAND**；
- 想要多卡集合通信？**NCCL**；
- 想要把 CUDA 接进真实工程，或者从 Python 调用？**CMake + `enable_language(CUDA)`**、`CUDA_SEPARABLE_COMPILATION`、`torch.utils.cpp_extension`、**CuPy / Numba / Triton**。

[第 7 章](ch07_生态库与工程化.md) 解决的就是这个问题：**「什么时候该自己写 kernel，什么时候该用库」**。你会看到用 cuBLAS 三行代码跑出比第 5～6 章手写版本快好几倍的 GEMM，也会看到一个完整的 CMake CUDA 工程长什么样，以及怎么把你的 kernel 封成 PyTorch 算子。

> **在进入第 7 章之前，请确认你已经理解本章 0.3 节那张「特性 × 硬件能力」表。** 后面你会大量使用这些库，而库的行为——支持哪些精度、需要什么架构、什么时候会回退到慢路径——**完全由这张表决定**。看不懂这张表，你就只能盲信库的默认行为；看懂了，你才知道什么时候该换库、什么时候该自己写。
