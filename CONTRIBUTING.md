# 教程编写规范（维护者文档）

> 这份文件面向**维护本教程的人**，不是给学习者看的。
> 学习者请从 [README.md](README.md) 开始。

本目录下每一章教程必须严格遵守以下规范，保证全书风格统一、代码可直接编译运行。

## 1. 目录结构

```
CUDALearning/
├── README.md                        # 总索引与学习路线（唯一入口）
├── CONTRIBUTING.md                  # 本文件：编写规范
├── docs/                            # 全部文档
│   ├── CUDA学习步骤与教程大纲.md      # 原始大纲
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
│   ├── common/                      # 公共头文件
│   │   ├── cuda_check.h
│   │   ├── cuda_timer.h
│   │   └── cuda_bench.h
│   ├── lessons/                     # 各章教学示例（按章节分目录）
│   │   ├── ch00_hello/
│   │   ├── ch01_indices/
│   │   ├── ch02_memory/
│   │   ├── ...
│   │   └── ch08_projects/
│   ├── projects/                    # 完整项目（可独立交付）
│   │   ├── sgemm/
│   │   ├── raytracer/
│   │   └── nbody/
│   └── CMakeLists.txt
└── tools/                           # 环境体检与维护脚本
    ├── check_env.sh
    ├── check_env.ps1
    ├── verify_all.ps1
    ├── cleanup.sh / cleanup.ps1     # 重组残留清理
    ├── fix_links.py
    └── reorganize.js
```

## 路径深度速查（写 include 时最容易错的地方）

示例代码的 `#include` 必须按**实际层级**写，不要凭感觉：

| 文件位置 | 到 `code/common/` 的前缀 |
|---|---|
| `code/lessons/chXX_yy/foo.cu` | `../../common/` |
| `code/lessons/ch03_sync/barrier/foo.cu` | `../../../common/` |
| `code/lessons/ch06_advanced/vmm/foo.cu` | `../../../common/` |
| `code/lessons/ch07_ecosystem/cublas/foo.cu` | `../../../common/` |
| `code/lessons/ch07_python/torch_ext/foo.cu` | `../../../../common/` |
| `code/projects/sgemm/sgemm_common.h` | `../../common/` |

> 新增文件后请用这个命令自查，应该**只**返回正确的前缀：
> ```bash
> grep -rn '#include "\.\./common/' code/lessons/ code/projects/
> ```

## 2. 每章必备结构（顺序固定）

```markdown
# 第 N 章 <标题>

> 本章目标 / 前置章节 / 预计耗时

## 0. 本章导读
   - 学完你能做什么（用具体能力描述）
   - 知识地图（可用简单文字图/表格）

## 1. 概念讲解
   - 由浅入深，先打比方，再给严谨定义
   - 每个新术语第一次出现时用 **加粗** 并解释

## 2. 代码实战
   - 每个知识点配一个**完整可编译**的程序
   - 代码块标注语言：```cpp / ```bash / ```cmake
   - 关键行加中文注释（注释解释"为什么"，不只是"做什么"）

## 3. 代码逐行解析
   - 对上面的代码分段讲解

## 4. 编译与运行
   - 给出确切的命令与**预期输出**（真实合理，不要编造夸张数字）

## 5. 动手练习
   - 3～6 题，由易到难，标注 [基础] [进阶] [挑战]

## 6. 常见错误与排查
   - 表格形式：错误信息 → 原因 → 解决办法

## 7. 本章小结
   - 要点清单 + 自检问题（附答案要点）

## 8. 下一章预告
```

## 3. 代码规范

- 所有示例必须能独立编译：给出完整 `main()`，不写伪代码片段。
- 统一错误检查（**强制**，全书格式必须一致，方便对照排查）：
  ```cpp
  #include "../../common/cuda_check.h"   // 注意：lessons/chXX_yy/ 下是两级
  CUDA_CHECK(cudaMalloc(...));
  ```
- 公共头文件分工（两个都可用，不要新增第三个）：
  | 头文件 | 提供 | 用在哪 |
  |---|---|---|
  | `common/cuda_check.h` | `CUDA_CHECK` / `CUDA_CHECK_KERNEL` / `CUDA_CHECK_SYNC` / `printDeviceInfo()` | 所有章节 |
  | `common/cuda_timer.h` | `CudaTimer` 类（`start`/`stopMs`/`measure`），面向对象风格 | 第 0~2 章 |
  | `common/cuda_bench.h` | `medianMs(fn, warmup, repeats)` / `gbps(bytes, ms)` / `gflopsOf(flops, ms)`，函数式风格 | 第 3~8 章 |
  两者都基于 `cudaEvent`，都做了"预热 + 多次取中位数"，可按偏好任选。
- **允许的例外**：为了让每个示例都能"复制单个 .cu 文件就编译"，
  许多示例会在文件内自带一个精简的 `CudaTimer` 类。这是刻意的取舍
  （自包含 > 零重复），不要为此把它们改成必须依赖 `common/` 目录。
  但**错误检查宏必须**统一用 `common/cuda_check.h`。
- 核函数命名使用小驼峰，如 `vectorAdd`。
- 数据规模用小常量（如 `N = 1 << 20`），保证小白机器也能跑。
- host 端计时统一用 `cudaEvent`。
- 浮点比较必须写容差：
  ```cpp
  bool ok = fabs(a - b) < 1e-5f;
  ```
- 避免使用需要 CC 8.0+ 才能编译的特性作为**必做**练习，必须使用时标注
  `// 需要 Compute Capability >= X.Y`。
- 不使用第三方依赖。

## 4. 写作语气

- 面向**零 CUDA 经验**的读者，假设他懂 C/C++ 基础，但不懂并行。
- 多用"你"，少用"我们"；句子短，术语先解释再使用。
- 每引入一个优化手段，必须回答三个问题：
  1. 原来慢在哪？
  2. 这个手段为什么能变快？
  3. 怎么用数据证明它变快了？
- 不要编造性能数字。基准数据用"在 XX 显卡上实测约为……，你的机器可能不同"表述。

## 5. 交叉引用

文档都在 `docs/` 下，因此：

- 引用其他章节用**同级文件名**：
  `[第 2 章](ch02_内存层次与访存优化.md)`
- 引用代码文件用相对路径，注意 `code/` 在 `docs/` 的上一级：
  `[example](../code/lessons/ch01_indices/vector_add.cu)`
- **不要**写成 `](code/...)`，那会解析成 `docs/code/...`，是坏链接。
  （`tools/fix_links.py` 可以批量检测并修复这类问题。）

## 6. 长度

- 每章 800～1500 行 Markdown 为宜，宁可拆细也不要压缩内容省略解释。
- 实际成稿在 1700～2800 行，原因是第 2 节要求贴出真实可编译代码。
  这是有意的取舍：**完整性优先于篇幅**。
