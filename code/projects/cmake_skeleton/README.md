# cmake_skeleton —— CUDA 工程化骨架

演示"同一个工程，有 GPU 用 GPU、没 GPU 自动退回 CPU"的完整写法。

> 位置：`code/projects/cmake_skeleton/`（原 `code/ch07_cmake/`，已随教程重组迁移）

## 目录结构

```
code/projects/cmake_skeleton/
├── CMakeLists.txt        # 构建脚本（含 CUDA 可选检测、arch 设定、测试注册）
├── include/gemm.h        # 对外接口（不含任何 CUDA 头文件）
├── src/gemm_api.cpp      # CPU + GPU 双实现，用 __CUDACC__ 条件编译
├── src/main.cpp          # 驱动程序，纯 C++，GPU/CPU 都能编
└── tests/test_gemm.cpp   # 单元测试（固定种子 + 边界用例）
```

## 构建与运行

```bash
cd code/projects/cmake_skeleton
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/gemm_demo

# 跑测试
ctest --test-dir build --output-on-failure
# 或使用自定义目标
cmake --build build --target check
```

## 关键设计点

1. **接口与实现分离**：`main.cpp` 和 `tests/` 里看不到 `cuda_runtime.h`，
   换后端不用改上层代码。
2. **CUDA 可选**：`check_language(CUDA)` 先试探，找不到就 `HAVE_CUDA=OFF`，
   工程照样编译、测试照样能跑。
3. **一份源码两条路径**：`src/gemm_api.cpp` 里 CPU 实现永远编译，
   GPU 实现包在 `#ifdef __CUDACC__` 里；CMake 决定用哪个编译器处理它。
4. **arch 显式设定**：`CMAKE_CUDA_ARCHITECTURES=native` 本机用；
   发布给别人时改成 `"70;75;80;86"` 这样的列表。
5. **测试一个都不能少**：`checkShape(1,1,1)` 和 `checkShape(17,31,33)`
   这类边界用例，往往比大矩阵测试更早发现 bug。

## 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `No CMAKE_CUDA_COMPILER could be found` | nvcc 不在 PATH | 安装 CUDA Toolkit 或 `export PATH=/usr/local/cuda/bin:$PATH` |
| `Unsupported gpu architecture 'compute_native'` | CMake < 3.24 | 改成显式架构，如 `-DCMAKE_CUDA_ARCHITECTURES=75` |
| nvcc 报 `error: identifier "__CUDACC__" ...` 之外的一堆语法错 | 文件被 C++ 编译器处理了 | 检查 `set_source_files_properties(... LANGUAGE CUDA)` 是否生效 |
| 链接错误 `undefined reference to sgemmGPU` | 没链接 `gemm_core` | `target_link_libraries(... PRIVATE gemm_core)` |
