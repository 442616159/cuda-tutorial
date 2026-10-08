# ============================================================
#  code/lessons/ch00_hello/CMakeLists.txt
#
#  用法（在 code/lessons/ch00_hello 目录下）：
#      cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=75
#      cmake --build build
#      ./build/hello
#
#  如果你是第一次配置，先别急着加 -DCMAKE_CUDA_ARCHITECTURES，
#  让 CMake 自己探测即可；一旦发现编译出的程序跑不动或性能异常，
#  再显式指定自己显卡的 CC（见第 4 节）。
# ============================================================

cmake_minimum_required(VERSION 3.18)   # CUDA 作为一等语言的完整支持从 3.18 起

project(cuda_ch00 LANGUAGES CXX CUDA)

# 用 C++17：CUDA 11+ 对 C++17 支持良好，且方便写现代 C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认构建类型给 Release：-O3 对 GPU 程序的性能影响非常大
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 不要用 nvcc 去链接 C++ 标准库的同名选项冲突；让 CMake 处理即可
# 关闭"把未知的 -arch 警告当成错误"这类激进行为，保证小白机器也能配通
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# 公共头文件目录（cuda_check.h / cuda_timer.h 都在 code/common 下）。
# 注意：示例代码里用的是 #include "../../common/cuda_check.h"，
#      所以其实不依赖这个 include 路径；这里加上只是方便你自己写新文件。
#      从 lessons/chXX_yy/ 到 code/common/ 是往上两级。
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../common)

# ---- 可执行文件 1：hello ----
add_executable(hello hello.cu)

# ---- 可执行文件 2：device_query ----
add_executable(device_query device_query.cu)

# 打印一下实际用的架构，方便你确认"是不是编到了我的卡上"
message(STATUS "CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")
