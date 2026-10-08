# ============================================================
#  code/lessons/ch01_indices/CMakeLists.txt
#
#  用法（在 code/lessons/ch01_indices 目录下）：
#      cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=75
#      cmake --build build --config Release
#      ./build/vector_add
# ============================================================

cmake_minimum_required(VERSION 3.18)
project(cuda_ch01 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 设备端也要 -O3；-Xptxas -v 会打印寄存器/共享内存用量，方便调试
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v --expt-relaxed-constexpr")

# 从 lessons/ch01_indices/ 到 code/common/ 是往上两级
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../common)

add_executable(vector_add   vector_add.cu)
add_executable(print_index  print_index.cu)
add_executable(matrix_add   matrix_add.cu)

message(STATUS "CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")
