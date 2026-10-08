# ============================================================
#  code/lessons/ch02_memory/CMakeLists.txt
#
#  用法（在 code/lessons/ch02_memory 目录下）：
#      cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=75
#      cmake --build build --config Release
#      ./build/matrix_mul
#
#  注意：如果你改动了任何 .cu 源文件里的宏，
#  CMake 会自己做依赖跟踪，不需要手动 make clean。
# ============================================================

cmake_minimum_required(VERSION 3.18)
project(cuda_ch02 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -Xptxas -v：打印每个核函数的寄存器/共享内存/spill 统计。
# 这对第 2 章来说是"标配调试信息"，强烈建议一直开着。
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v --expt-relaxed-constexpr")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../common)

add_executable(matrix_mul      matrix_mul.cu)
add_executable(transpose       transpose.cu)
add_executable(coalescing      coalescing.cu)
add_executable(constant_ldg    constant_ldg.cu)
add_executable(bank_conflict   bank_conflict.cu)
add_executable(unified_memory  unified_memory.cu)
add_executable(reduce_shared   reduce_shared.cu)
add_executable(occupancy       occupancy.cu)

# 把可执行文件集中到一个目录，方便命令行里 ./bin/xxx 直接跑
set_target_properties(matrix_mul transpose coalescing constant_ldg
                      bank_conflict unified_memory reduce_shared occupancy
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                      ${CMAKE_CURRENT_BINARY_DIR}/bin)

message(STATUS "CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")
