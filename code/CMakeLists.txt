# ============================================================
#  code/CMakeLists.txt —— 把 lessons/ 与 projects/ 统一纳入一个构建
#
#  用法（在 code/ 目录下）：
#      cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=75
#      cmake --build build -j
#
#  说明：
#    * 每个子目录都有自己独立的 CMakeLists.txt，可以单独构建。
#      这个顶层文件只是把它们 add_subdirectory 进来，方便一次编完。
#    * 需要特殊编译参数的示例（集群、动态并行、VMM）在各自的
#      子目录里单独处理，见 tools/ 下的说明与各章教程正文。
#    * 如果某个子目录缺 CMakeLists.txt，会被自动跳过，不会报错。
# ============================================================

cmake_minimum_required(VERSION 3.18)

project(cuda_learning_examples
        VERSION 1.0.0
        DESCRIPTION "CUDA 从零到实战 —— 全部示例代码"
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

# 默认让 CMake 探测本机 GPU；旧版本 CMake 请显式指定，例如：
#   cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND
   CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "目标计算能力")
endif()

# 公共头文件（cuda_check.h / cuda_timer.h / cuda_bench.h）
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# ------------------------------------------------------------
# 自动收集所有含 CMakeLists.txt 的子目录
# ------------------------------------------------------------
file(GLOB_RECURSE SUB_CMAKE
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/lessons/*/CMakeLists.txt
     ${CMAKE_CURRENT_SOURCE_DIR}/projects/*/CMakeLists.txt)

foreach(sub ${SUB_CMAKE})
    get_filename_component(subdir ${sub} DIRECTORY)
    message(STATUS "加入子目录: ${subdir}")
    add_subdirectory(${subdir})
endforeach()

# ------------------------------------------------------------
# 汇总信息
# ------------------------------------------------------------
message(STATUS "----------------------------------------")
message(STATUS "  CUDA 架构  : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  构建类型   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  子目录数   : ${CMAKE_CURRENT_LIST_LENGTH}")
message(STATUS "----------------------------------------")
message(STATUS "提示：只想编某章的话，直接进那个子目录单独 cmake 更快。")
