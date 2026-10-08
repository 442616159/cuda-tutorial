# ============================================================
#  CMakeLists.txt —— 一个"能上生产"的 CUDA 工程骨架
#
#  位置：code/projects/cmake_skeleton/CMakeLists.txt
#  （原位置为 code/ch07_cmake/CMakeLists.txt，已随教程重组迁移）
#
#  这个工程演示了四件工程化必备的事：
#    1) CUDA 是可选的：找不到 nvcc 也能编译（CPU 回退）
#    2) 同一个 .cpp 文件可以在"有 CUDA"时交给 nvcc 编译，
#       在"没 CUDA"时交给普通 C++ 编译器编译
#    3) 计算能力（arch）的检测与显式设定
#    4) 带单元测试的构建目标（ctest）
#
#  构建：
#    cd code/projects/cmake_skeleton
#    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#    cmake --build build -j
#    ctest --test-dir build --output-on-failure
# ============================================================

# CMake 3.17 才提供 find_package(CUDAToolkit)；
# 3.24 才支持 CMAKE_CUDA_ARCHITECTURES=native。这里要求 3.24。
cmake_minimum_required(VERSION 3.24)

project(cuda_cmake_demo
        VERSION 1.0.0
        DESCRIPTION "CUDA 工程化骨架示例"
        LANGUAGES CXX)          # 先只声明 C++；CUDA 稍后按需开启

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 不指定构建类型时默认 Release：Debug 下 CUDA 会关掉所有优化，
# 拿 Debug 版的性能数据去写报告是没有意义的。
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

# ------------------------------------------------------------
# 1. 检测 CUDA 是否可用
#    check_language(CUDA) 只是"试一试能不能找到 nvcc"，
#    找到了再 enable_language(CUDA)。这是 CMake 官方推荐的可选后端写法：
#    比直接 enable_language(CUDA) 更安全，因为后者找不到编译器会直接报错。
# ------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(HAVE_CUDA ON)
    message(STATUS "找到 CUDA 编译器: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA 版本: ${CMAKE_CUDA_COMPILER_VERSION}")

    # --------------------------------------------------------
    # 2. 计算能力（arch）设定
    #    CMAKE_CUDA_ARCHITECTURES 决定生成哪些 SM 的机器码。
    #    设成 native 让 CMake 探测本机 GPU（CMake >= 3.24）。
    #    交叉编译/打包分发时必须显式列出目标架构，例如 "70;75;80;86"。
    #
    #   * 只写 sm_80 的程序在 sm_75 的卡上会 JIT 重编译（首次启动慢），
    #     在老架构上甚至根本无法运行 —— 这是新手最常踩的"在我机器上能跑"。
    # --------------------------------------------------------
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "目标计算能力")
    endif()

    # 想同时给多个架构生成代码（发布用），可以这样写：
    #   set(CMAKE_CUDA_ARCHITECTURES "70-real;75-real;80-real;86-real")
    # 后缀 -real 表示"只要 cubin，不做 PTX JIT 兜底"，能显著减小体积。
else()
    set(HAVE_CUDA OFF)
    message(WARNING "未找到 CUDA 编译器，将以 CPU-only 模式构建")
endif()

# ------------------------------------------------------------
# 3. 可选项：让用户能在命令行开关某些优化
# ------------------------------------------------------------
option(GEMM_USE_FAST_MATH "使用 --use_fast_math（快但精度会下降）" OFF)
option(GEMM_SHOW_PTXAS    "编译时用 -Xptxas -v 打印寄存器/共享内存用量" OFF)

# ------------------------------------------------------------
# 4. 核心库
# ------------------------------------------------------------
add_library(gemm_core STATIC src/gemm_api.cpp)

target_include_directories(gemm_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(HAVE_CUDA)
    # --------------------------------------------------------
    # 关键技巧：把 .cpp 源文件的 LANGUAGE 改成 CUDA
    #   这样 nvcc 会处理它，文件里的 __CUDACC__ 分支生效。
    #   没 CUDA 时它就是个普通 C++ 文件，CPU 分支生效。
    #   一份源码，两条路径，接口零改动。
    # --------------------------------------------------------
    set_source_files_properties(src/gemm_api.cpp PROPERTIES LANGUAGE CUDA)

    # 只链接运行时要用的 CUDA::cudart，不引入多余的库
    target_link_libraries(gemm_core PUBLIC CUDA::cudart)

    # --------------------------------------------------------
    # CUDA_SEPARABLE_COMPILATION：
    #   默认情况下每个 .cu 是独立编译的，__device__ 函数不能跨文件调用。
    #   打开后 nvcc 会生成可重定位设备代码，并在最后做一次设备链接
    #   （需要 -rdc=true，CMake 会自动加）。
    #   代价：编译更慢、内联优化可能变差（跨文件调用无法内联）。
    #   所以：只有一个 .cu 或设备函数全在文件内时，保持 OFF 更好。
    # --------------------------------------------------------
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set_target_properties(gemm_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
        # 如果确实需要跨文件调用 __device__ 函数，改成：
        #   CUDA_SEPARABLE_COMPILATION ON
        # 同时确保最终的可执行文件也设置为 ON（设备链接在可执行文件阶段完成）
    endif()

    if(GEMM_USE_FAST_MATH)
        # --use_fast_math 会启用低精度数学函数近似、flush-to-zero 等，
        # 能快一点，但结果与 CPU 参考值的误差会明显变大。
        # 只有在确实验证过精度可接受时才打开。
        target_compile_options(gemm_core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    endif()

    if(GEMM_SHOW_PTXAS)
        # 注意这里必须写成 -Xptxas=-v（一个参数），
        # 写成 "-Xptxas -v" 会被当成单个 argv 传给 nvcc，导致报错。
        target_compile_options(gemm_core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
    endif()

    # 给上层代码一个宏，用于在需要时区分后端（可选）
    target_compile_definitions(gemm_core PUBLIC GEMM_HAS_CUDA=1)

    # 如果打开了 CUDA_SEPARABLE_COMPILATION，最终链接出可执行文件的那个
    # target 也必须打开同名属性，否则设备链接不会发生在最后一步：
    #   set_target_properties(gemm_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    target_compile_definitions(gemm_core PUBLIC GEMM_HAS_CUDA=0)
endif()

# ------------------------------------------------------------
# 5. 可执行文件
# ------------------------------------------------------------
add_executable(gemm_demo src/main.cpp)
target_link_libraries(gemm_demo PRIVATE gemm_core)

# ------------------------------------------------------------
# 6. 测试
#    enable_testing() + add_test() 之后，ctest 就能自动跑全部用例。
#    测试目标的源码里不许有 CUDA 语法，这样 CI 的 CPU 机器也能跑。
# ------------------------------------------------------------
enable_testing()
add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE gemm_core)
add_test(NAME gemm_correctness COMMAND test_gemm)

# 一条命令把测试和测试结果打印出来
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_gemm
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "运行全部测试")

# ------------------------------------------------------------
# 7. 构建信息汇总（排错时非常有用）
# ------------------------------------------------------------
message(STATUS "----------------------------------------")
message(STATUS "  CUDA 支持      : ${HAVE_CUDA}")
message(STATUS "  构建类型       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ 编译器     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(HAVE_CUDA)
    message(STATUS "  CUDA 架构      : ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "----------------------------------------")
