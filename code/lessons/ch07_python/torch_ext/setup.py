# ============================================================
#  setup.py —— 构建 PyTorch CUDA 扩展
#
#  推荐做法（也是本文件采用的）：
#      from torch.utils.cpp_extension import CUDAExtension, BuildExtension
#  不用自己写 nvcc 命令，PyTorch 会帮你补齐所有 include / lib 路径，
#  并且自动处理 ABI、arch 列表（TORCH_CUDA_ARCH_LIST）等繁琐细节。
#
#  构建：
#      python setup.py install            # 安装进当前 Python 环境
#      python setup.py build_ext --inplace  # 只生成 .pyd/.so 到当前目录
#  调试：把 extra_compile_args 里的注释打开，能看到寄存器用量和行号信息。
# ============================================================

import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 让 nvcc 生成的中间文件不污染源码目录
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(os.getcwd(), "build_ext"))

# 想让扩展支持多个 GPU 架构（发布给别人用时需要）：
#   export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
# 只在本机用，可以不设，PyTorch 会探测当前显卡。
if "TORCH_CUDA_ARCH_LIST" not in os.environ and not sys.platform.startswith("win"):
    print("[setup.py] 未设置 TORCH_CUDA_ARCH_LIST，将使用 PyTorch 自动探测的架构")

setup(
    name="vector_add_ext",
    version="1.0.0",
    description="PyTorch CUDA 扩展示例：向量加法",
    # CUDAExtension 会自动：
    #   * 用 nvcc 编译 .cu，用 g++/MSVC 编译 .cpp
    #   * 链接 cudart、libtorch、torch_python
    #   * 处理 -gencode 参数
    ext_modules=[
        CUDAExtension(
            name="vector_add_ext",
            sources=[
                "vector_add_cuda.cu",
            ],
            include_dirs=[],
            extra_compile_args={
                # nvcc 的参数
                "nvcc": [
                    "-O3",
                    # "-lineinfo",            # 想在 nsys/ncu 里看到源码行号就打开
                    # "-Xptxas", "-v",        # 打印寄存器/共享内存用量
                    # "--use_fast_math",      # 快，但精度下降，慎用
                ],
                # C++ 编译器的参数
                "cxx": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
