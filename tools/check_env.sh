#!/usr/bin/env bash
# ============================================================
#  check_env.sh —— 一键体检你的 CUDA 环境
#
#  位置：CUDALearning/tools/check_env.sh
#
#  用法（Linux / macOS / Git Bash / WSL）：
#      bash check_env.sh
#
#  Windows PowerShell 用户请用同目录的 check_env.ps1
# ============================================================

set -u

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { printf "${GREEN}[ OK ]${NC}  %s\n" "$1"; }
bad()  { printf "${RED}[FAIL]${NC}  %s\n" "$1"; }
warn() { printf "${YELLOW}[WARN]${NC}  %s\n" "$1"; }

echo "=============================================="
echo "        CUDA 学习环境体检"
echo "=============================================="

FAIL=0

# ---------- 1. 显卡驱动 ----------
echo
echo "--- 1. NVIDIA 驱动与显卡 ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "找到 nvidia-smi"
    if nvidia-smi >/dev/null 2>&1; then
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
        GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
        echo "        显卡型号   : ${GPUNAME:-未知}"
        echo "        驱动版本   : ${DRIVER:-未知}"
        echo "        计算能力   : ${CC:-未知}"
    else
        bad "nvidia-smi 存在但执行失败，驱动可能没装好或显卡不可用"
        FAIL=1
    fi
else
    bad "找不到 nvidia-smi，说明没装 NVIDIA 驱动，或没有 NVIDIA 显卡"
    FAIL=1
fi

# ---------- 2. CUDA Toolkit ----------
echo
echo "--- 2. CUDA Toolkit ---"
if command -v nvcc >/dev/null 2>&1; then
    NVCC_VER=$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -n1 | cut -d' ' -f2)
    ok "找到 nvcc，版本 release ${NVCC_VER:-未知}"
    echo "        路径       : $(command -v nvcc)"
    if [ -n "${NVCC_VER:-}" ]; then
        MAJOR=${NVCC_VER%%.*}
        if [ "${MAJOR:-0}" -lt 11 ] 2>/dev/null; then
            warn "CUDA 版本较老（<11.0），第 6 章的部分高级特性无法编译"
        fi
    fi
else
    bad "找不到 nvcc，CUDA Toolkit 没装或没加入 PATH"
    echo "        Linux  : export PATH=/usr/local/cuda/bin:\$PATH"
    echo "        Windows: 确认 CUDA\\v12.x\\bin 已加入系统环境变量 Path"
    FAIL=1
fi

# ---------- 3. 主机编译器 ----------
echo
echo "--- 3. 主机 C++ 编译器 ---"
if command -v g++ >/dev/null 2>&1; then
    ok "g++ $(g++ -dumpversion)"
elif command -v clang++ >/dev/null 2>&1; then
    ok "clang++ $(clang++ --version | head -n1)"
elif [ -n "${CXX:-}" ]; then
    warn "使用环境变量 CXX=$CXX"
else
    bad "找不到 g++ / clang++，无法编译主机端代码"
    FAIL=1
fi

# ---------- 4. 构建工具 ----------
echo
echo "--- 4. 构建工具 ---"
if command -v make >/dev/null 2>&1; then
    ok "make $(make --version | head -n1 | awk '{print $3}')"
else
    warn "找不到 make，可以用 nvcc 命令逐个手动编译"
fi
if command -v cmake >/dev/null 2>&1; then
    ok "cmake $(cmake --version | head -n1 | awk '{print $3}')"
else
    warn "找不到 cmake，第 7 章工程化部分需要它（非必须）"
fi

# ---------- 5. 性能分析工具 ----------
echo
echo "--- 5. 性能分析工具（第 4 章要用）---"
for t in ncu nsys compute-sanitizer; do
    if command -v $t >/dev/null 2>&1; then
        ok "$t"
    else
        warn "未找到 $t"
    fi
done

# ---------- 6. 实机编译测试 ----------
echo
echo "--- 6. 实机编译测试 ---"
TMPDIR_TEST=$(mktemp -d 2>/dev/null || echo "/tmp/cuda_test_$$")
mkdir -p "$TMPDIR_TEST"
cat > "$TMPDIR_TEST/t.cu" <<'EOF'
#include <cstdio>
__global__ void k(int *d) { *d = 42; }
int main() {
    int *d; cudaMalloc(&d, sizeof(int));
    k<<<1,1>>>(d);
    int h = 0; cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    printf("GPU 计算结果 = %d\n", h);
    return h == 42 ? 0 : 1;
}
EOF
if command -v nvcc >/dev/null 2>&1; then
    if nvcc -O2 "$TMPDIR_TEST/t.cu" -o "$TMPDIR_TEST/t" >"$TMPDIR_TEST/log" 2>&1; then
        if OUT=$("$TMPDIR_TEST/t" 2>&1); then
            ok "编译 + 运行成功：$OUT"
        else
            bad "编译成功但运行失败（显卡/驱动问题），输出: $OUT"
            FAIL=1
        fi
    else
        bad "编译失败，错误日志："
        sed 's/^/        /' "$TMPDIR_TEST/log"
        FAIL=1
    fi
fi
rm -rf "$TMPDIR_TEST"

# ---------- 结论 ----------
echo
echo "=============================================="
if [ "$FAIL" -eq 0 ]; then
    printf "${GREEN}环境体检通过，可以开始第 0 章了！${NC}\n"
else
    printf "${RED}存在问题，请先按上面提示修复。${NC}\n"
fi
echo "=============================================="
exit "$FAIL"
