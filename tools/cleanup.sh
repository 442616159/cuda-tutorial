#!/usr/bin/env bash
# ============================================================
#  cleanup.sh —— 清理重组后残留的旧文件与旧目录（Linux / macOS / WSL / Git Bash）
#
#  位置：CUDALearning/tools/cleanup.sh
#
#  背景：
#    code/ 目录从 40 个扁平目录重组为 lessons/ + projects/ 结构时，
#    新文件已经写好，但旧的源文件/源目录可能还留着，造成重复。
#
#  用法：
#    bash tools/cleanup.sh              # 预演（只报告，不删）
#    bash tools/cleanup.sh --execute    # 真正删除
#
#  安全措施：
#    * 默认预演模式，不加 --execute 不删任何东西
#    * 删除前检查"新位置确实存在且非空"，否则跳过
# ============================================================

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/code"

EXECUTE=0
[ "${1:-}" = "--execute" ] && EXECUTE=1

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; GRAY='\033[0;90m'; CYAN='\033[0;36m'; NC='\033[0m'

echo "=============================================="
echo "  重组残留清理"
if [ "$EXECUTE" -eq 1 ]; then
    echo -e "  模式：${RED}真正删除${NC}"
else
    echo -e "  模式：${YELLOW}预演（不会删除任何东西）${NC}"
    echo -e "  确认无误后加 ${YELLOW}--execute${NC} 重新运行"
fi
echo "=============================================="

# 旧目录 -> 新目录
MIGRATIONS="
ch00_hello:lessons/ch00_hello
ch01_vector_add:lessons/ch01_indices
ch02_matrix_mul:lessons/ch02_memory
ch03_sync:lessons/ch03_sync/barrier
ch03_syncwarp:lessons/ch03_sync/warp_sync
ch03_threadfence:lessons/ch03_sync/fence
ch03_shuffle_reduce:lessons/ch03_sync/shuffle
ch03_vote:lessons/ch03_sync/vote
ch03_atomics:lessons/ch03_atomics
ch03_histogram:lessons/ch03_atomics_histogram
ch03_reduce_cg:lessons/ch03_cooperative
ch03_streams:lessons/ch03_streams
ch03_graphs:lessons/ch03_graphs
ch04_timer:lessons/ch04_timing/timer
ch04_bandwidth_test:lessons/ch04_timing/bandwidth
ch04_roofline:lessons/ch04_perf_model
ch04_occupancy:lessons/ch04_occupancy
ch04_matmul_opt:lessons/ch04_case_study
ch04_sanitizer_bugs:lessons/ch04_debugging
ch05_reduce:lessons/ch05_reduce
ch05_scan:lessons/ch05_scan
ch05_stencil:lessons/ch05_stencil
ch05_sort:lessons/ch05_sort
ch05_spmv:lessons/ch05_spmv
ch05_bfs:lessons/ch05_graph
ch05_numeric:lessons/ch05_numeric
ch05_cg:lessons/ch05_cg
ch06_dynparallel:lessons/ch06_advanced/dynparallel
ch06_wmma:lessons/ch06_advanced/wmma
ch06_cpasync:lessons/ch06_advanced/cpasync
ch06_cluster:lessons/ch06_advanced/cluster
ch06_unified:lessons/ch06_advanced/unified
ch06_vmm:lessons/ch06_advanced/vmm
ch06_multigpu:lessons/ch06_advanced/multigpu
ch06_graphs:lessons/ch06_advanced/graphs_advanced
ch07_cublas:lessons/ch07_ecosystem/cublas
ch07_cufft:lessons/ch07_ecosystem/cufft
ch07_curand:lessons/ch07_ecosystem/curand
ch07_thrust:lessons/ch07_ecosystem/thrust
ch07_cub:lessons/ch07_ecosystem/cub
ch07_cusparse:lessons/ch07_ecosystem/cusparse
ch07_cudnn:lessons/ch07_ecosystem/cudnn
ch07_python:lessons/ch07_python
ch07_cmake:projects/cmake_skeleton
ch08_project_sgemm:projects/sgemm
ch08_project_raytracer:projects/raytracer
ch08_project_nbody:projects/nbody
"

echo
echo "--- 1. 清理 code/ 下的旧目录 ---"
removed=0; skipped=0

for entry in $MIGRATIONS; do
    old="${entry%%:*}"
    new="${entry##*:}"
    oldPath="$CODE/$old"
    newPath="$CODE/$new"

    [ -e "$oldPath" ] || continue

    if [ ! -e "$newPath" ]; then
        echo -e "  ${YELLOW}[跳过]${NC} $old —— 新位置不存在: $new"
        skipped=$((skipped+1)); continue
    fi

    count=$(find "$newPath" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -eq 0 ]; then
        echo -e "  ${YELLOW}[跳过]${NC} $old —— 新位置是空的: $new"
        skipped=$((skipped+1)); continue
    fi

    if [ "$EXECUTE" -eq 1 ]; then
        rm -rf "$oldPath"
        if [ -e "$oldPath" ]; then
            echo -e "  ${RED}[失败]${NC} $old"
            skipped=$((skipped+1))
        else
            echo -e "  ${GREEN}[删除]${NC} $old  (新: $new, $count 个文件)"
            removed=$((removed+1))
        fi
    else
        echo -e "  ${GRAY}[待删]${NC} $old -> $new ($count 个文件已就位)"
        removed=$((removed+1))
    fi
done

echo
echo "--- 2. 清理根目录下被取代的文件 ---"
SUPERSEDED="
check_env.sh:tools/check_env.sh
check_env.ps1:tools/check_env.ps1
verify_all.ps1:tools/verify_all.ps1
fix_links.py:tools/fix_links.py
STYLE_GUIDE.md:CONTRIBUTING.md
"

for entry in $SUPERSEDED; do
    old="${entry%%:*}"; new="${entry##*:}"
    oldPath="$ROOT/$old"; newPath="$ROOT/$new"

    [ -e "$oldPath" ] || continue

    if [ ! -e "$newPath" ]; then
        echo -e "  ${YELLOW}[跳过]${NC} $old —— 替代文件不存在: $new"
        skipped=$((skipped+1)); continue
    fi

    if [ "$EXECUTE" -eq 1 ]; then
        rm -f "$oldPath"
        if [ -e "$oldPath" ]; then
            echo -e "  ${RED}[失败]${NC} $old"; skipped=$((skipped+1))
        else
            echo -e "  ${GREEN}[删除]${NC} $old (已由 $new 取代)"; removed=$((removed+1))
        fi
    else
        echo -e "  ${GRAY}[待删]${NC} $old -> 已由 $new 取代"
        removed=$((removed+1))
    fi
done

echo
echo "=============================================="
if [ "$EXECUTE" -eq 1 ]; then
    echo -e "${CYAN}已删除 $removed 项，跳过 $skipped 项${NC}"
else
    echo -e "${CYAN}预演结果：将删除 $removed 项，跳过 $skipped 项${NC}"
    echo
    echo -e "${YELLOW}确认无误后执行：${NC}"
    echo "  bash tools/cleanup.sh --execute"
fi
echo "=============================================="
