#!/usr/bin/env bash
# ============================================================
#  bench_all.sh —— 一键跑完所有 SGEMM 版本，输出汇总表
#
#  用法：
#    ./bench_all.sh                 # 默认 1024 1024 1024
#    ./bench_all.sh 2048 2048 2048  # 指定形状
#    ./bench_all.sh 2>&1 | tee perf.txt
# ============================================================
set -u

M=${1:-1024}
N=${2:-1024}
K=${3:-1024}

BINS=(
  sgemm_v1_naive
  sgemm_v2_global_tile
  sgemm_v3_shared_tile
  sgemm_v4_thread_tile
  sgemm_v5_float4
  sgemm_v6_double_buffer
  sgemm_v7_reg_tile_8x8
  sgemm_cublas
)

echo "############################################################"
echo "#  SGEMM 基准测试  形状 = ${M} x ${N} x ${K}"
echo "#  日期: $(date)"
echo "#  设备: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "#  驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "#  nvcc: $(nvcc --version 2>/dev/null | tail -1)"
echo "############################################################"

fail=0
for b in "${BINS[@]}"; do
    if [ ! -x "./$b" ]; then
        echo "!! $b 不存在，先执行 make"
        fail=1
        continue
    fi
    echo
    echo "==================== $b ===================="
    if ! "./$b" "$M" "$N" "$K"; then
        echo "!! $b 执行失败（可能是正确性检查未通过）"
        fail=1
    fi
done

echo
echo "############################################################"
if [ "$fail" -eq 0 ]; then
    echo "# 全部版本执行完毕，且正确性检查通过。"
    echo "# 把上面的表格复制进 PERF_REPORT.md，并补齐环境信息。"
else
    echo "# 存在失败项，请先修复再填写性能报告。"
fi
echo "############################################################"
exit $fail
