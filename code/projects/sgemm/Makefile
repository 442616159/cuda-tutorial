# ============================================================
#  SGEMM 项目 Makefile
#
#  用法：
#    make            构建全部版本
#    make bench      跑全部版本并输出汇总
#    make verify     只做正确性验证
#    make profile    用 ncu 分析 v7（需要 Nsight Compute）
#    make clean
#
#  如果你的 nvcc 支持 -arch=native（CUDA 11.5+ 配合较新驱动），
#  可以让它自动探测本机架构；否则请显式指定，例如：
#    make ARCH=-arch=sm_75
#  默认为 sm_70，兼容性最好（几乎在任何 CUDA 11+ 上都能编过）。
# ============================================================

NVCC      ?= nvcc
ARCH      ?= -arch=sm_70
NVFLAGS   ?= -O3 $(ARCH) -lineinfo
CUBLAS    ?= -lcublas

VERSIONS  := v1_naive v2_global_tile v3_shared_tile v4_thread_tile \
             v5_float4 v6_double_buffer v7_reg_tile_8x8

BINS      := $(addprefix sgemm_,$(VERSIONS)) sgemm_cublas

# 测试形状。改这里就能把整套基准跑成另一个形状
SHAPE     ?= 1024 1024 1024

.PHONY: all bench verify profile clean list

all: $(BINS)

# 通用规则：sgemm_<name> 由 <name>.cu 生成
sgemm_%: %.cu sgemm_common.h ../../common/cuda_check.h
	$(NVCC) $(NVFLAGS) $< -o $@

# cuBLAS 参照需要额外链接
sgemm_cublas: cublas_ref.cu sgemm_common.h ../../common/cuda_check.h
	$(NVCC) $(NVFLAGS) $< -o $@ $(CUBLAS)

list:
	@echo $(BINS)

# 只跑正确性，不关心性能。任何一步失败都会中断整个 make。
verify: all
	@for b in $(BINS); do \
	    printf "%-24s" "$$b"; \
	    ./$$b $(SHAPE) > /dev/null 2>&1 && echo "PASS" || (echo "FAIL"; exit 1); \
	done

bench: all
	@for b in $(BINS); do \
	    echo "=============================================="; \
	    ./$$b $(SHAPE) || exit 1; \
	done

profile: sgemm_v7
	ncu --set full --launch-count 1 ./sgemm_v7 $(SHAPE)

nsys: sgemm_v7
	nsys profile --stats=true -o sgemm_v7_report ./sgemm_v7 $(SHAPE)

clean:
	rm -f $(BINS) *.o *.exp *.lib
	rm -rf sgemm_v7_report.*
