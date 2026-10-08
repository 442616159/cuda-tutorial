# ============================================================
#  CUDALearning 示例代码统一构建文件
#
#  位置：CUDALearning/code/Makefile
#
#  用法（在 code/ 目录下执行）：
#    make            # 编译所有示例
#    make list       # 列出所有会被编译的示例
#    make ch02       # 只编译第 2 章的示例
#    make clean      # 删除所有编译产物
#
#  可执行文件统一输出到 code/bin/，文件名由"相对路径压平"而成，
#  例如 lessons/ch02_memory/matrix_mul.cu -> bin/lessons_ch02_memory_matrix_mul
#
#  关于 -arch：
#    Makefile 会自动用 nvidia-smi 探测本机计算能力；探测不到时回退 sm_70。
#    你可以显式覆盖，例如：make ARCH="-arch=sm_86"
# ============================================================

NVCC      ?= nvcc

# 根据本机显卡自动检测计算能力；检测失败则回退到 7.0
DETECT_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
ifeq ($(DETECT_CC),)
  ARCH ?= -arch=sm_70
else
  ARCH ?= -arch=sm_$(shell echo $(DETECT_CC) | tr -d '.')
endif

NVCCFLAGS ?= -O3 $(ARCH) -lineinfo -std=c++17
LDFLAGS   ?= -lcublas -lcufft -lcurand -lcusparse

BIN_DIR   := bin
SRCS      := $(shell find . -name '*.cu' -not -path './bin/*' | sort)

# 把 lessons/ch02_memory/matrix_mul.cu 映射成 bin/lessons_ch02_memory_matrix_mul
TARGETS   := $(foreach s,$(SRCS),$(BIN_DIR)/$(subst /,_,$(patsubst ./%.cu,%,$(s))))

.PHONY: all list clean ch00 ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08

all: $(TARGETS)

list:
	@echo "检测到的计算能力: $(ARCH)"
	@echo "将编译以下示例:"
	@for t in $(TARGETS); do echo "  $$t"; done

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# ------------------------------------------------------------
# 通用规则
# ------------------------------------------------------------
$(BIN_DIR)/%: $(BIN_DIR)
	@real=""; \
	for f in $(SRCS); do \
	  key=$$(echo $$f | sed 's|^\./||; s|\.cu$$||; s|/|_|g'); \
	  if [ "$$key" = "$*" ]; then real=$$f; break; fi; \
	done; \
	if [ -z "$$real" ]; then echo "找不到源文件: $*"; exit 1; fi; \
	case "$$real" in \
	  *dynparallel*)  EXTRA="-rdc=true" ;; \
	  *cluster_dsm*)  EXTRA="-arch=sm_90" ;; \
	  *vmm_alloc*)    EXTRA="-lcuda" ;; \
	  *)              EXTRA="" ;; \
	esac; \
	echo "  NVCC  $$real $$EXTRA  ->  $@"; \
	$(NVCC) $(NVCCFLAGS) $$EXTRA $$real -o $@ $(LDFLAGS) || exit 1

ch00 ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08:
	@$(MAKE) --no-print-directory $(BIN_DIR)/lessons_$@_* $(BIN_DIR)/$@_*

clean:
	@rm -rf $(BIN_DIR)
	@echo "已清理 $(BIN_DIR)/"

# ------------------------------------------------------------
# 需要单独处理的示例（通用规则已自动处理，这里仅供你手动编译时参考）：
#
#   ch06 集群（需要 Hopper / CC 9.0）：
#     nvcc -O3 -arch=sm_90 lessons/ch06_advanced/cluster_dsm.cu -o cluster_dsm
#
#   ch06 动态并行（需要 -rdc=true）：
#     nvcc -O3 -arch=sm_70 -rdc=true lessons/ch06_advanced/dynparallel.cu -o dynparallel
#
#   ch06 虚拟内存管理（Driver API，需要链接 cuda 库）：
#     nvcc -O3 -arch=sm_70 lessons/ch06_advanced/vmm_alloc.cu -o vmm_alloc -lcuda
#
#   ch07 cuDNN（需要单独安装 cuDNN）：
#     nvcc -O3 -arch=sm_70 lessons/ch07_ecosystem/cudnn_conv_forward.cu -o conv -lcudnn
# ------------------------------------------------------------
