# SGEMM：从朴素到接近 cuBLAS

第 8 章主推项目的参考实现。7 个渐进版本，每个版本独立可编译，
共用 `sgemm_common.h` 里的计时/验证/报告基础设施。

## 目录结构

```
projects/sgemm/
├── sgemm_common.h          # 公共：CudaTimer、CPU 参考实现、benchmark、结果输出
├── v1_naive.cu             # 朴素三重循环
├── v2_global_tile.cu       # 寄存器分块（仍读全局内存）
├── v3_shared_tile.cu       # 共享内存分块
├── v4_thread_tile.cu       # 共享内存 + 4x4 线程级分块
├── v5_float4.cu            # + float4 向量化访存
├── v6_double_buffer.cu     # + 双缓冲软件流水线
├── v7_reg_tile_8x8.cu      # + 128x128 大块 + 8x8 寄存器分块
├── cublas_ref.cu           # cuBLAS 参照（性能天花板）
├── Makefile                # 一键构建/测试/清理
├── bench_all.sh            # 依次跑所有版本，输出汇总表
├── PERF_REPORT_TEMPLATE.md # 性能报告模板
└── README.md               # 本文件
```

## 构建

### 方式一：Makefile（Linux / WSL / MinGW）

```bash
make -j            # 构建全部 8 个可执行文件
make bench         # 跑全部版本并打印汇总表
make clean
```

### 方式二：直接调用 nvcc

```bash
nvcc -O3 -arch=sm_70 v1_naive.cu -o sgemm_v1
nvcc -O3 -arch=sm_70 v7_reg_tile_8x8.cu -Xptxas -v -o sgemm_v7
nvcc -O3 -arch=sm_70 cublas_ref.cu -o sgemm_cublas -lcublas
```

把 `sm_70` 换成你自己显卡的计算能力（用 `nvidia-smi --query-gpu=compute_cap --format=csv`
或 `code/lessons/ch00_hello/device_query` 查询）。CUDA 11.5+ 也可以直接用 `-arch=native`
让编译器自动探测。

## 运行与复现

```bash
./sgemm_v1 1024 1024 1024
./sgemm_v7 2048 2048 2048
./sgemm_cublas 2048 2048 2048
```

不带参数时默认 `1024 1024 1024`。**报告里必须写明测试的具体形状**，
因为不同形状下各版本的相对表现会变（K 很小的时候尤其明显）。

## 正确性验证

每个可执行文件都会：
1. 用 CPU 的 i-k-j 分块实现算出参考结果；
2. 用相对误差（阈值 `1e-3`）与 GPU 结果对比；
3. 正确性失败时返回非零退出码。

```bash
for f in sgemm_v1 sgemm_v2 sgemm_v3 sgemm_v4 sgemm_v5 sgemm_v6 sgemm_v7 sgemm_cublas; do
    ./$f 1024 1024 1024 > /dev/null && echo "$f PASS" || echo "$f FAIL"
done
```

## 复现步骤（写进报告用）

```bash
git clone <repo> && cd code/projects/sgemm
make -j
./bench_all.sh 2>&1 | tee perf_$(date +%Y%m%d).txt
ncu --set full --launch-count 1 ./sgemm_v7 1024 1024 1024
nsys profile -o sgemm_v7 --stats=true ./sgemm_v7 2048 2048 2048
```

## 已知限制

- `v5`~`v7` 在 M/N/K 不是分块大小整数倍时，走的是边界保护路径，
  性能会下降（功能仍然正确）。
- `v7` 的 `Bs` 在 float4 读取时存在 2 路 bank conflict，
  见第 8 章练习 5。
- 只实现了 `float`，没有做双精度（`double` 需要完全不同的寄存器预算）。
