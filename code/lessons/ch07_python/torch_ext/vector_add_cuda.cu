// ============================================================
//  vector_add_cuda.cu —— PyTorch 自定义 CUDA 扩展
//
//  用途：PyTorch 里没有的算子，或者你想替换掉某个性能不够的算子时，
//  写一个 CUDA 扩展，然后在 Python 里像调用普通 torch 函数一样调用它。
//
//  这个文件**不能**单独用 nvcc 编译，必须通过 setup.py 构建，
//  因为它要链接 PyTorch 的头文件和 libtorch。
//
//  构建与使用见同目录 setup.py 与 test_ext.py：
//    python setup.py install
//    python test_ext.py
//
//  三件必须做对的事（新手最常出错的地方）：
//    1) 输入张量必须 .contiguous()：PyTorch 张量可以是任意 stride 的视图，
//       直接按连续内存去索引会读到错的数据。
//    2) 输入张量必须在 CUDA 上、dtype 必须匹配：用 TORCH_CHECK 提前拦住。
//    3) 核函数启动后要 C10_CUDA_KERNEL_LAUNCH_CHECK()，
//       否则 kernel 里的错误会延迟到下一次同步才爆出来，极难定位。
// ============================================================

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// ------------------------------------------------------------
// 核函数：向量加法，和你在第 1 章写的一模一样
// 之所以元素数用 int64_t 的 numel 但要转成 int，
// 是因为 grid 计算用的是 int；超大张量要改用索引类型为 long long 的版本。
// ------------------------------------------------------------
__global__ void vectorAddKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ out,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];   // 越界检查不能省
}

// ------------------------------------------------------------
// C++ 启动函数：它是 Python 与 kernel 之间的"翻译层"
//   * 负责参数校验
//   * 负责准备输出张量
//   * 负责配置 grid/block 并启动 kernel
// ------------------------------------------------------------
torch::Tensor vectorAdd(torch::Tensor a, torch::Tensor b) {
    // TORCH_CHECK 检查失败会抛异常，比段错误友好得多
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "输入必须是 CUDA 张量");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                b.scalar_type() == torch::kFloat32, "只支持 float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "两个张量形状必须一致");

    // contiguous()：如果张量本身就是连续的，它不会拷贝，只返回自身，
    // 所以这是一个"零成本的安全网"。
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto out = torch::empty_like(a_c);   // 在同一个设备上分配输出

    const int n = (int)a_c.numel();
    if (n == 0) return out;

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // data_ptr<float>() 拿到设备指针；当前流用 at::cuda::getCurrentCUDAStream()，
    // 这样扩展就自动兼容 torch.cuda.stream(...) 的流语义，
    // 否则会跑到默认流上，和 PyTorch 的其它算子乱序执行。
    vectorAddKernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a_c.data_ptr<float>(), b_c.data_ptr<float>(), out.data_ptr<float>(), n);

    // 立即检查启动是否成功（内核错误是异步的，早检查早发现）
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// ------------------------------------------------------------
// 绑定到 Python 模块
//   PYBIND11_MODULE 的第一个参数必须与 setup.py 里 CUDAExtension 的名字一致
// ------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vectorAdd, "向量加法 (CUDA)，c = a + b");
}
