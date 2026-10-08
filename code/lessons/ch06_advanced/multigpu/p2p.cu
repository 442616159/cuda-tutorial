// ============================================================================
//  code/lessons/ch06_advanced/multigpu/p2p.cu
//
//  多 GPU 编程：点对点（Peer-to-Peer, P2P）直接访问
//
//  // 需要至少 2 张支持 CUDA 的显卡；单卡机器上程序会打印说明后正常退出
//
//  编译: nvcc -O3 -arch=sm_70 p2p.cu -o p2p
//  运行: ./p2p
//
//  两张卡之间搬数据本来有两条路：
//    A. 走主机：GPU0 -> 固定内存 -> CPU -> 固定内存 -> GPU1（两次 PCIe 往返）
//    B. 走 P2P：GPU0 -> PCIe/NVLink -> GPU1（一次直达）
//  cudaMemcpyPeer / cudaMemcpyPeerAsync 走的是 B；
//  开了 cudaDeviceEnablePeerAccess 之后，kernel 还能**直接解引用**另一张卡的指针。
//
//  单机多卡之上就是多机多卡，业界标准是 NCCL：
//    ncclCommInitAll / ncclAllReduce / ncclBroadcast / ncclSend / ncclRecv
//    它内部会自动选择 NVLink、PCIe P2P、共享内存或网络（IB/RoCE）作为传输层，
//    第 7 章会给出一个可运行的 ncclAllReduce 例子。
// ============================================================================
#include "../../../common/cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr size_t N = 1 << 24;  // 64 MB 的 float 数据

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

__global__ void scale(float* p, float a, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] *= a;
}

int main() {
    int nDev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDev));
    printf("\n[设备] 检测到 %d 个 CUDA 设备\n", nDev);
    for (int i = 0; i < nDev; ++i) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        printf("  dev %d: %-32s CC %d.%d  %.1f GB  %s\n", i, p.name, p.major, p.minor,
               p.totalGlobalMem / 1024.0 / 1024.0 / 1024.0,
               p.unifiedAddressing ? "统一寻址" : "非统一寻址");
    }

    if (nDev < 2) {
        printf("\n[跳过] 只有 1 张卡，无法演示 P2P。\n");
        printf("\nP2P 的判定与开启流程（多卡机器上照抄即可）：\n");
        printf("  1) cudaDeviceCanAccessPeer(&ok, i, j)  查询 i 能否直接访问 j；\n");
        printf("  2) cudaSetDevice(i); cudaDeviceEnablePeerAccess(j, 0);  开启；\n");
        printf("  3) 之后 cudaMemcpyPeer(dst, j, src, i, bytes) 就会走 P2P 通道；\n");
        printf("  4) 在 i 上启动的 kernel 可以直接解引用 j 上 cudaMalloc 的指针\n");
        printf("     （要求两卡通过 NVLink 或同一 PCIe 交换机相连）。\n");
        printf("\n常见坑：\n");
        printf("  * 某些主板/PCIe 拓扑下 cudaDeviceCanAccessPeer 返回 0；\n");
        printf("  * 开启 P2P 后两张卡的统一寻址（UVA）行为可能变化；\n");
        printf("  * 多进程多卡时，跨进程访问需要 CUDA IPC（cudaIpcGetMemHandle），\n");
        printf("    共享的是显存句柄而不是指针；\n");
        printf("  * 集合通信（AllReduce/Broadcast）不要自己写，用 NCCL。\n");
        return 0;
    }

    // ---- 1. 打印 P2P 能力矩阵 ----
    printf("\n[拓扑] cudaDeviceCanAccessPeer 能力矩阵（行=源，列=目的）\n     ");
    for (int j = 0; j < nDev; ++j) printf("  to%d", j);
    printf("\n");
    for (int i = 0; i < nDev; ++i) {
        printf("from%d", i);
        for (int j = 0; j < nDev; ++j) {
            if (i == j) {
                printf("    -");
                continue;
            }
            int ok = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&ok, i, j));
            printf("   %s", ok ? "Y" : "n");
        }
        printf("\n");
    }

    int can01 = 0, can10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    const bool p2pOK = (can01 != 0) && (can10 != 0);

    // ---- 2. 开启双向 P2P ----
    if (p2pOK) {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
        printf("\n[开启] 已开启 dev0 <-> dev1 的双向 P2P 访问\n");
    } else {
        printf("\n[提示] 本机两张卡之间不支持 P2P，下面只演示走主机的搬运路径。\n");
    }

    // ---- 3. 在两张卡上分配同规模显存 ----
    float* d0 = nullptr;
    float* d1 = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d0, 0, N * sizeof(float)));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&d1, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d1, 0, N * sizeof(float)));

    // 在 dev0 上填一个能验证的模式：所有元素都是 7.0
    {
        std::vector<float> h(N, 7.0f);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(d0, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    }

    CudaTimer timer;
    const double bytes = (double)N * sizeof(float);
    const int grid = (int)((N + 255) / 256);

    // ---- 4. 走主机的两条 PCIe 往返 ----
    float hostPathMs = 0.f;
    {
        std::vector<float> staging(N);
        timer.start();
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(staging.data(), d0, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(d1, staging.data(), bytes, cudaMemcpyHostToDevice));
        hostPathMs = timer.stop();
        printf("\n[搬运] 经主机中转 (D2H + H2D): %.3f ms, 有效带宽 %.2f GB/s\n",
               hostPathMs, 2.0 * bytes / ((double)hostPathMs * 1e-3) / 1e9);
    }

    // ---- 5. P2P 直达 ----
    if (p2pOK) {
        // 先清空 d1 以便验证
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemset(d1, 0, N * sizeof(float)));
        timer.start();
        CUDA_CHECK(cudaMemcpyPeer(d1, 1, d0, 0, bytes));
        const float peerMs = timer.stop();
        printf("[搬运] P2P 直达 (cudaMemcpyPeer)  : %.3f ms, 有效带宽 %.2f GB/s\n",
               peerMs, bytes / ((double)peerMs * 1e-3) / 1e9);
        printf("       提速约 %.2fx\n", hostPathMs / peerMs);
    }

    // ---- 6. 验证：dev1 上的数据确实是 dev0 上的 7.0 ----
    {
        std::vector<float> h(N, 0.f);
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(h.data(), d1, bytes, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (size_t i = 0; i < N; ++i)
            if (std::fabs(h[i] - 7.0f) > 1e-6f) ++bad;
        printf("\n[正确性] dev1 上不匹配元素 = %d  %s\n", bad, bad == 0 ? "[OK]" : "[FAIL]");
    }

    // ---- 7. 如果需要"用自己的 kernel 直接算另一张卡的数据" ----
    if (p2pOK) {
        // 这个 kernel 在 dev0 上运行，但读写的是 dev1 的指针。
        // 前提是 dev0 已经通过 cudaDeviceEnablePeerAccess(1) 打开了到 dev1 的访问。
        CUDA_CHECK(cudaSetDevice(0));
        scale<<<grid, 256>>>(d1, 2.0f, N);  // d1 是 dev1 的指针！
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h(N, 0.f);
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(h.data(), d1, bytes, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (size_t i = 0; i < N; ++i)
            if (std::fabs(h[i] - 14.0f) > 1e-6f) ++bad;
        printf("[远端 kernel] 在 dev0 上运行 kernel 修改 dev1 的显存: 不匹配 = %d  %s\n",
               bad, bad == 0 ? "[OK]" : "[FAIL]");
    }

    printf("\n[下一步] 从「两张卡互相拷」到「多卡协同训练」，中间缺的是集合通信。\n");
    printf("         NCCL 提供了 AllReduce / AllGather / ReduceScatter / Broadcast，\n");
    printf("         会自动选 NVLink / PCIe P2P / 共享内存 / IB 网络，\n");
    printf("         并在算法层面做 ring / tree 优化。第 7 章会写一个可运行的例子。\n");

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(d0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(d1));
    return 0;
}
