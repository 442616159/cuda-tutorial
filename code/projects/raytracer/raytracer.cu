// ============================================================
//  raytracer.cu —— 第 8 章项目二：GPU 并行光线追踪器
//
//  为什么选这个项目？
//    * 它是"每线程独立任务"的完美例子：每个像素一条主光线，
//      线程之间零通信 —— 正好对应你在第 1 章学的最简单的映射方式。
//    * 但它又不是"无脑并行"：加上反射、阴影、抗锯齿之后，
//      每个线程的工作量差异会很大，你会第一次真切体会到
//      "任务粒度"和"负载均衡"的重要性。
//    * 结果是一张图，肉眼就能验证对错，反馈非常直观。
//
//  本文件实现：
//    球体 + 地面平面、兰伯特漫反射、单光源硬阴影、多次反射、可选抗锯齿
//    输出 PPM（P6 二进制）图像
//
//  编译（在 code/projects/raytracer 目录下）：
//      nvcc -O3 -arch=sm_70 raytracer.cu -o raytracer
//  把 sm_70 换成你自己显卡的计算能力（用 nvidia-smi --query-gpu=compute_cap
//  查询，或跑 ch00_hello/device_query）。CUDA 11.5+ 也可以用 -arch=native。
//  运行： ./raytracer                    # 默认 800x600，1 次采样
//         ./raytracer 1920 1080 4        # 宽 高 每像素采样数
//  输出： output.ppm
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

#include "../../common/cuda_check.h"

// ============================================================
//  基础数学类型
//  GPU 上写图形代码，用最朴素的结构体比用类好：
//  没有虚函数、没有隐式堆分配，编译器更容易把它们放进寄存器。
// ============================================================
struct Vec3 {
    float x, y, z;
};

__host__ __device__ inline Vec3 makeVec(float x, float y, float z) {
    return Vec3{x, y, z};
}
__host__ __device__ inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return makeVec(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return makeVec(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline Vec3 operator*(const Vec3& a, float s) {
    return makeVec(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline Vec3 operator*(float s, const Vec3& a) { return a * s; }
__host__ __device__ inline Vec3 operator*(const Vec3& a, const Vec3& b) {
    return makeVec(a.x * b.x, a.y * b.y, a.z * b.z);
}
__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return makeVec(a.y * b.z - a.z * b.y,
                   a.z * b.x - a.x * b.z,
                   a.x * b.y - a.y * b.x);
}
__host__ __device__ inline Vec3 normalize(const Vec3& a) {
    // 这里刻意用 1/sqrtf 而不是 rsqrtf：rsqrtf 是设备端专用内在函数，
    // 而这个函数被 CPU 参考实现复用，写成 1/sqrtf 两边都能编译。
    const float len = sqrtf(dot(a, a));
    return a * (1.f / len);
}

// ============================================================
//  场景描述
// ============================================================
struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
    float reflectivity;   // 0 = 纯漫反射，1 = 全反射
};

struct Ray {
    Vec3 origin;
    Vec3 dir;   // 必须是单位向量，否则交点参数 t 的物理含义就不对了
};

// 场景放进 __constant__ 内存：所有线程在每个像素上读的都是同一份数据，
// 常量内存的广播机制正是为这种访问模式设计的。
// 注意 64 KB 上限；复杂场景要改用全局内存 + __ldg / 只读缓存。
constexpr int MAX_SPHERES = 16;
__constant__ Sphere c_spheres[MAX_SPHERES];
__constant__ int c_numSpheres;

// 光照参数用编译期常量，避免到处传参
constexpr float AMBIENT = 0.15f;

struct Hit {
    float t;
    Vec3 p;      // 交点位置
    Vec3 n;      // 单位法线
    Vec3 color;
    float refl;
};

// ============================================================
//  求交
// ============================================================

// 光线与球求交：解 |o + t*d - c|^2 = r^2
// 化简成 at^2 + 2*(b/2)t + c0 = 0，取最小的正根。
// 用 b 的一半能省一次乘法，是图形学里的标准写法。
__device__ bool hitSphere(const Sphere& s, const Ray& r,
                          float tMin, float tMax, float& tOut) {
    const Vec3 oc = r.origin - s.center;
    const float a = dot(r.dir, r.dir);
    const float halfB = dot(oc, r.dir);
    const float cc = dot(oc, oc) - s.radius * s.radius;
    const float disc = halfB * halfB - a * cc;
    if (disc < 0.f) return false;

    const float sq = sqrtf(disc);
    float t = (-halfB - sq) / a;
    if (t < tMin || t > tMax) {
        t = (-halfB + sq) / a;     // 近端不在范围里，试远端
        if (t < tMin || t > tMax) return false;
    }
    tOut = t;
    return true;
}

// 无限大平面 y = 0，用来当地面
__device__ bool hitPlane(const Ray& r, float tMin, float tMax, float& tOut) {
    if (fabsf(r.dir.y) < 1e-6f) return false;   // 光线平行于平面
    const float t = -r.origin.y / r.dir.y;
    if (t < tMin || t > tMax) return false;
    tOut = t;
    return true;
}

// 返回整个场景里最近的交点
__device__ bool traceScene(const Ray& r, float tMin, float tMax, Hit& hit) {
    bool found = false;
    float closest = tMax;

    for (int i = 0; i < c_numSpheres; ++i) {
        float t;
        if (hitSphere(c_spheres[i], r, tMin, closest, t)) {
            closest = t;
            hit.t = t;
            hit.p = r.origin + r.dir * t;
            hit.n = normalize(hit.p - c_spheres[i].center);
            hit.color = c_spheres[i].color;
            hit.refl = c_spheres[i].reflectivity;
            found = true;
        }
    }

    // 地面平面：带一点棋盘格纹理，让透视关系一眼可见
    float tp;
    if (hitPlane(r, tMin, closest, tp)) {
        closest = tp;
        hit.t = tp;
        hit.p = r.origin + r.dir * tp;
        hit.n = makeVec(0.f, 1.f, 0.f);
        const int cx = (int)floorf(hit.p.x);
        const int cz = (int)floorf(hit.p.z);
        const float g = ((cx + cz) & 1) ? 0.25f : 0.75f;
        hit.color = makeVec(g, g, g);
        hit.refl = 0.05f;
        found = true;
    }
    return found;
}

// 从 p 沿 toLight 方向看有没有遮挡 -> 硬阴影
__device__ bool inShadow(const Vec3& p, const Vec3& toLight) {
    const Ray shadowRay{p, toLight};
    Hit h;
    // tMin 取一个小正数，避免自交（shadow acne）
    return traceScene(shadowRay, 1e-3f, 1e30f, h);
}

// ============================================================
//  着色：兰伯特漫反射 + 环境光 + 反射
//
//  ★ 重要：这里**没有用递归**，而是展开成循环。
//    GPU 上的函数调用栈很浅，递归会带来巨大的开销（甚至编译不过），
//    而光线反弹本质上就是一个可以迭代的循环。
//    这个"把递归写成循环"的技巧在并行代码里非常常见。
// ============================================================
__device__ Vec3 shade(const Ray& r0, int maxDepth) {
    const Vec3 lightDir = normalize(makeVec(0.6f, 1.0f, 0.4f));
    const Vec3 lightColor = makeVec(1.0f, 0.98f, 0.92f);

    Vec3 acc = makeVec(0.f, 0.f, 0.f);     // 已经累积到的颜色
    Vec3 throughput = makeVec(1.f, 1.f, 1.f);  // 当前路径还剩多少能量
    Ray r = r0;

    for (int depth = 0; depth <= maxDepth; ++depth) {
        Hit hit;
        if (!traceScene(r, 1e-3f, 1e30f, hit)) {
            // 没打到东西：画背景（垂直渐变）。背景也要乘 throughput，
            // 否则透过镜面球看到的天空会亮得不合理。
            const float t = 0.5f * (r.dir.y + 1.f);
            const Vec3 bg = makeVec(1.f, 1.f, 1.f) * (1.f - t)
                          + makeVec(0.5f, 0.7f, 1.0f) * t;
            acc = acc + throughput * bg;
            break;
        }

        // 兰伯特项：法线与光线方向的夹角余弦，负数说明背光
        float ndotl = fmaxf(dot(hit.n, lightDir), 0.f);

        // 只有朝向光源的面才需要发阴影射线 —— 省掉大约一半的求交
        float shadowFactor = 1.f;
        if (ndotl > 0.f && inShadow(hit.p, lightDir)) shadowFactor = 0.f;

        const Vec3 local = hit.color * lightColor * (ndotl * shadowFactor)
                         + hit.color * AMBIENT;

        // 漫反射部分被吸收掉 (1 - refl)，其余能量进入反射路径
        acc = acc + throughput * local * (1.f - hit.refl);

        if (depth == maxDepth || hit.refl <= 0.f) break;

        throughput = throughput * hit.refl;
        // 反射方向：d - 2(d·n)n
        const Vec3 reflDir = normalize(r.dir - hit.n * (2.f * dot(r.dir, hit.n)));
        r = Ray{hit.p, reflDir};

        // 能量衰减到可以忽略时提前退出：这是最重要的性能优化之一，
        // 能让大量"擦边"像素少算好几层。
        if (throughput.x + throughput.y + throughput.z < 0.02f) break;
    }
    return acc;
}

// ============================================================
//  主核函数：一个线程负责一个像素
//  samplesPerPixel 次采样用于抗锯齿（像素内随机抖动）
// ============================================================
__global__ void renderKernel(unsigned char* out, int width, int height,
                             Vec3 camPos, Vec3 camForward, Vec3 camRight, Vec3 camUp,
                             float fovScaleX, float fovScaleY,
                             int samplesPerPixel, int maxDepth, unsigned int seed) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;   // 尺寸不是块的整数倍时必须有

    // 每个线程一个独立的伪随机状态。这里不引入 cuRAND，
    // 用最轻量的 xorshift 就够了 —— 抗锯齿只要求"看起来随机"。
    unsigned int rng = seed + (unsigned int)y * 1973u + (unsigned int)x * 9277u;

    Vec3 accum = makeVec(0.f, 0.f, 0.f);
    for (int s = 0; s < samplesPerPixel; ++s) {
        float jx = 0.f, jy = 0.f;
        if (samplesPerPixel > 1) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            jx = (float)(rng & 0xFFFFFFu) / 16777216.f - 0.5f;
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            jy = (float)(rng & 0xFFFFFFu) / 16777216.f - 0.5f;
        }
        // 归一化设备坐标：x 向右为正，y 向上为正
        const float u = ((x + 0.5f + jx) / width * 2.f - 1.f) * fovScaleX;
        const float v = (1.f - (y + 0.5f + jy) / height * 2.f) * fovScaleY;

        const Ray ray{camPos, normalize(camForward + camRight * u + camUp * v)};
        accum = accum + shade(ray, maxDepth);
    }
    Vec3 color = accum * (1.0f / samplesPerPixel);

    // gamma 校正（gamma 2.2 的常用近似：开平方）。
    // 不做这一步图像会明显偏暗。
    color.x = sqrtf(fminf(fmaxf(color.x, 0.f), 1.f));
    color.y = sqrtf(fminf(fmaxf(color.y, 0.f), 1.f));
    color.z = sqrtf(fminf(fmaxf(color.z, 0.f), 1.f));

    const size_t idx = ((size_t)y * width + x) * 3;
    out[idx + 0] = (unsigned char)(color.x * 255.f + 0.5f);
    out[idx + 1] = (unsigned char)(color.y * 255.f + 0.5f);
    out[idx + 2] = (unsigned char)(color.z * 255.f + 0.5f);
}

// ============================================================
//  CPU 参考实现
//  注意：它与 GPU 版本是"同一套逻辑的两份实现"，
//  必须逐行对照着写，否则对拍会因为实现差异而失败。
// ============================================================

// 把 CPU 版也写成循环结构，和 GPU 版一一对应
static bool cpuTrace(const Ray& r, const Sphere* sp, int n,
                     float tMin, float tMax, Hit& hit) {
    bool found = false;
    float closest = tMax;
    for (int i = 0; i < n; ++i) {
        const Sphere& s = sp[i];
        const Vec3 oc = r.origin - s.center;
        const float a = dot(r.dir, r.dir);
        const float halfB = dot(oc, r.dir);
        const float cc = dot(oc, oc) - s.radius * s.radius;
        const float disc = halfB * halfB - a * cc;
        if (disc < 0.f) continue;
        const float sq = sqrtf(disc);
        float t = (-halfB - sq) / a;
        if (t < tMin || t > closest) {
            t = (-halfB + sq) / a;
            if (t < tMin || t > closest) continue;
        }
        closest = t;
        hit.t = t;
        hit.p = r.origin + r.dir * t;
        hit.n = normalize(hit.p - s.center);
        hit.color = s.color;
        hit.refl = s.reflectivity;
        found = true;
    }
    if (fabsf(r.dir.y) > 1e-6f) {
        const float t = -r.origin.y / r.dir.y;
        if (t >= tMin && t <= closest) {
            closest = t;
            hit.t = t;
            hit.p = r.origin + r.dir * t;
            hit.n = makeVec(0.f, 1.f, 0.f);
            const int cx = (int)floorf(hit.p.x), cz = (int)floorf(hit.p.z);
            const float g = ((cx + cz) & 1) ? 0.25f : 0.75f;
            hit.color = makeVec(g, g, g);
            hit.refl = 0.05f;
            found = true;
        }
    }
    return found;
}

static Vec3 cpuShade(const Ray& r0, const Sphere* sp, int n, int maxDepth) {
    const Vec3 lightDir = normalize(makeVec(0.6f, 1.0f, 0.4f));
    const Vec3 lightColor = makeVec(1.0f, 0.98f, 0.92f);

    Vec3 acc = makeVec(0.f, 0.f, 0.f);
    Vec3 throughput = makeVec(1.f, 1.f, 1.f);
    Ray r = r0;

    for (int depth = 0; depth <= maxDepth; ++depth) {
        Hit hit;
        if (!cpuTrace(r, sp, n, 1e-3f, 1e30f, hit)) {
            const float t = 0.5f * (r.dir.y + 1.f);
            const Vec3 bg = makeVec(1.f, 1.f, 1.f) * (1.f - t)
                          + makeVec(0.5f, 0.7f, 1.0f) * t;
            acc = acc + throughput * bg;
            break;
        }
        float ndotl = fmaxf(dot(hit.n, lightDir), 0.f);
        float sf = 1.f;
        if (ndotl > 0.f) {
            const Ray sr{hit.p, lightDir};
            Hit h2;
            if (cpuTrace(sr, sp, n, 1e-3f, 1e30f, h2)) sf = 0.f;
        }
        const Vec3 local = hit.color * lightColor * (ndotl * sf)
                         + hit.color * AMBIENT;
        acc = acc + throughput * local * (1.f - hit.refl);

        if (depth == maxDepth || hit.refl <= 0.f) break;
        throughput = throughput * hit.refl;
        const Vec3 reflDir = normalize(r.dir - hit.n * (2.f * dot(r.dir, hit.n)));
        r = Ray{hit.p, reflDir};
        if (throughput.x + throughput.y + throughput.z < 0.02f) break;
    }
    return acc;
}

// ============================================================
//  写 PPM（P6：二进制 RGB）
// ============================================================
static bool writePPM(const char* path, const unsigned char* data, int w, int h) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(data, 1, (size_t)w * h * 3, f);
    fclose(f);
    return true;
}

class CudaTimer {
public:
    CudaTimer() { CUDA_CHECK(cudaEventCreate(&s_)); CUDA_CHECK(cudaEventCreate(&e_)); }
    ~CudaTimer() { cudaEventDestroy(s_); cudaEventDestroy(e_); }
    void start() { CUDA_CHECK(cudaEventRecord(s_, 0)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(e_, 0));
        CUDA_CHECK(cudaEventSynchronize(e_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s_, e_));
        return ms;
    }

private:
    cudaEvent_t s_{}, e_{};
};

int main(int argc, char** argv) {
    printDeviceInfo();

    int width = 800, height = 600, samples = 1;
    const int maxDepth = 3;
    if (argc >= 3) { width = atoi(argv[1]); height = atoi(argv[2]); }
    if (argc >= 4) { samples = atoi(argv[3]); }
    if (width <= 0 || height <= 0 || samples <= 0) {
        fprintf(stderr, "参数非法：宽 高 采样数必须为正整数\n");
        return EXIT_FAILURE;
    }

    // ---------------- 搭建场景 ----------------
    std::vector<Sphere> spheres;
    spheres.push_back({makeVec( 0.0f, 1.00f,  0.0f), 1.00f, makeVec(0.95f, 0.25f, 0.25f), 0.05f});
    spheres.push_back({makeVec(-2.2f, 0.80f, -0.5f), 0.80f, makeVec(0.25f, 0.85f, 0.35f), 0.05f});
    spheres.push_back({makeVec( 2.2f, 0.90f,  0.4f), 0.90f, makeVec(0.30f, 0.45f, 0.95f), 0.05f});
    spheres.push_back({makeVec( 1.1f, 0.45f, -1.8f), 0.45f, makeVec(0.90f, 0.90f, 0.95f), 0.75f});
    if ((int)spheres.size() > MAX_SPHERES) {
        fprintf(stderr, "球体数量超过 MAX_SPHERES=%d\n", MAX_SPHERES);
        return EXIT_FAILURE;
    }
    const int numSpheres = (int)spheres.size();
    CUDA_CHECK(cudaMemcpyToSymbol(c_spheres, spheres.data(), spheres.size() * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_numSpheres, &numSpheres, sizeof(int)));

    // ---------------- 相机 ----------------
    const Vec3 camPos = makeVec(0.f, 1.6f, 6.0f);
    const Vec3 camTarget = makeVec(0.f, 0.9f, 0.f);
    const Vec3 camForward = normalize(camTarget - camPos);
    const Vec3 worldUp = makeVec(0.f, 1.f, 0.f);
    const Vec3 camRight = normalize(cross(camForward, worldUp));
    const Vec3 camUp = cross(camRight, camForward);
    // 垂直半视场角 30 度；水平方向乘上宽高比，保证画面不被拉伸
    const float fovScaleY = tanf(30.f * 3.14159265f / 180.f);
    const float aspect = (float)width / (float)height;
    const float fovScaleX = fovScaleY * aspect;

    // ---------------- 显存 ----------------
    const size_t imgBytes = (size_t)width * height * 3;
    unsigned char* d_img = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img, imgBytes));
    std::vector<unsigned char> h_img(imgBytes);

    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 预热：第一次启动包含模块加载，不能算进正式耗时
    renderKernel<<<grid, block>>>(d_img, width, height, camPos, camForward, camRight, camUp,
                                  fovScaleX, fovScaleY, samples, maxDepth, 12345u);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_SYNC();

    CudaTimer timer;
    timer.start();
    renderKernel<<<grid, block>>>(d_img, width, height, camPos, camForward, camRight, camUp,
                                  fovScaleX, fovScaleY, samples, maxDepth, 12345u);
    CUDA_CHECK_KERNEL();
    const float ms = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_img.data(), d_img, imgBytes, cudaMemcpyDeviceToHost));

    printf("\n=== 光线追踪 %d x %d，每像素 %d 采样，最大反弹 %d 次 ===\n",
           width, height, samples, maxDepth);
    printf("GPU 渲染耗时: %.3f ms\n", ms);

    // ---------------- 正确性验证 ----------------
    // 挑几个确定的像素与 CPU 结果逐通道比较。
    // "图像看起来对"不算验证；必须有数值对拍。
    const int testPts[][2] = {
        {0, 0}, {width - 1, 0}, {0, height - 1}, {width - 1, height - 1},
        {width / 2, height / 2}, {width / 3, height / 2}, {2 * width / 3, height / 2}
    };
    int checked = 0, mismatched = 0;
    for (const auto& pt : testPts) {
        const int x = pt[0], y = pt[1];
        // 复现核函数的坐标映射（samples>1 时抖动非零，见下方说明）
        const float u = ((x + 0.5f) / width * 2.f - 1.f) * fovScaleX;
        const float v = (1.f - (y + 0.5f) / height * 2.f) * fovScaleY;
        const Ray ray{camPos, normalize(camForward + camRight * u + camUp * v)};
        const Vec3 c = cpuShade(ray, spheres.data(), numSpheres, maxDepth);
        const unsigned char expected[3] = {
            (unsigned char)(sqrtf(fminf(fmaxf(c.x, 0.f), 1.f)) * 255.f + 0.5f),
            (unsigned char)(sqrtf(fminf(fmaxf(c.y, 0.f), 1.f)) * 255.f + 0.5f),
            (unsigned char)(sqrtf(fminf(fmaxf(c.z, 0.f), 1.f)) * 255.f + 0.5f)
        };
        const size_t idx = ((size_t)y * width + x) * 3;
        int diff = 0;
        for (int k = 0; k < 3; ++k)
            diff = std::max(diff, std::abs((int)h_img[idx + k] - (int)expected[k]));
        if (diff > 2) ++mismatched;
        ++checked;
        printf("  像素(%4d,%4d) GPU=(%3d,%3d,%3d) CPU=(%3d,%3d,%3d) 最大差值=%d\n",
               x, y, h_img[idx], h_img[idx + 1], h_img[idx + 2],
               expected[0], expected[1], expected[2], diff);
    }
    printf("检查 %d 个像素，不一致(差值>2) %d 个 -> %s\n",
           checked, mismatched, mismatched == 0 ? "通过 ✔" : "失败 ✘");
    if (samples > 1) {
        printf("注意：samples>1 时 GPU 端有亚像素抖动，与 CPU 单点采样必然有差异，\n");
        printf("      上面的数值对拍只在 samples=1 时有意义。\n");
    }

    // ---------------- 写文件 ----------------
    const std::string path = "output.ppm";
    if (writePPM(path.c_str(), h_img.data(), width, height)) {
        printf("\n已写出图像: %s（%d x %d, PPM/P6）\n", path.c_str(), width, height);
        printf("查看方式：GIMP / IrfanView / Photoshop 直接打开，\n");
        printf("或 Python: from PIL import Image; Image.open('output.ppm').save('output.png')\n");
    } else {
        fprintf(stderr, "写出图像失败\n");
    }
    const double mrays = (double)width * height * samples * 1e-6;
    printf("主光线吞吐 ≈ %.2f Mrays/s（不含反射与阴影射线）\n", mrays / (ms * 1e-3));

    CUDA_CHECK(cudaFree(d_img));
    return (samples == 1 && mismatched != 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
