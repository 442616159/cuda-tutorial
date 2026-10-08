# ============================================================
#  check_env.ps1 —— 一键体检你的 CUDA 环境（Windows PowerShell）
#
#  位置：CUDALearning/tools/check_env.ps1
#
#  用法：
#      powershell -ExecutionPolicy Bypass -File check_env.ps1
#  或在 PowerShell 中：
#      .\check_env.ps1
# ============================================================

$ErrorActionPreference = 'Continue'
$script:Fail = 0

function Ok   ($m) { Write-Host "[ OK ]  $m" -ForegroundColor Green }
function Bad  ($m) { Write-Host "[FAIL]  $m" -ForegroundColor Red; $script:Fail = 1 }
function Warn ($m) { Write-Host "[WARN]  $m" -ForegroundColor Yellow }

Write-Host "=============================================="
Write-Host "        CUDA 学习环境体检 (Windows)"
Write-Host "=============================================="

# ---------- 1. 显卡驱动 ----------
Write-Host "`n--- 1. NVIDIA 驱动与显卡 ---"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    Ok "找到 nvidia-smi ($($smi.Source))"
    try {
        $info = & nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $info) {
            foreach ($line in ($info | Select-Object -First 4)) {
                Write-Host "        $line"
            }
        } else {
            Bad "nvidia-smi 执行失败，驱动可能有问题"
        }
    } catch {
        Bad "nvidia-smi 调用异常: $_"
    }
} else {
    Bad "找不到 nvidia-smi，可能没装 NVIDIA 驱动，或没有 NVIDIA 显卡"
}

# ---------- 2. CUDA Toolkit ----------
Write-Host "`n--- 2. CUDA Toolkit ---"
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) {
    $verOut = & nvcc --version 2>$null
    $rel = ($verOut | Select-String -Pattern 'release ([\d\.]+)').Matches.Groups[1].Value
    Ok "找到 nvcc，版本 release $rel"
    Write-Host "        路径       : $($nvcc.Source)"
    if ($rel -and [double]($rel.Split('.')[0]) -lt 11) {
        Warn "CUDA 版本较老 (<11.0)，第 6 章部分高级特性无法编译"
    }
} else {
    Bad "找不到 nvcc，CUDA Toolkit 没装或没加入 PATH"
    Write-Host "        请确认 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin 已加入 Path"
}

# ---------- 3. 主机编译器 ----------
Write-Host "`n--- 3. 主机 C++ 编译器 ---"
$cl = Get-Command cl -ErrorAction SilentlyContinue
$clang = Get-Command clang++ -ErrorAction SilentlyContinue
if ($cl) {
    Ok "找到 MSVC cl.exe"
} elseif ($clang) {
    Ok "找到 clang++"
} else {
    Warn "没找到 cl.exe / clang++。nvcc 在 Windows 上需要 MSVC 作为主机编译器"
    Write-Host "        请安装 Visual Studio 2019/2022 并勾选『使用 C++ 的桌面开发』"
}

# ---------- 4. 构建工具 ----------
Write-Host "`n--- 4. 构建工具 ---"
if (Get-Command cmake -ErrorAction SilentlyContinue) {
    Ok "cmake $((& cmake --version | Select-Object -First 1))"
} else {
    Warn "找不到 cmake（第 7 章工程化需要，非必须）"
}
if (Get-Command make -ErrorAction SilentlyContinue) {
    Ok "make 可用"
} else {
    Warn "找不到 make —— 不影响，每个示例都提供了 nvcc 单行编译命令"
}

# ---------- 5. 性能分析工具 ----------
Write-Host "`n--- 5. 性能分析工具（第 4 章要用）---"
foreach ($tool in @('ncu', 'nsys', 'compute-sanitizer')) {
    if (Get-Command $tool -ErrorAction SilentlyContinue) {
        Ok $tool
    } else {
        Warn "未找到 $tool"
    }
}

# ---------- 6. 实机编译测试 ----------
Write-Host "`n--- 6. 实机编译测试 ---"
if (-not $nvcc) {
    Warn "没有 nvcc，跳过编译测试"
} else {
    $tmp = Join-Path $env:TEMP ("cuda_check_" + [guid]::NewGuid().ToString('N').Substring(0, 8))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    $src = Join-Path $tmp 't.cu'
    @'
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
'@ | Set-Content -Path $src -Encoding UTF8

    $exe = Join-Path $tmp 't.exe'
    $log = Join-Path $tmp 'build.log'
    & nvcc -O2 $src -o $exe *> $log
    if ($LASTEXITCODE -eq 0) {
        $out = & $exe 2>&1
        if ($LASTEXITCODE -eq 0) {
            Ok "编译 + 运行成功：$out"
        } else {
            Bad "编译成功但运行失败（显卡/驱动问题）：$out"
        }
    } else {
        Bad "编译失败，错误日志："
        Get-Content $log | ForEach-Object { Write-Host "        $_" }
    }
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

# ---------- 结论 ----------
Write-Host "`n=============================================="
if ($script:Fail -eq 0) {
    Write-Host "环境体检通过，可以开始第 0 章了！" -ForegroundColor Green
} else {
    Write-Host "存在问题，请先按上面提示修复。" -ForegroundColor Red
}
Write-Host "=============================================="
exit $script:Fail
