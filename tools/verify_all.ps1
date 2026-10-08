# ============================================================
#  verify_all.ps1 —— 批量编译校验所有示例代码（Windows）
#
#  位置：CUDALearning/tools/verify_all.ps1
#
#  用法：
#      pwsh -File verify_all.ps1
#  或在 PowerShell 中：
#      .\verify_all.ps1
#
#  作用：遍历 code/ 下所有 .cu 文件逐个编译，报告成功/失败清单。
#        对需要特殊参数的示例（动态并行 / 集群 / VMM）会自动追加。
#
#  说明：个别示例依赖 cuDNN，未安装时会失败，这属于环境缺失而非代码错误。
# ============================================================

$ErrorActionPreference = 'Continue'
$root    = Split-Path -Parent $MyInvocation.MyCommand.Path   # .../CUDALearning/tools
$codeDir = Join-Path (Split-Path -Parent $root) 'code'       # .../CUDALearning/code
$binDir  = Join-Path $codeDir 'bin'

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Host "找不到 nvcc，无法校验。请先安装 CUDA Toolkit。" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $codeDir)) {
    Write-Host "找不到 code 目录: $codeDir" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Path $binDir -Force | Out-Null

$srcs = Get-ChildItem -Path $codeDir -Recurse -Filter *.cu -File |
        Where-Object { $_.FullName -notlike "*$binDir*" }

Write-Host "共发现 $($srcs.Count) 个 .cu 文件`n" -ForegroundColor Cyan

# 探测本机计算能力，作为 -arch 的默认值
$arch = '-arch=sm_70'
try {
    $cc = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null |
           Select-Object -First 1).Trim()
    if ($cc) { $arch = '-arch=sm_0' -replace 'sm_0', ('sm_' + ($cc -replace '\.', '')) }
} catch { }

Write-Host "使用架构: $arch`n" -ForegroundColor Cyan

$pass = @(); $fail = @(); $skip = @()

foreach ($s in $srcs) {
    # 用 "相对路径压平成下划线" 作为唯一可执行文件名
    $rel  = $s.FullName.Substring($codeDir.Length + 1) -replace '[\\/]', '_'
    $name = [System.IO.Path]::GetFileNameWithoutExtension($rel)
    $exe  = Join-Path $binDir $name

    $text  = Get-Content $s.FullName -Raw
    $extra = @()

    # 依赖第三方 SDK 的示例：没装就跳过，不算失败
    if ($text -match 'cudnn') {
        $skip += "$rel  (需要 cuDNN)"
        Write-Host "[SKIP] $rel  (未安装 cuDNN)" -ForegroundColor DarkYellow
        continue
    }

    if ($text -match 'cublas') { $extra += '-lcublas' }
    if ($text -match 'cufft')  { $extra += '-lcufft'  }
    if ($text -match 'curand') { $extra += '-lcurand' }
    if ($text -match 'cusparse') { $extra += '-lcusparse' }

    # 需要特殊处理的三类
    $thisArch = $arch
    if ($s.FullName -match 'dynparallel') { $extra += '-rdc=true' }
    if ($s.FullName -match 'cluster_dsm') { $thisArch = '-arch=sm_90' }  # 集群需要 Hopper
    if ($s.FullName -match 'vmm_alloc')   { $extra += '-lcuda' }

    $nvArgs = @('-O3', '-std=c++17', '-lineinfo', $thisArch) + $extra +
              @($s.FullName, '-o', $exe)
    $log = & nvcc @nvArgs 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[ OK ] $rel" -ForegroundColor Green
        $pass += $rel
    } else {
        Write-Host "[FAIL] $rel" -ForegroundColor Red
        $log | Select-Object -First 10 | ForEach-Object {
            Write-Host "        $_" -ForegroundColor DarkGray
        }
        $fail += $rel
    }
}

Write-Host "`n===== 校验汇总 =====" -ForegroundColor Cyan
Write-Host "通过: $($pass.Count)    失败: $($fail.Count)    跳过: $($skip.Count)"

if ($skip.Count -gt 0) {
    Write-Host "`n跳过（环境缺失，非代码问题）:" -ForegroundColor DarkYellow
    $skip | ForEach-Object { Write-Host "  $_" }
}

if ($fail.Count -gt 0) {
    Write-Host "`n失败清单:" -ForegroundColor Yellow
    $fail | ForEach-Object { Write-Host "  $_" }
    Write-Host @"

提示：
  * ch06 的 cluster_dsm.cu  需要 Hopper（CC 9.0）
  * ch06 的 cpasync_gemm.cu 需要 Ampere（CC 8.0）
  如果你的显卡不满足，这两个编译失败是预期行为，不影响其余章节。
"@ -ForegroundColor DarkYellow
    exit 1
}

Write-Host "`n全部通过。" -ForegroundColor Green
exit 0
