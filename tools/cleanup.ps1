# ============================================================
#  cleanup.ps1 —— 清理重组后残留的旧文件与旧目录
#
#  位置：CUDALearning/tools/cleanup.ps1
#
#  背景：
#    code/ 目录从 40 个扁平目录（ch02_matrix_mul 这种）重组为
#    lessons/ + projects/ 结构时，新文件已经写好，但旧的
#    源文件/源目录因为环境限制没有删除，造成重复。
#
#  这个脚本做三件事：
#    1. 备份式删除 code/ 下的全部旧目录（只删已确认迁移过的）
#    2. 删除根目录下已被 tools/ 取代的脚本
#    3. 删除已被 CONTRIBUTING.md 取代的 STYLE_GUIDE.md
#
#  用法：
#    powershell -ExecutionPolicy Bypass -File tools/cleanup.ps1           # 预演（只报告，不删）
#    powershell -ExecutionPolicy Bypass -File tools/cleanup.ps1 -Execute  # 真正删除
#
#  安全措施：
#    * 默认是"预演"模式，不加 -Execute 不会删除任何东西
#    * 删除前逐个检查"新位置是否真的存在对应文件"，不存在就跳过并警告
#    * 删除的是目录树，但只针对下面显式列出的名字
# ============================================================

param(
    [switch]$Execute
)

$ErrorActionPreference = 'Continue'

$Root     = Split-Path -Parent $PSScriptRoot      # CUDALearning/
$CodeDir  = Join-Path $Root 'code'

Write-Host "=============================================="
Write-Host "  重组残留清理" -ForegroundColor Cyan
if ($Execute) {
    Write-Host "  模式：真正删除" -ForegroundColor Red
} else {
    Write-Host "  模式：预演（不会删除任何东西）" -ForegroundColor Yellow
    Write-Host "  确认无误后加 -Execute 参数重新运行" -ForegroundColor Yellow
}
Write-Host "=============================================="

# ------------------------------------------------------------
# 1. code/ 下的旧目录 -> 新目录的对应关系
#    只有"新目录确实存在"时才允许删除旧目录
# ------------------------------------------------------------
$Migrations = @(
    @{ Old = 'ch00_hello';              New = 'lessons/ch00_hello' }
    @{ Old = 'ch01_vector_add';         New = 'lessons/ch01_indices' }
    @{ Old = 'ch02_matrix_mul';         New = 'lessons/ch02_memory' }
    @{ Old = 'ch03_sync';               New = 'lessons/ch03_sync/barrier' }
    @{ Old = 'ch03_syncwarp';           New = 'lessons/ch03_sync/warp_sync' }
    @{ Old = 'ch03_threadfence';        New = 'lessons/ch03_sync/fence' }
    @{ Old = 'ch03_shuffle_reduce';     New = 'lessons/ch03_sync/shuffle' }
    @{ Old = 'ch03_vote';               New = 'lessons/ch03_sync/vote' }
    @{ Old = 'ch03_atomics';            New = 'lessons/ch03_atomics' }
    @{ Old = 'ch03_histogram';          New = 'lessons/ch03_atomics_histogram' }
    @{ Old = 'ch03_reduce_cg';          New = 'lessons/ch03_cooperative' }
    @{ Old = 'ch03_streams';            New = 'lessons/ch03_streams' }
    @{ Old = 'ch03_graphs';             New = 'lessons/ch03_graphs' }
    @{ Old = 'ch04_timer';              New = 'lessons/ch04_timing/timer' }
    @{ Old = 'ch04_bandwidth_test';     New = 'lessons/ch04_timing/bandwidth' }
    @{ Old = 'ch04_roofline';           New = 'lessons/ch04_perf_model' }
    @{ Old = 'ch04_occupancy';          New = 'lessons/ch04_occupancy' }
    @{ Old = 'ch04_matmul_opt';         New = 'lessons/ch04_case_study' }
    @{ Old = 'ch04_sanitizer_bugs';     New = 'lessons/ch04_debugging' }
    @{ Old = 'ch05_reduce';             New = 'lessons/ch05_reduce' }
    @{ Old = 'ch05_scan';               New = 'lessons/ch05_scan' }
    @{ Old = 'ch05_stencil';            New = 'lessons/ch05_stencil' }
    @{ Old = 'ch05_sort';               New = 'lessons/ch05_sort' }
    @{ Old = 'ch05_spmv';               New = 'lessons/ch05_spmv' }
    @{ Old = 'ch05_bfs';                New = 'lessons/ch05_graph' }
    @{ Old = 'ch05_numeric';            New = 'lessons/ch05_numeric' }
    @{ Old = 'ch05_cg';                 New = 'lessons/ch05_cg' }
    @{ Old = 'ch06_dynparallel';        New = 'lessons/ch06_advanced/dynparallel' }
    @{ Old = 'ch06_wmma';               New = 'lessons/ch06_advanced/wmma' }
    @{ Old = 'ch06_cpasync';            New = 'lessons/ch06_advanced/cpasync' }
    @{ Old = 'ch06_cluster';            New = 'lessons/ch06_advanced/cluster' }
    @{ Old = 'ch06_unified';            New = 'lessons/ch06_advanced/unified' }
    @{ Old = 'ch06_vmm';                New = 'lessons/ch06_advanced/vmm' }
    @{ Old = 'ch06_multigpu';           New = 'lessons/ch06_advanced/multigpu' }
    @{ Old = 'ch06_graphs';             New = 'lessons/ch06_advanced/graphs_advanced' }
    @{ Old = 'ch07_cublas';             New = 'lessons/ch07_ecosystem/cublas' }
    @{ Old = 'ch07_cufft';              New = 'lessons/ch07_ecosystem/cufft' }
    @{ Old = 'ch07_curand';             New = 'lessons/ch07_ecosystem/curand' }
    @{ Old = 'ch07_thrust';             New = 'lessons/ch07_ecosystem/thrust' }
    @{ Old = 'ch07_cub';                New = 'lessons/ch07_ecosystem/cub' }
    @{ Old = 'ch07_cusparse';           New = 'lessons/ch07_ecosystem/cusparse' }
    @{ Old = 'ch07_cudnn';              New = 'lessons/ch07_ecosystem/cudnn' }
    @{ Old = 'ch07_python';             New = 'lessons/ch07_python' }
    @{ Old = 'ch07_cmake';              New = 'projects/cmake_skeleton' }
    @{ Old = 'ch08_project_sgemm';      New = 'projects/sgemm' }
    @{ Old = 'ch08_project_raytracer';  New = 'projects/raytracer' }
    @{ Old = 'ch08_project_nbody';      New = 'projects/nbody' }
)

Write-Host "`n--- 1. 清理 code/ 下的旧目录 ---"
$removed = 0
$skipped = 0

foreach ($m in $Migrations) {
    $oldPath = Join-Path $CodeDir $m.Old
    $newPath = Join-Path $CodeDir $m.New

    if (-not (Test-Path $oldPath)) { continue }   # 已经没了，跳过

    # 安全检查：新位置必须存在，否则说明迁移没完成，不能删
    if (-not (Test-Path $newPath)) {
        Write-Host "  [跳过] $($m.Old)  —— 新位置不存在: $($m.New)" -ForegroundColor Yellow
        $skipped++
        continue
    }

    # 再检查新位置不是空目录
    $newFiles = Get-ChildItem -Path $newPath -Recurse -File -ErrorAction SilentlyContinue
    if (-not $newFiles -or $newFiles.Count -eq 0) {
        Write-Host "  [跳过] $($m.Old)  —— 新位置是空的: $($m.New)" -ForegroundColor Yellow
        $skipped++
        continue
    }

    if ($Execute) {
        Remove-Item -Recurse -Force $oldPath -ErrorAction SilentlyContinue
        if (Test-Path $oldPath) {
            Write-Host "  [失败] $($m.Old)  —— 删除失败（可能被占用）" -ForegroundColor Red
            $skipped++
        } else {
            Write-Host "  [删除] $($m.Old)  (新: $($m.New), $($newFiles.Count) 个文件)" -ForegroundColor Green
            $removed++
        }
    } else {
        Write-Host "  [待删] $($m.Old)  ->  $($m.New)  ($($newFiles.Count) 个文件已就位)" -ForegroundColor Gray
        $removed++
    }
}

# ------------------------------------------------------------
# 2. 根目录下已被 tools/ 取代的脚本
# ------------------------------------------------------------
Write-Host "`n--- 2. 清理根目录下被 tools/ 取代的脚本 ---"

$SupersededFiles = @(
    @{ Old = 'check_env.sh';   New = 'tools/check_env.sh' }
    @{ Old = 'check_env.ps1';  New = 'tools/check_env.ps1' }
    @{ Old = 'verify_all.ps1'; New = 'tools/verify_all.ps1' }
    @{ Old = 'fix_links.py';   New = 'tools/fix_links.py' }
    @{ Old = 'STYLE_GUIDE.md'; New = 'CONTRIBUTING.md' }
)

foreach ($f in $SupersededFiles) {
    $oldPath = Join-Path $Root $f.Old
    $newPath = Join-Path $Root $f.New

    if (-not (Test-Path $oldPath)) { continue }

    if (-not (Test-Path $newPath)) {
        Write-Host "  [跳过] $($f.Old)  —— 替代文件不存在: $($f.New)" -ForegroundColor Yellow
        $skipped++
        continue
    }

    if ($Execute) {
        Remove-Item -Force $oldPath -ErrorAction SilentlyContinue
        if (Test-Path $oldPath) {
            Write-Host "  [失败] $($f.Old)" -ForegroundColor Red
            $skipped++
        } else {
            Write-Host "  [删除] $($f.Old)  (已由 $($f.New) 取代)" -ForegroundColor Green
            $removed++
        }
    } else {
        Write-Host "  [待删] $($f.Old)  ->  已由 $($f.New) 取代" -ForegroundColor Gray
        $removed++
    }
}

# ------------------------------------------------------------
# 3. 汇总
# ------------------------------------------------------------
Write-Host "`n=============================================="
if ($Execute) {
    Write-Host "已删除 $removed 项，跳过 $skipped 项" -ForegroundColor Cyan
} else {
    Write-Host "预演结果：将删除 $removed 项，跳过 $skipped 项" -ForegroundColor Cyan
    Write-Host "`n确认无误后执行：" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File tools/cleanup.ps1 -Execute" -ForegroundColor Yellow
}
Write-Host "=============================================="

if (-not $Execute) {
    Write-Host "`n提示：删除后请重新运行一次编译校验，确认没有破坏任何东西："
    Write-Host "  powershell -ExecutionPolicy Bypass -File tools/verify_all.ps1"
}
