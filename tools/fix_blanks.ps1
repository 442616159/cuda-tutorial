# ============================================================
#  fix_blanks.ps1 —— 清理 Markdown 中的连续空行
#
#  用法（在仓库根目录）：
#      powershell -NoProfile -ExecutionPolicy Bypass -File tools/fix_blanks.ps1
#      powershell ... -File tools/fix_blanks.ps1 -Only ch03
#
#  说明：代码块内部的空行不受影响（围栏内的内容原样保留），
#        只压缩正文中的连续空行。
# ============================================================
param([string]$Only = "")

# 本脚本位于仓库根的 tools/ 下，向上一级即仓库根
$root = Split-Path -Parent $PSScriptRoot
$docs = Join-Path $root "docs"

$files = Get-ChildItem $docs -Filter "*.md" | Where-Object {
    $Only -eq "" -or $_.Name -like "*$Only*"
} | Sort-Object Name

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

foreach ($file in $files) {
    # 显式按 UTF-8 读取，避免 PS 5.1 对无 BOM 的 UTF-8 退回 GBK 造成中文乱码
    $raw = [System.IO.File]::ReadAllText($file.FullName, [System.Text.Encoding]::UTF8)
    $lines = $raw -split "`r`n|`n|`r"
    if ($lines.Count -gt 0 -and $lines[$lines.Count - 1] -eq '') {
        $lines = $lines[0..($lines.Count - 2)]
    }
    $out = New-Object System.Collections.Generic.List[string]
    $inFence = $false
    $removed = 0

    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]

        # 围栏行本身照抄，并切换状态
        if ($line -match '^```') {
            $inFence = -not $inFence
            $out.Add($line)
            continue
        }

        # 围栏内部：原样保留（代码里的空行有意义）
        if ($inFence) { $out.Add($line); continue }

        # 正文：连续空行只保留一个
        if ($line -match '^\s*$' -and $out.Count -gt 0 -and $out[$out.Count - 1] -match '^\s*$') {
            $removed++
            continue
        }
        $out.Add($line)
    }

    if ($removed -gt 0) {
        [System.IO.File]::WriteAllLines($file.FullName, $out, $utf8NoBom)
        Write-Output "修复 $($file.Name): 删除 $removed 个多余空行 ($($lines.Count) -> $($out.Count))"
    } else {
        Write-Output "跳过 $($file.Name): 无需修复"
    }
}
