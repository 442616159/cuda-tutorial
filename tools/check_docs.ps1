# ============================================================
#  check_docs.ps1 —— 文档一致性校验工具
#
#  用法（在仓库根目录）：
#      pwsh -File code/tools/check_docs.ps1
#      pwsh -File code/tools/check_docs.ps1 -Only ch02
#
#  校验项：
#    1. 代码围栏是否配对（``` 必须成对）
#    2. Markdown 表格每个块的列数是否一致
#    3. 引用块（>）是否被非引用行打断
#    4. #### 子节标题是否 100% 带 emoji 前缀
#    5. 内部锚点链接 ](#xxx) 是否能解析到某个标题
#    6. 跨章节文件链接 ](chNN_xxx.md) 是否存在
#    7. 是否有连续空行、分割线前后是否留空
# ============================================================
param(
    [string]$Only = ""
)

$ErrorActionPreference = "Stop"
# 本脚本位于仓库根的 tools/ 下，向上一级即仓库根
$root = Split-Path -Parent $PSScriptRoot
$docs = Join-Path $root "docs"

# GitHub 风格锚点：小写 → 去标点(保留 -) → 空格转 -
function Get-Slug([string]$text) {
    $a = $text.ToLower()
    # 去掉除 字母/数字/下划线/中日韩/空格/连字符 之外的一切
    $a = $a -replace '[^\w\u4e00-\u9fff\s\-]', ''
    $a = $a -replace '\s', '-'
    return $a
}

$files = Get-ChildItem $docs -Filter "*.md" | Where-Object {
    $Only -eq "" -or $_.Name -like "*$Only*"
} | Sort-Object Name

$totalProblems = 0

foreach ($file in $files) {
    # 必须用 .NET 显式按 UTF-8 读取：
    # Windows PowerShell 5.1 的 Get-Content -Encoding UTF8 只在文件"带 BOM"时
    # 才按 UTF-8 解码，对无 BOM 的 UTF-8 会退回 ANSI 代码页(936/GBK)，
    # 导致中文全部乱码，锚点、emoji、表格列数等校验结果全部失真。
    $raw = [System.IO.File]::ReadAllText($file.FullName, [System.Text.Encoding]::UTF8)
    $lines = $raw -split "`r`n|`n|`r"
    # 文件末尾的换行会切出一个空元素，去掉它以免误报"连续空行"
    if ($lines.Count -gt 0 -and $lines[$lines.Count - 1] -eq '') {
        $lines = $lines[0..($lines.Count - 2)]
    }
    $problems = @()

    # ---- 1. 代码围栏配对 ----
    $fences = ($lines | Where-Object { $_ -match '^```' }).Count
    if ($fences % 2 -ne 0) {
        $problems += "代码围栏数量为奇数 ($fences)，有未闭合的代码块"
    }

    # ---- 收集标题锚点（用于第 5 项）----
    # GitHub 对不同层级的同名标题会加前缀 -，故同时登记两种形式
    $slugs = @{}
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^(#{1,6})\s+(.+?)\s*$') {
            $lvl = $Matches[1].Length
            $base = Get-Slug $Matches[2]
            $slugs[$base] = $true
            if ($lvl -ge 3) {
                $slugs[('-' * ($lvl - 2)) + $base] = $true
            }
        }
    }

    $inFence = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]

        if ($line -match '^```') { $inFence = -not $inFence; continue }
        if ($inFence) { continue }

        # ---- 2. 表格列数一致性 ----
        if ($line -match '^\s*\|' -and ($i + 1) -lt $lines.Count -and
            $lines[$i + 1] -match '^\s*\|[\s\-:|]+\|\s*$') {
            $hdr = ([regex]::Matches($line, '\|')).Count
            $j = $i + 2
            while ($j -lt $lines.Count -and $lines[$j] -match '^\s*\|') {
                $cols = ([regex]::Matches($lines[$j], '\|')).Count
                if ($cols -ne $hdr) {
                    $problems += "行 $($j+1): 表格列数 $cols != 表头 $hdr"
                }
                $j++
            }
        }

        # ---- 3. 引用块中断 ----
        if ($line -match '^>' -and ($i + 1) -lt $lines.Count) {
            $nxt = $lines[$i + 1]
            if ($nxt -notmatch '^>' -and $nxt -notmatch '^\s*$' -and
                $nxt -notmatch '^[|\-#]' -and $nxt -notmatch '^```') {
                $problems += "行 $($i+2): 引用块被非引用行打断"
            }
        }

        # ---- 4. #### 标题 emoji 前缀 ----
        # 判定标准：标题第一个字符不在 CJK/ASCII 范围内，即视为 emoji/符号
        if ($line -match '^####\s+(\S)') {
            $c = $Matches[1]
            $okEmoji = $false
            foreach ($ch in $c.ToCharArray()) {
                $cp = [int]$ch
                # CJK 统一表意文字、ASCII、中日韩标点 → 不算 emoji
                if (($cp -ge 0x4E00 -and $cp -le 0x9FFF) -or
                    ($cp -ge 0x3000 -and $cp -le 0x303F) -or
                    ($cp -ge 0xFF00 -and $cp -le 0xFFEF) -or
                    ($cp -ge 0x20 -and $cp -le 0x7E)) {
                    continue
                }
                $okEmoji = $true
            }
            if (-not $okEmoji) {
                $problems += "行 $($i+1): #### 标题缺少 emoji 前缀 -> $line"
            }
        }

        # ---- 5. 内部锚点链接 ----
        if ($line -match '\]\(#([^)]+)\)') {
            foreach ($m in [regex]::Matches($line, '\]\(#([^)]+)\)')) {
                $tgt = $m.Groups[1].Value
                if (-not $slugs.ContainsKey($tgt)) {
                    $problems += "行 $($i+1): 锚点无法解析 #$tgt"
                }
            }
        }

        # ---- 6. 跨章节文件链接 ----
        foreach ($m in [regex]::Matches($line, '\]\((ch\d+_[^)#]+\.md)\)')) {
            $rel = $m.Groups[1].Value
            if (-not (Test-Path (Join-Path $docs $rel))) {
                $problems += "行 $($i+1): 链接的文件不存在 -> $rel"
            }
        }

        # ---- 7a. 连续空行 ----
        if ($i -gt 0 -and $line -match '^\s*$' -and $lines[$i - 1] -match '^\s*$') {
            $problems += "行 $($i+1): 出现连续空行"
        }

        # ---- 7b. 分割线前后留空 ----
        if ($line -match '^---\s*$') {
            if ($i -gt 0 -and $lines[$i - 1] -notmatch '^\s*$') {
                $problems += "行 $($i+1): 分割线 --- 前面缺少空行"
            }
            if (($i + 1) -lt $lines.Count -and $lines[$i + 1] -notmatch '^\s*$') {
                $problems += "行 $($i+1): 分割线 --- 后面缺少空行"
            }
        }
    }

    $status = if ($problems.Count -eq 0) { "OK  " } else { "FAIL" }
    Write-Output "$status $($file.Name)  ($($lines.Count) 行)"
    foreach ($p in $problems) { Write-Output "       - $p" }
    $totalProblems += $problems.Count
}

Write-Output ""
Write-Output "=========================================="
Write-Output "总问题数: $totalProblems"
if ($totalProblems -eq 0) { Write-Output "全部通过 ✅" } else { Write-Output "存在问题 ❌" }
exit $totalProblems
