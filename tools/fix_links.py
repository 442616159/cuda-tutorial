"""
fix_links.py —— 文档链接检查与修复工具

位置：CUDALearning/tools/fix_links.py

背景：
    教程正文在 docs/ 目录，示例代码在 code/ 目录（与 docs/ 同级）。
    如果在正文里把链接写成 `](code/xxx)`，会被解析成 docs/code/xxx，
    是一个坏链接。正确写法是 `](../code/xxx)`。

用法（在 CUDALearning 目录下）：
    python tools/fix_links.py          # 检查并修复
    python tools/fix_links.py --check  # 只检查，不修改

说明：
    本脚本只处理 `](code/` 这一种已知的错误形式。
    如果你的 shell 不可用，也可以直接用编辑器批量替换
    `](code/` → `](../code/`。
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# 只匹配 ](code/ 形式；已经写成 ](../code/ 的不会被二次替换
PATTERN = re.compile(r"\]\(code/")

CHECK_ONLY = "--check" in sys.argv


def main() -> int:
    if not DOCS.is_dir():
        print(f"找不到 docs 目录: {DOCS}")
        return 1

    total = 0
    changed = []

    for md in sorted(DOCS.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        matches = PATTERN.findall(text)
        if not matches:
            continue

        n = len(matches)
        total += n
        changed.append((md.name, n))

        if not CHECK_ONLY:
            md.write_text(PATTERN.sub("](../code/", text), encoding="utf-8")

    if total == 0:
        print("检查通过：docs/ 下没有发现 `](code/` 形式的坏链接。")
        return 0

    print(f"发现 {total} 处坏链接：")
    for name, n in changed:
        print(f"  {name}: {n} 处")

    if CHECK_ONLY:
        print("\n（--check 模式，未修改文件）")
        return 1

    print(f"\n已修复 {total} 处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
