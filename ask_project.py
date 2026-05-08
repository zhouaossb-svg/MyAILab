#!/usr/bin/env python3
"""
ask_project.py — 自然语言查询 MyAILab 项目逻辑细节

用法:
  python ask_project.py -q "image_processor 怎么工作的"
  python ask_project.py -q "数据流是怎样的"
  python ask_project.py -q "模型结构"
  python ask_project.py         # 交互模式
"""

import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT = Path(__file__).resolve().parent


# ════════════════════════════════════════════════════════════════
#  知识库：解析 PROJECT_MAP.md 为可搜索的章节
# ════════════════════════════════════════════════════════════════

def _load_knowledge_base() -> List[Dict]:
    """将 PROJECT_MAP.md 拆成带标题和权重标签的章节块。"""
    md_path = PROJECT / "PROJECT_MAP.md"
    if not md_path.exists():
        return [{"title": "（无文档）", "content": "PROJECT_MAP.md 不存在。", "tags": []}]

    text = md_path.read_text(encoding="utf-8")
    # 按 ## 二级标题分节
    sections = []
    for block in re.split(r"\n(?=## )", text):
        title_match = re.match(r"## (.+)", block)
        title = title_match.group(1) if title_match else "（前言）"
        # 章节内的一级标签作为搜索关键词
        tags = re.findall(r"`[^`]+`|\b\w{3,}\b", block.lower())
        # 文件路径标签
        file_tags = re.findall(r"[\w.-]+\.(?:py|pth|png|jpg|html)", block.lower())
        sections.append({
            "title": title,
            "content": block.strip(),
            "tags": list(set(t.lower() for t in tags + file_tags)),
        })
    return sections


_KB: Optional[List[Dict]] = None

def knowledge_base() -> List[Dict]:
    global _KB
    if _KB is None:
        _KB = _load_knowledge_base()
    return _KB


# ════════════════════════════════════════════════════════════════
#  代码检索：解析 .py 文件中的函数/类定义
# ════════════════════════════════════════════════════════════════

def _extract_code_nodes(filepath: Path) -> List[Dict]:
    """AST 解析一个 .py 文件，提取函数和类的签名+文档+行号。"""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    source_lines = filepath.read_text(encoding="utf-8").splitlines()
    total = len(source_lines)

    nodes: List[Dict] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node) or ""
            end = min(node.lineno + 30, total)
            lines = source_lines[node.lineno - 1: end]

            nodes.append({
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "name": node.name,
                "doc": doc.strip(),
                "lineno": node.lineno,
                "source": "\n".join(lines),
                "file": filepath.name,
            })
    return nodes


def _load_all_code() -> List[Dict]:
    """加载 MyAILab 下所有 .py 的代码节点。"""
    nodes = []
    for py in sorted(PROJECT.glob("*.py")):
        if py.name == "ask_project.py":
            continue
        nodes.extend(_extract_code_nodes(py))
    return nodes


_CODE_CACHE: Optional[List[Dict]] = None

def code_base() -> List[Dict]:
    global _CODE_CACHE
    if _CODE_CACHE is None:
        _CODE_CACHE = _load_all_code()
    return _CODE_CACHE


# ════════════════════════════════════════════════════════════════
#  关键词匹配 & 排序
# ════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> set:
    """提取中文+英文关键词"""
    # 中文分词：按字 + 双字组合
    chinese_chars = re.findall(r"[一-鿿]+", text)
    tokens = list(chinese_chars)
    # 中文双字组合
    for s in chinese_chars:
        if len(s) >= 2:
            tokens.extend(s[i:i + 2] for i in range(len(s) - 1))
    # 英文关键词（3 个字母以上，排除纯数字）
    tokens.extend(re.findall(r"[a-zA-Z_]\w{2,}", text))
    # 文件扩展名特意保留
    tokens.extend(re.findall(r"[\w.-]+\.\w+", text))
    return set(t.lower() for t in tokens)


def _relevance(q_tokens: set, tags: list) -> int:
    """计算查询与标签集合的匹配分值。"""
    q_tokens_lower = {t.lower() for t in q_tokens}
    tags_lower = {t.lower() for t in tags}
    return len(q_tokens_lower & tags_lower)


def _smart_search(query: str) -> List[Tuple[str, str, int]]:
    """
    返回 (类型, 内容, 相关度) 列表，按相关度降序。
    类型: "doc" | "code"
    """
    q_tokens = _tokenize(query)

    results: List[Tuple[str, str, int]] = []

    # 1) 搜索文档章节
    for sec in knowledge_base():
        score = _relevance(q_tokens, sec["tags"])
        # 查询中的文件名直接命中加分
        for qt in q_tokens:
            if qt in sec["title"].lower():
                score += 3
        if score > 0:
            results.append(("doc", f"**{sec['title']}**\n\n{sec['content']}", score))

    # 2) 搜索代码节点
    for node in code_base():
        tags = _tokenize(node["name"]) | _tokenize(node["doc"]) | {node["file"]}
        score = _relevance(q_tokens, tags)
        if score > 0:
            snippet = node["source"]
            if len(snippet) > 600:
                snippet = snippet[:600] + "\n    ..."
            header = f"`{node['file']}:{node['lineno']}`  **{node['name']}**"
            if node["doc"]:
                header += f"\n> {node['doc'][:200]}"
            results.append(("code", f"{header}\n\n```python\n{snippet}\n```", score * 2))

    # 3) 搜索文件名（对所有文件，包括 .pth / .md）
    all_files = list(PROJECT.glob("*"))
    for f in all_files:
        score = _relevance(q_tokens, {f.name, f.stem})
        if score > 0 and f.suffix in {".py", ".pth", ".md", ".html", ".png", ".jpg"}:
            size = f.stat().st_size
            if f.suffix == ".pth":
                extra = f"  model: {_pth_info(f)}"
            else:
                extra = ""
            results.append(("file", f"`{f.name}`  ({_human_size(size)}){extra}", score))

    # 去重 + 按分数降序
    seen = set()
    unique = []
    for typ, content, score in sorted(results, key=lambda x: -x[2]):
        # 用前 80 个字符做去重 key
        key = content[:80]
        if key not in seen:
            seen.add(key)
            unique.append((typ, content, score))

    return unique


_PTH_INFO = {
    "mnist_cnn_model.pth":   "MNIST_CNN · 225,034 参数 · 8 层 · 主力手写模型",
    "mnist_deep_model.pth":  "DeepNet · 669,706 参数 · 10 层 · 5 层全连接",
    "mnist_model.pth":       "SimpleNet · 109,386 参数 · 6 层 · 3 层全连接基线",
    "fashion_cnn_model.pth": "FashionCNN · 31,338 参数 · 8 层 · FashionMNIST 版",
}

def _pth_info(path: Path) -> str:
    return _PTH_INFO.get(path.name, "")


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# ════════════════════════════════════════════════════════════════
#  输出格式
# ════════════════════════════════════════════════════════════════

_COLORS = {"doc": "\x1b[36m", "code": "\x1b[33m", "file": "\x1b[32m", "reset": "\x1b[0m"}


def _color(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def _sanitize(text: str) -> str:
    """Remove/ replace characters that cannot be displayed in a GBK
    terminal (emoji, box-drawing, unusual symbols)."""
    # Replace multi-byte symbols with ASCII fallbacks
    text = text.replace("✅", "[OK]").replace("❌", "[NO]")
    text = text.replace("→", "->").replace("↓", "|")
    text = text.replace("△", "▲").replace("▽", "▼")
    # Strip remaining characters outside BMP or in private use
    result = []
    for ch in text:
        try:
            ch.encode("gbk")
            result.append(ch)
        except UnicodeEncodeError:
            result.append(" ")
    return "".join(result)


def _print_answer(results: List[Tuple[str, str, int]], query: str) -> None:
    if not results:
        print(f"\n  [无匹配] 没找到与「{query}」相关的内容。试试换个问法。\n")
        return

    top_score = results[0][2]

    for typ, content, score in results:
        # 只显示最高分的条目，或相关度足够高的
        if score < top_score * 0.4 and score < 2:
            continue

        prefix = {"doc": "[文档]", "code": "[代码]", "file": "[文件]"}.get(typ, "[信息]")
        label = _color(f"── {prefix} ───────────────────────────────────", typ)

        print(f"\n{label}")
        content = _sanitize(content)
        for line in content.split("\n"):
            if line.startswith("```"):
                print(_color(line, typ))
            else:
                print(line)

    # 搜索提示
    print(_color("\n── [提示] ──────────────────────────────────", "doc"))
    print(f"  共匹配 {len(results)} 项。输入 -q \"你的问题\" 继续查询。")


# ════════════════════════════════════════════════════════════════
#  交互模式
# ════════════════════════════════════════════════════════════════

def _interactive_loop() -> None:
    print(_color("===== MyAILab 项目问答助手 =====", "doc"))
    print(_color("输入问题（自然语言），输入 q 退出", "doc"))

    while True:
        try:
            q = input(_color("\n> ", "doc")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            break
        results = _smart_search(q)
        _print_answer(results, q)


# ════════════════════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="自然语言查询 MyAILab 项目逻辑细节",
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        default=None,
        help='自然语言问题，例如 "image_processor 怎么工作的"',
    )
    parser.add_argument(
        "--list-files", action="store_true", help="列出项目所有文件"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="列出所有预训练模型及参数",
    )

    args = parser.parse_args()

    if args.list_files:
        print("\n[文件清单] 项目文件清单：\n")
        for f in sorted(PROJECT.glob("*")):
            if f.is_dir():
                continue
            size = _human_size(f.stat().st_size)
            print(f"  {f.name:40s} {size}")
        print()
        return

    if args.list_models:
        print("\n[模型清单] 预训练模型：\n")
        for f in sorted(PROJECT.glob("*.pth")):
            print(f"  {f.name:35s} {_pth_info(f)}")
        print()
        return

    # 只预加载知识库，代码库按需惰性加载
    _ = knowledge_base()

    if args.question:
        results = _smart_search(args.question)
        _print_answer(results, args.question)
    else:
        _interactive_loop()


if __name__ == "__main__":
    main()
