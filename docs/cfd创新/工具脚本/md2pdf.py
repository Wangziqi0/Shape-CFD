#!/usr/bin/env python3
"""
将 Markdown 文件转换为精美 PDF
使用 markdown + weasyprint，支持中文、表格、代码块等
"""

import markdown
from weasyprint import HTML
import sys
import os

# ---------- 配置 ----------
INPUT_FILE = "CFD-RAG研究完整总结.md"
OUTPUT_FILE = "CFD-RAG研究完整总结.pdf"

# ---------- 读取 Markdown ----------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    md_text = f.read()

# ---------- Markdown → HTML ----------
extensions = [
    "tables",
    "fenced_code",
    "codehilite",
    "toc",
    "nl2br",
    "sane_lists",
    "smarty",
]
extension_configs = {
    "codehilite": {
        "css_class": "codehilite",
        "linenums": False,
        "guess_lang": False,
    }
}

html_body = markdown.markdown(
    md_text, extensions=extensions, extension_configs=extension_configs
)

# ---------- CSS 样式 ----------
css = """
@page {
    size: A4;
    margin: 2cm 2.2cm 2cm 2.2cm;
    
    @top-center {
        content: "AD-Rank：对流-扩散图排序";
        font-size: 8pt;
        color: #94a3b8;
        font-family: "Noto Sans CJK SC", sans-serif;
    }
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 8pt;
        color: #94a3b8;
        font-family: "Noto Sans CJK SC", sans-serif;
    }
}

@page :first {
    @top-center { content: none; }
    margin-top: 3cm;
}

/* ---- 基础 ---- */
body {
    font-family: "Noto Serif CJK SC", "Noto Sans CJK SC", "Source Han Serif SC", serif;
    font-size: 10.5pt;
    line-height: 1.75;
    color: #1e293b;
    text-align: justify;
}

/* ---- 标题系统 ---- */
h1 {
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 22pt;
    font-weight: 900;
    color: #0f172a;
    text-align: center;
    margin-top: 0;
    margin-bottom: 8pt;
    padding-bottom: 12pt;
    border-bottom: 3px solid #3b82f6;
    letter-spacing: 1pt;
}

h2 {
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 15pt;
    font-weight: 700;
    color: #1e40af;
    margin-top: 28pt;
    margin-bottom: 10pt;
    padding-bottom: 5pt;
    border-bottom: 1.5px solid #93c5fd;
    page-break-after: avoid;
}

h3 {
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 12pt;
    font-weight: 600;
    color: #1d4ed8;
    margin-top: 18pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}

h4 {
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 10.5pt;
    font-weight: 600;
    color: #2563eb;
    margin-top: 12pt;
    margin-bottom: 4pt;
}

/* ---- 段落 ---- */
p {
    margin-top: 4pt;
    margin-bottom: 8pt;
    orphans: 3;
    widows: 3;
}

/* ---- 强调 ---- */
strong {
    color: #0f172a;
    font-weight: 700;
}

em {
    color: #475569;
}

/* ---- 引用块 ---- */
blockquote {
    margin: 12pt 0;
    padding: 10pt 16pt;
    background: linear-gradient(135deg, #eff6ff, #f0f9ff);
    border-left: 4px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    color: #334155;
    font-size: 10pt;
}

blockquote p {
    margin: 4pt 0;
}

/* ---- 代码块 ---- */
pre {
    background: #0f172a;
    color: #e2e8f0;
    padding: 14pt 16pt;
    border-radius: 8px;
    font-family: "JetBrains Mono", "Fira Code", "Noto Sans Mono CJK SC", monospace;
    font-size: 9pt;
    line-height: 1.6;
    margin: 10pt 0;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    page-break-inside: avoid;
}

code {
    font-family: "JetBrains Mono", "Fira Code", "Noto Sans Mono CJK SC", monospace;
    font-size: 9pt;
    background: #e0e7ff;
    color: #3730a3;
    padding: 1pt 4pt;
    border-radius: 3px;
}

pre code {
    background: none;
    color: inherit;
    padding: 0;
    border-radius: 0;
}

/* ---- 表格 ---- */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

thead {
    background: linear-gradient(135deg, #1e40af, #3b82f6);
}

thead th {
    color: #ffffff;
    font-weight: 600;
    padding: 8pt 10pt;
    text-align: left;
    font-family: "Noto Sans CJK SC", sans-serif;
    border: none;
}

thead th:first-child {
    border-radius: 6px 0 0 0;
}

thead th:last-child {
    border-radius: 0 6px 0 0;
}

tbody tr:nth-child(even) {
    background: #f8fafc;
}

tbody tr:nth-child(odd) {
    background: #ffffff;
}

tbody tr:hover {
    background: #eff6ff;
}

td {
    padding: 7pt 10pt;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}

tbody tr:last-child td:first-child {
    border-radius: 0 0 0 6px;
}

tbody tr:last-child td:last-child {
    border-radius: 0 0 6px 0;
}

/* ---- 列表 ---- */
ul, ol {
    margin: 6pt 0;
    padding-left: 22pt;
}

li {
    margin-bottom: 3pt;
}

li > ul, li > ol {
    margin-top: 2pt;
}

/* ---- 分割线 ---- */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #93c5fd, transparent);
    margin: 20pt 0;
}

/* ---- 链接 ---- */
a {
    color: #2563eb;
    text-decoration: none;
}

/* ---- codehilite 代码高亮修正 ---- */
.codehilite {
    background: #0f172a;
    border-radius: 8px;
    margin: 10pt 0;
    padding: 0;
}

.codehilite pre {
    margin: 0;
    border-radius: 8px;
}

/* ---- 删除线 ---- */
del {
    color: #94a3b8;
    text-decoration: line-through;
}

/* ---- 首页元信息 ---- */
body > blockquote:first-of-type {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-left: 4px solid #0ea5e9;
    text-align: center;
    font-size: 10pt;
    color: #475569;
}

/* ---- 章节编号样式增强 ---- */
h2 + h3 {
    margin-top: 10pt;
}

/* ---- 避免分页 ---- */
h2, h3, h4 {
    page-break-after: avoid;
}

table, pre, blockquote {
    page-break-inside: avoid;
}

/* ---- emoji 和符号修正 ---- */
"""

# ---------- 组装完整 HTML ----------
full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>"""

# ---------- HTML → PDF ----------
print(f"正在生成 PDF: {OUTPUT_FILE}")
HTML(string=full_html).write_pdf(OUTPUT_FILE)
print(f"✅ PDF 已生成: {os.path.abspath(OUTPUT_FILE)}")
print(f"   文件大小: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
