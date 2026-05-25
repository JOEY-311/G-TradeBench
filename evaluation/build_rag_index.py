#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地法规文件 RAG 索引构建脚本
扫描 各国法规和部分数据集/ 目录，提取文本，按国家切块，保存为 JSON

依赖（按需安装）：
  pip install pdfplumber python-docx

用法：
  python build_rag_index.py           # 构建完整索引
  python build_rag_index.py --test    # 只处理前 5 个文件（测试用）
"""

import hashlib, json, os, re, sys, zipfile
from pathlib import Path

# ════════════════════════════════════════════════════════════════
#  配置
# ════════════════════════════════════════════════════════════════
RAG_ROOT  = Path('E:/论文/跨境对齐/各国法规和部分数据集')
OUT_PATH  = Path('E:/论文/跨境对齐/评测阶段/rag_index.json')

CHUNK_SIZE    = 600    # 每块字符数
CHUNK_OVERLAP = 80     # 相邻块重叠字符数
MAX_PER_FILE  = 120    # 每个文件最多保留块数（防止超大文件撑爆索引）
TEST_MODE     = '--test' in sys.argv

# 文件夹名 → 国家标签（一个文件夹可对应多个国家）
COUNTRY_MAP: dict = {
    '中国':   ['中国'],
    '日本':   ['日本'],
    '韩国':   ['韩国'],
    '美国':   ['美国'],
    '德法':   ['德国', '法国'],   # 德法目录同时服务两国
}

# 跳过的文件名关键词（非法规文件）
SKIP_KEYWORDS = ['EU_Agent_Bench', '参考阅读', 'multiagent', 'SkillRL',
                 'Reflexion', 'ToT', 'Language', 'OmniGAIA', '学习笔记',
                 '研究计划', '选题', '研讨', '切入点']

# GUID 正则（跳过重复的 _origin 子目录文件）
GUID_PAT = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
    re.I)


# ════════════════════════════════════════════════════════════════
#  文件读取
# ════════════════════════════════════════════════════════════════
def read_pdf(path: Path) -> str:
    # 优先 pdfplumber（效果更好）
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    pages.append(t)
        return '\n'.join(pages)
    except ImportError:
        pass
    except Exception as e:
        print(f'    pdfplumber失败: {e}')

    # 降级 PyPDF2
    try:
        import PyPDF2
        pages = []
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    pages.append(t)
        return '\n'.join(pages)
    except ImportError:
        pass
    except Exception as e:
        print(f'    PyPDF2失败: {e}')

    return ''


def read_docx(path: Path) -> str:
    # 优先 python-docx
    try:
        from docx import Document
        doc = Document(path)
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        pass
    except Exception as e:
        print(f'    python-docx失败: {e}')

    # 降级：直接解 ZIP 读 XML
    try:
        with zipfile.ZipFile(path) as z:
            xml_bytes = z.read('word/document.xml')
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_bytes)
        ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        texts = [node.text for node in root.iter(f'{ns}t') if node.text]
        return ' '.join(texts)
    except Exception as e:
        print(f'    ZIP解析失败: {e}')

    return ''


def read_txt(path: Path) -> str:
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin-1'):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ''


# ════════════════════════════════════════════════════════════════
#  文本处理
# ════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    text = re.sub(r'[ \t]{2,}', ' ', text)      # 多空格 → 单空格
    text = re.sub(r'\n{3,}', '\n\n', text)       # 多空行 → 双换行
    text = re.sub(r'[^\S\n]+\n', '\n', text)     # 行尾空白
    return text.strip()


def chunk_text(text: str,
               size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list:
    text = clean_text(text)
    if not text or len(text) < 30:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 30]


def text_hash(text: str) -> str:
    return hashlib.md5(text[:200].encode('utf-8', errors='ignore')).hexdigest()[:8]


# ════════════════════════════════════════════════════════════════
#  国家推断
# ════════════════════════════════════════════════════════════════
def infer_countries(path: Path) -> list:
    parts_str = '/'.join(path.parts)
    for folder, countries in COUNTRY_MAP.items():
        if f'/{folder}/' in parts_str or parts_str.endswith(f'/{folder}'):
            return countries
    return []


# ════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════
def should_skip(path: Path) -> bool:
    name = path.name
    path_str = str(path)
    # 跳过 GUID 文件（重复的 _origin）
    if GUID_PAT.search(name):
        return True
    # 跳过非法规关键词
    if any(kw in path_str for kw in SKIP_KEYWORDS):
        return True
    return False


def main():
    exts = {'.pdf', '.docx', '.txt'}
    all_files = sorted(
        f for f in RAG_ROOT.rglob('*')
        if f.is_file() and f.suffix.lower() in exts and not should_skip(f)
    )

    if TEST_MODE:
        all_files = all_files[:5]
        print(f'[测试模式] 只处理前 {len(all_files)} 个文件')
    else:
        print(f'共发现 {len(all_files)} 个有效法规文件，开始建索引...')

    index = []
    seen_hashes: set = set()
    country_stats: dict = {}

    for i, fp in enumerate(all_files, 1):
        countries = infer_countries(fp)
        if not countries:
            continue   # 无法归属国家的文件跳过

        rel = str(fp.relative_to(RAG_ROOT))
        suffix = fp.suffix.lower()
        print(f'[{i:>3}/{len(all_files)}] {countries} {fp.name[:45]}', end=' ... ')

        # 读取文本
        if suffix == '.pdf':
            raw = read_pdf(fp)
        elif suffix == '.docx':
            raw = read_docx(fp)
        else:
            raw = read_txt(fp)

        if not raw or len(raw.strip()) < 50:
            print('(空)')
            continue

        chunks = chunk_text(raw)
        # 限制每文件块数，优先保留前段（通常是核心条款）
        if len(chunks) > MAX_PER_FILE:
            # 均匀采样：头部 60% + 尾部 40%
            head = int(MAX_PER_FILE * 0.6)
            tail = MAX_PER_FILE - head
            step = max(1, len(chunks) // tail)
            sampled_tail = chunks[head::step][:tail]
            chunks = chunks[:head] + sampled_tail

        added = 0
        for chunk in chunks:
            h = text_hash(chunk)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            for country in countries:
                index.append({
                    'country': country,
                    'source':  rel[:80],
                    'text':    chunk,
                })
                country_stats[country] = country_stats.get(country, 0) + 1
                added += 1

        print(f'{added} 块')

    print(f'\n{"─"*60}')
    print(f'索引总块数: {len(index)}')
    print('各国分布:')
    for c, n in sorted(country_stats.items()):
        print(f'  {c}: {n} 块')

    OUT_PATH.write_text(
        json.dumps(index, ensure_ascii=False, separators=(',', ':')),
        encoding='utf-8')
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f'\n索引已保存至: {OUT_PATH}  ({size_mb:.1f} MB)')
    print('下一步：python run_claude_strategies.py  （自动加载索引）')


if __name__ == '__main__':
    main()
