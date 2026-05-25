"""
knowledge_base/kg_builder.py
知识图谱增量构建器

从非结构化法规文本中提取 SPO 三元组，写入本地 KG。
支持：Markdown 文件、PDF 文本提取结果、JSON 法规条目

引用：
  Nandini et al. (2025) — LLM 从贸易协定文本提取 SPO 三元组，
    零样本和少样本 prompting 均优于传统 NLP 方法
  Agarwal et al. (2024) — KG 清洗、去重、规范化、增量更新 pipeline
  Lu & Wang (2025) KARMA 框架 — schema-guided multi-agent extraction
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Iterator

from openai import OpenAI
from config.settings import MODEL, OPENROUTER_API_KEY
from skills.regulation_retriever import append_triples

_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ── SPO 抽取 Prompt ───────────────────────────────────────────────
_EXTRACT_SYSTEM = """\
你是知识图谱构建专家，专注于食品法规领域。
从用户提供的法规文本中提取 SPO（主语-谓语-宾语）三元组。

每个三元组必须包含：
  subject        — 实体（食品类别、物质名、法规名、国家）
  predicate      — 关系（限量为、禁用于、生效日期、适用范围、修订为）
  object         — 值或实体（限量数值+单位、日期、食品名称）
  country        — 适用国家/地区
  effective_date — 法规生效日期（YYYY-MM 或 YYYY-MM-DD），无则填 "Unknown"
  legal_status   — Proposed | Immediately Effective | Notification of Update
  source         — 法规编号或来源（如 GB 2762-2022、EU Reg 2023/915）
  confidence     — 抽取置信度（0.0–1.0）

仅输出 JSON 数组，禁止任何其他文字。示例：
[
  {
    "subject": "铅",
    "predicate": "在畜禽肉中限量为",
    "object": "0.2 mg/kg",
    "country": "中国",
    "effective_date": "2023-06",
    "legal_status": "Immediately Effective",
    "source": "GB 2762-2022",
    "confidence": 0.95
  }
]
"""

def extract_triples_from_text(text: str, source_label: str = "") -> list[dict]:
    """
    用 LLM 从法规文本段落中提取 SPO 三元组。

    参数：
      text         — 原始法规文本（建议 ≤ 2000 字符/块，超长请先分块）
      source_label — 文件名或法规编号，附加到每个三元组的 source 字段

    返回：三元组列表（可能为空）
    """
    if not text.strip():
        return []

    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user",   "content": f"【来源】{source_label}\n\n{text[:3000]}"},
            ],
            temperature=0,
            max_tokens=2000,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        triples = json.loads(raw)
        if isinstance(triples, list):
            return triples
    except Exception as e:
        print(f"  ⚠ triple 抽取失败 [{source_label}]: {e}")
    return []


# ── 文本分块 ──────────────────────────────────────────────────────
def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> Iterator[str]:
    """将长文本按固定窗口分块，相邻块有 overlap 字符重叠以保留上下文。"""
    start = 0
    while start < len(text):
        yield text[start: start + chunk_size]
        start += chunk_size - overlap


# ── 从文件批量构建 KG ─────────────────────────────────────────────
def build_kg_from_markdown(md_path: Path, delay: float = 1.0) -> int:
    """
    从 Markdown 法规文件逐段抽取三元组并写入本地 KG。

    参数：
      md_path — Markdown 文件路径
      delay   — 每次 API 调用后的等待秒数（避免触发限速）

    返回：成功写入的新三元组数量
    """
    text = md_path.read_text(encoding="utf-8")
    source_label = md_path.stem
    total_added = 0

    for i, chunk in enumerate(_chunk_text(text)):
        print(f"  处理块 {i+1}（{len(chunk)} 字符）...")
        triples = extract_triples_from_text(chunk, source_label=source_label)
        if triples:
            added = append_triples(triples)
            total_added += added
            print(f"    ✓ 新增 {added} 个三元组")
        time.sleep(delay)

    return total_added


def build_kg_from_json_list(json_path: Path) -> int:
    """
    从已结构化的 JSON 法规条目列表直接写入 KG（无需 LLM 抽取）。
    JSON 格式：[{"subject": ..., "predicate": ..., "object": ..., ...}]
    """
    entries = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"{json_path} 应为 JSON 数组")
    return append_triples(entries)


# ── 去重与置信度过滤 ──────────────────────────────────────────────
def clean_kg(min_confidence: float = 0.70) -> int:
    """
    过滤 KG 中置信度过低的条目，返回剩余条目数。

    引用：Agarwal et al. (2024) — KG cleaning and normalization pipeline。
    """
    from skills.regulation_retriever import _KG, KG_PATH
    original_len = len(_KG)
    cleaned = [t for t in _KG if float(t.get("confidence", 1.0)) >= min_confidence]

    from skills import regulation_retriever as rr
    rr._KG = cleaned
    from config.settings import KG_PATH as _KG_PATH
    _KG_PATH.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

    removed = original_len - len(cleaned)
    print(f"  KG 清洗：移除 {removed} 个低置信三元组，剩余 {len(cleaned)} 个")
    return len(cleaned)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：python kg_builder.py <法规文件.md 或 .json>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if path.suffix == ".md":
        n = build_kg_from_markdown(path)
    elif path.suffix == ".json":
        n = build_kg_from_json_list(path)
    else:
        print("仅支持 .md 和 .json 文件")
        sys.exit(1)
    print(f"完成，共新增 {n} 个三元组")
