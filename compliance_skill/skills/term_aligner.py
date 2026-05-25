"""
skills/term_aligner.py
Skill 1 — 术语对齐（语义版）

改进点（对比原 standard_aligner.py）：
  - 原版：JSON 字典精确匹配，找不到直接返回错误
  - 新版：embedding 语义相似度 → 词典精确匹配 → LLM 零样本推断，三级降级
  - 引用：Nandini et al. (2025) 用 LLM 从贸易协定文本中抽取 SPO 三元组，
          验证了 LLM 在法律语言细微差别上优于传统 NLP 方法。
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional

from config.settings import TERM_MAP_PATH, MODEL, OPENROUTER_API_KEY
from openai import OpenAI

_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ── 载入本地词典 ──────────────────────────────────────────────────
def _load_term_map() -> dict:
    if TERM_MAP_PATH.exists():
        return json.loads(TERM_MAP_PATH.read_text(encoding="utf-8"))
    return {}

_TERM_MAP: dict = _load_term_map()


# ── 层 1：精确字典匹配 ────────────────────────────────────────────
def _exact_match(term: str) -> Optional[dict]:
    """大小写不敏感的精确查找。"""
    key = term.lower().strip()
    return _TERM_MAP.get(key) or _TERM_MAP.get(term.strip())


# ── 层 2：embedding 余弦相似度匹配 ───────────────────────────────
def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b))
    return dot / norm if norm else 0.0

def _embed(text: str) -> list[float]:
    """
    调用 embedding 端点（text-embedding-3-small）。
    生产环境建议缓存，避免重复计费。
    """
    resp = _client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding

def _semantic_match(term: str, threshold: float = 0.82) -> Optional[dict]:
    """
    对词典中所有条目计算余弦相似度，返回最高分且超过阈值的结果。
    词典规模小时可接受；规模大时应换用向量数据库（如 Chroma / FAISS）。

    参考：Kommineni et al. (2024) 及 KARMA 框架（Lu & Wang 2025）中的
    schema-guided entity normalization。
    """
    if not _TERM_MAP:
        return None
    try:
        q_vec = _embed(term)
    except Exception:
        return None

    best_val, best_entry = 0.0, None
    for key, entry in _TERM_MAP.items():
        try:
            k_vec = _embed(key)
            score = _cosine(q_vec, k_vec)
            if score > best_val:
                best_val, best_entry = score, entry
        except Exception:
            continue

    if best_val >= threshold and best_entry is not None:
        return {**best_entry, "_match_score": round(best_val, 4), "_match_method": "semantic"}
    return None


# ── 层 3：LLM 零样本推断 ─────────────────────────────────────────
_ALIGN_SYSTEM = """你是食品化学专家。将输入的外文/缩写食品添加剂名称映射为中国国标（GB）规范名称。
仅返回 JSON，格式：
{"国标名": "...", "CAS号": "...", "备注": "..."}
如无法确定，返回：{"国标名": null, "备注": "无法识别"}
禁止输出任何其他文字。"""

def _llm_infer(term: str) -> dict:
    """
    用 LLM 做零样本推断（最终兜底）。
    标记 _match_method=llm_infer，供下游置信度门控参考。
    引用：Agarwal et al. (2024) RAGulating Compliance 框架中的
    agent-based knowledge normalization。
    """
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _ALIGN_SYSTEM},
                {"role": "user",   "content": term},
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["_match_method"] = "llm_infer"
        return result
    except Exception as e:
        return {"国标名": None, "备注": f"LLM 推断失败: {e}", "_match_method": "llm_infer"}


# ── 公开接口 ──────────────────────────────────────────────────────
def align_term(foreign_term: str) -> dict:
    """
    三级降级的术语对齐主函数。

    返回字段：
      国标名        str | None  — 规范化后的国标名称
      CAS号         str | None  — 可选
      备注          str         — 来源说明
      _match_method str         — exact | semantic | llm_infer
      _confidence   str         — high | medium | low（供置信度门控使用）
    """
    # 层 1：精确匹配
    result = _exact_match(foreign_term)
    if result:
        return {**result, "_match_method": "exact", "_confidence": "high"}

    # 层 2：语义相似度
    result = _semantic_match(foreign_term)
    if result:
        confidence = "high" if result.get("_match_score", 0) >= 0.90 else "medium"
        return {**result, "_confidence": confidence}

    # 层 3：LLM 零样本推断
    result = _llm_infer(foreign_term)
    return {**result, "_confidence": "low"}


# ── 知识图谱更新（供 kg_builder 调用）────────────────────────────
def update_term_map(new_entries: dict[str, dict]) -> int:
    """
    向本地词典追加新条目，持久化到 TERM_MAP_PATH。
    返回实际新增条目数。
    """
    global _TERM_MAP
    added = 0
    for k, v in new_entries.items():
        if k.lower() not in _TERM_MAP:
            _TERM_MAP[k.lower()] = v
            added += 1
    TERM_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    TERM_MAP_PATH.write_text(
        json.dumps(_TERM_MAP, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return added
