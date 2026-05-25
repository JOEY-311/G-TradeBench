"""
skills/regulation_retriever.py
Skill 2 — 法规检索（KG + GraphRAG 版）

改进点（对比原 rule_updater.py）：
  - 原版：Markdown 关键词行匹配，无法处理关系型查询
  - 新版：
      层 1 — 官方 API 实时查询（USA Federal Register、EU EUR-Lex SPARQL 等）
      层 2 — 本地知识图谱（SPO 三元组）遍历 + 置信度评分
      层 3 — 向量 RAG 回退（适用于非结构化法规文本）
      层 4 — 静态知识库最终兜底

引用：
  Agarwal et al. (2024) — KG + RAG 多智能体合规 QA 框架
  Edge et al. (2024, ICLR 2025) — GraphRAG：通过图结构 KG 进行 RAG 检索
  Nandini et al. (2025) — LLM 从贸易协定文本抽取 SPO 三元组
"""
from __future__ import annotations
import json
import re
import time
import html as html_module
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from config.settings import (
    KG_PATH, NET_TIMEOUT, SUPPORTED_COUNTRIES, OPENROUTER_API_KEY, MODEL
)
from openai import OpenAI

_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

_HEADERS = {
    "User-Agent": "ComplianceSkill/2.0 (research; contact: researcher@example.com)",
    "Accept": "application/json, text/html",
}

# ── 知识图谱加载 ──────────────────────────────────────────────────
def _load_kg() -> list[dict]:
    """
    加载本地 KG（SPO 三元组列表）。
    格式：[{"subject": ..., "predicate": ..., "object": ...,
             "country": ..., "source": ..., "effective_date": ...,
             "confidence": float}, ...]
    """
    if KG_PATH.exists():
        try:
            return json.loads(KG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

_KG: list[dict] = _load_kg()


# ── HTTP 工具 ─────────────────────────────────────────────────────
def _http_get(url: str, params: dict | None = None, as_json: bool = False):
    try:
        if _HAS_REQUESTS:
            r = _requests.get(url, params=params, headers=_HEADERS, timeout=NET_TIMEOUT)
            r.raise_for_status()
            return r.json() if as_json else r.text
        else:
            full = url + ("?" + urllib.parse.urlencode(params) if params else "")
            req = urllib.request.Request(full, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw) if as_json else raw
    except Exception:
        return None

def _strip_html(text: str, max_chars: int = 400) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", html_module.unescape(text)).strip()
    return text[:max_chars]


# ── 层 1：各国官方 API ────────────────────────────────────────────
def _fetch_usa(keywords: str) -> list[dict]:
    """Federal Register REST API — 免费，无需密钥"""
    data = _http_get(
        "https://www.federalregister.gov/api/v1/documents.json",
        params={
            "conditions[term]": keywords,
            "conditions[agencies][]": "food-and-drug-administration",
            "conditions[type][]": ["RULE", "PRULE", "NOTICE"],
            "per_page": 5,
            "order": "newest",
            "fields[]": ["title", "type", "effective_on", "publication_date", "abstract", "html_url"],
        },
        as_json=True,
    )
    if not data:
        return []
    results = []
    for doc in data.get("results", []):
        doc_type = doc.get("type", "")
        status = (
            "Proposed" if doc_type == "PRULE"
            else "Notification of Update" if doc_type == "NOTICE"
            else "Immediately Effective"
        )
        results.append({
            "title": doc.get("title", ""),
            "legal_status": status,
            "effective_date": doc.get("effective_on") or doc.get("publication_date", "Not specified"),
            "abstract": (doc.get("abstract") or "")[:300],
            "url": doc.get("html_url", ""),
            "source": "Federal Register API",
            "retrieval_layer": 1,
        })
    return results

def _fetch_eu(keywords: str) -> list[dict]:
    """EUR-Lex SPARQL 端点，降级为页面抓取"""
    kw_esc = keywords.replace('"', '\\"')
    sparql = f"""
    PREFIX eli: <http://data.europa.eu/eli/ontology#>
    PREFIX dc:  <http://purl.org/dc/elements/1.1/>
    SELECT ?work ?title ?date WHERE {{
      ?work eli:date_document ?date ; dc:title ?title .
      FILTER(LANG(?title) = 'en')
      FILTER(CONTAINS(LCASE(STR(?title)), LCASE("{kw_esc}")))
      FILTER(?date >= "2020-01-01"^^xsd:date)
    }} ORDER BY DESC(?date) LIMIT 5
    """
    data = _http_get(
        "https://publications.europa.eu/webapi/rdf/sparql",
        params={"query": sparql, "format": "application/sparql-results+json"},
        as_json=True,
    )
    results = []
    if data:
        for b in data.get("results", {}).get("bindings", []):
            results.append({
                "title": b.get("title", {}).get("value", ""),
                "legal_status": "Immediately Effective",
                "effective_date": b.get("date", {}).get("value", "")[:10],
                "url": b.get("work", {}).get("value", ""),
                "source": "EUR-Lex SPARQL",
                "retrieval_layer": 1,
            })
    if not results:
        # 降级：普通搜索页
        raw = _http_get(
            "https://eur-lex.europa.eu/search.html",
            params={"scope": "EURLEX", "type": "quick", "lang": "en", "text": keywords},
        )
        if raw:
            titles = re.findall(r'class="title"[^>]*>(.*?)</(?:span|a|div)', raw, re.S)
            dates  = re.findall(r"(\d{2}/\d{2}/\d{4})", raw)
            for i, t in enumerate(titles[:5]):
                results.append({
                    "title": _strip_html(t, 150),
                    "legal_status": "Immediately Effective",
                    "effective_date": dates[i] if i < len(dates) else "Not specified",
                    "url": "https://eur-lex.europa.eu",
                    "source": "EUR-Lex search page (fallback)",
                    "retrieval_layer": 1,
                })
    return results

def _fetch_china(keywords: str) -> list[dict]:
    """
    中国：食品安全国家标准数据检索平台（CFSA）+ SAMR 市场监管总局
    参考：https://sppt.cfsa.net.cn:8086/db
    """
    results = []
    html_text = _http_get(
        "https://sppt.cfsa.net.cn:8086/db",
        params={"keyword": keywords, "pageSize": 5},
    )
    if html_text:
        items = re.findall(
            r"(GB\s*\d[\d\-\.]+)[^\n]*?(\d{4}[-年]\d{1,2}[-月]\d{1,2}日?)",
            html_text,
        )
        for std_no, date in items[:5]:
            results.append({
                "title":          std_no.strip(),
                "legal_status":   "Immediately Effective",
                "effective_date": date.strip(),
                "url":            "https://sppt.cfsa.net.cn:8086/db",
                "source":         "CFSA 食品安全国家标准平台",
                "retrieval_layer": 1,
            })
    if not results:
        samr_html = _http_get(
            "https://www.samr.gov.cn/search/search_result.shtml",
            params={"searchword": keywords, "channelid": "228976"},
        )
        if samr_html:
            anchors = re.findall(
                r'<a[^>]+href="([^"]+)"[^>]*>([^<]{5,80})</a>', samr_html
            )
            for url_path, title in anchors[:5]:
                if any(kw in title for kw in ["食品", "标准", "公告", "规定", "限量"]):
                    results.append({
                        "title":          title.strip(),
                        "legal_status":   "Unknown – verify on SAMR",
                        "effective_date": "Not specified",
                        "url":            "https://www.samr.gov.cn" + url_path,
                        "source":         "SAMR 市场监管总局",
                        "retrieval_layer": 1,
                    })
    return results


def _fetch_japan(keywords: str) -> list[dict]:
    """
    日本：厚生劳动省食品安全相关通知检索
    https://www.mhlw.go.jp/
    """
    html_text = _http_get(
        "https://www.mhlw.go.jp/cgi-bin/indexpage/search.cgi",
        params={"keyword": keywords, "category": "shokuhin", "num": 5},
    )
    results = []
    if html_text:
        links = re.findall(
            r'href="(https://www\.mhlw\.go\.jp[^"]+)"[^>]*>\s*([^<]{5,120})</a>',
            html_text,
        )
        dates = re.findall(
            r"(\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{2}/\d{2}|\d{4}-\d{2}-\d{2})",
            html_text,
        )
        for i, (url, title) in enumerate(links[:5]):
            results.append({
                "title":          _strip_html(title, 120),
                "legal_status":   "Immediately Effective",
                "effective_date": dates[i] if i < len(dates) else "Not specified",
                "url":            url,
                "source":         "厚生労働省 MHLW",
                "retrieval_layer": 1,
            })
    return results


def _fetch_korea(keywords: str) -> list[dict]:
    """
    韩国：MFDS 식품안전나라 공지사항 검색
    https://www.foodsafetykorea.go.kr
    """
    html_text = _http_get(
        "https://www.foodsafetykorea.go.kr/portal/board/board.do",
        params={
            "menu_grp":   "MENU_NEW01",
            "menu_no":    "2815",
            "searchWord": keywords,
        },
    )
    results = []
    if html_text:
        items = re.findall(
            r'<td[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</td>.*?(\d{4}\.\d{2}\.\d{2})',
            html_text,
            re.S,
        )
        for title, date in items[:5]:
            results.append({
                "title":          _strip_html(title, 120),
                "legal_status":   "Immediately Effective",
                "effective_date": date,
                "url":            "https://www.foodsafetykorea.go.kr",
                "source":         "MFDS 식품안전나라",
                "retrieval_layer": 1,
            })
    return results


_COUNTRY_FETCHERS = {
    # 中文键
    "美国": _fetch_usa,  "欧盟": _fetch_eu,
    "德国": _fetch_eu,   "法国": _fetch_eu,
    "中国": _fetch_china, "日本": _fetch_japan, "韩国": _fetch_korea,
    # 英文/缩写键
    "usa": _fetch_usa,  "us":  _fetch_usa,  "united states": _fetch_usa,
    "eu":  _fetch_eu,   "europe": _fetch_eu, "germany": _fetch_eu, "france": _fetch_eu,
    "china": _fetch_china, "cn":  _fetch_china,
    "japan": _fetch_japan, "jp":  _fetch_japan,
    "korea": _fetch_korea, "kr":  _fetch_korea, "south korea": _fetch_korea,
    # 澳大利亚（官网 FSANZ，降级到 Layer 2/4）
    "australia": None, "au": None, "澳大利亚": None,
}


# ── 层 2：本地 KG 遍历（GraphRAG 核心）──────────────────────────
def _kg_search(keywords: list[str], country: str, top_n: int = 5) -> list[dict]:
    """
    在本地 SPO 知识图谱中检索相关三元组。

    算法：
      1. 过滤 country 字段（精确 + 前缀）
      2. 对每个三元组计算关键词覆盖得分
      3. 返回 top_n 个置信度加权得分最高的条目

    引用：Edge et al. (2024) GraphRAG — 通过图遍历而非纯向量检索提升
    跨文档关系推理能力；本实现为简化版，适合中小规模 KG。
    """
    pool = [
        t for t in _KG
        if not country or country.lower() in t.get("country", "").lower()
    ]
    if not pool:
        return []

    kw_lower = [k.lower() for k in keywords]

    def score(triple: dict) -> float:
        text = " ".join([
            triple.get("subject", ""),
            triple.get("predicate", ""),
            triple.get("object", ""),
        ]).lower()
        kw_hits = sum(1 for kw in kw_lower if kw in text)
        kg_conf = float(triple.get("confidence", 0.8))
        return kw_hits * kg_conf

    ranked = sorted(pool, key=score, reverse=True)
    top = [t for t in ranked[:top_n] if score(t) > 0]
    return [
        {
            "title": f"{t['subject']} — {t['predicate']} — {t['object']}",
            "legal_status": t.get("legal_status", "Unknown"),
            "effective_date": t.get("effective_date", "Not specified"),
            "source": t.get("source", "Local KG"),
            "confidence": t.get("confidence", 0.8),
            "retrieval_layer": 2,
        }
        for t in top
    ]


# ── 层 3：RAG 索引检索（rag_index.json）──────────────────────────
import os as _os

_RAG_INDEX_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),  # skill root
    "..",  # 上一级（评测阶段/）
    "..",  # 再上一级（跨境对齐/）
    "评测阶段",
    "rag_index.json",
)

_RAG_INDEX: list[dict] = []

def _load_rag_index() -> None:
    """懒加载 rag_index.json（首次调用时初始化）。"""
    global _RAG_INDEX
    if _RAG_INDEX:
        return
    try:
        import json as _json
        from pathlib import Path as _Path
        # 优先环境变量，其次相对路径推导
        rag_path = _os.environ.get("RAG_INDEX_PATH", "")
        if not rag_path:
            rag_path = str(
                _Path(__file__).resolve().parent.parent.parent.parent
                / "评测阶段" / "rag_index.json"
            )
        if _Path(rag_path).exists():
            _RAG_INDEX = _json.loads(_Path(rag_path).read_text(encoding="utf-8"))
    except Exception:
        pass  # 索引加载失败时静默降级


def _vector_rag_search(keywords: list[str], country: str, top_n: int = 4) -> list[dict]:
    """
    基于关键词覆盖率的 RAG 检索（rag_index.json）。

    rag_index.json 格式：[{"country": str, "source": str, "text": str}, ...]

    引用：Agarwal et al. (2024) — Transformer-based embedding 训练于
    eCFR 法规文本；此处用关键词 TF 近似（与 run_claude_skills.py 一致），
    生产环境可替换为 Chroma / FAISS embedding 检索。
    """
    _load_rag_index()
    if not _RAG_INDEX:
        return []

    # 过滤国家
    pool = (
        [c for c in _RAG_INDEX if c.get("country") == country]
        if country else _RAG_INDEX
    )
    if not pool:
        pool = _RAG_INDEX   # 无匹配国家时全库检索

    kw_lower = [k.lower() for k in keywords]

    def score(chunk: dict) -> int:
        text = chunk.get("text", "").lower()
        return sum(1 for kw in kw_lower if kw in text)

    ranked = sorted(pool, key=score, reverse=True)
    top = [item for item in ranked[:top_n] if score(item) > 0]
    return [
        {
            "title": (
                item.get("source", "法规文本块")
                + (" | " + item["text"][:100].replace("\n", " "))
            ),
            "legal_status":   "Unknown – from local RAG index",
            "effective_date": "Not specified",
            "source":         item.get("source", "Local RAG index"),
            "confidence":     0.70,
            "retrieval_layer": 3,
            "text_snippet":   item.get("text", "")[:300],
        }
        for item in top
    ]


# ── 层 4：静态知识库兜底 ─────────────────────────────────────────
_STATIC_KB: list[dict] = [
    # ── 欧盟 ─────────────────────────────────────────────────────────
    {
        "keywords": ["titanium dioxide", "二氧化钛", "E171", "EU", "欧盟"],
        "legal_status": "Immediately Effective", "effective_date": "2022-08",
        "country": "欧盟",
        "detail": "EU Reg 2022/63: 禁止将二氧化钛(E171)用作食品添加剂（2022年8月起）",
    },
    {
        "keywords": ["lead", "铅", "2023/915", "EU", "欧盟", "contaminant"],
        "legal_status": "Immediately Effective", "effective_date": "2023-07",
        "country": "欧盟",
        "detail": "EU Reg 2023/915: 全面收严铅限量（畜禽肉0.1/水产0.3/谷物0.2 mg/kg）",
    },
    {
        "keywords": ["acrylamide", "丙烯酰胺", "benchmark", "EU", "欧盟"],
        "legal_status": "Immediately Effective", "effective_date": "2023-01",
        "country": "欧盟",
        "detail": "EU Reg 2023/194: 更新丙烯酰胺基准水平（薯片/饼干/咖啡等）",
    },
    {
        "keywords": ["cadmium", "镉", "cocoa", "chocolate", "可可", "巧克力", "EU"],
        "legal_status": "Notification of Update", "effective_date": "2025-01",
        "country": "欧盟",
        "detail": "EU 拟议进一步降低可可/巧克力镉限量（2025年起）",
    },
    {
        "keywords": ["sodium benzoate", "苯甲酸钠", "E211", "benzoic acid", "EU", "欧盟"],
        "legal_status": "Immediately Effective", "effective_date": "2011-01",
        "country": "欧盟",
        "detail": "EU Reg 1333/2008 Annex II: 软饮料中苯甲酸(E211)最大使用量 150 mg/kg",
    },
    {
        "keywords": ["allergen", "过敏原", "labelling", "标签", "1169/2011", "EU"],
        "legal_status": "Immediately Effective", "effective_date": "2014-12",
        "country": "欧盟",
        "detail": "EU Reg 1169/2011: 强制标注14种过敏原（花生/坚果/麸质/乳/蛋等）",
    },
    {
        "keywords": ["health claim", "nutrition claim", "健康声称", "1924/2006", "EU"],
        "legal_status": "Immediately Effective", "effective_date": "2007-07",
        "country": "欧盟",
        "detail": "EU Reg 1924/2006: 营养及健康声称须经EFSA批准并列于正面清单",
    },
    {
        "keywords": ["pesticide", "农药", "MRL", "396/2005", "EU", "欧盟"],
        "legal_status": "Immediately Effective", "effective_date": "2005-02",
        "country": "欧盟",
        "detail": "EU Reg 396/2005: 统一农药最大残留限量，默认0.01 mg/kg",
    },
    # ── 中国 ─────────────────────────────────────────────────────────
    {
        "keywords": ["GB 2762", "lead", "铅", "cadmium", "镉", "China", "中国", "contaminant"],
        "legal_status": "Immediately Effective", "effective_date": "2023-06",
        "country": "中国",
        "detail": "GB 2762-2022: 食品污染物（铅/镉/汞/砷）国家标准，2023年6月实施",
    },
    {
        "keywords": ["GB 2763", "pesticide", "农药", "MRL", "China", "中国"],
        "legal_status": "Immediately Effective", "effective_date": "2021-09",
        "country": "中国",
        "detail": "GB 2763-2021: 食品中农药最大残留限量国家标准，2021年9月实施",
    },
    {
        "keywords": ["GACC", "registration", "注册", "overseas", "境外", "China", "中国"],
        "legal_status": "Immediately Effective", "effective_date": "2022-01",
        "country": "中国",
        "detail": "GACC 248/249令: 境外食品生产企业须向海关总署注册，2022年1月实施",
    },
    {
        "keywords": ["GB 2760", "additive", "添加剂", "China", "中国", "防腐剂"],
        "legal_status": "Immediately Effective", "effective_date": "2015-05",
        "country": "中国",
        "detail": "GB 2760-2014: 食品添加剂使用标准，规定各类添加剂允许范围和最大使用量",
    },
    {
        "keywords": ["GB 7718", "label", "标签", "预包装", "China", "中国"],
        "legal_status": "Immediately Effective", "effective_date": "2012-04",
        "country": "中国",
        "detail": "GB 7718-2011: 预包装食品标签通则，须中文标注名称/配料/保质期等",
    },
    {
        "keywords": ["苯甲酸钠", "sodium benzoate", "benzoate", "防腐剂", "China", "中国"],
        "legal_status": "Immediately Effective", "effective_date": "2015-05",
        "country": "中国",
        "detail": "GB 2760-2014: 碳酸饮料中苯甲酸钠最大使用量 0.2 g/kg",
    },
    # ── 日本 ─────────────────────────────────────────────────────────
    {
        "keywords": ["glyphosate", "草甘膦", "oat", "燕麦", "Japan", "日本"],
        "legal_status": "Immediately Effective", "effective_date": "2018-12",
        "country": "日本",
        "detail": "日本将燕麦草甘膦MRL从0.2提高至30 mg/kg（肯定リスト制度 2018修订）",
    },
    {
        "keywords": ["positive list", "肯定リスト", "0.01", "pesticide", "Japan", "日本"],
        "legal_status": "Immediately Effective", "effective_date": "2006-05",
        "country": "日本",
        "detail": "日本肯定リスト制度: 未制定MRL的农药统一适用一律基准 0.01 mg/kg",
    },
    {
        "keywords": ["allergen", "过敏原", "labelling", "标签", "Japan", "日本"],
        "legal_status": "Immediately Effective", "effective_date": "2015-04",
        "country": "日本",
        "detail": "食品表示法(2015): 强制标注7大过敏原，建议标注20种",
    },
    # ── 美国 ─────────────────────────────────────────────────────────
    {
        "keywords": ["FSMA", "food safety", "modernization", "USA", "美国"],
        "legal_status": "Immediately Effective", "effective_date": "2017-09",
        "country": "美国",
        "detail": "FDA FSMA预防控制规则全面生效，适用进口食品设施注册与风险预防",
    },
    {
        "keywords": ["PFAS", "polyfluoroalkyl", "FDA", "USA", "美国", "food contact"],
        "legal_status": "Proposed", "effective_date": "TBD",
        "country": "美国",
        "detail": "FDA 拟议PFAS行动水平，用于食品接触材料，最终规则待定",
    },
    {
        "keywords": ["pesticide", "农药", "tolerance", "EPA CFR 40", "USA", "美国"],
        "legal_status": "Immediately Effective", "effective_date": "N/A",
        "country": "美国",
        "detail": "EPA CFR 40: 美国农药残留容忍量规定，食品类别农药通常 ≤0.1 mg/kg",
    },
    # ── 韩国 ─────────────────────────────────────────────────────────
    {
        "keywords": ["MFDS", "import", "declaration", "申报", "Korea", "韩国"],
        "legal_status": "Immediately Effective", "effective_date": "2020-01",
        "country": "韩国",
        "detail": "韩国MFDS修订进口食品申报及检验规定，2020年1月实施",
    },
    {
        "keywords": ["KFDA", "식품공전", "MRL", "pesticide", "Korea", "韩国"],
        "legal_status": "Immediately Effective", "effective_date": "N/A",
        "country": "韩国",
        "detail": "KFDA 식품공전: 韩国食品标准及规格，含农药MRL和重金属限量",
    },
    # ── 澳大利亚 ─────────────────────────────────────────────────────
    {
        "keywords": ["FSANZ", "Australia", "澳大利亚", "food standards", "import"],
        "legal_status": "Immediately Effective", "effective_date": "2002-01",
        "country": "澳大利亚",
        "detail": "FSANZ食品标准法典: 规定进口食品的限量/标签/添加剂/进口许可要求",
    },
    {
        "keywords": ["Australia", "澳大利亚", "biosecurity", "检验检疫", "DAWE"],
        "legal_status": "Immediately Effective", "effective_date": "N/A",
        "country": "澳大利亚",
        "detail": "澳大利亚出口食品须符合目标国法规；进口食品须经DAWE生物安全检查",
    },
]

def _static_search(keywords: list[str], country: str) -> Optional[dict]:
    kw_lower = [k.lower() for k in keywords]
    best, best_score = None, 0
    for entry in _STATIC_KB:
        if country and country not in entry.get("country", ""):
            continue
        s = sum(1 for ek in entry["keywords"]
                if any(ek.lower() in kw or kw in ek.lower() for kw in kw_lower))
        if s > best_score:
            best, best_score = entry, s
    if best and best_score > 0:
        return {
            "title": best["detail"],
            "legal_status": best["legal_status"],
            "effective_date": best["effective_date"],
            "source": "Static fallback KB",
            "retrieval_layer": 4,
        }
    return None


# ── 置信度评分 ────────────────────────────────────────────────────
def _compute_confidence(results: list[dict], layer: int) -> float:
    """
    基于检索层次和结果数量计算综合置信度。
    层次越低（官方 API），置信度越高。

    参考：Meng et al. (2025) 动态阈值策略：合规查询阈值 0.85–0.90。
    """
    base = {1: 0.95, 2: 0.85, 3: 0.75, 4: 0.60}.get(layer, 0.50)
    count_bonus = min(0.05, len(results) * 0.01)
    return round(min(base + count_bonus, 1.0), 3)


# ── 公开接口 ──────────────────────────────────────────────────────
def search_regulations(keywords: str, country: str = "") -> dict:
    """
    四层降级法规检索主函数。

    参数：
      keywords  str  — 法规关键词，空格或逗号分隔
      country   str  — 目标国（可选），支持中英文

    返回：
      {
        "found": bool,
        "results": [...],
        "retrieval_layer": int,        # 实际命中的层次（1-4）
        "confidence": float,           # 综合置信度
        "below_threshold": bool,       # True 时建议触发置信度门控
        "source": str,
      }
    """
    from config.settings import CONFIDENCE_THRESHOLDS
    kw_list  = [k.strip() for k in re.split(r"[,\s]+", keywords) if k.strip()]
    std_country = SUPPORTED_COUNTRIES.get(country.lower().strip(), country)

    # 层 1：官方 API（fetcher=None 表示该国无配置，跳过）
    fetcher = _COUNTRY_FETCHERS.get(std_country)
    if fetcher is None:
        fetcher = _COUNTRY_FETCHERS.get(country.lower().strip())
    if fetcher:
        results = fetcher(keywords)
        if results:
            conf = _compute_confidence(results, 1)
            return {
                "found": True, "results": results,
                "retrieval_layer": 1, "confidence": conf,
                "below_threshold": conf < CONFIDENCE_THRESHOLDS["compliance"],
                "source": results[0].get("source", "Official API"),
            }

    # 层 2：本地 KG
    results = _kg_search(kw_list, std_country)
    if results:
        conf = _compute_confidence(results, 2)
        return {
            "found": True, "results": results,
            "retrieval_layer": 2, "confidence": conf,
            "below_threshold": conf < CONFIDENCE_THRESHOLDS["compliance"],
            "source": "Local Knowledge Graph",
        }

    # 层 3：向量 RAG
    results = _vector_rag_search(kw_list, std_country)
    if results:
        conf = _compute_confidence(results, 3)
        return {
            "found": True, "results": results,
            "retrieval_layer": 3, "confidence": conf,
            "below_threshold": conf < CONFIDENCE_THRESHOLDS["compliance"],
            "source": "Vector RAG index",
        }

    # 层 4：静态 KB
    result = _static_search(kw_list, std_country)
    if result:
        conf = _compute_confidence([result], 4)
        return {
            "found": True, "results": [result],
            "retrieval_layer": 4, "confidence": conf,
            "below_threshold": True,   # 静态 KB 始终触发门控
            "source": "Static fallback KB",
        }

    return {
        "found": False, "results": [],
        "retrieval_layer": 0, "confidence": 0.0,
        "below_threshold": True,
        "source": "none",
        "message": (
            f'No regulation found for "{keywords}" / "{country}". '
            "Suggested manual sources: "
            "USA→federalregister.gov | EU→eur-lex.europa.eu | "
            "CN→samr.gov.cn | JP→mhlw.go.jp | KR→mfds.go.kr"
        ),
    }


# ── KG 增量更新接口 ───────────────────────────────────────────────
def append_triples(new_triples: list[dict]) -> int:
    """
    向本地 KG 追加新三元组，去重后持久化。
    返回实际新增数量。

    引用：Agarwal et al. (2024) 的 KG 清洗、去重、增量更新 pipeline。
    """
    global _KG
    existing = {(t["subject"], t["predicate"], t["object"]) for t in _KG}
    added = 0
    for t in new_triples:
        key = (t.get("subject", ""), t.get("predicate", ""), t.get("object", ""))
        if key not in existing:
            _KG.append(t)
            existing.add(key)
            added += 1
    KG_PATH.parent.mkdir(parents=True, exist_ok=True)
    KG_PATH.write_text(json.dumps(_KG, ensure_ascii=False, indent=2), encoding="utf-8")
    return added
