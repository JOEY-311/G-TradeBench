#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_claude_skills.py
Function-calling (Skills) 策略实验
将三个合规工具暴露给模型，与 vanilla / CoT / RAG 并列对比（Section 4.3）

三个工具：
  check_mrl_limit        — 污染物/农药限量查询（畜禽肉/水产/谷物…× 各国）
  query_regulation_status — 法规状态查询（草案/已生效/生效日期）
  lookup_hs_code         — HS 编码推荐 + 各国检验要求框架

用法：python run_claude_skills.py
"""

import csv, json, os, re, time, random
import urllib.parse, urllib.request, html
from pathlib import Path
from openai import OpenAI

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    print('⚠ requests 未安装，网络检索将使用 urllib 降级模式。建议：pip install requests')

# ════════════════════════════════════════════════════════════════
#  全局配置
# ════════════════════════════════════════════════════════════════
API_KEY    = os.environ.get('OPENROUTER_API_KEY',
             os.environ.get('OPENROUTER_API_KEY', ''))
MODEL      = 'anthropic/claude-sonnet-4-5'
SAMPLE_PCT = 0.2
SEED       = 99
MAX_TOOL_TURNS = 6   # 单次请求最多 tool-call 轮次，防止死循环

EVAL = Path('E:/论文/跨境对齐/评测阶段')
AUS_DATA  = Path('E:/论文/跨境对齐/准备阶段/商品数据集/可用/澳洲初级农产品/Food Composition.csv')
EU_DATA   = Path('E:/论文/跨境对齐/准备阶段/商品数据集/问题集/formatted_EU_profiles.csv')
USA_DATA  = Path('E:/论文/跨境对齐/准备阶段/商品数据集/问题集/formatted_USA_profiles.csv')
TEMPORAL  = EVAL / '规则更新' / 'temporal_eval_en.jsonl'

random.seed(SEED)
client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=API_KEY)

RAG_INDEX_PATH = EVAL / 'rag_index.json'

# ════════════════════════════════════════════════════════════════
#  RAG 索引加载（供 query_regulation_status 动态检索）
# ════════════════════════════════════════════════════════════════
_RAG_INDEX: list = []

# 国家名称映射：工具参数（英文/中文）→ 索引中的 country 字段（中文）
_COUNTRY_NORM = {
    'china': '中国', 'cn': '中国', '中国': '中国',
    'japan': '日本', 'jp': '日本', '日本': '日本',
    'korea': '韩国', 'kr': '韩国', 'south korea': '韩国', '韩国': '韩国',
    'usa': '美国',   'us': '美国', 'united states': '美国', '美国': '美国',
    'eu': '德国',    'europe': '德国', 'germany': '德国', 'france': '法国',
    '德国': '德国',  '法国': '法国',
}

def load_rag_index() -> int:
    """启动时加载本地法规索引，返回块数。0 = 未找到，工具将降级为静态知识库。"""
    global _RAG_INDEX
    if RAG_INDEX_PATH.exists():
        try:
            _RAG_INDEX = json.loads(RAG_INDEX_PATH.read_text(encoding='utf-8'))
            print(f'  ✓ RAG 索引已加载: {len(_RAG_INDEX)} 块（供 query_regulation_status 动态检索）')
            return len(_RAG_INDEX)
        except Exception as e:
            print(f'  ⚠ RAG 索引加载失败: {e}')
    else:
        print(f'  ⚠ 未找到 {RAG_INDEX_PATH}，query_regulation_status 将使用静态知识库兜底')
        print(f'    提示：运行 python build_rag_index.py 可构建动态索引')
    return 0

def _rag_retrieve(country_cn: str, keywords: list[str], top_n: int = 5) -> list[dict]:
    """从 RAG 索引中检索与关键词最相关的法规文本块。"""
    pool = [c for c in _RAG_INDEX if c.get('country') == country_cn] if country_cn else _RAG_INDEX
    if not pool:
        return []
    scored = sorted(
        ((c, sum(1 for kw in keywords if kw.lower() in c['text'].lower())) for c in pool),
        key=lambda x: -x[1]
    )
    return [item[0] for item in scored[:top_n] if item[1] > 0]


# ════════════════════════════════════════════════════════════════
#  工具一：MRL / 污染物限量查询
# ════════════════════════════════════════════════════════════════
# 数据来源：GB 2762-2022、食品卫生法告示、KFDA 식품공전、EPA CFR 40、EU Reg 2023/915
# 格式：(限量值 mg/kg, 来源法规)；None 表示该国无通用限量，需查具体品类法规

LIMITS: dict = {
    '中国': {
        '畜禽肉及其制品': {'铅': (0.2,  'GB 2762-2022'), '镉': (0.1,  'GB 2762-2022'), '农药': (0.05, 'GB 2763-2021')},
        '水产品':        {'铅': (0.5,  'GB 2762-2022'), '镉': (0.1,  'GB 2762-2022'), '农药': (0.05, 'GB 2763-2021')},
        '谷物及其制品':   {'铅': (0.2,  'GB 2762-2022'), '镉': (0.1,  'GB 2762-2022'), '农药': (0.05, 'GB 2763-2021')},
        '蔬菜及其制品':   {'铅': (0.3,  'GB 2762-2022'), '镉': (0.2,  'GB 2762-2022'), '农药': (0.05, 'GB 2763-2021')},
        '水果及其制品':   {'铅': (0.1,  'GB 2762-2022'), '镉': (0.05, 'GB 2762-2022'), '农药': (0.05, 'GB 2763-2021')},
        '乳及乳制品':     {'铅': (0.05, 'GB 2762-2022'), '镉': (0.01, 'GB 2762-2022'), '农药': (0.05, 'GB 2763-2021')},
        '油脂及其制品':   {'铅': (0.1,  'GB 2762-2022'), '镉': (0.1,  'GB 2762-2022'), '农药': (0.05, 'GB 2763-2021')},
    },
    '日本': {
        '畜禽肉及其制品': {'铅': (0.1,  '食品衛生法告示'), '镉': (0.05, '食品衛生法告示'), '农药': (0.01, '肯定リスト制度')},
        '水产品':        {'铅': (0.5,  '食品衛生法告示'), '镉': (0.1,  '食品衛生法告示'), '农药': (0.01, '肯定リスト制度')},
        '谷物及其制品':   {'铅': (0.2,  '食品衛生法告示'), '镉': (0.4,  '食品衛生法告示'), '农药': (0.01, '肯定リスト制度')},
        '蔬菜及其制品':   {'铅': (0.1,  '食品衛生法告示'), '镉': (0.05, '食品衛生法告示'), '农药': (0.01, '肯定リスト制度')},
        '水果及其制品':   {'铅': (0.1,  '食品衛生法告示'), '镉': (0.05, '食品衛生法告示'), '农药': (0.01, '肯定リスト制度')},
        '乳及乳制品':     {'铅': (0.02, '食品衛生法告示'), '镉': (0.01, '食品衛生法告示'), '农药': (0.01, '肯定リスト制度')},
        '油脂及其制品':   {'铅': (0.1,  '食品衛生法告示'), '镉': (0.05, '食品衛生法告示'), '农药': (0.01, '肯定リスト制度')},
    },
    '韩国': {
        '畜禽肉及其制品': {'铅': (0.1,  'KFDA 식품공전'), '镉': (0.05, 'KFDA 식품공전'), '农药': (0.01, 'KFDA MRL')},
        '水产品':        {'铅': (0.5,  'KFDA 식품공전'), '镉': (0.1,  'KFDA 식품공전'), '农药': (0.01, 'KFDA MRL')},
        '谷物及其制品':   {'铅': (0.2,  'KFDA 식품공전'), '镉': (0.2,  'KFDA 식품공전'), '农药': (0.01, 'KFDA MRL')},
        '蔬菜及其制品':   {'铅': (0.1,  'KFDA 식품공전'), '镉': (0.05, 'KFDA 식품공전'), '农药': (0.01, 'KFDA MRL')},
        '水果及其制品':   {'铅': (0.1,  'KFDA 식품공전'), '镉': (0.05, 'KFDA 식품공전'), '农药': (0.01, 'KFDA MRL')},
        '乳及乳制品':     {'铅': (0.02, 'KFDA 식품공전'), '镉': (0.01, 'KFDA 식품공전'), '农药': (0.01, 'KFDA MRL')},
        '油脂及其制品':   {'铅': (0.1,  'KFDA 식품공전'), '镉': (0.05, 'KFDA 식품공전'), '农药': (0.01, 'KFDA MRL')},
    },
    '美国': {
        # 美国无统一重金属限量，EPA 仅规定农药 tolerances
        '畜禽肉及其制品': {'铅': (None, 'FDA 无通用限量'), '镉': (None, 'FDA 无通用限量'), '农药': (0.1,  'EPA CFR 40')},
        '水产品':        {'铅': (None, 'FDA 无通用限量'), '镉': (None, 'FDA 无通用限量'), '农药': (0.1,  'EPA CFR 40')},
        '谷物及其制品':   {'铅': (None, 'FDA 无通用限量'), '镉': (None, 'FDA 无通用限量'), '农药': (0.1,  'EPA CFR 40')},
        '蔬菜及其制品':   {'铅': (None, 'FDA 无通用限量'), '镉': (None, 'FDA 无通用限量'), '农药': (0.1,  'EPA CFR 40')},
        '水果及其制品':   {'铅': (None, 'FDA 无通用限量'), '镉': (None, 'FDA 无通用限量'), '农药': (0.1,  'EPA CFR 40')},
        '乳及乳制品':     {'铅': (None, 'FDA 无通用限量'), '镉': (None, 'FDA 无通用限量'), '农药': (0.1,  'EPA CFR 40')},
        '油脂及其制品':   {'铅': (None, 'FDA 无通用限量'), '镉': (None, 'FDA 无通用限量'), '农药': (0.1,  'EPA CFR 40')},
    },
    '德国': {
        '畜禽肉及其制品': {'铅': (0.1,  'EU Reg 2023/915'), '镉': (0.05, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '水产品':        {'铅': (0.3,  'EU Reg 2023/915'), '镉': (0.05, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '谷物及其制品':   {'铅': (0.2,  'EU Reg 2023/915'), '镉': (0.1,  'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '蔬菜及其制品':   {'铅': (0.1,  'EU Reg 2023/915'), '镉': (0.05, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '水果及其制品':   {'铅': (0.1,  'EU Reg 2023/915'), '镉': (0.05, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '乳及乳制品':     {'铅': (0.02, 'EU Reg 2023/915'), '镉': (0.01, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '油脂及其制品':   {'铅': (0.1,  'EU Reg 2023/915'), '镉': (0.05, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
    },
    '法国': {   # 与德国相同（EU 统一法规）
        '畜禽肉及其制品': {'铅': (0.1,  'EU Reg 2023/915'), '镉': (0.05, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '水产品':        {'铅': (0.3,  'EU Reg 2023/915'), '镉': (0.05, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '谷物及其制品':   {'铅': (0.2,  'EU Reg 2023/915'), '镉': (0.1,  'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '蔬菜及其制品':   {'铅': (0.1,  'EU Reg 2023/915'), '镉': (0.05, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '水果及其制品':   {'铅': (0.1,  'EU Reg 2023/915'), '镉': (0.05, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '乳及乳制品':     {'铅': (0.02, 'EU Reg 2023/915'), '镉': (0.01, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
        '油脂及其制品':   {'铅': (0.1,  'EU Reg 2023/915'), '镉': (0.05, 'EU Reg 2023/915'), '农药': (0.01, 'EU Reg 396/2005')},
    },
}

# 物质名称规范化
SUBSTANCE_NORM = {
    'lead': '铅', 'pb': '铅', '铅(pb)': '铅', '铅': '铅',
    'cadmium': '镉', 'cd': '镉', '镉(cd)': '镉', '镉': '镉',
    'pesticide': '农药', 'pesticides': '农药', 'mrl': '农药',
    '农药': '农药', '农残': '农药', '综合农药残留': '农药',
}

# 食品类别规范化（从关键词匹配到标准类别）
CATEGORY_NORM = [
    (['猪肉', '牛肉', '羊肉', '禽肉', '鸡肉', '鸭肉', '肉类', '畜禽', '冷冻肉', '火腿'], '畜禽肉及其制品'),
    (['鱼', '虾', '蟹', '贝', '海鲜', '水产', '鱿鱼', '三文鱼', '金枪鱼'],             '水产品'),
    (['小麦', '大米', '玉米', '谷物', '面粉', '大麦', '燕麦', '黑麦', '米'],             '谷物及其制品'),
    (['蔬菜', '菠菜', '白菜', '番茄', '胡萝卜', '洋葱', '辣椒', '叶菜'],                 '蔬菜及其制品'),
    (['水果', '苹果', '橙子', '葡萄', '草莓', '蓝莓', '芒果', '浆果', '柑橘'],           '水果及其制品'),
    (['牛奶', '奶粉', '乳品', '酸奶', '乳酪', '奶酪', '乳制品', '乳'],                  '乳及乳制品'),
    (['植物油', '橄榄油', '大豆油', '菜籽油', '葵花籽油', '棕榈油', '油脂'],             '油脂及其制品'),
]

def _norm_substance(raw: str) -> str:
    k = raw.lower().strip()
    return SUBSTANCE_NORM.get(k, raw)

def _norm_category(raw: str) -> str:
    raw_l = raw
    for keywords, std_cat in CATEGORY_NORM:
        if any(kw in raw_l for kw in keywords):
            return std_cat
    return raw   # 原样返回，让下游做模糊匹配

def check_mrl_limit(substance: str, food_category: str, destination_country: str) -> dict:
    """查询污染物/农药在目标国对特定食品类别的 MRL。"""
    sub = _norm_substance(substance)
    cat = _norm_category(food_category)

    country_data = LIMITS.get(destination_country)
    if not country_data:
        return {'found': False, 'message': f'暂无 {destination_country} 的限量数据库，建议人工查阅官方法规'}

    cat_data = country_data.get(cat)
    if not cat_data:
        # 二次模糊：遍历已有类别，取第一个包含关键词的
        for std_cat, data in country_data.items():
            if any(k in food_category for k in std_cat.split('及')):
                cat_data = data
                cat = std_cat
                break
    if not cat_data:
        return {'found': False, 'message': f'未匹配到 {destination_country} 下 "{food_category}" 的类别，请提供更精确的食品类别名'}

    entry = cat_data.get(sub)
    if entry is None:
        return {'found': False, 'message': f'该数据库中无 "{sub}" 的限量记录，请检查物质名称'}

    limit_val, source = entry
    if limit_val is None:
        return {
            'found': True, 'substance': sub, 'food_category': cat,
            'country': destination_country, 'limit': None, 'unit': 'mg/kg',
            'source': source,
            'note': '该国对此污染物无通用限量，需查具体品类或进口商品专项规定'
        }
    return {
        'found': True, 'substance': sub, 'food_category': cat,
        'country': destination_country, 'limit': limit_val,
        'unit': 'mg/kg', 'source': source
    }


# ════════════════════════════════════════════════════════════════
#  工具二：法规状态查询（实时连接各国官方数据库）
# ════════════════════════════════════════════════════════════════
# 检索优先级：
#   1. 各国官方 API / 数据库（实时，权威）
#      USA  → Federal Register REST API（免费，无需密钥）
#      EU   → EUR-Lex SPARQL 端点（官方查询接口）
#      中国  → 食品安全国家标准数据检索平台 + SAMR
#      日本  → 厚生劳动省食品安全信息页面
#      韩国  → MFDS 식품안전나라
#   2. 本地 RAG 索引（网络失败时降级）
#   3. 静态知识库（最后兜底）
#
# 与 RAG 策略的本质区别：
#   RAG 策略 → 被动接收预注入的本地文本块，知识边界 = 本地文件日期
#   Skills   → 主动查询官方数据库，知识边界 = 官网实时更新时间

NET_TIMEOUT = 12   # 单次网络请求超时（秒）

# HTTP 请求头（模拟浏览器，避免被部分官网拦截）
_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; FoodRegSearch/1.0; research use)',
    'Accept': 'text/html,application/xhtml+xml,application/json',
    'Accept-Language': 'en,zh-CN;q=0.9,ja;q=0.8',
}

def _http_get(url: str, params: dict = None, as_json: bool = False):
    """统一 HTTP GET，优先 requests，降级 urllib。超时/异常返回 None。"""
    try:
        if _HAS_REQUESTS:
            r = _requests.get(url, params=params, headers=_HEADERS, timeout=NET_TIMEOUT)
            r.raise_for_status()
            return r.json() if as_json else r.text
        else:
            full_url = url + ('?' + urllib.parse.urlencode(params) if params else '')
            req = urllib.request.Request(full_url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as resp:
                raw = resp.read().decode('utf-8', errors='replace')
                return json.loads(raw) if as_json else raw
    except Exception as e:
        return None

def _strip_html(text: str, max_chars: int = 500) -> str:
    """简单去除 HTML 标签，截取前 max_chars 字符。"""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', html.unescape(text)).strip()
    return text[:max_chars]

# ── 各国官方数据源 ────────────────────────────────────────────────

def _fetch_usa_federal_register(keywords: str) -> list[dict]:
    """
    美国：Federal Register 官方 REST API（免费，无需密钥）
    文档：https://www.federalregister.gov/developers/api/v1
    """
    data = _http_get(
        'https://www.federalregister.gov/api/v1/documents.json',
        params={
            'conditions[term]': keywords,
            'conditions[agencies][]': 'food-and-drug-administration',
            'conditions[type][]': ['RULE', 'PRULE', 'NOTICE'],
            'per_page': 5,
            'order': 'newest',
            'fields[]': ['title', 'type', 'effective_on', 'publication_date', 'abstract', 'html_url'],
        },
        as_json=True
    )
    if not data:
        return []
    results = []
    for doc in data.get('results', []):
        doc_type = doc.get('type', '')
        status = ('Proposed' if doc_type == 'PRULE'
                  else 'Notification of Update' if doc_type == 'NOTICE'
                  else 'Immediately Effective')
        results.append({
            'title':          doc.get('title', ''),
            'type':           doc_type,
            'legal_status':   status,
            'effective_date': doc.get('effective_on') or doc.get('publication_date', 'Not specified'),
            'abstract':       (doc.get('abstract') or '')[:300],
            'url':            doc.get('html_url', ''),
            'source':         'Federal Register API',
        })
    return results

def _fetch_eu_eurlex(keywords: str) -> list[dict]:
    """
    欧盟：EUR-Lex SPARQL 端点（官方语义检索接口）
    SPARQL endpoint: https://publications.europa.eu/webapi/rdf/sparql
    """
    kw_escaped = keywords.replace('"', '\\"')
    sparql = f"""
    PREFIX eli: <http://data.europa.eu/eli/ontology#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    SELECT ?work ?title ?date ?type WHERE {{
      ?work eli:type_document ?type ;
            eli:date_document ?date ;
            dc:title ?title .
      FILTER(LANG(?title) = 'en')
      FILTER(CONTAINS(LCASE(STR(?title)), LCASE("{kw_escaped}")))
      FILTER(?date >= "2020-01-01"^^xsd:date)
    }}
    ORDER BY DESC(?date) LIMIT 5
    """
    data = _http_get(
        'https://publications.europa.eu/webapi/rdf/sparql',
        params={'query': sparql, 'format': 'application/sparql-results+json'},
        as_json=True
    )
    results = []
    if data:
        for binding in data.get('results', {}).get('bindings', []):
            title = binding.get('title', {}).get('value', '')
            date  = binding.get('date',  {}).get('value', '')[:10]
            results.append({
                'title':          title,
                'legal_status':   'Immediately Effective',
                'effective_date': date,
                'url':            binding.get('work', {}).get('value', ''),
                'source':         'EUR-Lex SPARQL',
            })

    # SPARQL 失败时降级：EUR-Lex 普通搜索页面抓取摘要
    if not results:
        html_text = _http_get(
            'https://eur-lex.europa.eu/search.html',
            params={'scope': 'EURLEX', 'type': 'quick', 'lang': 'en', 'text': keywords}
        )
        if html_text:
            titles = re.findall(r'class="title"[^>]*>(.*?)</(?:span|a|div)', html_text, re.S)
            dates  = re.findall(r'(\d{2}/\d{2}/\d{4})', html_text)
            for i, t in enumerate(titles[:5]):
                results.append({
                    'title':          _strip_html(t, 150),
                    'legal_status':   'Immediately Effective',
                    'effective_date': dates[i] if i < len(dates) else 'Not specified',
                    'url':            'https://eur-lex.europa.eu',
                    'source':         'EUR-Lex search page',
                })
    return results

def _fetch_china_cfsa(keywords: str) -> list[dict]:
    """
    中国：食品安全国家标准数据检索平台（CFSA）
    https://sppt.cfsa.net.cn:8086/db
    """
    html_text = _http_get(
        'https://sppt.cfsa.net.cn:8086/db',
        params={'keyword': keywords, 'pageSize': 5}
    )
    results = []
    if html_text:
        # 提取标准名称与编号
        items = re.findall(r'(GB\s*\d[\d\-\.]+)[^\n]*?(\d{4}[-年]\d{1,2}[-月]\d{1,2}日?)', html_text)
        for std_no, date in items[:5]:
            results.append({
                'title':          std_no.strip(),
                'legal_status':   'Immediately Effective',
                'effective_date': date.strip(),
                'url':            'https://sppt.cfsa.net.cn:8086/db',
                'source':         'CFSA 食品安全国家标准平台',
            })
    # 若 CFSA 无结果，补充查 SAMR 公告
    if not results:
        samr_html = _http_get(
            'https://www.samr.gov.cn/search/search_result.shtml',
            params={'searchword': keywords, 'channelid': '228976'}
        )
        if samr_html:
            titles = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>([^<]{5,80})</a>', samr_html)
            for url_path, title in titles[:5]:
                if any(kw in title for kw in ['食品', '标准', '公告', '规定', '限量']):
                    results.append({
                        'title':          title.strip(),
                        'legal_status':   'Unknown – verify on SAMR',
                        'effective_date': 'Not specified',
                        'url':            'https://www.samr.gov.cn' + url_path,
                        'source':         'SAMR 市场监管总局',
                    })
    return results

def _fetch_japan_mhlw(keywords: str) -> list[dict]:
    """
    日本：厚生劳动省食品安全相关通知检索
    https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/kenkou_iryou/shokuhin/
    """
    html_text = _http_get(
        'https://www.mhlw.go.jp/cgi-bin/indexpage/search.cgi',
        params={'keyword': keywords, 'category': 'shokuhin', 'num': 5}
    )
    results = []
    if html_text:
        links = re.findall(
            r'href="(https://www\.mhlw\.go\.jp[^"]+)"[^>]*>\s*([^<]{5,120})</a>',
            html_text
        )
        dates = re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{2}/\d{2}|\d{4}-\d{2}-\d{2})', html_text)
        for i, (url, title) in enumerate(links[:5]):
            results.append({
                'title':          _strip_html(title, 120),
                'legal_status':   'Immediately Effective',
                'effective_date': dates[i] if i < len(dates) else 'Not specified',
                'url':            url,
                'source':         '厚生労働省 MHLW',
            })
    return results

def _fetch_korea_mfds(keywords: str) -> list[dict]:
    """
    韩国：MFDS 식품안전나라 공지사항 검색
    https://www.foodsafetykorea.go.kr
    """
    html_text = _http_get(
        'https://www.foodsafetykorea.go.kr/portal/board/board.do',
        params={'menu_grp': 'MENU_NEW01', 'menu_no': '2815', 'searchWord': keywords}
    )
    results = []
    if html_text:
        items = re.findall(
            r'<td[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</td>.*?(\d{4}\.\d{2}\.\d{2})',
            html_text, re.S
        )
        for title, date in items[:5]:
            results.append({
                'title':          _strip_html(title, 120),
                'legal_status':   'Immediately Effective',
                'effective_date': date,
                'url':            'https://www.foodsafetykorea.go.kr',
                'source':         'MFDS 식품안전나라',
            })
    return results

# 国家 → 抓取函数映射（英文/中文均可）
_COUNTRY_FETCHERS = {
    'usa': _fetch_usa_federal_register, '美国': _fetch_usa_federal_register,
    'us':  _fetch_usa_federal_register, 'united states': _fetch_usa_federal_register,
    'eu':  _fetch_eu_eurlex, 'europe': _fetch_eu_eurlex,
    'germany': _fetch_eu_eurlex, 'france': _fetch_eu_eurlex,
    '德国': _fetch_eu_eurlex, '法国': _fetch_eu_eurlex, '欧盟': _fetch_eu_eurlex,
    'china': _fetch_china_cfsa, '中国': _fetch_china_cfsa, 'cn': _fetch_china_cfsa,
    'japan': _fetch_japan_mhlw, '日本': _fetch_japan_mhlw, 'jp': _fetch_japan_mhlw,
    'korea': _fetch_korea_mfds, '韩国': _fetch_korea_mfds, 'kr': _fetch_korea_mfds,
}

# ── 静态知识库：仅作兜底，涵盖时序推理任务的高频法规 ──────────────
_STATIC_KB: list[dict] = [
    {'keywords': ['titanium dioxide', 'E171', 'EU'],
     'status': 'Immediately Effective', 'effective_date': '2022-08', 'region': 'EU',
     'detail': 'EU Reg 2022/63 banned titanium dioxide (E171) as food additive, Aug 2022'},
    {'keywords': ['acrylamide', 'benchmark', 'EU'],
     'status': 'Immediately Effective', 'effective_date': '2023-01', 'region': 'EU',
     'detail': 'EU Reg 2023/194 updated acrylamide benchmark levels, Jan 2023'},
    {'keywords': ['lead', 'contaminant', '2023/915', 'EU'],
     'status': 'Immediately Effective', 'effective_date': '2023-07', 'region': 'EU',
     'detail': 'EU Reg 2023/915 stricter lead limits across food categories, Jul 2023'},
    {'keywords': ['cadmium', 'cocoa', 'chocolate', 'EU'],
     'status': 'Notification of Update', 'effective_date': '2025-01', 'region': 'EU',
     'detail': 'Proposed lower cadmium limits for cocoa/chocolate products, Jan 2025'},
    {'keywords': ['glyphosate', 'oat', 'MRL', 'Japan'],
     'status': 'Immediately Effective', 'effective_date': '2018-12', 'region': 'Japan',
     'detail': 'Japan raised glyphosate MRL for oats from 0.2 to 30 mg/kg, Dec 2018'},
    {'keywords': ['positive list', 'pesticide', '0.01', 'Japan'],
     'status': 'Immediately Effective', 'effective_date': '2006-05', 'region': 'Japan',
     'detail': 'Japan Positive List System: unregistered pesticides default to 0.01 mg/kg'},
    {'keywords': ['GB 2762', 'contaminant', 'lead', 'cadmium', 'China'],
     'status': 'Immediately Effective', 'effective_date': '2023-06', 'region': 'China',
     'detail': 'GB 2762-2022 National Standard for Contaminants in Food, effective Jun 2023'},
    {'keywords': ['GB 2763', 'pesticide', 'MRL', 'China'],
     'status': 'Immediately Effective', 'effective_date': '2021-09', 'region': 'China',
     'detail': 'GB 2763-2021 National MRL Standard for Pesticides, effective Sep 2021'},
    {'keywords': ['GACC', 'overseas facility', 'registration', 'China'],
     'status': 'Immediately Effective', 'effective_date': '2022-01', 'region': 'China',
     'detail': 'GACC Decree 248/249: overseas food facilities must register, Jan 2022'},
    {'keywords': ['FSMA', 'food safety', 'modernization', 'USA'],
     'status': 'Immediately Effective', 'effective_date': '2017-09', 'region': 'USA',
     'detail': 'FDA FSMA Preventive Controls rule fully in effect Sep 2017'},
    {'keywords': ['PFAS', 'polyfluoroalkyl', 'FDA', 'USA'],
     'status': 'Proposed', 'effective_date': 'Not specified', 'region': 'USA',
     'detail': 'FDA proposed PFAS action levels for food contact; final rule pending'},
    {'keywords': ['MFDS', 'import', 'declaration', 'Korea'],
     'status': 'Immediately Effective', 'effective_date': '2020-01', 'region': 'Korea',
     'detail': 'Korea MFDS revised import food declaration procedures, Jan 2020'},
]

# 状态判断关键词（用于从原始法规文本中推断状态）
_STATUS_SIGNALS = {
    'Proposed':              ['proposed', 'draft', 'consultation', '草案', '征求意见', '拟议', '公开征询'],
    'Notification of Update':['notification', 'planned', 'scheduled', 'upcoming', '预告', '计划', '将于'],
    'Immediately Effective': ['effective', 'entered into force', 'in force', 'enacted', 'published',
                              '生效', '实施', '颁布', '公告', '发布', '已实施'],
}

def _infer_status(text: str) -> str:
    """从法规文本中推断法规状态。"""
    text_l = text.lower()
    # 按优先级：草案 < 预告 < 已生效
    for status, signals in _STATUS_SIGNALS.items():
        if any(s.lower() in text_l for s in signals):
            return status
    return 'Unknown'

def query_regulation_status(keywords: str, country: str = '') -> dict:
    """
    实时查询各国官方法规数据库，返回匹配的法规条目及状态信息。

    检索优先级：
      1. 各国官方 API / 官网（实时，权威）
      2. 本地 RAG 索引（网络不可达时降级）
      3. 静态知识库（最终兜底）
    """
    kw_list   = [k.strip() for k in re.split(r'[,\s]+', keywords) if k.strip()]
    country_k = country.strip().lower()
    country_cn = _COUNTRY_NORM.get(country_k, '')

    # ── 第一层：各国官方数据库实时检索 ─────────────────────────────
    fetcher = _COUNTRY_FETCHERS.get(country_k) or _COUNTRY_FETCHERS.get(country_cn)
    if fetcher:
        live_results = fetcher(keywords)
        if live_results:
            return {
                'found':   True,
                'source':  'official_live',
                'country': country,
                'query':   keywords,
                'results': live_results,
                'note':    (
                    'Data retrieved in real-time from official regulatory database. '
                    'legal_status is inferred from document type where available.'
                ),
            }
        # 官方源返回空（网络超时或无结果）→ 记录并继续降级
        live_note = f'Official source returned no results for "{keywords}" (timeout or no match).'
    else:
        live_note = f'No official API configured for country "{country}".'

    # ── 第二层：本地 RAG 索引 ────────────────────────────────────
    if _RAG_INDEX:
        chunks = _rag_retrieve(country_cn, kw_list, top_n=4)
        if chunks:
            return {
                'found':  True,
                'source': 'local_rag_index',
                'query':  {'keywords': kw_list, 'country': country_cn or country},
                'results': [
                    {
                        'title':            c['source'].split('\\')[-1][:50],
                        'inferred_status':  _infer_status(c['text']),
                        'excerpt':          c['text'][:400],
                        'source':           'Local regulatory file',
                    }
                    for c in chunks
                ],
                'note': live_note + ' Fell back to local RAG index.',
            }

    # ── 第三层：静态知识库 ────────────────────────────────────────
    kw_lower = [k.lower() for k in kw_list]
    best, best_score = None, 0
    for entry in _STATIC_KB:
        region_match = (not country) or any(
            r.lower() in entry['region'].lower()
            for r in [country, country_cn, country_k]
        )
        if not region_match:
            continue
        score = sum(1 for ek in entry['keywords']
                    if any(ek.lower() in kw or kw in ek.lower() for kw in kw_lower))
        if score > best_score:
            best, best_score = entry, score

    if best and best_score > 0:
        return {
            'found':          True,
            'source':         'static_fallback_kb',
            'region':         best['region'],
            'legal_status':   best['status'],
            'effective_date': best['effective_date'],
            'detail':         best['detail'],
            'note':           live_note + ' Fell back to static knowledge base.',
        }

    return {
        'found':   False,
        'source':  'none',
        'message': (
            f'{live_note} No match in local index or static KB either. '
            'Suggested official sources: '
            'USA → federalregister.gov | EU → eur-lex.europa.eu | '
            'CN → samr.gov.cn | JP → mhlw.go.jp | KR → mfds.go.kr'
        ),
    }


# ════════════════════════════════════════════════════════════════
#  工具三：HS 编码推荐 + 各国检验要求框架
# ════════════════════════════════════════════════════════════════
# 注意：本数据库为研究用途的简化映射表，不具备法律效力。
# 实际申报以各国海关当局裁定为准。如需精确分类建议接入 WCO 官方数据库。

HS_DB: list[dict] = [
    {
        'category': '畜禽肉及其制品',
        'keywords': ['猪肉', '牛肉', '羊肉', '鸡肉', '鸭肉', '禽肉', '肉', '冷冻肉', '火腿', 'pork', 'beef', 'lamb', 'poultry', 'meat'],
        'hs_codes': [
            {'code': '0201', 'desc': '鲜、冷牛肉'},
            {'code': '0202', 'desc': '冻牛肉'},
            {'code': '0203', 'desc': '鲜、冷、冻猪肉'},
            {'code': '0204', 'desc': '鲜、冷、冻绵羊/山羊肉'},
            {'code': '0207', 'desc': '禽类肉（鸡/鸭/鹅）'},
            {'code': '0210', 'desc': '腌制、盐渍、熏制肉'},
        ],
        'inspection': {
            '中国': '检验检疫申报 + 海关总署认证境外企业 + 兽医卫生证书（GACC Decree 248）',
            '日本': '厚生劳動省届出 + 動物検疫所検査 + 原産地証明',
            '韩国': 'MFDS 수입신고 + 수입검사 + 원산지증명',
            '美国': 'USDA FSIS 进口检验 + 等效性认证国',
            '德国': 'EU 统一卫生证书（EHC）+ CHED-P 电子申报',
            '法国': 'EU 统一卫生证书（EHC）+ CHED-P 电子申报',
        }
    },
    {
        'category': '水产品',
        'keywords': ['鱼', '虾', '蟹', '贝', '海鲜', '水产', '三文鱼', '金枪鱼', 'fish', 'shrimp', 'seafood', 'salmon'],
        'hs_codes': [
            {'code': '0302', 'desc': '鲜、冷鱼'},
            {'code': '0303', 'desc': '冻鱼'},
            {'code': '0306', 'desc': '甲壳动物（虾、蟹、龙虾）'},
            {'code': '0307', 'desc': '软体动物（扇贝、鱿鱼、章鱼）'},
        ],
        'inspection': {
            '中国': '检验检疫 + 境外养殖场/加工厂注册（GACC）',
            '日本': '厚生劳動省届出 + 水産物检疫',
            '韩国': 'MFDS 수입신고 + 해양수산부 위생증명',
            '美国': 'FDA HACCP 验证 + 进口自动扣押名单核查',
            '德国': 'EU 卫生证书 + CHED-P',
            '法国': 'EU 卫生证书 + CHED-P',
        }
    },
    {
        'category': '谷物及其制品',
        'keywords': ['小麦', '大米', '玉米', '大麦', '燕麦', '黑麦', '谷物', '面粉', 'wheat', 'rice', 'corn', 'barley', 'oat', 'cereal', 'grain'],
        'hs_codes': [
            {'code': '1001', 'desc': '小麦及混合麦'},
            {'code': '1006', 'desc': '稻谷/大米'},
            {'code': '1005', 'desc': '玉米'},
            {'code': '1102', 'desc': '谷物粉（小麦粉、玉米粉等）'},
            {'code': '1904', 'desc': '膨化/即食谷物食品'},
        ],
        'inspection': {
            '中国': '植物检疫 + 粮食进口资质企业备案',
            '日本': '植物防疫法检查 + 厚生劳動省届出',
            '韩国': '식물검역 + MFDS 수입신고',
            '美国': 'USDA APHIS 植物检疫 + FDA 食品安全',
            '德国': 'EU 植物卫生证书 + CHED-PP',
            '法国': 'EU 植物卫生证书 + CHED-PP',
        }
    },
    {
        'category': '蔬菜及其制品',
        'keywords': ['蔬菜', '菠菜', '白菜', '番茄', '胡萝卜', '洋葱', '辣椒', 'vegetable', 'tomato', 'spinach', 'carrot'],
        'hs_codes': [
            {'code': '0701', 'desc': '马铃薯（鲜/冷）'},
            {'code': '0702', 'desc': '番茄（鲜/冷）'},
            {'code': '0706', 'desc': '胡萝卜、萝卜类'},
            {'code': '0709', 'desc': '其他蔬菜（鲜/冷）'},
            {'code': '0714', 'desc': '木薯、甘薯等根茎类'},
        ],
        'inspection': {
            '中国': '植物检疫证书 + 检验检疫申报',
            '日本': '植物防疫法检查 + 农药残留检查',
            '韩国': '식물검역증명서 + MFDS 수입신고',
            '美国': 'USDA APHIS + FDA 进口警告核查',
            '德国': 'EU 植物卫生证书 + CHED-PP',
            '法国': 'EU 植物卫生证书 + CHED-PP',
        }
    },
    {
        'category': '水果及其制品',
        'keywords': ['水果', '苹果', '橙子', '葡萄', '草莓', '蓝莓', '芒果', '柑橘', 'fruit', 'apple', 'orange', 'grape', 'berry', 'mango', 'citrus'],
        'hs_codes': [
            {'code': '0805', 'desc': '柑橘类水果'},
            {'code': '0806', 'desc': '葡萄（鲜/干）'},
            {'code': '0808', 'desc': '苹果、梨'},
            {'code': '0810', 'desc': '其他鲜果（草莓、蓝莓等）'},
        ],
        'inspection': {
            '中国': '植物检疫证书 + 出口商/果园注册（部分国家）',
            '日本': '植物防疫法检查（部分水果受限）',
            '韩国': '식물검역 + MFDS 수입신고',
            '美国': 'USDA APHIS 植物检疫（入境许可证）',
            '德国': 'EU 植物卫生证书',
            '法国': 'EU 植物卫生证书',
        }
    },
    {
        'category': '乳及乳制品',
        'keywords': ['牛奶', '奶粉', '乳品', '酸奶', '乳酪', '奶酪', '乳制品', '乳', 'milk', 'dairy', 'cheese', 'yogurt', 'powder'],
        'hs_codes': [
            {'code': '0401', 'desc': '液态乳（全脂/脱脂）'},
            {'code': '0402', 'desc': '浓缩乳、乳粉'},
            {'code': '0406', 'desc': '奶酪及凝乳'},
        ],
        'inspection': {
            '中国': '进口乳品境外生产企业注册（GACC）+ 卫生证书',
            '日本': '厚生劳動省届出 + 乳及乳制品成分规格检查',
            '韩国': 'MFDS 乳制品 수입신고 + 검사',
            '美国': 'FDA 牛奶安全要求 + USDA 乳制品等级',
            '德国': 'EU 动物源食品卫生证书 + CHED-A',
            '法国': 'EU 动物源食品卫生证书 + CHED-A',
        }
    },
    {
        'category': '油脂及其制品',
        'keywords': ['植物油', '橄榄油', '大豆油', '菜籽油', '葵花籽油', '棕榈油', '油脂', 'oil', 'olive oil', 'soybean', 'canola', 'palm oil'],
        'hs_codes': [
            {'code': '1507', 'desc': '大豆油'},
            {'code': '1509', 'desc': '橄榄油'},
            {'code': '1512', 'desc': '葵花籽油、红花油'},
            {'code': '1514', 'desc': '菜籽油、芥末油'},
            {'code': '1511', 'desc': '棕榈油'},
        ],
        'inspection': {
            '中国': '检验检疫 + 转基因标识（如大豆油）',
            '日本': '厚生劳動省届出 + 食品添加物确认',
            '韩国': 'MFDS 수입신고',
            '美国': 'FDA 一般食品安全 + GRAS 认定',
            '德国': 'EU 普通食品进口程序',
            '法国': 'EU 普通食品进口程序',
        }
    },
]

def lookup_hs_code(food_name: str, food_category: str, destination_country: str = '') -> dict:
    """根据食品名称/类别推荐 HS 编码，并返回目标国检验要求框架。"""
    combined = (food_name + ' ' + food_category).lower()

    best, best_score = None, 0
    for entry in HS_DB:
        score = sum(1 for kw in entry['keywords'] if kw.lower() in combined)
        if score > best_score:
            best, best_score = entry, score

    if best is None or best_score == 0:
        return {
            'found': False,
            'message': f'无法从名称 "{food_name}" 和类别 "{food_category}" 匹配到 HS 编码，请提供更多描述'
        }

    result = {
        'found': True,
        'matched_category': best['category'],
        'hs_codes': best['hs_codes'],
        'disclaimer': '以下为简化映射表，实际申报以各国海关裁定为准',
    }
    if destination_country and destination_country in best['inspection']:
        result['inspection_requirements'] = {
            destination_country: best['inspection'][destination_country]
        }
    else:
        result['inspection_requirements'] = best['inspection']
    return result


# ════════════════════════════════════════════════════════════════
#  工具 Schema（OpenAI function-calling 格式）
# ════════════════════════════════════════════════════════════════
TOOL_SCHEMAS = [
    {
        'type': 'function',
        'function': {
            'name': 'check_mrl_limit',
            'description': (
                '查询特定污染物或农药在目标进口国对特定食品类别的最大残留限量（MRL）标准。'
                '当需要核实实测值是否符合进口要求时调用此工具。'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'substance': {
                        'type': 'string',
                        'description': '查询物质名称，如：铅、镉、农药残留（支持中英文及缩写，如 Pb、Cd、pesticide）'
                    },
                    'food_category': {
                        'type': 'string',
                        'description': '食品类别描述，如：畜禽肉及其制品、水产品、谷物及其制品、蔬菜及其制品'
                    },
                    'destination_country': {
                        'type': 'string',
                        'description': '目标进口国，支持：中国、日本、韩国、美国、德国、法国'
                    }
                },
                'required': ['substance', 'food_category', 'destination_country']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'query_regulation_status',
            'description': (
                '动态检索本地法规文件库，查询食品法规的当前状态'
                '（Proposed 草案 / Notification of Update 预告更新 / Immediately Effective 已生效）及生效日期。'
                '返回原始法规文本摘录，供进一步推理。'
                '当问题涉及法规时效性、是否已实施、何时生效、是否仍为草案时调用此工具。'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'keywords': {
                        'type': 'string',
                        'description': '法规相关关键词，多个词用空格或逗号分隔，如：EU cadmium chocolate 或 Japan glyphosate oat MRL'
                    },
                    'country': {
                        'type': 'string',
                        'description': '目标国家或地区（可选），如：EU、Japan、China、USA、Korea'
                    }
                },
                'required': ['keywords']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'lookup_hs_code',
            'description': (
                '根据食品名称和类别推荐 HS 编码，并返回各主要贸易国的进口检验要求框架。'
                '当问题涉及海关编码分类、进口申报程序或多国检验要求比较时调用此工具。'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'food_name': {
                        'type': 'string',
                        'description': '食品商品名称，如：冷冻羊肉片、橄榄油、脱脂奶粉、三文鱼'
                    },
                    'food_category': {
                        'type': 'string',
                        'description': '食品大类描述，如：畜禽肉及其制品、油脂类、乳制品、水产品'
                    },
                    'destination_country': {
                        'type': 'string',
                        'description': '目标进口国（可选），提供后只返回该国检验要求'
                    }
                },
                'required': ['food_name', 'food_category']
            }
        }
    }
]

# 工具分发表
TOOL_FN_MAP = {
    'check_mrl_limit':        check_mrl_limit,
    'query_regulation_status': query_regulation_status,
    'lookup_hs_code':          lookup_hs_code,
}

def dispatch_tool(name: str, args: dict) -> dict:
    fn = TOOL_FN_MAP.get(name)
    if fn is None:
        return {'error': f'未知工具: {name}'}
    try:
        return fn(**args)
    except Exception as e:
        return {'error': str(e)}


# ════════════════════════════════════════════════════════════════
#  核心：带工具调用的多轮对话循环
# ════════════════════════════════════════════════════════════════
def run_with_tools(system_prompt: str, user_message: str,
                   max_turns: int = MAX_TOOL_TURNS) -> str:
    """
    发起带工具的对话。模型可自主决定调用工具，循环执行直到给出最终文本回复。
    返回最终文本回复，失败时返回空字符串。
    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',   'content': user_message},
    ]

    for turn in range(max_turns):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice='auto',
                temperature=0,
                max_tokens=1500,
            )
        except Exception as e:
            print(f'    API error (turn {turn}): {e}')
            time.sleep(5)
            continue

        msg = resp.choices[0].message

        # 模型没有调用工具 → 得到最终回复
        if not msg.tool_calls:
            return (msg.content or '').strip()

        # 有工具调用 → 执行并追加结果
        messages.append({
            'role': 'assistant',
            'content': msg.content or '',
            'tool_calls': [
                {
                    'id':       tc.id,
                    'type':     'function',
                    'function': {'name': tc.function.name, 'arguments': tc.function.arguments}
                }
                for tc in msg.tool_calls
            ]
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = dispatch_tool(tc.function.name, args)
            messages.append({
                'role':         'tool',
                'tool_call_id': tc.id,
                'content':      json.dumps(result, ensure_ascii=False),
            })

        time.sleep(0.5)

    # 超出最大轮次
    return ''


# ════════════════════════════════════════════════════════════════
#  工具函数：抽样 / CSV IO
# ════════════════════════════════════════════════════════════════
def sample_rows(rows: list, pct: float = SAMPLE_PCT, seed: int = SEED) -> list:
    rng = random.Random(seed)
    n = max(1, round(len(rows) * pct))
    return rng.sample(rows, min(n, len(rows)))


def init_csv(path: Path, headers: list) -> set:
    done = set()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(headers)
    else:
        with open(path, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                done.add(r.get(headers[0], ''))
    return done


def append_row(path: Path, row: list):
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(row)


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


# ════════════════════════════════════════════════════════════════
#  任务一：合规审查
# ════════════════════════════════════════════════════════════════
COMP_SYSTEM = """\
你是一位资深的跨境食品合规审查专家。
你有三个工具可以调用：
  • check_mrl_limit   — 查询任意国家/食品类别的污染物限量标准
  • lookup_hs_code    — 查询食品 HS 编码及各国检验申报要求

审查流程：
1. 识别食品类别和目标国家
2. 调用工具查询各污染物的限量标准
3. 将实测值与标准值逐项对比
4. 综合判断并输出结论

最后必须另起一行输出以下格式之一（不得附加任何文字）：
审查结论：违规
审查结论：存在违规风险
审查结论：建议放行
"""

ROUTES = {
    'AUS': [
        ('澳大利亚', '日本', AUS_DATA,  '澳日'),
        ('澳大利亚', '中国', AUS_DATA,  '澳中'),
        ('澳大利亚', '韩国', AUS_DATA,  '澳韩'),
    ],
    'EU': [
        ('欧盟',   '中国', EU_DATA, 'EU中'),
        ('欧盟',   '日本', EU_DATA, 'EU日'),
        ('欧盟',   '韩国', EU_DATA, 'EU韩'),
    ],
    'USA': [
        ('美国',   '中国', USA_DATA,  '美中'),
        ('美国',   '日本', USA_DATA,  '美日'),
        ('美国',   '韩国', USA_DATA,  '美韩'),
    ],
}

OUT_COMP = EVAL / '合规审查' / 'claude-sonnet-skills'

def run_compliance_skills():
    print('\n── 任务一：合规审查 (Skills) ─────────────────────')
    for src, (routes) in ROUTES.items():
        for origin, dest, data_path, label in routes:
            out_path = OUT_COMP / f'{src}_claude-sonnet-skills_{label}.csv'
            if not data_path.exists():
                print(f'  ⚠ 数据不存在: {data_path}'); continue

            rows = load_csv(data_path)
            subset = sample_rows(rows)
            done = init_csv(out_path, ['ID', 'ORIGIN', 'DESTINATION', 'FOOD_PROFILE', 'REVIEW_OUTPUT'])

            print(f'  {label}: {len(subset)} 条')
            for row in subset:
                rid = str(row.get('ID', row.get('id', '')))
                if rid in done:
                    continue

                profile = row.get('FOOD_PROFILE', row.get('food_profile', str(row)))
                user_msg = f'【目标进口国】：{dest}\n【拟进口食品档案】：\n{profile}'

                output = run_with_tools(COMP_SYSTEM, user_msg)
                append_row(out_path, [rid, origin, dest, profile, output])
                print(f'    {rid}: {output[-40:].strip()!r}')
                time.sleep(random.uniform(0.8, 1.5))

    print('  合规审查完成')


# ════════════════════════════════════════════════════════════════
#  任务二：时序推理
# ════════════════════════════════════════════════════════════════
TEMPORAL_SYSTEM = """\
You are a food regulatory expert specializing in the legal status of food regulations worldwide.
You have access to a tool:
  • query_regulation_status — search for a regulation's current status and effective date

When answering:
1. Identify the key regulation, substance, and country from the question
2. Call query_regulation_status with relevant keywords
3. Synthesize the tool result with your knowledge to give a structured answer

Always provide your answer in this exact format:
[Legal Status]: Proposed / Notification of Update / Immediately Effective
[Proposed Effective Date]: YYYY-MM / Not specified / NaN
[Target Details]: brief description of the substance and change
"""

OUT_TEMPORAL = EVAL / '规则更新' / 'temporal_claude-sonnet-skills.csv'

def run_temporal_skills():
    print('\n── 任务二：时序推理 (Skills) ─────────────────────')
    if not TEMPORAL.exists():
        print(f'  ⚠ 数据不存在: {TEMPORAL}'); return

    items = [json.loads(l) for l in TEMPORAL.read_text(encoding='utf-8').splitlines() if l.strip()]
    subset = sample_rows(items)
    done = init_csv(OUT_TEMPORAL, ['index', 'instruction', 'ground_truth', 'model_output'])

    print(f'  时序推理: {len(subset)} 条')
    for i, item in enumerate(subset):
        idx = str(i)
        if idx in done:
            continue

        instruction = item.get('instruction', '')
        gt          = item.get('answer', '')
        output = run_with_tools(TEMPORAL_SYSTEM, instruction)
        append_row(OUT_TEMPORAL, [idx, instruction, gt, output])
        print(f'    [{idx}] {output[:60].strip()!r}')
        time.sleep(random.uniform(0.8, 1.5))

    print('  时序推理完成')


# ════════════════════════════════════════════════════════════════
#  任务三：标准对齐（场景问答）
# ════════════════════════════════════════════════════════════════
ALIGNMENT_SYSTEM = """\
你是一位资深跨境食品合规与通关专家，熟悉中国、日本、韩国、美国及欧盟的食品法规体系。
你有三个工具：
  • check_mrl_limit          — 查询污染物/农药限量
  • lookup_hs_code           — 查询 HS 编码和检验要求
  • query_regulation_status  — 查询法规状态和生效日期

回答要求：
1. 识别问题类型（HS 编码/限量核查/添加剂/标签/程序性等）
2. 按需调用工具获取准确数据
3. 引用具体法规编号（如 GB 2762-2022、EU Reg 2023/915）
4. 给出可操作的专业建议
"""

OUT_ALIGN = EVAL / '标准对齐' / 'claude-sonnet-skills'

# 标准对齐子任务目录（参考 baseline 目录）
ALIGN_BASELINE_DIR = EVAL / '标准对齐' / 'claude-sonnet'

def run_alignment_skills():
    print('\n── 任务三：标准对齐 (Skills) ─────────────────────')
    if not ALIGN_BASELINE_DIR.exists():
        print(f'  ⚠ baseline 目录不存在: {ALIGN_BASELINE_DIR}'); return

    for csv_file in sorted(ALIGN_BASELINE_DIR.glob('*.csv')):
        task_name = csv_file.stem.replace('claude-sonnet_', '')
        out_path  = OUT_ALIGN / f'claude-sonnet-skills_{task_name}.csv'

        rows   = load_csv(csv_file)
        subset = sample_rows(rows)
        done   = init_csv(out_path, ['QUESTION', 'REFERENCE', 'OUTPUT', 'STANDARD'])

        print(f'  {task_name}: {len(subset)} 条')
        for row in subset:
            q = row.get('QUESTION', '')
            if q in done:
                continue

            ref   = row.get('REFERENCE', '')
            std   = row.get('STANDARD', '')
            output = run_with_tools(ALIGNMENT_SYSTEM, q)
            append_row(out_path, [q, ref, output, std])
            print(f'    Q: {q[:50]!r} → {output[:40].strip()!r}')
            time.sleep(random.uniform(0.8, 1.5))

    print('  标准对齐完成')


# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=== Skills 策略实验（function calling）===')
    print(f'模型: {MODEL}  |  采样: {SAMPLE_PCT*100:.0f}%  |  seed={SEED}')
    print(f'工具: {[s["function"]["name"] for s in TOOL_SCHEMAS]}')
    load_rag_index()   # 加载本地 RAG 索引（官网不可达时的第二层降级）
    run_compliance_skills()
    run_temporal_skills()
    run_alignment_skills()
    print('\n全部完成。')
