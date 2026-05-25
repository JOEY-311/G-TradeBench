"""
config/settings.py
全局配置 — 所有环境相关变量通过环境变量或 .env 注入，禁止硬编码。
"""
import os
from pathlib import Path

# ── API ──────────────────────────────────────────────────────────
# 通过环境变量注入，绝不在代码中写默认密钥
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY",
    os.environ.get('OPENROUTER_API_KEY', ''))
MODEL: str              = os.getenv("COMPLIANCE_MODEL", "anthropic/claude-sonnet-4-5")

# ── 知识库路径 ────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
KB_DIR        = BASE_DIR / "knowledge_base"
KG_PATH       = KB_DIR / "regulation_kg.json"        # 知识图谱（SPO 三元组）
TERM_MAP_PATH = KB_DIR / "term_mapping.json"          # 术语规范化词典
EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# ── 置信度阈值（参考 Meng et al. 2025 多层幻觉缓解框架）────────────
CONFIDENCE_THRESHOLDS = {
    "general":    0.75,   # 一般信息性查询
    "product":    0.80,   # 产品相关查询
    "compliance": 0.85,   # 合规 / 法规查询（高风险）
}

# ── 多智能体参数 ──────────────────────────────────────────────────
MAX_TOOL_TURNS  = 6    # 单轮最多工具调用次数，防死循环
DEBATE_ROUNDS   = 2    # 辩论 agent 最大轮次
VOTE_THRESHOLD  = 2    # 裁决所需最低票数（3 个 agent 中）
NET_TIMEOUT     = 12   # 官方 API 请求超时（秒）

# ── 实验 / 评测参数 ───────────────────────────────────────────────
SAMPLE_PCT = float(os.getenv("SAMPLE_PCT", "0.2"))
SEED       = int(os.getenv("SEED", "99"))

# ── 支持的目标国 ──────────────────────────────────────────────────
SUPPORTED_COUNTRIES = {
    # 归一化键 → 标准名
    "china": "中国", "cn": "中国", "中国": "中国",
    "japan": "日本", "jp": "日本", "日本": "日本",
    "korea": "韩国", "kr": "韩国", "south korea": "韩国", "韩国": "韩国",
    "usa": "美国",   "us": "美国", "united states": "美国", "美国": "美国",
    "eu": "欧盟",    "europe": "欧盟", "germany": "欧盟", "france": "欧盟",
    "德国": "欧盟",  "法国": "欧盟", "欧盟": "欧盟",
}
