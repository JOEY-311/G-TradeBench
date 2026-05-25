#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_skill_openrouter.py
用 OpenRouter Tool Calling 方式完整评测 compliance_skill
三个任务均抽取 10% 样本（固定随机种子）

工具暴露：
  run_compliance_review   — 调用完整 5 步流水线（合规审查任务）
  search_regulations      — 4 层降级法规检索（规则更新时序推理任务）
  align_term              — 3 级术语对齐（标准对齐任务辅助）

运行：
  python eval_skill_openrouter.py
  python eval_skill_openrouter.py --task compliance   # 只跑合规审查
  python eval_skill_openrouter.py --task temporal     # 只跑规则更新
  python eval_skill_openrouter.py --task alignment    # 只跑标准对齐
  python eval_skill_openrouter.py --model google/gemini-2.0-flash-001
  python eval_skill_openrouter.py --resume            # 断点续跑
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ══════════════════════════════════════════════════════════════════════
#  路径 & 全局配置
# ══════════════════════════════════════════════════════════════════════
EVAL_BASE  = Path(__file__).parent
SKILL_ROOT = Path(r"E:\论文\跨境对齐\贡献\compliance_skill")

API_KEY       = os.environ.get('OPENROUTER_API_KEY', '')
DEFAULT_MODEL = "google/gemini-3.1-pro-preview"
JUDGE_MODEL   = "anthropic/claude-haiku-4-5"

SAMPLE_PCT    = 0.10    # 保留备用
SAMPLE_N      = 10      # 每个任务固定总条数
SEED          = 99
MAX_TURNS     = 5       # 单请求最大 tool-call 轮次
INTER_DELAY   = 1.5     # 请求间隔（秒）

OUT_DIR = EVAL_BASE / "results" / "skill_openrouter_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── skill 模块注入 ───────────────────────────────────────────────────
os.environ["OPENROUTER_API_KEY"] = API_KEY
os.environ.setdefault("COMPLIANCE_MODEL", DEFAULT_MODEL)
sys.path.insert(0, str(SKILL_ROOT))

# ══════════════════════════════════════════════════════════════════════
#  OpenRouter 客户端
# ══════════════════════════════════════════════════════════════════════
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# ══════════════════════════════════════════════════════════════════════
#  工具定义（Tool Schema）
# ══════════════════════════════════════════════════════════════════════
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_compliance_review",
            "description": (
                "对食品成分/产品运行完整的 5 步跨境合规审查流水线："
                "① 术语对齐（外文→国标名）"
                "② 法规检索（官方API→KG→RAG→静态KB 四层降级）"
                "③ 置信度门控"
                "④ 结构化 CoT 审查（STEP1-6，引用具体法规编号）"
                "⑤ 多智能体辩论裁决（ReviewAgent→DebateAgent→ArbiterAgent）\n"
                "返回：最终裁决（违规/存在违规风险/建议放行/需人工复核）及完整推理链。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredient": {
                        "type": "string",
                        "description": "待审食品成分或产品名称（可为外文/商品名，如 'Natriumbenzoat'、'Apple, royal gala'）",
                    },
                    "question": {
                        "type": "string",
                        "description": "完整合规问题，包含产地、目标国、关键成分信息",
                    },
                    "destination_country": {
                        "type": "string",
                        "description": "目标进口国，如 '中国'、'日本'、'欧盟'、'美国'",
                    },
                },
                "required": ["ingredient", "question", "destination_country"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_regulations",
            "description": (
                "查询各国食品法规/标准的最新状态，支持四层检索降级：\n"
                "Layer 1 — 各国官方 API 实时检索（Federal Register / EUR-Lex / CFSA / MHLW / MFDS）\n"
                "Layer 2 — 本地知识图谱（SPO 三元组，涵盖 GB 2762/2763、EU Reg 2023/915 等）\n"
                "Layer 3 — 法规文本 RAG 索引（6,389 块法规原文）\n"
                "Layer 4 — 静态知识库（高频法规兜底）\n"
                "适用场景：法规生效状态查询、时序推理（草案/已生效/拟议通知）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "法规关键词，如 'glyphosate oat Japan' 或 'GB 2762 铅 中国'",
                    },
                    "country": {
                        "type": "string",
                        "description": "目标国（可选），如 '日本'、'欧盟'、'美国'",
                    },
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "align_term",
            "description": (
                "将外文/缩写食品术语规范化为中国国标（GB）名称，三级降级：\n"
                "① 精确字典匹配（含英/德/法/日/中文及 EU E 编号）\n"
                "② Embedding 语义相似度匹配\n"
                "③ LLM 零样本推断（兜底）\n"
                "适用场景：HS 编码归类、标准限量对齐、配料准入查询中的术语规范化。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "待规范化的术语，如 'Natriumbenzoat'、'E171'、'titanium dioxide'",
                    },
                },
                "required": ["term"],
            },
        },
    },
]

# ══════════════════════════════════════════════════════════════════════
#  工具派发（调用 skill 内部函数）
# ══════════════════════════════════════════════════════════════════════
def dispatch_tool(name: str, args: dict) -> dict:
    """将 LLM 的 tool_call 路由到对应 skill 函数。"""
    try:
        if name == "run_compliance_review":
            from main_workflow import process_case
            case = process_case(
                foreign_ingredient=args["ingredient"],
                question=args["question"],
                destination_country=args.get("destination_country", ""),
                verbose=False,
            )
            return {
                "final_verdict": case.final_verdict,
                "flagged":       case.flagged_for_human,
                "audit_summary": (case.audit_trail or "")[:500],
            }

        elif name == "search_regulations":
            from skills.regulation_retriever import search_regulations
            result = search_regulations(
                keywords=args["keywords"],
                country=args.get("country", ""),
            )
            # 截断 results 避免 token 溢出
            result = dict(result)
            if "results" in result:
                result["results"] = result["results"][:5]
            return result

        elif name == "align_term":
            from skills.term_aligner import align_term
            return align_term(args["term"])

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════
#  多轮 Tool-Calling 循环
# ══════════════════════════════════════════════════════════════════════
def run_with_tools(system_prompt: str, user_prompt: str, model: str) -> dict:
    """
    完整的 OpenRouter tool-calling 对话循环。

    返回：{
        "answer":       str,   # 模型最终文本回答
        "tool_calls":   list,  # 所有 tool_call 记录 [{name, args, result}]
        "turns":        int,   # tool-call 轮次数
        "error":        str | None,
    }
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    tool_log = []

    for turn in range(MAX_TURNS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
                max_tokens=2000,
            )
        except Exception as e:
            return {"answer": "", "tool_calls": tool_log, "turns": turn, "error": str(e)}

        msg = resp.choices[0].message

        # ── 无 tool call → 最终回答 ──────────────────────────────
        if not msg.tool_calls:
            return {
                "answer":     (msg.content or "").strip(),
                "tool_calls": tool_log,
                "turns":      turn,
                "error":      None,
            }

        # ── 有 tool call → 执行并追加结果 ────────────────────────
        # 将 assistant 消息（含 tool_calls）追加到 messages
        messages.append({
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # 执行每个工具，追加 tool role 消息
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = dispatch_tool(tc.function.name, args)
            tool_log.append({
                "name":   tc.function.name,
                "args":   args,
                "result": result,
            })

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result, ensure_ascii=False),
            })

    # 超过 MAX_TURNS → 请求最终回答
    try:
        final = client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "user", "content": "请给出最终答案。"}],
            temperature=0,
            max_tokens=1000,
        )
        answer = (final.choices[0].message.content or "").strip()
    except Exception as e:
        answer = ""

    return {"answer": answer, "tool_calls": tool_log, "turns": MAX_TURNS, "error": None}


# ══════════════════════════════════════════════════════════════════════
#  任务一：合规审查
# ══════════════════════════════════════════════════════════════════════
COMPLIANCE_DATA_DIR = EVAL_BASE / "合规审查" / "deepseekv4"

COMPLIANCE_SYSTEM = """\
你是专业的跨境食品合规顾问。
使用 run_compliance_review 工具对食品进行完整的 5 步合规审查。
根据工具返回的裁决，给出简洁结论：
最终裁决应为以下三项之一（最后一行必须且只能是）：
  审查结论：违规
  审查结论：存在违规风险
  审查结论：建议放行
"""

def _extract_verdict(text: str) -> str:
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if "审查结论：" in line:
            return line.split("审查结论：", 1)[-1].strip()
    # 尝试从工具返回的 final_verdict 解析
    for kw in ["违规", "存在违规风险", "建议放行", "需人工复核"]:
        if kw in (text or ""):
            return kw
    return "无法判断"

def _normalize_gt_verdict(raw: str) -> str:
    raw = (raw or "").strip()
    for prefix in ["裁判结：", "裁判结:", "审查结论：", "审查结论:"]:
        if prefix in raw:
            raw = raw.split(prefix, 1)[-1].strip()
            break
    return raw.strip()

def _extract_ingredient(food_profile: str) -> str:
    patterns = [
        r'产品/食品名称[：:]\s*(.+)',
        r'产品/样本名称[：:]\s*(.+)',
        r'产品名称[：:]\s*(.+)',
        r'品名[：:]\s*(.+)',
    ]
    for pat in patterns:
        m = re.search(pat, food_profile, re.IGNORECASE)
        if m:
            name = m.group(1).strip().split("\n")[0].strip()
            if name:
                return name[:100]
    return food_profile.strip()[:80]

def _dest_from_filename(fname: str) -> str:
    DEST_MAP = {
        "CN": "中国", "JP": "日本", "KR": "韩国",
        "US": "美国", "USA": "美国", "FR": "法国",
        "GM": "德国", "AU": "澳大利亚",
    }
    m = re.search(r"_to_([A-Z]+)", fname, re.IGNORECASE)
    return DEST_MAP.get(m.group(1).upper(), m.group(1)) if m else ""

def load_compliance_data(total_n: int, seed: int) -> list[dict]:
    """从 9 个 CSV 文件中均匀抽取，合计恰好 total_n 条。"""
    all_rows = []
    for csv_path in sorted(COMPLIANCE_DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            continue
        if not {"FOOD_PROFILE", "REVIEW_OUTPUT"}.issubset(df.columns):
            continue
        df = df.dropna(subset=["FOOD_PROFILE", "REVIEW_OUTPUT"])
        df = df[df["REVIEW_OUTPUT"].str.strip().ne("")]
        dest = _dest_from_filename(csv_path.name)
        for _, r in df.iterrows():
            all_rows.append({
                "task": "compliance",
                "file": csv_path.name,
                "dest": dest,
                "food_profile": str(r["FOOD_PROFILE"]),
                "gt_verdict":   _normalize_gt_verdict(str(r["REVIEW_OUTPUT"])),
                "origin":       str(r.get("ORIGIN", "")),
                "id":           str(r.get("ID", f"{csv_path.stem}_{_}")),
            })
    random.seed(seed)
    return random.sample(all_rows, min(total_n, len(all_rows)))

def eval_compliance_row(row: dict, model: str) -> dict:
    ingredient = _extract_ingredient(row["food_profile"])
    question   = (
        f"以下食品从 {row['origin']} 出口到 {row['dest']}，请审查其合规性：\n\n"
        f"{row['food_profile']}"
    )
    result = run_with_tools(COMPLIANCE_SYSTEM, question, model)

    # 优先从工具调用结果中提取裁决
    skill_verdict = ""
    for tc in result["tool_calls"]:
        if tc["name"] == "run_compliance_review":
            fv = tc["result"].get("final_verdict", "")
            if fv:
                skill_verdict = _normalize_gt_verdict(fv)
                break
    if not skill_verdict:
        skill_verdict = _extract_verdict(result["answer"])

    correct = (skill_verdict == row["gt_verdict"])
    return {
        **row,
        "skill_verdict": skill_verdict,
        "correct":       correct,
        "model_answer":  result["answer"][:300],
        "tool_calls_n":  len(result["tool_calls"]),
        "turns":         result["turns"],
        "error":         result["error"],
    }


# ══════════════════════════════════════════════════════════════════════
#  任务二：规则更新（时序推理）
# ══════════════════════════════════════════════════════════════════════
TEMPORAL_DATA = EVAL_BASE / "规则更新" / "temporal_eval_en.jsonl"

TEMPORAL_SYSTEM = """\
你是食品法规专家。
使用 search_regulations 工具查询相关法规的最新状态，然后回答问题。
回答须包含：
  [Legal Status]: Immediately Effective / Proposed / Notification of Update
  [Effective Date]: 生效日期或拟议日期
  [Target Substance/Product]: 目标物质或产品
并在最后一行给出简短的中文结论。
"""

# LLM-as-Judge 评分
JUDGE_SYSTEM = """\
你是严格的评测员，对下列回答按 0–10 分评分。
评分标准：
  - 法规状态（Immediately Effective/Proposed/Notification of Update）是否正确：3 分
  - 目标物质/产品识别是否正确：2 分
  - 生效/提案日期是否接近正确：2 分
  - 逻辑清晰，有法规编号引用：3 分
仅输出一个整数，范围 0–10，不附加任何解释。
"""

def llm_judge(question: str, reference: str, prediction: str, model: str) -> int:
    prompt = (
        f"【问题】{question}\n\n"
        f"【参考答案】{reference[:400]}\n\n"
        f"【模型回答】{prediction[:400]}\n\n"
        "评分（0-10）："
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\d+", text)
        return min(10, max(0, int(m.group()))) if m else 0
    except Exception:
        return 0

def load_temporal_data(total_n: int, seed: int) -> list[dict]:
    random.seed(seed)
    with open(TEMPORAL_DATA, encoding="utf-8") as f:
        all_rows = [json.loads(l) for l in f if l.strip()]
    sampled = random.sample(all_rows, min(total_n, len(all_rows)))
    return [
        {
            "task":        "temporal",
            "id":          f"temporal_{i}",
            "instruction": r["instruction"],
            "gt_answer":   r["answer"],
        }
        for i, r in enumerate(sampled)
    ]

def eval_temporal_row(row: dict, model: str, judge_model: str) -> dict:
    result = run_with_tools(TEMPORAL_SYSTEM, row["instruction"], model)
    score  = llm_judge(row["instruction"], row["gt_answer"], result["answer"], judge_model)
    return {
        **row,
        "model_answer": result["answer"][:400],
        "score_10":     score,
        "tool_calls_n": len(result["tool_calls"]),
        "turns":        result["turns"],
        "error":        result["error"],
    }


# ══════════════════════════════════════════════════════════════════════
#  任务三：标准对齐
# ══════════════════════════════════════════════════════════════════════
ALIGNMENT_BASE = EVAL_BASE / "标准对齐"

# 使用已评测模型之一的输出作为问题来源（选 gpt-5.4 目录，文件格式最稳定）
ALIGNMENT_SOURCE_DIRS = ["gpt-5.5", "gpt-5.4", "deepseek-V3.2", "gemini-baseline"]

ALIGNMENT_SYSTEM = """\
你是跨境食品贸易专家，精通 HS 编码归类、各国食品标准限量、配料准入、标签法规。
可以使用以下工具辅助回答：
  - align_term：将外文成分/添加剂规范化为中国国标名称
  - search_regulations：查询各国相关法规限量/状态
请给出专业、有据可查的回答。
"""

ALIGNMENT_JUDGE_SYSTEM = """\
你是严格的评测员，对下列回答按 0–10 分评分。严格执行以下规则，不得因答案"听起来合理"而放宽标准。

【评分细则】

1. 精确数值/编码（3分）
   - HS编码：必须精确到8位数字，仅给出4位章节号得1分，未给出得0分
   - 限量值：必须给出精确数字+单位（如"0.5 g/kg"），只说"符合限量"或"在允许范围内"得0分
   - 合规结论：必须依托具体检测值与标准值对比，笼统下结论得0分

2. 法规引用精确性（4分）
   - 引用标准必须含编号+年份（如"GB 2760-2014"），只说"GB标准"或"国家标准"得1分
   - 同时引用具体条款/附录（如"附录A 表A.2"，"第X条"）再得1分
   - 共可得2分上限；如引用了不存在的标准编号或条款，该项得0分并总分扣1分
   - 参考答案涉及的每个关键法规若均正确引用，额外得2分

3. 关键问题覆盖（2分）
   - 对照参考答案，每遗漏一个核心要点（如陷阱识别、补救措施、分类依据）扣1分
   - 最低得0分

4. 准确性惩罚（-1分）
   - 若出现明显错误事实（如错误的限量数值、错误的标准归属），总分扣1分

仅输出一个整数（0–10），不附加任何解释。
"""

def llm_judge_alignment(question: str, reference: str, standard: str,
                         prediction: str, model: str) -> int:
    prompt = (
        f"【问题】{question[:300]}\n\n"
        f"【参考答案摘要】{reference[:300]}\n\n"
        f"【评分标准】{standard[:300]}\n\n"
        f"【模型回答】{prediction[:400]}\n\n"
        "评分（0-10）："
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ALIGNMENT_JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\d+", text)
        return min(10, max(0, int(m.group()))) if m else 0
    except Exception:
        return 0

def _load_alignment_xlsx(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(path, engine="openpyxl")
        if {"QUESTION", "REFERENCE", "STANDARD"}.issubset(df.columns):
            return df
    except Exception:
        pass
    return None

def _task_from_filename(fname: str) -> str:
    """从文件名推断子任务类型（HSCODE / 限量对齐 / 冲突判断 / 配料准入 / 准入程序 / 标签对齐）"""
    fname_l = fname.lower()
    for kw in ["hscode", "hs_code", "hscode"]:
        if kw in fname_l:
            return "HSCODE"
    for kw in ["限量", "limit"]:
        if kw in fname_l:
            return "限量对齐"
    for kw in ["冲突", "conflict"]:
        if kw in fname_l:
            return "冲突判断"
    for kw in ["配料", "ingredient"]:
        if kw in fname_l:
            return "配料准入"
    for kw in ["程序", "procedure", "准入程序"]:
        if kw in fname_l:
            return "准入程序"
    for kw in ["标签", "label"]:
        if kw in fname_l:
            return "标签对齐"
    return "其他"

def load_alignment_data(total_n: int, seed: int) -> list[dict]:
    """从各子任务文件中各取 1 条，合计恰好 total_n 条（循环补齐）。"""
    all_rows = []
    seen_files = set()
    for src_dir_name in ALIGNMENT_SOURCE_DIRS:
        src_dir = ALIGNMENT_BASE / src_dir_name
        if not src_dir.exists():
            continue
        for xlsx_path in sorted(src_dir.glob("*.xlsx")):
            task_key = _task_from_filename(xlsx_path.name)
            if task_key in seen_files:
                continue
            df = _load_alignment_xlsx(xlsx_path)
            if df is None or len(df) == 0:
                continue
            seen_files.add(task_key)
            for i, r in df.iterrows():
                all_rows.append({
                    "task":      "alignment",
                    "subtask":   task_key,
                    "file":      xlsx_path.name,
                    "id":        f"align_{task_key}_{i}",
                    "question":  str(r.get("QUESTION", "")),
                    "reference": str(r.get("REFERENCE", "")),
                    "standard":  str(r.get("STANDARD", "")),
                })
    random.seed(seed)
    return random.sample(all_rows, min(total_n, len(all_rows)))

def eval_alignment_row(row: dict, model: str, judge_model: str) -> dict:
    result = run_with_tools(ALIGNMENT_SYSTEM, row["question"], model)
    score  = llm_judge_alignment(
        row["question"], row["reference"], row["standard"],
        result["answer"], judge_model,
    )
    return {
        **row,
        "model_answer": result["answer"][:400],
        "score_10":     score,
        "tool_calls_n": len(result["tool_calls"]),
        "turns":        result["turns"],
        "error":        result["error"],
    }


# ══════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════
def run_all(args):
    model       = args.model or DEFAULT_MODEL
    judge_model = JUDGE_MODEL
    os.environ["COMPLIANCE_MODEL"] = model

    out_path = OUT_DIR / f"results_{model.replace('/', '_')}.json"
    all_results: list[dict] = []
    done_ids: set[str] = set()

    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            all_results = json.load(f)
        done_ids = {r["id"] for r in all_results if not r.get("error")}
        print(f"Resume: {len(done_ids)} 条已完成")

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── 加载各任务数据 ─────────────────────────────────────────────
    tasks_to_run = []
    if args.task in (None, "compliance"):
        tasks_to_run += load_compliance_data(SAMPLE_N, SEED)
    if args.task in (None, "temporal"):
        tasks_to_run += load_temporal_data(SAMPLE_N, SEED)
    if args.task in (None, "alignment"):
        tasks_to_run += load_alignment_data(SAMPLE_N, SEED)

    print(f"\n模型：{model}")
    print(f"总样本：{len(tasks_to_run)} 条（每任务固定 {SAMPLE_N} 条）\n")

    # ── 逐条评测 ──────────────────────────────────────────────────
    by_task: dict[str, list] = defaultdict(list)

    for i, row in enumerate(tasks_to_run, 1):
        rid = row.get("id", f"row_{i}")
        if rid in done_ids:
            continue

        task = row["task"]
        print(f"[{i}/{len(tasks_to_run)}] task={task} id={rid}")

        try:
            if task == "compliance":
                result = eval_compliance_row(row, model)
                mark = "OK" if result["correct"] else "XX"
                info = f"GT={result['gt_verdict']} Skill={result['skill_verdict']}"
            elif task == "temporal":
                result = eval_temporal_row(row, model, judge_model)
                mark = f"{result['score_10']}/10"
                info = ""
            else:  # alignment
                result = eval_alignment_row(row, model, judge_model)
                mark = f"{result['score_10']}/10"
                info = f"subtask={result['subtask']}"

            print(f"  {mark}  tools_used={result['tool_calls_n']}  {info}")
            if result.get("error"):
                print(f"  ERROR: {result['error']}")

        except Exception as e:
            result = {**row, "error": str(e)}
            print(f"  EXCEPTION: {e}")

        all_results.append(result)
        by_task[task].append(result)

        if len(all_results) % 5 == 0:
            save()
        time.sleep(INTER_DELAY)

    save()

    # ── 汇总打印 ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  compliance_skill OpenRouter Tool-Calling 评测汇总")
    print(f"  模型：{model}")
    print("=" * 65)

    # 合规审查
    comp = [r for r in all_results if r.get("task") == "compliance" and not r.get("error")]
    if comp:
        acc = sum(r["correct"] for r in comp) / len(comp)
        print(f"\n合规审查准确率：{acc*100:.1f}%  (N={len(comp)})")
        by_dest: dict[str, list] = defaultdict(list)
        for r in comp:
            by_dest[r.get("dest", "?")].append(r["correct"])
        for dest, vals in sorted(by_dest.items()):
            print(f"  {dest:<10}: {sum(vals)/len(vals)*100:.1f}%  (n={len(vals)})")
        tool_used = sum(1 for r in comp if r.get("tool_calls_n", 0) > 0)
        print(f"  工具调用率：{tool_used}/{len(comp)}")

    # 规则更新
    temp = [r for r in all_results if r.get("task") == "temporal" and not r.get("error")]
    if temp:
        avg = sum(r["score_10"] for r in temp) / len(temp)
        print(f"\n规则更新平均分：{avg:.2f}/10  (N={len(temp)})")

    # 标准对齐
    align = [r for r in all_results if r.get("task") == "alignment" and not r.get("error")]
    if align:
        avg = sum(r["score_10"] for r in align) / len(align)
        print(f"\n标准对齐平均分：{avg:.2f}/10  (N={len(align)})")
        by_sub: dict[str, list] = defaultdict(list)
        for r in align:
            by_sub[r.get("subtask", "?")].append(r["score_10"])
        for sub, vals in sorted(by_sub.items()):
            print(f"  {sub:<12}: {sum(vals)/len(vals):.2f}/10  (n={len(vals)})")

    # 保存汇总
    summary = {
        "model": model,
        "sample_pct": SAMPLE_PCT,
        "compliance": {
            "n": len(comp),
            "accuracy": round(sum(r["correct"] for r in comp) / len(comp) * 100, 1) if comp else 0,
            "tool_call_rate": round(sum(1 for r in comp if r.get("tool_calls_n", 0) > 0) / len(comp) * 100, 1) if comp else 0,
        } if comp else {},
        "temporal": {
            "n": len(temp),
            "avg_score_10": round(sum(r["score_10"] for r in temp) / len(temp), 2) if temp else 0,
        } if temp else {},
        "alignment": {
            "n": len(align),
            "avg_score_10": round(sum(r["score_10"] for r in align) / len(align), 2) if align else 0,
            "by_subtask": {
                sub: round(sum(v) / len(v), 2)
                for sub, v in {
                    s: [r["score_10"] for r in align if r.get("subtask") == s]
                    for s in set(r.get("subtask", "?") for r in align)
                }.items()
            },
        } if align else {},
    }
    sum_path = OUT_DIR / f"summary_{model.replace('/', '_')}.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果：{out_path}")
    print(f"汇总结果：{sum_path}")


# ══════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compliance_skill OpenRouter Tool-Calling 评测")
    parser.add_argument("--task",  choices=["compliance", "temporal", "alignment"],
                        default=None, help="只跑指定任务（默认全部）")
    parser.add_argument("--model", type=str, default=None,
                        help=f"模型（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑")
    args = parser.parse_args()
    run_all(args)
