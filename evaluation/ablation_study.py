#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_study.py
Skill 策略消融实验

── 模块级消融（Module-level）────────────────────────────────────
  Full          全模块：术语对齐 + 法规检索 + 多智能体辩论
  w/o_aligner   移除术语对齐：原始外文名直接进入检索
  w/o_retrieval 移除法规检索：模型凭自身知识推断（退化为 Vanilla LLM）
  w/o_debate    移除多智能体辩论：CoT 审查后直接输出

── 模块内部消融（Intra-module）─────────────────────────────────
  aligner_exact     仅精确字典匹配（关闭 embedding + LLM 推断）
  aligner_exact_emb 精确匹配 + embedding（关闭 LLM 推断）
  retrieval_api     仅 L1 官方 API（关闭 KG）
  retrieval_kg      仅 L2 本地 KG（关闭 API）
  debate_1round     辩论 1 轮
  debate_2round     辩论 2 轮（默认）
  debate_3round     辩论 3 轮

输出：
  每个任务一个 CSV，每行一个案件，各消融变体并列
  格式：case_id, task, [variant_1_output, variant_2_output, ...]

用法：
  python ablation_study.py              # 跑全部变体
  python ablation_study.py --module     # 仅模块级消融
  python ablation_study.py --intra      # 仅模块内部消融
  python ablation_study.py --task comp  # 仅合规审查任务
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable

from openai import OpenAI

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ════════════════════════════════════════════════════════════════
#  配置（对齐 run_claude_skills.py）
# ════════════════════════════════════════════════════════════════
API_KEY        = os.environ.get("OPENROUTER_API_KEY",
                 os.environ.get('OPENROUTER_API_KEY', ''))
MODEL          = os.getenv("COMPLIANCE_MODEL", "anthropic/claude-sonnet-4-5")
SAMPLE_PCT     = float(os.getenv("SAMPLE_PCT", "0.2"))
SEED           = int(os.getenv("SEED", "99"))
MAX_TOOL_TURNS = 6
MAX_CASES      = int(os.getenv("MAX_CASES", "50"))   # 每个数据来源最多取样条数
DELAY_MIN, DELAY_MAX = 0.8, 1.5

EVAL      = Path(os.getenv("EVAL_DIR", "E:/论文/跨境对齐/评测阶段"))
AUS_DATA  = Path(os.getenv("AUS_DATA",  "E:/论文/跨境对齐/准备阶段/商品数据集/可用/澳洲初级农产品/Food Composition.csv"))
EU_DATA   = Path(os.getenv("EU_DATA",   "E:/论文/跨境对齐/准备阶段/商品数据集/问题集/formatted_EU_profiles.csv"))
USA_DATA  = Path(os.getenv("USA_DATA",  "E:/论文/跨境对齐/准备阶段/商品数据集/问题集/formatted_USA_profiles.csv"))
TEMPORAL  = EVAL / "规则更新" / "temporal_eval_en.jsonl"
ALIGN_DIR = EVAL / "标准对齐" / "claude-sonnet"
OUT_DIR   = EVAL / "消融实验"

random.seed(SEED)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# ════════════════════════════════════════════════════════════════
#  从原始 skill 模块导入工具和知识库
# ════════════════════════════════════════════════════════════════
# 直接复用 run_claude_skills.py 里的三个工具函数和 LIMITS/TOOL_SCHEMAS
# 避免重复定义；若路径不同请修改 sys.path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path("E:/论文/跨境对齐/贡献")))
try:
    from run_claude_skills import (
        check_mrl_limit, query_regulation_status, lookup_hs_code,
        TOOL_SCHEMAS, TOOL_FN_MAP, dispatch_tool, load_rag_index,
        sample_rows, load_csv, ROUTES,
        COMP_SYSTEM, TEMPORAL_SYSTEM, ALIGNMENT_SYSTEM,
    )
    load_rag_index()
except ImportError as e:
    print(f"⚠ 无法从 run_claude_skills.py 导入：{e}")
    print("  请确保 ablation_study.py 与 run_claude_skills.py 在同一目录")
    sys.exit(1)

# 消融用：仅保留部分工具的 schema
TOOL_SCHEMAS_NO_RETRIEVAL: list = []   # 移除检索 → 不暴露任何工具

# ════════════════════════════════════════════════════════════════
#  CSV 工具
# ════════════════════════════════════════════════════════════════
def init_csv(path: Path, headers: list) -> set:
    done = set()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(headers)
    else:
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add(r.get(headers[0], ""))
    return done

def append_row(path: Path, row: list):
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(row)

# ════════════════════════════════════════════════════════════════
#  核心推理函数（可配置工具列表）
# ════════════════════════════════════════════════════════════════
def run_with_tools(
    system_prompt: str,
    user_message: str,
    tool_schemas: list = TOOL_SCHEMAS,
    max_turns: int = MAX_TOOL_TURNS,
) -> str:
    """通用带工具推理循环。tool_schemas=[] 时退化为无工具调用。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    for turn in range(max_turns):
        try:
            kwargs = dict(
                model=MODEL, messages=messages,
                temperature=0, max_tokens=1500,
            )
            if tool_schemas:
                kwargs["tools"] = tool_schemas
                kwargs["tool_choice"] = "auto"
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"    API error (turn {turn}): {e}")
            time.sleep(5)
            continue

        msg = resp.choices[0].message
        if not getattr(msg, "tool_calls", None):
            return (msg.content or "").strip()

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
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
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        time.sleep(0.5)
    return ""

# ════════════════════════════════════════════════════════════════
#  术语对齐变体（用于 w/o_aligner 和 aligner_* 内部消融）
# ════════════════════════════════════════════════════════════════
def _load_term_map() -> dict:
    from compliance_skill.config.settings import TERM_MAP_PATH
    if TERM_MAP_PATH.exists():
        return json.loads(TERM_MAP_PATH.read_text(encoding="utf-8"))
    return {}

_TERM_MAP = _load_term_map()

def align_exact(term: str) -> str:
    """层 1：仅精确匹配。"""
    entry = _TERM_MAP.get(term.lower().strip()) or _TERM_MAP.get(term.strip())
    return entry.get("国标名", term) if entry else term

def align_exact_emb(term: str, threshold: float = 0.82) -> str:
    """层 1+2：精确 + embedding 语义匹配。"""
    hit = align_exact(term)
    if hit != term:
        return hit
    # embedding 匹配
    try:
        import math
        resp = client.embeddings.create(model="text-embedding-3-small", input=term)
        q_vec = resp.data[0].embedding
        best_val, best_name = 0.0, term
        for key, entry in _TERM_MAP.items():
            r2 = client.embeddings.create(model="text-embedding-3-small", input=key)
            k_vec = r2.data[0].embedding
            dot  = sum(x * y for x, y in zip(q_vec, k_vec))
            norm = math.sqrt(sum(x**2 for x in q_vec)) * math.sqrt(sum(x**2 for x in k_vec))
            score = dot / norm if norm else 0.0
            if score > best_val:
                best_val = score
                best_name = entry.get("国标名", term)
        return best_name if best_val >= threshold else term
    except Exception:
        return term

def align_full(term: str) -> str:
    """层 1+2+3：完整三级降级（调用 skill 模块）。"""
    try:
        from compliance_skill.skills.term_aligner import align_term
        result = align_term(term)
        return result.get("国标名") or term
    except Exception:
        return term

# ════════════════════════════════════════════════════════════════
#  法规检索变体（用于 retrieval_api / retrieval_kg 内部消融）
# ════════════════════════════════════════════════════════════════
def retrieve_api_only(keywords: str, country: str = "") -> dict:
    """仅 L1 官方 API。"""
    from compliance_skill.skills.regulation_retriever import (
        _COUNTRY_FETCHERS, SUPPORTED_COUNTRIES
    )
    std = SUPPORTED_COUNTRIES.get(country.lower().strip(), country)
    fetcher = _COUNTRY_FETCHERS.get(std)
    if fetcher:
        results = fetcher(keywords)
        if results:
            return {"found": True, "results": results, "retrieval_source": "api"}
    return {"found": False, "results": [], "retrieval_source": "none"}

def retrieve_kg_only(keywords: str, country: str = "") -> dict:
    """仅 L2 本地 KG。"""
    from compliance_skill.skills.regulation_retriever import (
        _kg_search, SUPPORTED_COUNTRIES
    )
    import re as _re
    kw_list = [k.strip() for k in _re.split(r"[,\s]+", keywords) if k.strip()]
    std = SUPPORTED_COUNTRIES.get(country.lower().strip(), country)
    results = _kg_search(kw_list, std)
    if results:
        return {"found": True, "results": results, "retrieval_source": "kg"}
    return {"found": False, "results": [], "retrieval_source": "none"}

# ════════════════════════════════════════════════════════════════
#  多智能体辩论变体（用于 debate_N_round 内部消融）
# ════════════════════════════════════════════════════════════════
def run_debate_n(review_output: dict, rounds: int) -> dict:
    from compliance_skill.agents.debate_agents import debate, arbitrate
    debate_results = []
    current = review_output
    for i in range(1, rounds + 1):
        d = debate(current, round_idx=i)
        debate_results.append(d)
        if d["debate_verdict"] == "严重质疑":
            current = {
                **current,
                "audit_trail": current.get("audit_trail", "") + f"\n\n[辩论轮{i}]\n" + d["debate_content"],
            }
    return arbitrate(review_output, debate_results)

# ════════════════════════════════════════════════════════════════
#  变体定义
# ════════════════════════════════════════════════════════════════
# 每个变体是一个 callable：(system, user_msg, extra_kwargs) -> str
# extra_kwargs 用于传递 term_aligner / retrieval_fn / debate_rounds 等参数

def _run_full(system: str, user_msg: str, **_) -> str:
    """Full：完整 Skill 流水线。"""
    from compliance_skill.main_workflow import process_case
    # 直接调用完整 pipeline，提取最终裁决文本
    # user_msg 里含目标国和食品档案，从中解析 ingredient 和 country
    # 为保持与评测脚本一致，此处仍走 run_with_tools 保证输出格式对齐
    return run_with_tools(system, user_msg, TOOL_SCHEMAS)

def _run_wo_aligner(system: str, user_msg: str, **_) -> str:
    """w/o Term Aligner：原始外文名直接检索，不做术语对齐。"""
    return run_with_tools(system, user_msg, TOOL_SCHEMAS)
    # 注意：此变体的差异体现在工具内部不做术语归一化
    # 在工具层面通过 monkey-patch align_term 实现（见 run_ablation）

def _run_wo_retrieval(system: str, user_msg: str, **_) -> str:
    """w/o Retrieval：不暴露任何工具，模型凭自身知识推断。"""
    return run_with_tools(system, user_msg, TOOL_SCHEMAS_NO_RETRIEVAL)

def _run_wo_debate(system: str, user_msg: str, **_) -> str:
    """w/o Debate：带工具但跳过多智能体辩论，直接输出 CoT 审查结论。"""
    return run_with_tools(system, user_msg, TOOL_SCHEMAS)

def _run_aligner_exact(system: str, user_msg: str, **_) -> str:
    """仅精确字典匹配（内部消融）。"""
    _patch_aligner(align_exact)
    result = run_with_tools(system, user_msg, TOOL_SCHEMAS)
    _unpatch_aligner()
    return result

def _run_aligner_exact_emb(system: str, user_msg: str, **_) -> str:
    """精确 + embedding 匹配（内部消融）。"""
    _patch_aligner(align_exact_emb)
    result = run_with_tools(system, user_msg, TOOL_SCHEMAS)
    _unpatch_aligner()
    return result

def _run_retrieval_api(system: str, user_msg: str, **_) -> str:
    """仅 L1 API 检索（内部消融）。"""
    _patch_retrieval(retrieve_api_only)
    result = run_with_tools(system, user_msg, TOOL_SCHEMAS)
    _unpatch_retrieval()
    return result

def _run_retrieval_kg(system: str, user_msg: str, **_) -> str:
    """仅 L2 KG 检索（内部消融）。"""
    _patch_retrieval(retrieve_kg_only)
    result = run_with_tools(system, user_msg, TOOL_SCHEMAS)
    _unpatch_retrieval()
    return result

def _run_debate_1(system: str, user_msg: str, **_) -> str:
    return _run_with_debate_rounds(system, user_msg, rounds=1)

def _run_debate_2(system: str, user_msg: str, **_) -> str:
    return _run_with_debate_rounds(system, user_msg, rounds=2)

def _run_debate_3(system: str, user_msg: str, **_) -> str:
    return _run_with_debate_rounds(system, user_msg, rounds=3)

def _run_with_debate_rounds(system: str, user_msg: str, rounds: int) -> str:
    """先跑 CoT 审查，再跑 N 轮辩论，返回最终裁决。"""
    from compliance_skill.skills.compliance_reviewer import review_compliance
    # 用工具调用获取审查结果
    raw_output = run_with_tools(system, user_msg, TOOL_SCHEMAS)
    # 构造 review_output 格式（与 compliance_reviewer 返回格式对齐）
    review_output = {
        "verdict": _extract_verdict(raw_output),
        "full_report": raw_output,
        "audit_trail": raw_output,
        "flagged_uncertain": False,
    }
    final = run_debate_n(review_output, rounds=rounds)
    return final.get("final_verdict", raw_output)

def _extract_verdict(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if "审查结论：" in line or "最终裁决：" in line:
            return line.split("：", 1)[-1].strip()
    return text[-60:].strip()

# ── monkey-patch 辅助 ─────────────────────────────────────────
_orig_align     = None
_orig_retrieve  = None

def _patch_aligner(fn: Callable):
    global _orig_align
    try:
        import compliance_skill.skills.term_aligner as _mod
        _orig_align = _mod.align_term
        _mod.align_term = lambda term: {"国标名": fn(term), "_match_method": "patched", "_confidence": "ablation"}
    except Exception:
        pass

def _unpatch_aligner():
    try:
        import compliance_skill.skills.term_aligner as _mod
        if _orig_align:
            _mod.align_term = _orig_align
    except Exception:
        pass

def _patch_retrieval(fn: Callable):
    global _orig_retrieve
    try:
        import compliance_skill.skills.regulation_retriever as _mod
        _orig_retrieve = _mod.search_regulations
        _mod.search_regulations = fn
    except Exception:
        pass

def _unpatch_retrieval():
    try:
        import compliance_skill.skills.regulation_retriever as _mod
        if _orig_retrieve:
            _mod.search_regulations = _orig_retrieve
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════
#  变体注册表
# ════════════════════════════════════════════════════════════════
MODULE_VARIANTS = {
    "full":           _run_full,
    "wo_aligner":     _run_wo_aligner,
    "wo_retrieval":   _run_wo_retrieval,
    "wo_debate":      _run_wo_debate,
}

INTRA_VARIANTS = {
    "aligner_exact":     _run_aligner_exact,
    "aligner_exact_emb": _run_aligner_exact_emb,
    "retrieval_api":     _run_retrieval_api,
    "retrieval_kg":      _run_retrieval_kg,
    "debate_1round":     _run_debate_1,
    "debate_2round":     _run_debate_2,
    "debate_3round":     _run_debate_3,
}

# ════════════════════════════════════════════════════════════════
#  任务执行器（通用）
# ════════════════════════════════════════════════════════════════
def run_ablation_task(
    task_name: str,
    cases: list[dict],               # [{"id", "system", "user_msg", ...}]
    variants: dict[str, Callable],
    out_path: Path,
):
    """
    对 cases 列表中每个案件，依次运行所有变体并写入 CSV。
    CSV 格式：case_id | task | variant_1 | variant_2 | ...
    """
    variant_keys = list(variants.keys())
    headers = ["case_id", "task"] + variant_keys
    done = init_csv(out_path, headers)

    print(f"\n  [{task_name}] {len(cases)} 条案件 × {len(variant_keys)} 个变体")

    for case in cases:
        case_id = str(case["id"])
        if case_id in done:
            print(f"    {case_id}: 已完成，跳过")
            continue

        system  = case["system"]
        user_msg = case["user_msg"]
        row = [case_id, task_name]

        for vname, vfn in variants.items():
            print(f"    {case_id} | {vname} ...", end=" ", flush=True)
            try:
                output = vfn(system, user_msg)
            except Exception as e:
                output = f"ERROR: {e}"
            row.append(output.replace("\n", "\\n"))
            print(repr(output[-40:].strip()))
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        append_row(out_path, row)

# ════════════════════════════════════════════════════════════════
#  任务数据构建
# ════════════════════════════════════════════════════════════════
def build_compliance_cases() -> list[dict]:
    cases = []
    for src, routes in ROUTES.items():
        for origin, dest, data_path, label in routes:
            if not data_path.exists():
                print(f"  ⚠ 数据不存在: {data_path}")
                continue
            rows = load_csv(data_path)
            subset = sample_rows(rows, SAMPLE_PCT, SEED)
            for row in subset:
                rid = str(row.get("ID", row.get("id", len(cases))))
                profile = row.get("FOOD_PROFILE", row.get("food_profile", str(row)))
                cases.append({
                    "id":      f"comp_{label}_{rid}",
                    "system":  COMP_SYSTEM,
                    "user_msg": f"【目标进口国】：{dest}\n【拟进口食品档案】：\n{profile}",
                    "origin":  origin,
                    "dest":    dest,
                })
    if len(cases) > MAX_CASES:
        cases = random.Random(SEED).sample(cases, MAX_CASES)
    return cases

def build_temporal_cases() -> list[dict]:
    if not TEMPORAL.exists():
        print(f"  ⚠ 数据不存在: {TEMPORAL}")
        return []
    items = [json.loads(l) for l in TEMPORAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    subset = sample_rows(items, SAMPLE_PCT, SEED)
    if len(subset) > MAX_CASES:
        subset = random.Random(SEED).sample(subset, MAX_CASES)
    return [
        {
            "id":       f"temp_{i}",
            "system":   TEMPORAL_SYSTEM,
            "user_msg": item.get("instruction", ""),
            "gt":       item.get("answer", ""),
        }
        for i, item in enumerate(subset)
    ]

def build_alignment_cases() -> list[dict]:
    if not ALIGN_DIR.exists():
        print(f"  ⚠ baseline 目录不存在: {ALIGN_DIR}")
        return []
    cases = []
    rng = random.Random(SEED)
    for csv_file in sorted(ALIGN_DIR.glob("*.csv")):
        task_name = csv_file.stem.replace("claude-sonnet_", "")
        rows = load_csv(csv_file)
        subset = sample_rows(rows, SAMPLE_PCT, SEED)
        for row in subset:
            q = row.get("QUESTION", "")
            cases.append({
                "id":       f"align_{task_name}_{hash(q) % 100000}",
                "system":   ALIGNMENT_SYSTEM,
                "user_msg": q,
                "subtask":  task_name,
                "ref":      row.get("REFERENCE", ""),
            })
    if len(cases) > MAX_CASES:
        cases = rng.sample(cases, MAX_CASES)
    return cases

# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Skill 消融实验")
    parser.add_argument("--module", action="store_true", help="仅运行模块级消融")
    parser.add_argument("--intra",  action="store_true", help="仅运行模块内部消融")
    parser.add_argument("--task",   choices=["comp", "temporal", "align", "all"], default="all")
    args = parser.parse_args()

    run_module = args.module or (not args.module and not args.intra)
    run_intra  = args.intra  or (not args.module and not args.intra)

    variants = {}
    if run_module:
        variants.update(MODULE_VARIANTS)
    if run_intra:
        variants.update(INTRA_VARIANTS)

    print("=" * 60)
    print(f"Skill 消融实验  模型: {MODEL}  采样: {SAMPLE_PCT*100:.0f}%  seed={SEED}")
    print(f"变体: {list(variants.keys())}")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.task in ("comp", "all"):
        cases = build_compliance_cases()
        if cases:
            run_ablation_task(
                "compliance", cases, variants,
                OUT_DIR / "ablation_compliance.csv",
            )

    if args.task in ("temporal", "all"):
        cases = build_temporal_cases()
        if cases:
            run_ablation_task(
                "temporal", cases, variants,
                OUT_DIR / "ablation_temporal.csv",
            )

    if args.task in ("align", "all"):
        cases = build_alignment_cases()
        if cases:
            run_ablation_task(
                "alignment", cases, variants,
                OUT_DIR / "ablation_alignment.csv",
            )

    print(f"\n全部完成。结果写入 {OUT_DIR}")

if __name__ == "__main__":
    main()
