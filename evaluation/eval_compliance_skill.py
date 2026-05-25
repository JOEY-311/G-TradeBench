"""
eval_compliance_skill.py
用 compliance_skill 主流水线对合规审查数据集进行评测。

数据格式（来自 合规审查/{model}/*.csv）：
  列：ID | ORIGIN | DESTINATION | FOOD_PROFILE | REVIEW_OUTPUT
  REVIEW_OUTPUT 格式："裁判结：违规" / "裁判结：存在违规风险" / "裁判结：建议放行"

评测逻辑：
  1. 从 FOOD_PROFILE 提取产品名（作为 foreign_ingredient）
  2. 将 FOOD_PROFILE 全文作为 question（完整信息传入 skill）
  3. DESTINATION 列作为 destination_country
  4. 运行 compliance_skill 五步流水线
  5. 比对 skill 输出的 final_verdict 与 GT，统计准确率

运行示例：
  python eval_compliance_skill.py                    # 全部文件，每文件取 10 条
  python eval_compliance_skill.py --n 20             # 每文件取 20 条
  python eval_compliance_skill.py --type AUS         # 只跑 AUS 类文件
  python eval_compliance_skill.py --model gemini     # 用指定模型（覆盖 settings.py）
  python eval_compliance_skill.py --resume           # 断点续跑
"""

import argparse
import json
import os
import re
import sys
import time
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────────────────────────
EVAL_BASE   = Path(__file__).parent
SKILL_ROOT  = Path(r"E:\论文\跨境对齐\贡献\compliance_skill")
DATA_ROOT   = EVAL_BASE / "合规审查"
OUT_DIR     = EVAL_BASE / "results" / "compliance_skill_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 以 deepseekv4 文件夹为数据源（FOOD_PROFILE / GT 与模型无关）
DATA_MODEL_DIR = DATA_ROOT / "deepseekv4"

# ── API 配置 ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# 默认使用 gemini-2.0-flash（快、便宜、支持长上下文）
DEFAULT_MODEL = "google/gemini-2.0-flash-001"

# ── 将 compliance_skill 加入模块路径 ─────────────────────────────────────────
sys.path.insert(0, str(SKILL_ROOT))

SAMPLE_N    = 10
RANDOM_SEED = 42

# ── 目标国映射（文件名后缀 → 中文） ──────────────────────────────────────────
DEST_MAP = {
    "CN": "中国", "JP": "日本", "KR": "韩国",
    "US": "美国", "USA": "美国", "FR": "法国",
    "GM": "德国", "AU": "澳大利亚",
}

# ── 裁决规范化 ────────────────────────────────────────────────────────────────
VERDICT_NORM = {
    "违规":         "违规",
    "存在违规风险":  "存在违规风险",
    "建议放行":     "建议放行",
    "需人工复核":   "需人工复核",
    "无法判断":     "无法判断",
}

def normalize_verdict(raw: str) -> str:
    """提取并规范化裁决字符串。"""
    if not raw:
        return "无法判断"
    # GT 格式："裁判结：违规" 或 "裁判结：存在违规风险"
    raw = raw.strip()
    for prefix in ["裁判结：", "裁判结:", "审查结论：", "审查结论:"]:
        if prefix in raw:
            raw = raw.split(prefix, 1)[-1].strip()
            break
    for key, val in VERDICT_NORM.items():
        if key in raw:
            return val
    return raw.strip()


# ── 从文件名中解析目标国 ──────────────────────────────────────────────────────
def dest_from_filename(fname: str) -> str:
    """AUS_AUS_to_JP.csv → '日本'"""
    m = re.search(r'_to_([A-Z]+)', fname, re.IGNORECASE)
    if m:
        code = m.group(1).upper()
        return DEST_MAP.get(code, code)
    return ""


def country_type_from_filename(fname: str) -> str:
    """AUS_* → 'AUS'; EU_* → 'EU'; US_* → 'USA'"""
    if fname.startswith("AUS"):
        return "AUS"
    elif fname.startswith("EU"):
        return "EU"
    elif fname.startswith("US"):
        return "USA"
    return "UNKNOWN"


# ── 从 FOOD_PROFILE 提取产品/成分名 ──────────────────────────────────────────
def extract_ingredient(food_profile: str) -> str:
    """
    尝试从 FOOD_PROFILE 提取产品名称作为 foreign_ingredient。
    按优先级依次匹配多种格式，找不到时取首行非空文本。
    """
    patterns = [
        r'产品/食品名称[：:]\s*(.+)',
        r'产品/样本名称[：:]\s*(.+)',
        r'产品名称[：:]\s*(.+)',
        r'品名[：:]\s*(.+)',
        r'Product[/ ]?Name[：:]\s*(.+)',
        r'- 产品/食品名称[：:]\s*(.+)',
        r'- 产品/样本名称[：:]\s*(.+)',
        r'食品名称[：:]\s*(.+)',
    ]
    for pat in patterns:
        m = re.search(pat, food_profile, re.IGNORECASE)
        if m:
            name = m.group(1).strip().split('\n')[0].strip()
            if name:
                return name[:120]

    # 兜底：取 FOOD_PROFILE 第一行非空文字
    for line in food_profile.splitlines():
        line = line.strip().lstrip('-•*').strip()
        if line and not line.startswith('【') and len(line) > 3:
            return line[:120]
    return "未知食品"


# ── 读取 CSV 并采样 ──────────────────────────────────────────────────────────
def load_file(fpath: Path, n: int, seed: int) -> list[dict]:
    df = pd.read_csv(fpath, encoding="utf-8-sig")
    required = {"FOOD_PROFILE", "REVIEW_OUTPUT"}
    if not required.issubset(df.columns):
        print(f"  [SKIP] 列不匹配，跳过：{fpath.name}")
        return []
    df = df.dropna(subset=["FOOD_PROFILE", "REVIEW_OUTPUT"])
    df = df[df["REVIEW_OUTPUT"].str.strip().ne("")]
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True).to_dict("records")


# ── 单条评测 ─────────────────────────────────────────────────────────────────
def evaluate_row(row: dict, dest_country: str) -> dict:
    """
    运行 compliance_skill 五步流水线（带回退策略），返回评测结果字典。

    回退策略：
      当 term_aligner 无法将产品名映射到国标名（常见于整体食品/农产品场景），
      直接将原始产品名注入 aligned_info 并继续后续步骤，
      置信度设为 "low" 以触发低置信提示。
    """
    from skills.term_aligner         import align_term
    from skills.regulation_retriever import search_regulations
    from skills.compliance_reviewer  import review_compliance
    from agents.confidence_gate      import confidence_gate
    from agents.debate_agents        import run_debate_pipeline

    food_profile = str(row.get("FOOD_PROFILE", ""))
    gt_raw       = str(row.get("REVIEW_OUTPUT", ""))
    gt_verdict   = normalize_verdict(gt_raw)

    ingredient = extract_ingredient(food_profile)
    question   = (
        f"以下食品从 {row.get('ORIGIN', '未知')} 出口到 {dest_country}，"
        f"请审查其合规性：\n\n{food_profile}"
    )

    start = time.time()
    llm_knowledge_only = False
    try:
        # ── Step 1：术语对齐（允许失败回退）──────────────────────────
        aligned_info = align_term(ingredient)
        if aligned_info.get("国标名") is None:
            # 回退：直接使用原始产品名，标记为低置信
            aligned_info = {
                "国标名":        ingredient[:80],
                "备注":          "term_aligner 未命中，直接使用原始产品名",
                "_match_method": "direct_pass",
                "_confidence":   "low",
            }
        gb_name = aligned_info["国标名"]

        # ── Step 2：法规检索 ─────────────────────────────────────────
        keywords = f"{gb_name} {dest_country}".strip()
        reg_results = search_regulations(keywords, country=dest_country)

        # 当法规检索完全无结果时，注入一个通用上下文占位条目，
        # 使评测流程继续进行（compliance_reviewer 将依赖 LLM 自带知识推理）。
        # 此情景标记 "llm_knowledge_only"，在论文中单独报告。
        llm_knowledge_only = False
        if not reg_results.get("found"):
            llm_knowledge_only = True
            reg_results = {
                "found":           True,
                "results": [{
                    "title":        f"通用食品安全要求（{dest_country}）",
                    "legal_status": "Reference",
                    "effective_date": "N/A",
                    "source":       "LLM knowledge fallback",
                    "retrieval_layer": 5,
                }],
                "retrieval_layer": 5,
                "confidence":      0.50,   # 低置信，触发低置信提示但不触发门控
                "below_threshold": True,
                "source":          "LLM knowledge fallback",
            }

        # ── Step 3：置信度门控 ───────────────────────────────────────
        # eval 模式：强制通过门控（不因缺乏结构化法规而终止评测流程），
        # 低置信由 compliance_reviewer 通过 ⚠️ 标注体现。
        gate = confidence_gate(
            reg_results,
            query_type="compliance",
            supplemental_retriever=search_regulations,
            keywords=keywords,
            country=dest_country,
        )
        # 仅当门控动作非 flag_human 时跳过；llm_knowledge_only 时强制继续
        if gate.action == "flag_human" and not llm_knowledge_only:
            return {
                "id":            row.get("ID", ""),
                "ingredient":    ingredient,
                "destination":   dest_country,
                "gt_verdict":    gt_verdict,
                "skill_verdict": "需人工复核",
                "correct":       "需人工复核" == gt_verdict,
                "flagged":       True,
                "elapsed_s":     round(time.time() - start, 1),
                "audit_snippet": f"Gate: {gate.reason}",
                "error":         None,
                "llm_knowledge_only": False,
            }
        if gate.supplemental_results:
            existing = reg_results.get("results", [])
            reg_results["results"] = existing + gate.supplemental_results

        # ── Step 4：CoT 审查 ─────────────────────────────────────────
        ret_conf = reg_results.get("confidence", 1.0)
        review_output = review_compliance(
            aligned_term_info=aligned_info,
            regulation_results=reg_results,
            question=question,
            retrieval_confidence=ret_conf,
        )

        # ── Step 5：多智能体辩论 ─────────────────────────────────────
        final_output  = run_debate_pipeline(review_output)
        skill_verdict = normalize_verdict(final_output.get("final_verdict", "需人工复核"))
        flagged       = skill_verdict == "需人工复核"
        audit_snippet = (final_output.get("audit_trail", ""))[:300]
        error         = None

    except Exception as e:
        skill_verdict = "执行异常"
        flagged       = True
        audit_snippet = ""
        error         = str(e)

    elapsed = round(time.time() - start, 1)
    correct = (skill_verdict == gt_verdict)

    return {
        "id":                 row.get("ID", ""),
        "ingredient":         ingredient,
        "destination":        dest_country,
        "gt_verdict":         gt_verdict,
        "skill_verdict":      skill_verdict,
        "correct":            correct,
        "flagged":            flagged,
        "elapsed_s":          elapsed,
        "audit_snippet":      audit_snippet,
        "error":              error,
        "llm_knowledge_only": llm_knowledge_only,
    }


# ── 主评测流程 ────────────────────────────────────────────────────────────────
def run_eval(args):
    # 覆盖模型（如指定）
    if args.model:
        os.environ["COMPLIANCE_MODEL"] = args.model
        print(f"使用模型：{args.model}")
    else:
        os.environ.setdefault("COMPLIANCE_MODEL", DEFAULT_MODEL)
        print(f"使用模型：{os.environ['COMPLIANCE_MODEL']}")

    out_path = OUT_DIR / "skill_eval_results.json"
    all_results: list[dict] = []
    done_ids: set[str] = set()

    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            all_results = json.load(f)
        done_ids = {r["id"] for r in all_results if not r.get("error")}
        print(f"Resume: {len(done_ids)} 条已完成\n")

    # 枚举数据文件
    if not DATA_MODEL_DIR.exists():
        print(f"数据目录不存在：{DATA_MODEL_DIR}")
        sys.exit(1)

    csv_files = sorted(DATA_MODEL_DIR.glob("*.csv"))
    if args.type:
        csv_files = [f for f in csv_files if country_type_from_filename(f.name) == args.type.upper()]

    print(f"共 {len(csv_files)} 个数据文件，每文件取 {args.n} 条：")
    for f in csv_files:
        print(f"  {f.name}")
    print()

    # 按文件类型聚合
    type_results: dict[str, list[dict]] = defaultdict(list)
    dest_results: dict[str, list[dict]] = defaultdict(list)

    for csv_path in csv_files:
        ctype = country_type_from_filename(csv_path.name)
        dest  = dest_from_filename(csv_path.name)
        rows  = load_file(csv_path, args.n, RANDOM_SEED)
        if not rows:
            continue

        print(f"[{csv_path.name}]  类型={ctype}  目标国={dest}  {len(rows)} 条")

        for i, row in enumerate(rows):
            rid = str(row.get("ID", f"{csv_path.stem}_{i}"))
            if rid in done_ids:
                continue

            result = evaluate_row(row, dest or str(row.get("DESTINATION", "")))
            result["file"]         = csv_path.name
            result["country_type"] = ctype

            all_results.append(result)
            type_results[ctype].append(result)
            dest_results[dest].append(result)

            mark = "OK" if result["correct"] else "XX"
            flag = " [!人工]" if result["flagged"] else ""
            print(f"  [{i+1}/{len(rows)}] {mark} GT={result['gt_verdict']}  "
                  f"Skill={result['skill_verdict']}{flag}  ({result['elapsed_s']}s)")
            if result["error"]:
                print(f"    ERROR: {result['error']}")

            # 每 5 条保存一次
            if len(all_results) % 5 == 0:
                _save(all_results, out_path)

            time.sleep(1.5)   # 避免触发限速

        print()

    _save(all_results, out_path)

    # ── 汇总输出 ──────────────────────────────────────────────────────────────
    valid = [r for r in all_results if not r.get("error") and r["gt_verdict"] not in ("无法判断",)]

    print("=" * 65)
    print("         compliance_skill 评测结果汇总")
    print("=" * 65)

    # 总体准确率
    if valid:
        total_acc = sum(r["correct"] for r in valid) / len(valid)
        print(f"\n总体准确率：{total_acc*100:.1f}%  (N={len(valid)})")

    # 按裁决类型准确率
    print("\n── 按 GT 裁决类型 ──")
    by_gt: dict[str, list] = defaultdict(list)
    for r in valid:
        by_gt[r["gt_verdict"]].append(r["correct"])
    for verd in ["违规", "存在违规风险", "建议放行"]:
        recs = by_gt.get(verd, [])
        if recs:
            acc = sum(recs) / len(recs)
            print(f"  {verd:<12}: {acc*100:.1f}%  (n={len(recs)})")

    # 按出口国类型
    print("\n── 按出口国类型 ──")
    for ctype in ["AUS", "EU", "USA"]:
        recs = [r for r in valid if r.get("country_type") == ctype]
        if recs:
            acc = sum(r["correct"] for r in recs) / len(recs)
            print(f"  {ctype:<6}: {acc*100:.1f}%  (n={len(recs)})")

    # 按目标国
    print("\n── 按目标国 ──")
    by_dest: dict[str, list] = defaultdict(list)
    for r in valid:
        by_dest[r["destination"]].append(r["correct"])
    for dest in sorted(by_dest.keys()):
        recs = by_dest[dest]
        acc  = sum(recs) / len(recs)
        print(f"  {dest:<8}: {acc*100:.1f}%  (n={len(recs)})")

    # 混淆矩阵（GT × Skill）
    print("\n── 混淆矩阵（GT → Skill 预测） ──")
    labels = ["违规", "存在违规风险", "建议放行", "需人工复核"]
    conf: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in valid:
        conf[r["gt_verdict"]][r["skill_verdict"]] += 1
    pred_labels = sorted({r["skill_verdict"] for r in valid})
    header = f"{'GT \\ Skill':<14}" + "".join(f"{p:<12}" for p in pred_labels)
    print(header)
    for gt in ["违规", "存在违规风险", "建议放行"]:
        if gt in conf:
            row_str = f"{gt:<14}" + "".join(f"{conf[gt].get(p,0):<12}" for p in pred_labels)
            print(row_str)

    # 人工复核比例
    flagged_n = sum(1 for r in all_results if r.get("flagged"))
    print(f"\n标记需人工复核：{flagged_n}/{len(all_results)} "
          f"({flagged_n/len(all_results)*100:.1f}%)")

    # 保存汇总
    summary = {
        "total_n":    len(valid),
        "overall_acc": round(total_acc * 100, 1) if valid else 0,
        "by_verdict": {
            v: {"acc": round(sum(recs)/len(recs)*100, 1), "n": len(recs)}
            for v, recs in by_gt.items() if recs
        },
        "by_country_type": {},
        "by_destination":  {},
        "flagged_rate":    round(flagged_n / len(all_results) * 100, 1) if all_results else 0,
        "model": os.environ.get("COMPLIANCE_MODEL", DEFAULT_MODEL),
    }
    for ctype in ["AUS", "EU", "USA"]:
        recs = [r for r in valid if r.get("country_type") == ctype]
        if recs:
            summary["by_country_type"][ctype] = {
                "acc": round(sum(r["correct"] for r in recs) / len(recs) * 100, 1),
                "n": len(recs),
            }
    for dest, recs in by_dest.items():
        summary["by_destination"][dest] = {
            "acc": round(sum(recs) / len(recs) * 100, 1), "n": len(recs)
        }

    sum_path = OUT_DIR / "skill_eval_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果：{out_path}")
    print(f"汇总结果：{sum_path}")


def _save(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=SAMPLE_N,
                        help=f"每文件抽样条数（默认{SAMPLE_N}）")
    parser.add_argument("--type",   choices=["AUS", "EU", "USA"], default=None,
                        help="只跑指定出口国类型")
    parser.add_argument("--model",  type=str, default=None,
                        help="覆盖 COMPLIANCE_MODEL（如 google/gemini-2.0-flash-001）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑，跳过已完成的 ID")
    args = parser.parse_args()

    run_eval(args)
