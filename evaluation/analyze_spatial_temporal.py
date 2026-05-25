"""
分析两个维度的模型表现差异：
1. 空间差异：按出口国分类，合规审查任务的平均准确率（AUS / EU / USA）
2. 时间差异：按法规发布年份分类，规则更新任务的平均得分

输出：JSON + 控制台表格
"""

import json
import os
import pandas as pd
from collections import defaultdict
from pathlib import Path

BASE = Path(r"E:\论文\跨境对齐\评测阶段")

# ── 哪些模型是"主模型"（不含策略变体） ───────────────────────────────────────
STRATEGY_SUFFIXES = ["-cot", "-rag", "-rag_cot", "-baseline"]

def is_base_model(name: str) -> bool:
    return not any(name.endswith(s) or s in name for s in STRATEGY_SUFFIXES)

# ════════════════════════════════════════════════════════════════════════════════
# Part 1: 空间差异 — 合规审查 × 出口国
# ════════════════════════════════════════════════════════════════════════════════

def analyze_spatial():
    print("\n" + "=" * 70)
    print("Part 1: 空间差异 — 合规审查任务 × 出口国")
    print("=" * 70)

    with open(BASE / "results/合规审查/compliance_all_results.json", encoding="utf-8") as f:
        all_data = json.load(f)

    base_models = [m for m in all_data if is_base_model(m["model"])]
    print(f"纳入分析的基础模型（{len(base_models)} 个）：{[m['model'] for m in base_models]}\n")

    # 按出口国聚合 (AUS / EU / USA)
    # 文件名前缀: AUS_* → 澳大利亚; EU_* → 欧盟; US_* → 美国
    country_map = {"AUS": [], "EU": [], "USA": []}

    # 每个模型 × 每个国家的准确率
    model_country_acc = {}  # {model: {AUS: x, EU: x, USA: x}}

    for m in base_models:
        model_name = m["model"]
        aus_accs, eu_accs, usa_accs = [], [], []
        for fr in m.get("file_results", []):
            fname = fr["file"]
            acc = fr["acc"]
            if fname.startswith("AUS"):
                aus_accs.append(acc)
            elif fname.startswith("EU"):
                eu_accs.append(acc)
            elif fname.startswith("US") or fname.startswith("USA"):
                usa_accs.append(acc)

        model_country_acc[model_name] = {
            "AUS": round(sum(aus_accs) / len(aus_accs), 1) if aus_accs else None,
            "EU":  round(sum(eu_accs)  / len(eu_accs),  1) if eu_accs  else None,
            "USA": round(sum(usa_accs) / len(usa_accs), 1) if usa_accs else None,
        }

    # 打印逐模型表格
    print(f"{'模型':<22} {'AUS(%)':<10} {'EU(%)':<10} {'USA(%)':<10} {'All(%)':<10}")
    print("-" * 62)
    for mn, acc in model_country_acc.items():
        vals = [v for v in acc.values() if v is not None]
        avg_all = round(sum(vals) / len(vals), 1) if vals else None
        aus_s  = f"{acc['AUS']:.1f}" if acc['AUS'] is not None else "N/A"
        eu_s   = f"{acc['EU']:.1f}"  if acc['EU']  is not None else "N/A"
        usa_s  = f"{acc['USA']:.1f}" if acc['USA'] is not None else "N/A"
        all_s  = f"{avg_all:.1f}"    if avg_all    is not None else "N/A"
        print(f"{mn:<22} {aus_s:<10} {eu_s:<10} {usa_s:<10} {all_s}")

    # 跨模型平均
    print("\n--- 跨模型平均（空间差异汇总）---")
    for country in ["AUS", "EU", "USA"]:
        vals = [acc[country] for acc in model_country_acc.values() if acc[country] is not None]
        if vals:
            avg = round(sum(vals) / len(vals), 1)
            std = round((sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5, 1)
            print(f"  {country:<6}: avg={avg:.1f}%  std={std:.1f}%  n={len(vals)}  "
                  f"range=[{min(vals):.1f}%, {max(vals):.1f}%]")

    # 额外：按 destination country（文件名包含 _to_XX 的）
    print("\n--- 目标国维度（按目的地国，仅含明确命名的文件）---")
    dest_accs = defaultdict(list)
    for m in base_models:
        for fr in m.get("file_results", []):
            fname = fr["file"]
            # 提取 _to_XX 部分
            if "_to_" in fname:
                dest = fname.split("_to_")[-1].replace(".csv", "").strip()
                # 标准化目的地名称
                dest_norm = {
                    "CN": "中国(CN)", "JP": "日本(JP)", "KR": "韩国(KR)",
                    "FR": "法国(FR)", "US": "美国(US)", "USA": "美国(US)",
                    "GM": "德国(GM)", "AU": "澳大利亚(AU)"
                }.get(dest.upper(), dest.upper())
                dest_accs[dest_norm].append(fr["acc"])

    if dest_accs:
        print(f"{'目标国':<16} {'平均准确率':<12} {'样本数':<8} {'范围'}")
        print("-" * 50)
        for dest in sorted(dest_accs.keys()):
            vals = dest_accs[dest]
            avg = round(sum(vals) / len(vals), 1)
            print(f"  {dest:<14} {avg:<12.1f} {len(vals):<8} [{min(vals):.1f}%, {max(vals):.1f}%]")
    else:
        print("  （无法解析目标国信息）")

    # 保存结果
    result = {
        "model_by_country": model_country_acc,
        "cross_model_avg": {},
        "by_destination": {},
    }
    for country in ["AUS", "EU", "USA"]:
        vals = [acc[country] for acc in model_country_acc.values() if acc[country] is not None]
        if vals:
            avg = round(sum(vals) / len(vals), 1)
            result["cross_model_avg"][country] = {"avg": avg, "n": len(vals),
                                                  "min": min(vals), "max": max(vals)}
    for dest, vals in dest_accs.items():
        result["by_destination"][dest] = {
            "avg": round(sum(vals) / len(vals), 1), "n": len(vals)
        }
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Part 2: 时间差异 — 规则更新 × 法规发布年份
# ════════════════════════════════════════════════════════════════════════════════

def analyze_temporal():
    print("\n" + "=" * 70)
    print("Part 2: 时间差异 — 规则更新任务 × 法规发布年份")
    print("=" * 70)

    # 读取文档元数据（含 publication_date）
    docs_df = pd.read_csv(
        BASE / "规则更新/documents_matching_food_additive_and_of_type_rule.csv",
        encoding="utf-8-sig"
    )
    docs_df["pub_year"] = pd.to_datetime(docs_df["publication_date"], errors="coerce").dt.year
    # index 在 scores 文件中是 1-based，对应 docs_df 的 0-based 行
    docs_df["score_index"] = range(1, len(docs_df) + 1)
    # 只取前200行（与评测集对应）
    docs_meta = docs_df[docs_df["score_index"] <= 200].set_index("score_index")
    print(f"文档元数据已加载：{len(docs_meta)} 条（index 1-{len(docs_meta)}）")
    year_dist = docs_meta["pub_year"].value_counts().sort_index().to_dict()
    print(f"年份分布：{year_dist}\n")

    # 读取各模型得分文件
    score_files = {
        "deepseek-V3.2": BASE / "规则更新/temporal_ds_scores.json",
        "gemini-3.1pro": BASE / "规则更新/temporal_gemini_scores.json",
        "gpt-5.4":       BASE / "规则更新/temporal_gpt_scores.json",
        "grok-4.20":     BASE / "规则更新/temporal_grok_scores.json",
        "qwen-3.6plus":  BASE / "规则更新/temporal_qwen_scores.json",
    }
    # gpt-5.5 和 deepseek-V4 用临时处理
    extra_files = {
        "gpt-5.5":    BASE / "规则更新/temporal_gpt-5.5.csv" if (BASE / "规则更新/temporal_gpt-5.5.csv").exists() else None,
        "deepseek-V4": BASE / "规则更新/temporal_test_results_deepseekv4.csv" if (BASE / "规则更新/temporal_test_results_deepseekv4.csv").exists() else None,
    }

    # 合并所有模型的 {index → normalized_score} 数据
    all_scores = defaultdict(list)  # {score_index: [score, ...]}
    model_count = 0

    for model_name, path in score_files.items():
        if not path.exists():
            print(f"  {model_name}: 文件不存在，跳过")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model_count += 1
        for item in data:
            idx = int(item["index"])
            ns = item.get("normalized_score", 0.0)
            all_scores[idx].append(ns)
        f1_avg = sum(item["F1"]["score"] for item in data) / len(data)
        f2_avg = sum(item["F2"]["score"] for item in data) / len(data)
        f3_avg = sum(item["F3"]["score"] for item in data) / len(data)
        ns_avg = sum(item["normalized_score"] for item in data) / len(data)
        print(f"  {model_name:<16}: F1={f1_avg:.2f}/2  F2={f2_avg:.2f}/2  F3={f3_avg:.2f}/2  norm={ns_avg:.1f}%")

    # 分组：按年份
    year_scores = defaultdict(list)
    for idx, scores in all_scores.items():
        if idx not in docs_meta.index:
            continue
        year = docs_meta.loc[idx, "pub_year"]
        if pd.isna(year):
            continue
        year_scores[int(year)].extend(scores)

    # 为了可读性，合并成5年段
    def year_to_period(y):
        if y < 2000:
            return "1994-1999"
        elif y < 2005:
            return "2000-2004"
        elif y < 2010:
            return "2005-2009"
        elif y < 2015:
            return "2010-2014"
        elif y < 2020:
            return "2015-2019"
        elif y < 2023:
            return "2020-2022"
        else:
            return "2023-2026"

    period_scores = defaultdict(list)
    for year, scores in year_scores.items():
        period = year_to_period(year)
        period_scores[period].extend(scores)

    print(f"\n已加载 {model_count} 个模型，共 {sum(len(v) for v in all_scores.values())} 条得分记录")

    # 按年份输出
    print("\n--- 按发布年份 × 平均得分（跨模型）---")
    print(f"{'年份':<10} {'平均得分(%)':<14} {'样本数':<8} {'范围'}")
    print("-" * 50)
    for year in sorted(year_scores.keys()):
        vals = year_scores[year]
        avg = round(sum(vals) / len(vals), 1)
        print(f"  {year:<8} {avg:<14.1f} {len(vals):<8} [{min(vals):.1f}%, {max(vals):.1f}%]")

    # 按5年段输出
    period_order = ["1994-1999", "2000-2004", "2005-2009", "2010-2014", "2015-2019", "2020-2022", "2023-2026"]
    print("\n--- 按法规时段 × 平均得分（5年分组）---")
    print(f"{'时段':<14} {'平均得分(%)':<14} {'文档数':<10} {'得分样本数'}")
    print("-" * 55)
    period_result = {}
    for period in period_order:
        vals = period_scores.get(period, [])
        if not vals:
            continue
        avg = round(sum(vals) / len(vals), 1)
        n_docs = len(set(idx for idx in all_scores.keys()
                         if idx in docs_meta.index
                         and not pd.isna(docs_meta.loc[idx, "pub_year"])
                         and year_to_period(int(docs_meta.loc[idx, "pub_year"])) == period))
        print(f"  {period:<14} {avg:<14.1f} {n_docs:<10} {len(vals)}")
        period_result[period] = {"avg": avg, "n_docs": n_docs, "n_scores": len(vals)}

    # gt_type A vs B 跨模型平均（类型A=常规更新，B=更复杂/特殊）
    print("\n--- gt_type 分布 × 得分（A=常规通知型，B=废除/强制型）---")
    type_a_scores, type_b_scores = [], []
    for model_name, path in score_files.items():
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            ns = item.get("normalized_score", 0.0)
            if item["gt_type"] == "A":
                type_a_scores.append(ns)
            else:
                type_b_scores.append(ns)
    if type_a_scores:
        print(f"  Type A（常规更新）: avg={sum(type_a_scores)/len(type_a_scores):.1f}%  n={len(type_a_scores)}")
    if type_b_scores:
        print(f"  Type B（废除/强制）: avg={sum(type_b_scores)/len(type_b_scores):.1f}%  n={len(type_b_scores)}")

    return {
        "by_year": {str(k): {"avg": round(sum(v)/len(v), 1), "n": len(v)}
                    for k, v in sorted(year_scores.items())},
        "by_period": period_result,
        "type_a_avg": round(sum(type_a_scores)/len(type_a_scores), 1) if type_a_scores else None,
        "type_b_avg": round(sum(type_b_scores)/len(type_b_scores), 1) if type_b_scores else None,
    }


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    spatial_result  = analyze_spatial()
    temporal_result = analyze_temporal()

    out = {
        "spatial_compliance":  spatial_result,
        "temporal_rule_update": temporal_result,
    }
    out_path = BASE / "results/spatial_temporal_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n\n结果已保存至：{out_path}")
