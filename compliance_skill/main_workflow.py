"""
main_workflow.py
主编排器 — 将四层 skill 串联为完整的合规审查流水线

流程：
  Step 1  术语对齐    (term_aligner)
  Step 2  法规检索    (regulation_retriever)
  Step 3  置信度门控  (confidence_gate)
  Step 4  CoT 审查    (compliance_reviewer)
  Step 5  多智能体辩论 (debate_agents)
  Output  结构化报告

对比原 main_workflow.py：
  - 新增置信度门控（Step 3）
  - 新增多智能体辩论（Step 5）
  - 输出包含完整可审计推理链（audit_trail）
  - 所有路径配置通过 config/settings.py 注入，无硬编码路径
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field


@dataclass
class ComplianceCase:
    """单个合规审查案件的全部中间状态和最终结果。"""
    foreign_ingredient: str
    question: str

    # 各步骤输出
    aligned_info: dict = field(default_factory=dict)
    regulation_results: dict = field(default_factory=dict)
    gate_decision: object = None          # GateDecision
    review_output: dict = field(default_factory=dict)
    final_output: dict = field(default_factory=dict)

    # 最终字段
    final_verdict: str = "未执行"
    audit_trail: str = ""
    flagged_for_human: bool = False


def process_case(
    foreign_ingredient: str,
    question: str,
    destination_country: str = "",
    verbose: bool = True,
) -> ComplianceCase:
    """
    完整合规审查流水线主入口。

    参数：
      foreign_ingredient  — 待审外文成分名（如 "Natriumbenzoat"）
      question            — 用户原始问题
      destination_country — 目标进口国（如 "中国"、"Japan"）
      verbose             — 是否打印每步进度

    返回：ComplianceCase（包含最终裁决和完整推理链）
    """
    from skills.term_aligner       import align_term
    from skills.regulation_retriever import search_regulations
    from skills.compliance_reviewer  import review_compliance
    from agents.confidence_gate      import confidence_gate
    from agents.debate_agents        import run_debate_pipeline

    case = ComplianceCase(foreign_ingredient=foreign_ingredient, question=question)

    def log(msg: str):
        if verbose:
            print(msg)

    # ── Step 1：术语对齐 ─────────────────────────────────────────
    log(f"\n{'='*60}")
    log(f"🚀 开始处理：{foreign_ingredient}")
    log("─" * 60)
    log("Step 1 ▶ 术语对齐")

    case.aligned_info = align_term(foreign_ingredient)
    log(f"   国标名：{case.aligned_info.get('国标名')}  "
        f"[{case.aligned_info.get('_match_method')} / "
        f"{case.aligned_info.get('_confidence')}]")

    if case.aligned_info.get("国标名") is None:
        log("   ⚠ 成分无法识别，流程终止。")
        case.final_verdict = "流程终止：成分无法识别"
        case.flagged_for_human = True
        return case

    gb_name = case.aligned_info["国标名"]

    # ── Step 2：法规检索 ─────────────────────────────────────────
    log("Step 2 ▶ 法规检索")
    keywords = f"{gb_name} {destination_country}".strip()
    case.regulation_results = search_regulations(keywords, country=destination_country)
    log(f"   检索层次：{case.regulation_results.get('retrieval_layer')}  "
        f"置信度：{case.regulation_results.get('confidence', 'N/A')}  "
        f"命中：{len(case.regulation_results.get('results', []))} 条")

    # ── Step 3：置信度门控 ───────────────────────────────────────
    log("Step 3 ▶ 置信度门控")
    case.gate_decision = confidence_gate(
        case.regulation_results,
        query_type="compliance",
        supplemental_retriever=search_regulations,
        keywords=keywords,
        country=destination_country,
    )
    log(f"   {case.gate_decision.action.upper()} — {case.gate_decision.reason}")

    if case.gate_decision.action == "flag_human":
        case.final_verdict = "需人工复核"
        case.flagged_for_human = True
        case.audit_trail = f"置信度门控拦截：{case.gate_decision.reason}"
        log("   流程因置信度不足被门控拦截，已标记人工复核。")
        return case

    # 若有补充检索结果，合并进去
    if case.gate_decision.supplemental_results:
        existing = case.regulation_results.get("results", [])
        case.regulation_results["results"] = existing + case.gate_decision.supplemental_results

    # ── Step 4：结构化 CoT 审查 ──────────────────────────────────
    log("Step 4 ▶ 结构化 CoT 合规审查")
    case.review_output = review_compliance(
        aligned_term_info=case.aligned_info,
        regulation_results=case.regulation_results,
        question=question,
        retrieval_confidence=case.regulation_results.get("confidence", 1.0),
    )
    log(f"   初步结论：{case.review_output.get('verdict')}")

    # ── Step 5：多智能体辩论裁决 ────────────────────────────────
    log("Step 5 ▶ 多智能体辩论")
    case.final_output = run_debate_pipeline(case.review_output)
    case.final_verdict  = case.final_output.get("final_verdict", "需人工复核")
    case.audit_trail    = case.final_output.get("audit_trail", "")
    case.flagged_for_human = (case.final_verdict == "需人工复核")

    log(f"\n{'='*60}")
    log(f"✅ 最终裁决：【{case.final_verdict}】")
    log(f"{'='*60}")

    return case


# ── 批量评测接口 ──────────────────────────────────────────────────
def run_batch(cases: list[dict], delay: float = 1.2) -> list[dict]:
    """
    批量运行多个案件，返回结果列表。

    参数：
      cases — [{"ingredient": ..., "question": ..., "country": ...}]
      delay — 案件间隔（秒），避免触发 API 限速

    返回：[{"ingredient", "verdict", "flagged", "audit_trail"}, ...]
    """
    results = []
    for i, c in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {c.get('ingredient', '')}")
        try:
            case = process_case(
                foreign_ingredient=c.get("ingredient", ""),
                question=c.get("question", ""),
                destination_country=c.get("country", ""),
                verbose=True,
            )
            results.append({
                "ingredient":   case.foreign_ingredient,
                "verdict":      case.final_verdict,
                "flagged":      case.flagged_for_human,
                "audit_trail":  case.audit_trail[:500] + "...",   # 截断，完整版需单独保存
            })
        except Exception as e:
            results.append({
                "ingredient": c.get("ingredient", ""),
                "verdict":    f"执行异常: {e}",
                "flagged":    True,
                "audit_trail": "",
            })
        time.sleep(delay)
    return results


# ── 本地测试 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        {
            "ingredient": "Natriumbenzoat",
            "question": "这款德国饮料含有限量为 0.5 g/kg 的 Natriumbenzoat，可以进口到中国吗？",
            "country": "中国",
        },
        {
            "ingredient": "Titanium Dioxide E171",
            "question": "含有 E171（二氧化钛）的糖果是否符合欧盟进口要求？",
            "country": "欧盟",
        },
    ]
    results = run_batch(test_cases)
    print("\n\n===== 批量结果汇总 =====")
    for r in results:
        print(f"  {r['ingredient']:25s} → {r['verdict']}"
              + (" ⚠️需人工" if r["flagged"] else ""))
