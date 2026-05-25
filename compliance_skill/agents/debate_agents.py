"""
agents/debate_agents.py
多智能体辩论 + 裁决系统

架构（引用 Sun et al. 2024 Markov Chain 多智能体辩论幻觉检测框架）：
  ReviewAgent  — 给出初步合规结论（= compliance_reviewer 输出）
  DebateAgent  — 扮演"挑剔律师"，质疑推理链中的弱点
  ArbiterAgent — 综合双方论点，形成最终共识裁决

设计原则：
  1. 每轮辩论只针对前一轮输出，避免信息累积导致幻觉传播
  2. 若辩论后仍存在 ⚠️ 标记项，裁决自动降级为"存在违规风险"
  3. 完整辩论记录纳入 audit_trail，满足可审计要求

引用：
  Sun et al. (2024) — Markov Chain-based Multi-Agent Debate Framework
  Agarwal et al. (2024) — Multi-Agent consensus validation
  NCBl review (2024) — Multi-agent cross-validation reduces hallucination
"""
from __future__ import annotations
import json
from openai import OpenAI
from config.settings import MODEL, OPENROUTER_API_KEY, DEBATE_ROUNDS

_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ── Agent 系统提示 ────────────────────────────────────────────────
_DEBATE_SYSTEM = """\
你是一名极其挑剔的食品法律顾问。你的任务是对另一位审查员给出的【合规报告】进行质疑。

请重点检查：
1. 是否遗漏了任何污染物、添加剂或程序性违规项？
2. 引用的法规编号和限量数值是否准确（警惕幻觉）？
3. 置信度低于 0.85 的条款是否被正确标注？
4. 结论是否与推理过程一致？

输出格式：
- 【质疑点 N】：描述具体疑问
- 最后一行：辩论结论：维持 | 需要修正 | 严重质疑
"""

_ARBITER_SYSTEM = """\
你是最终裁决仲裁员。你将收到【初步审查报告】和【辩论质疑】，综合两者给出最终结论。

裁决原则：
1. 若辩论质疑指出了实质性错误，采纳修正后的结论
2. 若质疑仅为措辞/格式问题，维持原结论
3. 若存在任何 ⚠️[需人工核实] 标记，最终结论不得高于"存在违规风险"

最后一行必须且只能是：
最终裁决：违规
最终裁决：存在违规风险
最终裁决：建议放行
最终裁决：需人工复核
"""

# ── DebateAgent ───────────────────────────────────────────────────
def debate(review_output: dict, round_idx: int = 1) -> dict:
    """
    对 ReviewAgent 输出发起质疑。

    参数：
      review_output — compliance_reviewer.review_compliance() 的返回值
      round_idx     — 当前辩论轮次（用于日志）

    返回：
      {
        "debate_verdict": "维持" | "需要修正" | "严重质疑",
        "debate_content": str,
        "round": int,
      }
    """
    audit_trail = review_output.get("audit_trail", review_output.get("full_report", ""))
    conf = review_output.get("confidence", 1.0)

    user_msg = (
        f"【初步审查报告（检索置信度 {conf:.2f}）】\n{audit_trail}\n\n"
        "请对上述报告进行逐项质疑，输出所有质疑点，最后一行输出辩论结论。"
    )

    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _DEBATE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,    # 略高于 0，使质疑更多样
            max_tokens=1200,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        content = f"DebateAgent 调用失败: {e}"

    verdict = "维持"
    for line in reversed(content.splitlines()):
        line = line.strip()
        if line.startswith("辩论结论："):
            verdict = line.replace("辩论结论：", "").strip()
            break

    return {"debate_verdict": verdict, "debate_content": content, "round": round_idx}


# ── ArbiterAgent ──────────────────────────────────────────────────
def arbitrate(review_output: dict, debate_results: list[dict]) -> dict:
    """
    综合初步报告和所有辩论轮次，给出最终裁决。

    返回：
      {
        "final_verdict": str,
        "arbiter_report": str,
        "audit_trail": str,     # 完整推理链（初步 + 辩论 + 裁决）
      }
    """
    audit_trail = review_output.get("audit_trail", "")
    flagged = review_output.get("flagged_uncertain", False)
    conf = review_output.get("confidence", 1.0)

    debate_summary = "\n\n".join(
        f"【辩论轮 {d['round']}】\n{d['debate_content']}" for d in debate_results
    )

    user_msg = (
        f"【初步审查报告】\n{audit_trail}\n\n"
        f"【辩论质疑汇总】\n{debate_summary}\n\n"
        f"检索置信度：{conf:.2f}，存在不确定标记：{'是' if flagged else '否'}\n\n"
        "请综合以上信息给出最终裁决，最后一行输出最终裁决标签。"
    )

    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _ARBITER_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_tokens=1000,
        )
        arbiter_report = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        arbiter_report = f"ArbiterAgent 调用失败: {e}"

    final_verdict = _extract_final_verdict(arbiter_report)

    # 安全降级：有 ⚠️ 且结论为"建议放行"时强制降级
    if flagged and final_verdict == "建议放行":
        final_verdict = "存在违规风险"
        arbiter_report += "\n\n[系统降级] 存在标记项，自动降级为【存在违规风险】。"

    full_trail = (
        "=== 初步审查 ===\n" + audit_trail
        + "\n\n=== 辩论记录 ===\n" + debate_summary
        + "\n\n=== 裁决报告 ===\n" + arbiter_report
    )

    return {
        "final_verdict": final_verdict,
        "arbiter_report": arbiter_report,
        "audit_trail": full_trail,
    }


def _extract_final_verdict(report: str) -> str:
    for line in reversed(report.strip().splitlines()):
        line = line.strip()
        if line.startswith("最终裁决："):
            return line.replace("最终裁决：", "").strip()
    return "需人工复核"


# ── 完整辩论流程（主入口）────────────────────────────────────────
def run_debate_pipeline(review_output: dict, rounds: int = DEBATE_ROUNDS) -> dict:
    """
    运行 N 轮辩论 + 最终裁决。

    参数：
      review_output — ReviewAgent（compliance_reviewer）输出
      rounds        — 辩论轮次（默认 2）

    返回：arbitrate() 的输出，包含 final_verdict + audit_trail
    """
    debate_results = []
    current = review_output
    for i in range(1, rounds + 1):
        d = debate(current, round_idx=i)
        debate_results.append(d)
        # 若质疑严重，将辩论结果合并回 current 供下一轮参考
        if d["debate_verdict"] == "严重质疑":
            current = {
                **current,
                "audit_trail": current.get("audit_trail", "") + f"\n\n[辩论轮{i}质疑]\n" + d["debate_content"],
            }

    return arbitrate(review_output, debate_results)
