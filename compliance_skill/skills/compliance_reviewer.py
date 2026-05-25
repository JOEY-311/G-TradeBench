"""
skills/compliance_reviewer.py
Skill 3 — 结构化 CoT 合规审查

改进点（对比原 compliance_reviewer.py）：
  - 原版：把全部推理交给单次 LLM 调用，无法自我纠错，无推理链
  - 新版：
      1. 结构化 CoT —— 逐项核验（限量值 / 添加剂 / 标签 / 程序性），
         每步显式引用法规编号
      2. 输出中间推理链（audit_trail）供下游 DebateAgent 使用
      3. 置信度感知 —— 传入检索置信度，低置信时提示模型标记不确定项

引用：
  Agarwal et al. (2024) — 合规 QA 需要"完整可追溯的决策路径"
  Hassani et al. (2024) — LLM 合规自动化需显式推理步骤
"""
from __future__ import annotations
import json
from openai import OpenAI
from config.settings import MODEL, OPENROUTER_API_KEY

_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

_COT_SYSTEM = """\
你是资深跨境食品合规审查员，精通 GB 2762/2763、EU Reg 2023/915、EU Reg 396/2005、
日本食品衛生法告示、肯定リスト制度、韩国 KFDA 식품공전、美国 EPA CFR 40 / FDA FSMA。

【审查流程 — 必须逐步输出，禁止跳过任何步骤】
STEP 1  成分/物质识别：列出所有待审成分及其实测值（若有）
STEP 2  限量核验：逐项对比实测值与目标国限量标准，引用具体法规编号
STEP 3  添加剂合规：核查使用范围和最大用量是否符合目标国规定
STEP 4  标签合规：检查标签、语言、过敏原标注要求
STEP 5  程序性要求：检验检疫、注册/备案、卫生证书等申报程序
STEP 6  综合结论：汇总违规项，给出可操作建议

【置信度提示】
若检索置信度 < 0.85，在对应条款末尾标注 ⚠️[需人工核实] 。

【输出格式 — 严格遵守】
最后一行必须且只能是以下三项之一（不附加任何文字）：
审查结论：违规
审查结论：存在违规风险
审查结论：建议放行
"""


def review_compliance(
    aligned_term_info: dict,
    regulation_results: dict,
    question: str,
    retrieval_confidence: float = 1.0,
) -> dict:
    """
    结构化 CoT 合规审查。

    参数：
      aligned_term_info     — term_aligner 返回的术语规范化结果
      regulation_results    — regulation_retriever 返回的法规检索结果
      question              — 用户原始问题
      retrieval_confidence  — 检索置信度（float 0–1）

    返回：
      {
        "verdict":     "违规" | "存在违规风险" | "建议放行" | "无法判断",
        "full_report": str,   # 完整 CoT 报告（含各步骤推理）
        "audit_trail": str,   # 同 full_report，供 DebateAgent 使用
        "confidence":  float, # 检索置信度透传
        "flagged_uncertain": bool,  # 是否存在⚠️标记项
      }
    """
    # 构造 context
    reg_text = _format_regulations(regulation_results)
    conf_note = (
        f"\n【检索置信度：{retrieval_confidence:.2f}】"
        + (" — 低置信，请对不确定项标注 ⚠️[需人工核实]"
           if retrieval_confidence < 0.85 else "")
    )

    user_msg = (
        f"【成分信息】\n{json.dumps(aligned_term_info, ensure_ascii=False, indent=2)}\n\n"
        f"【检索到的法规】\n{reg_text}{conf_note}\n\n"
        f"【用户问题】\n{question}\n\n"
        "请按 STEP 1–6 逐步输出审查意见，最后一行输出审查结论。"
    )

    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _COT_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_tokens=2000,
        )
        full_report = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {
            "verdict": "无法判断",
            "full_report": f"LLM 调用失败: {e}",
            "audit_trail": "",
            "confidence": retrieval_confidence,
            "flagged_uncertain": True,
        }

    verdict = _extract_verdict(full_report)
    return {
        "verdict": verdict,
        "full_report": full_report,
        "audit_trail": full_report,          # DebateAgent 使用同一份推理链
        "confidence": retrieval_confidence,
        "flagged_uncertain": "⚠️" in full_report,
    }


def _format_regulations(reg_results: dict) -> str:
    """将结构化检索结果序列化为可读文本，供 Prompt 注入。"""
    if not reg_results.get("found"):
        return reg_results.get("message", "未检索到相关法规。")
    items = reg_results.get("results", [])
    lines = [f"（检索层次：{reg_results.get('retrieval_layer')}，"
             f"置信度：{reg_results.get('confidence', 'N/A')}）"]
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. [{item.get('legal_status', '?')}] {item.get('title', '')} "
            f"| 生效日期：{item.get('effective_date', 'N/A')} "
            f"| 来源：{item.get('source', 'N/A')}"
        )
    return "\n".join(lines)


def _extract_verdict(report: str) -> str:
    """从报告末尾提取审查结论标签。"""
    for line in reversed(report.strip().splitlines()):
        line = line.strip()
        if line.startswith("审查结论："):
            return line.replace("审查结论：", "").strip()
    return "无法判断"
