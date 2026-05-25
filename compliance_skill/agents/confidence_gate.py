"""
agents/confidence_gate.py
置信度门控中间件

当检索置信度低于阈值时，触发补充检索或将案件标记为需人工复核，
而不是让 LLM 在不确定的基础上继续推理。

引用：
  Meng et al. (2025) — Multi-Layered Framework for LLM Hallucination
  Mitigation: 合规查询动态阈值 0.85–0.90
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from config.settings import CONFIDENCE_THRESHOLDS


@dataclass
class GateDecision:
    passed: bool                        # True = 可继续推理；False = 需干预
    action: str                         # "proceed" | "retry" | "flag_human"
    reason: str                         # 日志说明
    supplemental_results: list = field(default_factory=list)  # 补充检索结果


def confidence_gate(
    retrieval_result: dict,
    query_type: str = "compliance",
    supplemental_retriever: Optional[Callable[[str, str], dict]] = None,
    keywords: str = "",
    country: str = "",
) -> GateDecision:
    """
    置信度门控主函数。

    参数：
      retrieval_result      — regulation_retriever.search_regulations() 返回值
      query_type            — "general" | "product" | "compliance"
      supplemental_retriever — 可选的补充检索函数，签名 (keywords, country) -> dict
      keywords / country    — 传递给补充检索器的参数

    返回：GateDecision
    """
    threshold = CONFIDENCE_THRESHOLDS.get(query_type, 0.80)
    conf = retrieval_result.get("confidence", 0.0)
    layer = retrieval_result.get("retrieval_layer", 0)
    found = retrieval_result.get("found", False)

    # 未找到任何法规 → 直接标记人工复核
    if not found:
        return GateDecision(
            passed=False,
            action="flag_human",
            reason=f"无法检索到相关法规（confidence=0），需人工查阅官方数据库。",
        )

    # 置信度达标 → 直接通过
    if conf >= threshold:
        return GateDecision(
            passed=True,
            action="proceed",
            reason=f"置信度 {conf:.2f} ≥ 阈值 {threshold}，继续推理。",
        )

    # 置信度不足，尝试补充检索
    if supplemental_retriever and keywords:
        try:
            extra = supplemental_retriever(keywords, country)
            extra_conf = extra.get("confidence", 0.0)
            if extra_conf >= threshold:
                return GateDecision(
                    passed=True,
                    action="retry",
                    reason=f"补充检索后置信度提升至 {extra_conf:.2f}，继续推理。",
                    supplemental_results=extra.get("results", []),
                )
        except Exception as e:
            pass  # 补充检索失败，继续走下面的人工标记逻辑

    # 仍低于阈值 → 标记人工复核，但允许 LLM 继续（带 ⚠️ 标注）
    action = "flag_human" if layer >= 3 else "proceed_with_warning"
    return GateDecision(
        passed=(action != "flag_human"),
        action=action,
        reason=(
            f"置信度 {conf:.2f} < 阈值 {threshold}（检索层次 {layer}）。"
            + (" 已标记需人工核实。" if action == "flag_human"
               else " 将带 ⚠️ 标注继续推理。")
        ),
    )
