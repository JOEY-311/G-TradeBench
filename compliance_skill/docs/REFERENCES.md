# 参考文献

> 跨境贸易合规审查 Skill 设计的核心文献依据

---

## 1. 知识图谱 + RAG 合规框架

**[1] Agarwal, B., Jomraj, H.S., Kaplunov, S., Krolick, J., & Rojkova, V. (2024).**
*RAGulating Compliance: A Multi-Agent Knowledge Graph for Regulatory QA.*
MasterControl AI Research. arXiv:2408.09893 (preprint).

> **对应设计决策**：
> - 用 KG（SPO 三元组）替代纯文本 chunk 作为法规知识的存储单元
> - 多智能体分工：构建 KG 的 agent、检索 KG 的 agent、生成回答的 agent
> - KG 的清洗、去重、规范化、增量更新 pipeline
> - 强调"完整可追溯的决策路径"（audit trail）是合规场景的强制需求
>
> **引用位置**：`regulation_retriever.py`（KG 遍历检索），`kg_builder.py`（增量更新），`compliance_reviewer.py`（可审计推理链）

---

**[2] Edge, D. et al. (2024, published ICLR 2025).**
*From Local to Global: A Graph RAG Approach to Query-Focused Summarization.*
Microsoft Research. arXiv:2404.16130. Proceedings of ICLR 2025.

> **对应设计决策**：
> - Skill 2（法规检索）采用图遍历而非纯向量相似度检索
> - 图结构更擅长处理跨文档关系型查询（如"哪些物质在 2023 年后被修订"）
> - 本项目为简化版 GraphRAG，适合中小规模 KG；大规模可替换为 Microsoft GraphRAG 库
>
> **引用位置**：`regulation_retriever.py` → `_kg_search()` 注释

---

## 2. 贸易法规 SPO 三元组抽取

**[3] Nandini, D., Koch, R., & Schönfeld, M. (2025).**
*Towards Structured Knowledge: Advancing Triple Extraction from Regional Trade Agreements using Large Language Models.*
University of Bayreuth. arXiv:2510.05121.

> **对应设计决策**：
> - 验证了用 LLM（Llama 3.1）从贸易协定法律文本中零样本抽取 SPO 三元组的可行性
> - 强调法律语言的细微差别对传统 NLP 方法构成挑战，LLM 优于标准 IE 模型
> - 本项目 `kg_builder.py` 的 Prompt 设计参考本文的 few-shot 示例格式
>
> **引用位置**：`kg_builder.py` → `extract_triples_from_text()`，`term_aligner.py` → `_llm_infer()`

---

## 3. 幻觉缓解与多智能体辩论

**[4] Sun, X., Li, J., Zhong, Y., Zhao, D., & Yan, R. (2024).**
*Towards Detecting LLMs Hallucination via Markov Chain-based Multi-Agent Debate Framework.*
Peking University / Renmin University of China. arXiv:2406.03075.

> **对应设计决策**：
> - 三角色架构（ReviewAgent → DebateAgent → ArbiterAgent）直接对应本文的 claim-debate-verify 流程
> - Markov Chain 结构：每轮辩论仅依赖前一轮输出，避免信息累积导致错误传播
> - 多智能体投票在"concise claim verification"任务上显著优于单 agent
>
> **引用位置**：`agents/debate_agents.py` — 全文核心设计依据

---

**[5] Anonymous (2024).**
*Agentic AI and Large Language Models in Radiology: Opportunities and Hallucination Challenges.*
NCBI / PubMed. PMC12729288. (2024–2025 review)

> **对应设计决策**：
> - 多智能体 role-based 系统通过交叉验证降低幻觉率
> - RAG 策略将响应锚定在已验证文献上
> - 本项目将二者结合：RAG 提供法规事实，多智能体辩论做推理验证
>
> **引用位置**：`agents/debate_agents.py` → `debate()` 函数注释

---

## 4. 合规 QA 与 LLM 推理

**[6] A Multi-Agent RAG Framework for Regulatory Compliance Checking. (2024).**
*Proceedings, ACM Digital Library.* DOI: 10.1145/3785472.

> **对应设计决策**：
> - 指出"知识截止与法规演变"是合规 LLM 最核心的风险
> - 提出"动态法规检索 + 多 agent 共识验证 + 可追溯决策路径"的三要素框架
> - 本项目以四层降级检索（官方 API → KG → 向量 RAG → 静态 KB）解决知识截止问题
>
> **引用位置**：整体架构设计，`regulation_retriever.py` 四层降级逻辑

---

**[7] Hassani, S., Sabetzadeh, M., Amyot, D., & Liao, J. (2024).**
*Rethinking Legal Compliance Automation: Opportunities with Large Language Models.*
IEEE 32nd International Requirements Engineering Conference (RE 2024), pp. 432–440.

> **对应设计决策**：
> - LLM 合规自动化需要显式推理步骤（Chain-of-Thought），不能依赖黑盒输出
> - 验证框架（verification framework）是实现"可信合规"的必要条件
>
> **引用位置**：`skills/compliance_reviewer.py` → STEP 1–6 结构化 CoT 设计

---

## 5. 置信度阈值与幻觉分层缓解

**[8] Meng, X. et al. (2025).**
*Multi-Layered Framework for LLM Hallucination Mitigation in High-Stakes Applications: A Tutorial.*
MDPI Computers, 14(8), 332.

> **对应设计决策**：
> - 动态置信度阈值：一般查询 0.75 / 产品查询 0.80 / **合规查询 0.85–0.90**
> - "Thought → Action → Observation" ReAct 模式
> - 三层参考架构：输入治理、证据锚定生成、响应后验证
>
> **引用位置**：`config/settings.py` → `CONFIDENCE_THRESHOLDS`，`agents/confidence_gate.py` — 核心设计依据

---

## 6. 知识图谱构建与实体规范化

**[9] Kommineni, J. et al. (2024).**
*LLM-empowered Knowledge Graph Construction: A Survey.*
arXiv:2510.20345.

> **对应设计决策**：
> - 预定义本体结构（fixed ontological schema）保证精确性和可解释性
> - LLM 先生成 Competency Questions 划定知识范围，再进行 ABox 填充
> - 本项目 KG schema（subject/predicate/object/country/effective_date/confidence）参考此思路
>
> **引用位置**：`knowledge_base/kg_builder.py` — schema 设计，`term_aligner.py` → `_semantic_match()` 注释

---

**[10] Lu, H., & Wang, Y. (2025). KARMA Framework.**
*Multi-agent architecture for schema-guided extraction with entity normalization.*
（引用自 Kommineni et al. 2024 综述，Section: KARMA）

> **对应设计决策**：
> - 每个 agent 执行 schema-guided 抽取任务，保证实体规范化和关系分类准确性
> - 本项目 `term_aligner.py` 三级降级（精确 → 语义 → LLM）对应 KARMA 的规范化策略
>
> **引用位置**：`skills/term_aligner.py` 整体架构

---

## 7. 贸易合规 LLM 基准

**[11] Wang, J. et al. (2025).**
*LLM-based HSE Compliance Assessment: Benchmark, Performance, and Advancements.*
arXiv:2505.22959.

> **背景参考**：验证 LLM 在 HSE（健康、安全、环境）合规评估中的适用性，
> 为食品跨境合规场景提供方法论参照。

---

## 引用索引（按模块）

| 模块文件 | 主要引用文献 |
|---|---|
| `config/settings.py` | [8] Meng et al. 2025 |
| `skills/term_aligner.py` | [3] Nandini 2025, [9] Kommineni 2024, [10] KARMA |
| `skills/regulation_retriever.py` | [1] Agarwal 2024, [2] Edge 2024 (GraphRAG), [6] ACM 2024 |
| `skills/compliance_reviewer.py` | [7] Hassani 2024, [1] Agarwal 2024 |
| `agents/debate_agents.py` | [4] Sun 2024, [5] NCBI 2024 |
| `agents/confidence_gate.py` | [8] Meng 2025 |
| `knowledge_base/kg_builder.py` | [3] Nandini 2025, [1] Agarwal 2024, [9] Kommineni 2024 |
| `main_workflow.py` | [6] ACM 2024（三要素框架） |
