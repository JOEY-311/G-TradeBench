# 跨境贸易合规审查 Skill

面向食品跨境贸易合规场景的多智能体 RAG 系统骨架。

## 项目结构

```
compliance_skill/
├── config/
│   └── settings.py              # 全局配置（环境变量注入，无硬编码）
├── skills/
│   ├── term_aligner.py          # Skill 1：术语对齐（精确→语义→LLM 三级降级）
│   ├── regulation_retriever.py  # Skill 2：法规检索（官方API→KG→向量RAG→静态KB）
│   └── compliance_reviewer.py   # Skill 3：结构化 CoT 合规审查
├── agents/
│   ├── confidence_gate.py       # 置信度门控中间件
│   └── debate_agents.py         # 多智能体辩论（ReviewAgent + DebateAgent + ArbiterAgent）
├── knowledge_base/
│   ├── kg_builder.py            # KG 增量构建（SPO 三元组抽取与更新）
│   ├── regulation_kg.json       # 本地知识图谱（运行时生成）
│   └── term_mapping.json        # 术语规范化词典
├── docs/
│   └── REFERENCES.md            # 完整参考文献（含引用位置说明）
├── main_workflow.py             # 主编排器
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
export OPENROUTER_API_KEY="your-key-here"

# 3. 构建初始知识图谱（可选，有静态 KB 兜底）
python knowledge_base/kg_builder.py knowledge_base/GB2762_2022.md

# 4. 运行测试案例
python main_workflow.py
```

## 五步流程

```
输入（商品档案）
    ↓
Step 1  术语对齐       外文→国标名，三级降级
    ↓
Step 2  法规检索       四层降级：官方API→KG→向量RAG→静态KB
    ↓
Step 3  置信度门控     低置信触发补充检索或人工标记
    ↓
Step 4  CoT 合规审查   STEP1-6 结构化推理，显式引用法规编号
    ↓
Step 5  多智能体辩论   ReviewAgent → DebateAgent → ArbiterAgent
    ↓
输出（结构化报告 + audit_trail）
```

## 参考文献

见 `docs/REFERENCES.md`，共 11 篇文献，涵盖：
- KG+RAG 合规框架（Agarwal 2024, Edge 2024）
- SPO 三元组抽取（Nandini 2025）
- 多智能体幻觉检测（Sun 2024）
- 置信度阈值框架（Meng 2025）
- LLM 合规推理（Hassani 2024）
