# Cross-Border Food Compliance Benchmark

A benchmark dataset and multi-agent system for evaluating large language models on cross-border food regulatory compliance tasks, covering regulations from China, the US, the EU (Germany/France), Japan, and South Korea.

## Overview

Cross-border food trade requires navigating heterogeneous regulatory frameworks across jurisdictions. This project provides:

- **CrossFoodAlign** — a benchmark with three task types: standard alignment, compliance review, and temporal reasoning
- **A multi-agent compliance skill** — a RAG + knowledge-graph + multi-agent debate system for automated regulatory compliance checking
- **Evaluation scripts** — for assessing multiple LLMs (GPT-4o, Claude, Gemini, DeepSeek, Qwen, Grok) under various augmentation strategies (Vanilla, RAG, CoT, RAG+CoT, Skill)

## Repository Structure

```
.
├── data/                          # Benchmark datasets
│   ├── benchmark_aus.csv          # Australia product profiles (raw)
│   ├── benchmark_eu.csv           # EU product profiles (raw)
│   ├── benchmark_usa.csv          # USA product profiles (raw, LFS)
│   ├── formatted_AUS_profiles.csv # Australia formatted benchmark
│   ├── formatted_EU_profiles.csv  # EU formatted benchmark
│   ├── formatted_USA_profiles.csv # USA formatted benchmark (LFS)
│   ├── NOCHINA_HSCODE.xlsx        # HS code alignment tasks
│   ├── NOCHINA_俗名映射.xlsx       # Common name mapping tasks
│   ├── NOCHINA_冲突判断.xlsx       # Conflict detection tasks
│   ├── NOCHINA_准入程序.xlsx       # Market access procedure tasks
│   ├── NOCHINA_多国流通.xlsx       # Multi-jurisdiction circulation tasks
│   ├── NOCHINA_标签对齐.xlsx       # Label alignment tasks
│   ├── NOCHINA_配料准入.xlsx       # Ingredient admission tasks
│   ├── NOCHINA_限量对齐.xlsx       # Limit value alignment tasks
│   └── conduct.py                 # Dataset construction utilities
│
├── skill/                         # Claude Code skill definition
│   ├── SKILL.md                   # Skill manifest and capability map
│   └── references/
│       ├── capabilities/          # Per-capability reference docs
│       └── project/               # Project-level context (paths, models, formats)
│
├── knowledge_base/                # Regulatory knowledge graph
│   ├── regulation_kg.json         # Prebuilt KG (SPO triples, 3 MB)
│   ├── term_mapping.json          # Cross-lingual term normalization dictionary
│   ├── kg_builder.py              # Incremental KG builder (LLM-based SPO extraction)
│   ├── build_kg_and_terms.py      # Full KG + term mapping construction pipeline
│   └── batch_build_kg.py          # Batch construction with checkpoint/resume
│
├── compliance_skill/              # Multi-agent compliance system (main contribution)
│   ├── agents/
│   │   ├── confidence_gate.py     # Confidence-gated retrieval middleware
│   │   └── debate_agents.py       # Review → Debate → Arbiter multi-agent pipeline
│   ├── config/
│   │   └── settings.py            # Config (all secrets via environment variables)
│   ├── skills/
│   │   ├── term_aligner.py        # Skill 1: three-tier term alignment
│   │   ├── regulation_retriever.py# Skill 2: four-tier regulation retrieval
│   │   └── compliance_reviewer.py # Skill 3: structured CoT compliance review
│   ├── knowledge_base/            # Runtime KG + term mapping (symlinked from root)
│   ├── tests/
│   │   └── test_compliance_skill.py
│   ├── docs/
│   │   └── REFERENCES.md
│   ├── main_workflow.py           # Main orchestrator
│   └── requirements.txt
│
└── evaluation/                    # LLM evaluation scripts
    ├── eval_alignment.py          # Standard alignment evaluation
    ├── eval_compliance.py         # Compliance review evaluation
    ├── eval_compliance_all.py     # Multi-model compliance evaluation
    ├── eval_temporal.py           # Temporal reasoning evaluation
    ├── eval_rag_cot.py            # RAG / CoT strategy evaluation
    ├── eval_compliance_skill.py   # Skill-augmented evaluation
    ├── eval_skill_openrouter.py   # Multi-model skill evaluation via OpenRouter
    ├── ablation_study.py          # Ablation experiments
    ├── build_rag_index.py         # RAG index construction
    ├── request_*_rag_cot.py       # Batch inference with RAG+CoT
    ├── compliance_judge_prompt.md # LLM-as-judge prompt for compliance
    └── conflict_and_temporal_judge_prompts.md
```

## Task Types

| Task | Description | Metric |
|------|-------------|--------|
| Standard Alignment | Align product ingredient/limit values across regulatory frameworks | Accuracy |
| Compliance Review | Determine whether a product violates target-market regulations | F1 |
| Temporal Reasoning | Identify regulation effective dates and draft/active status | F1/F2 |

## Quick Start

### 1. Install dependencies

```bash
pip install -r compliance_skill/requirements.txt
```

### 2. Set environment variables

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
# or set ANTHROPIC_API_KEY / OPENAI_API_KEY depending on the models you use
```

### 3. Build the knowledge graph (optional — prebuilt KG included)

```bash
python knowledge_base/batch_build_kg.py
```

### 4. Run compliance evaluation

```bash
python evaluation/eval_compliance_all.py
```

### 5. Run the multi-agent skill

```bash
python compliance_skill/main_workflow.py
```

## Datasets

Large files (`benchmark_usa.csv`, `formatted_USA_profiles.csv`) are tracked with **Git LFS**. Install Git LFS before cloning:

```bash
git lfs install
git clone <repo-url>
```

The benchmark covers food products exported to **USA**, **EU** (France/Germany), and **Australia**, with product profiles sourced from Open Food Facts, FoodData Central, and RASFF.

## Models Evaluated

GPT-4o, GPT-4.1, Claude Sonnet, Gemini 2.0 Flash, DeepSeek-V3, Qwen-Plus, Grok-3 — evaluated under five strategies: Vanilla, RAG, CoT, RAG+CoT, and Skill (multi-agent).

## License

MIT
