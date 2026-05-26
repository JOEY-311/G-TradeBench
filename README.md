# Cross-Border Food Compliance Benchmark

A benchmark dataset and multi-agent system for evaluating large language models on cross-border food regulatory compliance tasks, covering regulations from China, the US, the EU (Germany/France), Japan, and South Korea.

## Overview

Cross-border food trade requires navigating heterogeneous regulatory frameworks across jurisdictions. This project provides:

- **CrossFoodAlign** — a benchmark with three task types: standard alignment, compliance review, and temporal reasoning
- **A multi-agent compliance skill** — a RAG + knowledge-graph + multi-agent debate system for automated regulatory compliance checking
- **Evaluation scripts** — for assessing multiple LLMs (GPT-4o, Claude, Gemini, DeepSeek, Qwen, Grok) under various augmentation strategies (Vanilla, RAG, CoT, RAG+CoT, Skill)

## Task Types

| Task | Description |
|------|-------------|
| Standard Alignment | Align product ingredient/limit values across regulatory frameworks |
| Compliance Review | Determine whether a product violates target-market regulations |
| Temporal Reasoning | Identify regulation effective dates and draft/active status |

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
