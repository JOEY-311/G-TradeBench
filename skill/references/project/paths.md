# 项目路径速查

所有路径基于 `E:\论文\跨境对齐\` 根目录。

---

## 评测阶段根目录

```
E:\论文\跨境对齐\评测阶段\
```

---

## 主要脚本

| 脚本 | 功能 |
|------|------|
| `run_claude_sonnet.py` | Claude Sonnet 全流程生成（3个任务，9条路线） |
| `eval_claude_sonnet.py` | Claude Sonnet 评测（规则 + LLM-judge） |
| `run_claude_strategies.py` | RAG/CoT 策略生成（cot/rag/rag_cot，20% 子集） |
| `eval_claude_strategies.py` | 四策略对比评测 |
| `build_rag_index.py` | 构建本地法规 RAG 索引 |

---

## 输入数据

| 数据 | 路径 |
|------|------|
| 澳洲初级农产品 | `E:\论文\跨境对齐\准备阶段\商品数据集\可用\澳洲初级农产品\Food Composition.csv` |
| EU 食品档案 | `E:\论文\跨境对齐\准备阶段\商品数据集\问题集\formatted_EU_profiles.csv` |
| USA 食品档案 | `E:\论文\跨境对齐\准备阶段\商品数据集\问题集\formatted_USA_profiles.csv` |
| 时序推理题目 | `E:\论文\跨境对齐\评测阶段\规则更新\temporal_eval_en.jsonl` |
| 标准对齐题目 | 各模型输出目录下的 `标准对齐/{model}/` 中的 CSV |

---

## 输出目录结构

```
评测阶段/
├── 合规审查/
│   ├── claude-sonnet/          ← baseline (450行 × 9文件)
│   ├── claude-sonnet-cot/      ← CoT策略 (90行 × 9文件)
│   ├── claude-sonnet-rag/      ← RAG策略
│   ├── claude-sonnet-rag_cot/  ← RAG+CoT策略
│   ├── gemini/
│   ├── gemini-baseline/
│   ├── gemini-cot/
│   ├── gemini-rag/
│   └── ...（其他模型）
│
├── 规则更新/
│   ├── temporal_claude-sonnet.csv     ← baseline (200行)
│   ├── temporal_claude-sonnet-cot.csv ← CoT (40行)
│   ├── temporal_claude-sonnet-rag.csv
│   ├── temporal_claude-sonnet-rag_cot.csv
│   ├── temporal_gemini.csv
│   └── ...
│
├── 标准对齐/
│   ├── claude-sonnet/          ← baseline (400行 × 8文件)
│   ├── claude-sonnet-cot/      ← CoT (80行 × 8文件)
│   ├── claude-sonnet-rag/
│   ├── claude-sonnet-rag_cot/
│   └── ...
│
└── results/
    ├── final_summary.json          ← 所有模型综合得分排名
    ├── rag_cot_summary.json        ← 各模型策略对比结果
    └── claude-sonnet_eval.json     ← Claude 单模型详细评测
```

---

## 本地法规文件

```
E:\论文\跨境对齐\各国法规和部分数据集\
├── 中国\法规\（海关总署公告、国家法律等）
├── 日本\法规\（食品卫生法、厚生劳动省告示等）
├── 韩国\法规\（进口食品申报规定等）
├── 美国\法规\（联邦法规 CFR 15/19卷等）
└── 德法\法规\（EU 食品标签/添加剂/新型食品法规等）
```

RAG 索引文件：
```
E:\论文\跨境对齐\评测阶段\rag_index.json
```
（由 `build_rag_index.py` 生成，约 5~20 MB）
