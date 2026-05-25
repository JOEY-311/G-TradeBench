# 时序推理能力

判断食品法规的时效状态：是草案还是已生效？生效日期是什么时候？涉及什么物质？

---

## 任务本质

给定一段描述某项食品法规变动的文字或问题，模型需要回答三件事：

1. **法律状态（F1）**：这条法规是"已提议/草案"还是"已生效/正式实施"？
2. **涉及对象（F2）**：是什么物质/添加剂/限量在被修改？变化方向是什么？
3. **生效日期（F3）**：具体何时生效？如果不确定，要如何表述？

---

## 关键判断词汇

### 草案状态（Proposed）
```
proposed, draft, under consideration, open for comment,
consultation, not yet enacted, to be enacted, seeking comment,
NPRM（美国预先制定规则通知）, 征求意见稿, 草案
```

### 已生效状态（Enacted）
```
in effect, effective, enacted, adopted, promulgated,
notification of update, immediately effective, officially,
has been, shall apply from, 已实施, 公告发布, 正式生效
```

---

## 各监管机构的流程特征

| 机构 | 草案阶段 | 生效阶段 |
|------|---------|---------|
| FDA（美国）| NPRM 发布 | Final Rule 发布 |
| EU | EFSA Opinion → 委员会提案 | Commission Regulation 发布 |
| 日本厚生劳动省 | 意见征集 | 官报（官報）告示即生效 |
| Codex CAC | Step 3/5/8 草案 | Plenary 全体会议通过 |
| 中国 NHC/SAMR | 征求意见稿 | 公告发布并注明实施日期 |

---

## Ground Truth 格式

```
[Legal Status]: Proposed / Notification of Update / Immediately Effective / ...
[Proposed Effective Date]: 2024-01 / Not specified / After a certain transition period / NaN
[Target Details]: 物质名称 + 变化内容（如"将亚硝酸钠在腌制肉类中的限量从150降至100 ppm"）
```

---

## 三个评测维度

### F1 — 法律状态识别（规则打分）

```python
_PROPOSAL_KW = ['proposed', 'draft', 'under consideration', 'open for comment',
                'consultation', 'not yet', 'to be enacted', 'seeking comment']
_ENACTED_KW  = ['in effect', 'effective', 'enacted', 'adopted', 'promulgated',
                'notification of update', 'immediately', 'officially', 'has been']

def score_f1(gt, output):
    status = gt['legal_status'].lower()
    o = output.lower()
    is_proposed = any(k in status for k in ['proposed', 'draft'])
    is_enacted  = any(k in status for k in ['notification of update',
                                              'immediately effective', 'enacted'])
    if is_proposed: return 1 if any(k in o for k in _PROPOSAL_KW) else 0
    if is_enacted:  return 1 if any(k in o for k in _ENACTED_KW)  else 0
    return 1 if any(k in o for k in _ENACTED_KW + _PROPOSAL_KW) else 0
```

### F2 — 物质与变化识别（LLM-as-Judge）

规则无法判断，需要 LLM judge：

```python
F2_PROMPT = """
Question: {question}
Key info: {target_details}
Model answer: {answer}

Score 1 if the model correctly identified both the target substance AND the type of change, else 0.
Output JSON only: {{"score": 0 or 1, "reason": "brief"}}
"""
```

### F3 — 日期准确性（规则打分）

```python
_YEAR_PAT  = re.compile(r'\b(20\d{2})\b')
_MONTH_PAT = re.compile(r'\b(january|...|december)\b', re.I)
_NA_GT     = re.compile(r'^(nan|not applicable|n/a|none)\s*$', re.I)
_VAGUE_GT  = re.compile(r'after a certain period|not specified|pending|tbd', re.I)

def score_f3(gt, output):
    gd = gt['effective_date'].strip()
    o = output.lower()
    # GT 本身就是"无日期"时，模型不应该编造日期
    if _NA_GT.match(gd):
        return 0 if (_YEAR_PAT.search(output) or _MONTH_PAT.search(output)) else 1
    # GT 是模糊表述时，模型应该也表述模糊
    if _VAGUE_GT.search(gd):
        return 1 if re.search(r'after|period|transition|unclear|uncertain|pending', o) else 0
    # GT 有具体年份时，模型输出中必须包含相同年份
    gy = set(_YEAR_PAT.findall(gd))
    if gy: return 1 if gy & set(_YEAR_PAT.findall(output)) else 0
    return 0
```

---

## 综合得分

```
F_norm = (F1 + F2 + F3) / 3 × 100%
```

---

## Prompt 设计

这是英文任务，系统 prompt 和用户 prompt 都应使用英文：

```python
SYSTEM = ("You are a professional assistant specializing in food regulatory "
          "analysis and temporal reasoning. Provide accurate, concise, and "
          "logically consistent answers.")

USER = "Please provide a detailed answer in English:\n\n{instruction}"
```

---

## 注意事项

- **RAG 对本任务可能有负面影响**：如果 RAG 注入的是中文法规文档，
  会干扰模型对英文问题的推理，实测 F2 从 95% 跌至 75%。
  如果要用 RAG，应只注入**英文**法规文档，且内容需与问题匹配。
- **F3 普遍偏低**：模型普遍难以准确提取日期，约 45% 左右，
  这是本任务的固有难度，不代表模型质量差。
- **CoT 对 F1/F2 有帮助**：让模型先分析法律状态再作答，F1 可从 90% 提升至 100%。
