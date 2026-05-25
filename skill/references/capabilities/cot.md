# CoT 增强能力

通过 Chain-of-Thought 提示让模型在给出答案之前先进行分步推理，提升复杂判断的准确性。

---

## 核心思路

CoT 的价值在于：强迫模型"先想清楚再开口"。
对于需要多步逻辑（识别食品类别 → 查限量 → 比较数值 → 综合判断）的任务，
直接要求结论往往比分步推理准确率低。

---

## 三类任务的 CoT 模板

### 合规判断 CoT

```
请严格按以下步骤分析，完成每步后再给出结论：
步骤1：识别食品类别及主要合规风险点
步骤2：逐项核查铅(Pb)、镉(Cd)、农药残留的实测值与【目标国】限量差距（注明数值）
步骤3：判断各指标状态——合规(<85%限量) / 边界风险(≥85%且≤100%) / 超标(>限量)
步骤4：综合判断

完成以上步骤后，最后必须另起一行，严格输出以下格式之一（不得附加任何文字）：
审查结论：违规
审查结论：存在违规风险
审查结论：建议放行
```

### 时序推理 CoT

```
Please analyze step by step before answering:
Step 1 - Legal Status: Is it Proposed/Draft or Enacted/In Effect? Look for status keywords.
Step 2 - Target Substance & Change: What substance/additive/limit is being changed?
Step 3 - Effective Date: Extract exact date if stated; acknowledge uncertainty if vague.
Step 4 - Final Answer: Provide clear structured answer based on Steps 1-3.
```

### 场景问答 CoT

```
请严格按以下步骤进行分析：
步骤1：准确识别核心合规问题（是什么问题，涉及哪些国家/法规体系）
步骤2：引用具体适用法规（含编号，如GB 2762-2022、EU Regulation No 1333/2008）
步骤3：逐步推导解决方案关键行动项（可操作性优先）
步骤4：给出最终专业建议（可含替代方案）
```

---

## CoT 输出的结论提取（关键！）

**CoT 与普通模式的提取方式完全不同。**

普通模式：模型第一行就输出结论 → 取第一个"审查结论："
CoT 模式：模型先输出推理链，结论在最后 → 取**最后一个**"审查结论："

```python
all_matches = list(re.finditer(r'审查结论[：:]\s*(.{2,20})', text))
if all_matches:
    m = all_matches[-1]   # CoT：取最后一个
    # m = all_matches[0]  # 非 CoT：取第一个
```

如果全文没有"审查结论："标记，从输出**末尾 5 行**中找关键词：

```python
lines = [l.strip() for l in text.split('\n') if l.strip()]
for line in reversed(lines[-5:]):
    if '存在违规风险' in line: return '存在违规风险'
    if '违规' in line and '风险' not in line: return '违规'
    if '合规' in line or '放行' in line: return '合规'
```

---

## 各任务的 CoT 效果

| 任务 | CoT 是否推荐 | 实测效果 |
|------|------------|---------|
| 合规判断 | ⚠️ 需谨慎 | 推理质量提升，但结论提取失败率约 50%，综合准确率反而下降 |
| 时序推理 | ✅ 推荐 | F1 从 90% → 100%，F2 从 95% → 97.5% |
| 场景问答 | ✅ 轻微推荐 | D2（法规引用）小幅提升，总体变化不大 |

---

## max_tokens 设置

CoT 会产生更长的输出，需要相应增加 token 上限：

| 任务 | 普通模式 | CoT 模式 |
|------|---------|---------|
| 合规判断 | 100~150 | 800~1200 |
| 时序推理 | 400~500 | 700~800 |
| 场景问答 | 1500~2000 | 2000（不变，已足够）|

---

## 注意事项

- **合规 CoT 的格式约束极其重要**：如果不在最后明确要求"必须另起一行输出审查结论："，
  模型很可能将结论嵌入正文而不是单独输出，导致提取失败。
- **不要在 CoT 结论行后加任何内容**：模型有时会在结论行后补充"综上所述..."，
  这会干扰提取。如果发现这个问题，在 prompt 中加"结论行之后不得附加任何文字"。
- **token 成本**：CoT 每次调用的 token 消耗约是普通模式的 4~8 倍，批量任务时注意成本控制。
