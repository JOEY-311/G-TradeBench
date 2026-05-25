# 数据格式说明

---

## 合规审查输出 CSV

列名：`ID, ORIGIN, DESTINATION, FOOD_PROFILE, REVIEW_OUTPUT`

| 列 | 说明 |
|----|------|
| `ID` | 食品档案唯一编号 |
| `ORIGIN` | 出口国（中文，如"澳大利亚"） |
| `DESTINATION` | 进口目标国（中文，如"日本"） |
| `FOOD_PROFILE` | 完整食品档案文本，包含食品类别、成分、污染物实测值等 |
| `REVIEW_OUTPUT` | 模型输出，理想格式："审查结论：违规/存在违规风险/建议放行" |

**文件命名规则：**
- baseline：`{SOURCE}_{MODEL}_{路线}.csv`，如 `AUS_claude-sonnet_澳日.csv`
- 策略：`{SOURCE}_{MODEL}-{策略}_{路线}.csv`，如 `AUS_claude-sonnet-rag_澳日.csv`

---

## 规则更新输出 CSV

列名：`index, instruction, ground_truth, model_output`

| 列 | 说明 |
|----|------|
| `index` | 题目序号（字符串，用于断点续传匹配） |
| `instruction` | 英文问题文本 |
| `ground_truth` | 结构化 GT，包含三个字段（见下） |
| `model_output` | 模型的英文回答 |

**ground_truth 格式：**
```
[Legal Status]: Proposed / Notification of Update / Immediately Effective
[Proposed Effective Date]: 2024-03 / Not specified / NaN
[Target Details]: 描述涉及的物质和变化内容
```

**输入 JSONL 格式（`temporal_eval_en.jsonl`）：**
```json
{"instruction": "问题文本", "answer": "GT文本（同上格式）"}
```

---

## 标准对齐输出 CSV

列名：`QUESTION, REFERENCE, OUTPUT, STANDARD`

| 列 | 说明 |
|----|------|
| `QUESTION` | 业务场景题目（作为断点续传的 key） |
| `REFERENCE` | 参考答案（评测时给 judge 看） |
| `OUTPUT` | 模型回答 |
| `STANDARD` | 评分标准描述（评测时给 judge 看） |

**文件命名规则：**
- baseline：`{MODEL}_{任务类型}.csv`，如 `claude-sonnet_HSCODE.csv`
- 策略：`{MODEL}-{策略}_{任务类型}.csv`，如 `claude-sonnet-rag_限量对齐.csv`

---

## 食品档案（FOOD_PROFILE）字段示例

```
食品类别：畜禽肉及其制品
商品名称：冷冻羊肉片
产地：澳大利亚
生产日期：2024-03-15
保质期：24个月

检测结果：
铅(Pb)含量：0.085 mg/kg
镉(Cd)含量：0.041 mg/kg
综合农药残留：0.008 mg/kg

配料：羊肉100%
添加剂：无
包装形式：真空包装
净重：500g
```

其中污染物字段（铅/镉/农药）的提取正则见 `compliance-check.md`。

---

## 策略实验的抽样一致性

所有策略（cot/rag/rag_cot）从 baseline 数据抽取 **20%** 子集，
使用固定种子 `seed=99`，保证三种策略处理完全相同的样本。

| 任务 | baseline 行数 | 策略子集行数 |
|------|-------------|------------|
| 合规审查（每条路线）| ~50 | ~10 |
| 规则更新 | 200 | 40 |
| 标准对齐（每类任务）| 50 | 10 |
