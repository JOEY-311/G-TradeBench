# 合规审查任务 — LLM-as-Judge Prompt 模板

---

## 使用说明

本模板用于对「澳韩/澳中等跨境合规审查」任务的模型输出进行自动打分。  
评分模型（judge）需要具备基本的食品法规理解能力，推荐使用 Claude Sonnet / GPT-4o 等。  
每条样本独立调用一次，输出结构化 JSON。

---

## System Prompt

```
你是一位食品进出口合规领域的资深评审专家，同时担任大模型评测裁判（judge）。
你的任务是：根据提供的「食品档案」和「模型输出」，按照评分细则对模型输出进行客观评分。

【评分原则】
1. 只根据「食品档案」中的数值和「目标国法规限量」进行事实核查，不凭主观印象打分。
2. 每个维度必须给出评分依据，引用模型输出的原文片段作为证据。
3. 如果模型输出的法规数值与你所知有出入，以模型输出为待评对象，说明其是否与目标国实际法规相符。
4. 不因语言风格、篇幅长短影响评分；只看信息准确性与完整性。
5. 输出严格按照指定 JSON 格式，不添加任何额外说明文字。
```

---

## User Prompt 模板

> 将 `{{...}}` 替换为实际数据后使用。

```
## 评分任务

### 食品档案（输入信息）
{{FOOD_PROFILE}}

### 出口目标国
{{DESTINATION}}（如：韩国、中国）

### 模型输出（待评分）
{{REVIEW_OUTPUT}}

---

## 评分细则

请按以下 4 个维度逐一评分，总分 10 分。

---

### D1 — 结论类别准确性（满分 3 分）

判断模型给出的最终结论（违规 / 存在违规风险 / 合规）是否与食品档案数值一致。

| 分值 | 标准 |
|------|------|
| 3 | 结论分类正确，且与所有指标的实测值/限量值比对结果完全一致 |
| 1 | 结论大方向正确，但严重程度分类有偏差（如将明确超标判为"存在风险"，或反之） |
| 0 | 结论与数值事实相反（将超标项判为合规，或将合规项判为违规） |

评分时请注意：
- "违规"指实测值已超过目标国限量值；
- "存在违规风险"指实测值接近限量但未超标，或因法规适用性不确定存在通关风险；
- 若档案中无任何超标/风险指标，正确结论应为"合规"。

---

### D2 — 法规引用正确性（满分 3 分）

检查模型引用的目标国限量值是否准确，且与具体食品品类匹配。

| 分值 | 标准 |
|------|------|
| 3 | 每项超标指标均引用了正确的限量值，且品类描述与产品分类匹配 |
| 2 | 限量值正确，但品类描述有一处不准确或过于笼统 |
| 1 | 限量值有部分错误，或引用了不适用该品类的标准 |
| 0 | 法规数值明显错误，或完全未引用任何限量值 |

评分时请注意：
- 如果模型未明确写出限量值但结论正确，最多给 1 分；
- 农药残留肯定列表（PLS）的默认限值 0.01 mg/kg 属于通用规则，引用正确可得分；
- 对于你无法确认真实限量值的指标，请在备注中标注"无法核实"，该指标不扣分也不加分。

---

### D3 — 违规点完整性（满分 2 分）

检查食品档案中所有超标/风险指标是否均被识别并说明。

首先，请你根据食品档案中的实测数据，列出你认为应被识别的超标/风险指标清单（作为参考基准）。

| 分值 | 标准 |
|------|------|
| 2 | 全部应识别的超标/风险指标均有对应说明，无遗漏 |
| 1 | 遗漏 1 个超标/风险指标，或有指标被一笔带过而非显式识别 |
| 0 | 遗漏 2 个及以上超标/风险指标，或仅说结论未列明具体指标 |

---

### D4 — 推理链可追溯性（满分 2 分）

检查每条违规/风险理由是否包含完整的「实测值 → 限量值 → 对比结论」三要素。

| 分值 | 标准 |
|------|------|
| 2 | 每条理由均明确写出实测值、限量值，并说明超出量或超出比例 |
| 1 | 含实测值和结论，但省略了限量值；或含限量值但未写实测值 |
| 0 | 仅给出结论性语言，无任何数值支撑 |

---

## 输出格式

请严格按照以下 JSON 格式输出，不要添加任何 JSON 之外的文字：

{
  "D1": {
    "score": <0|1|3>,
    "evidence": "<引用模型输出的原文片段，20字以内>",
    "reason": "<评分依据，说明为何给此分值>"
  },
  "D2": {
    "score": <0|1|2|3>,
    "evidence": "<引用模型输出的原文片段，20字以内>",
    "reason": "<评分依据>",
    "unverifiable_items": ["<无法核实限量值的指标名称，若无则为空列表>"]
  },
  "D3": {
    "reference_violations": ["<你识别的应被覆盖的超标/风险指标列表>"],
    "score": <0|1|2>,
    "missed_items": ["<模型遗漏的指标，若无则为空列表>"],
    "reason": "<评分依据>"
  },
  "D4": {
    "score": <0|1|2>,
    "weakest_item": "<推理链最薄弱的那条违规理由的简短描述，若全部完整则填 null>",
    "reason": "<评分依据>"
  },
  "total": <D1+D2+D3+D4之和>,
  "pass": <true（total>=6 且 D1>=3 且 D2>=2）| false>
}
```

---

## 调用示例（Python）

```python
import json
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """你是一位食品进出口合规领域的资深评审专家，同时担任大模型评测裁判（judge）。
你的任务是：根据提供的「食品档案」和「模型输出」，按照评分细则对模型输出进行客观评分。

【评分原则】
1. 只根据「食品档案」中的数值和「目标国法规限量」进行事实核查，不凭主观印象打分。
2. 每个维度必须给出评分依据，引用模型输出的原文片段作为证据。
3. 如果模型输出的法规数值与你所知有出入，以模型输出为待评对象，说明其是否与目标国实际法规相符。
4. 不因语言风格、篇幅长短影响评分；只看信息准确性与完整性。
5. 输出严格按照指定 JSON 格式，不添加任何额外说明文字。"""

USER_TEMPLATE = """## 评分任务

### 食品档案（输入信息）
{food_profile}

### 出口目标国
{destination}

### 模型输出（待评分）
{review_output}

---

## 评分细则
[此处插入完整评分细则，同上文 User Prompt 模板中的内容]
"""

def judge_single(food_profile: str, destination: str, review_output: str) -> dict:
    user_prompt = USER_TEMPLATE.format(
        food_profile=food_profile,
        destination=destination,
        review_output=review_output,
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = response.content[0].text.strip()
    # 去掉可能的 markdown 代码块标记
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def batch_judge(df, n_repeat: int = 2):
    """
    对 DataFrame 中每条样本运行 n_repeat 次 judge，
    取各维度得分的众数/均值作为最终得分（降低 judge 自身方差）。
    """
    import pandas as pd
    from collections import Counter

    results = []
    for _, row in df.iterrows():
        scores_list = []
        for _ in range(n_repeat):
            try:
                result = judge_single(
                    food_profile=row["FOOD_PROFILE"],
                    destination=row["DESTINATION"],
                    review_output=row["REVIEW_OUTPUT"],
                )
                scores_list.append(result)
            except Exception as e:
                print(f"Judge error for ID {row.get('ID', '?')}: {e}")

        if not scores_list:
            continue

        # 取各维度得分的众数
        final = {"ID": row.get("ID")}
        for dim in ["D1", "D2", "D3", "D4"]:
            dim_scores = [s[dim]["score"] for s in scores_list]
            final[f"{dim}_score"] = Counter(dim_scores).most_common(1)[0][0]
        final["total"] = sum(final[f"{d}_score"] for d in ["D1", "D2", "D3", "D4"])
        final["pass"] = (
            final["total"] >= 6
            and final["D1_score"] >= 3
            and final["D2_score"] >= 2
        )
        # 保留首次运行的详细理由供人工复核
        final["detail"] = scores_list[0]
        results.append(final)

    return pd.DataFrame(results)
```

---

## 通过门槛说明

| 指标 | 阈值 | 含义 |
|------|------|------|
| `pass = true` | total ≥ 6 **且** D1 ≥ 3 **且** D2 ≥ 2 | 强通过：结论正确、法规引用无误 |
| 通过率 | pass 样本数 / 总样本数 | 类 acc. 主指标 |
| 维度均分 | 各维度得分均值 | 诊断模型短板用 |

> **注意**：通过门槛要求 D1 和 D2 同时达标，原因是这两项直接影响合规结论的可信度；D3/D4 失分代表输出质量问题，但不构成「错误答案」。

---

## 人工抽验说明

建议对自动评分结果随机抽取 10% 做人工复核，计算以下指标：

- **Spearman ρ**（人工总分 vs judge 总分）：目标 ≥ 0.80
- **D1 维度一致率**：目标 ≥ 90%（结论分类最关键）
- **D2 维度一致率**：目标 ≥ 80%（法规数值核实难度较高）

若一致率不达标，优先检查 judge 的 system prompt 中「法规引用正确性」部分是否需要补充目标国法规背景知识。
