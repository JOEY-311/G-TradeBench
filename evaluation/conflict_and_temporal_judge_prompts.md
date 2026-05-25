# 规则冲突判断 & 标准对齐（规则更新）任务
# LLM-as-Judge Prompt 模板

---

# 任务一：规则冲突判断

## 数据集特征说明

## 阶段一：STANDARD 解析脚本

```python
import re

def parse_standard(standard_text: str) -> list[dict]:
    """
    将 STANDARD 字段解析为评分点列表。
    兼容三种格式：
      A: "准确识别法规冲突得 3 分。"
      B: "准确识别法规冲突（3 分）"
      C: "**准确识别合规决策选项**（5 分）：答出退运处理得 2 分，提及特批得 1 分..." （带嵌套子项）
    """
    items = []

    # 格式 A：「...得 X 分」
    pattern_a = re.compile(r'[-•]\s*(.+?)[，。]?得\s*(\d+)\s*分[。\n]?', re.MULTILINE)
    # 格式 B/C：「...（X 分）」，可能后跟子项说明
    pattern_b = re.compile(r'[-•*]\s*\*{0,2}(.+?)\*{0,2}[（(](\d+)\s*分[）)]([^。\n]*)', re.MULTILINE)

    matched_a = [(m.group(1).strip(), int(m.group(2)), '', m.start()) for m in pattern_a.finditer(standard_text)]
    matched_b = [(m.group(1).strip(), int(m.group(2)), m.group(3).strip(), m.start()) for m in pattern_b.finditer(standard_text)]

    # 合并去重（按位置排序，避免同一行被两个 pattern 重复匹配）
    all_matches = sorted(matched_a + matched_b, key=lambda x: x[3])
    seen = set()
    deduped = []
    for desc, score, sub_text, pos in all_matches:
        if not any(abs(pos - p) < 15 for p in seen):
            deduped.append((desc, score, sub_text, pos))
            seen.add(pos)

    # 陷阱关键词
    trap_kw = ['陷阱', '关键', '隐含', '易忽略', '特批', '不明显', '实际上', '注意']

    for desc, score, sub_text, _ in deduped:
        is_trap = any(kw in desc for kw in trap_kw)

        # 解析嵌套子项（如「答出退运得2分，提及特批得1分」）
        sub_items = []
        if sub_text:
            subs = re.findall(r'(?:答出|提及|提出|识别|引用|说明|建议)(.{2,20}?)得\s*(\d+)\s*分', sub_text)
            sub_items = [(s.strip(), int(v)) for s, v in subs]

        items.append({
            "description": desc,
            "max_score": score,
            "is_trap": is_trap,
            "sub_items": sub_items,   # [] 表示无子项，直接整体评分
        })

    return items


def get_total_score(standard_text: str, parsed_items: list[dict]) -> int:
    """提取总分；若无「总分 X 分」则对各项求和。"""
    m = re.search(r'总分\s*(\d+)\s*分', standard_text)
    if m:
        return int(m.group(1))
    return sum(it["max_score"] for it in parsed_items)


def build_scoring_items_json(parsed_items: list[dict]) -> str:
    """生成注入 User Prompt 的评分点 JSON 字符串。"""
    import json
    return json.dumps(parsed_items, ensure_ascii=False, indent=2)
```

---

## 阶段二：System Prompt

```
你是一位食品进出口合规领域的资深专家，同时担任大模型评测裁判（judge）。

你的任务是：逐一检查「模型输出」是否覆盖了给定的评分要点，并给出得分和原文证据。

【评分原则】
1. 以「是否覆盖评分点的实质内容」为唯一标准，不要求措辞与要点完全相同。
2. 每个评分点必须引用模型输出的原文片段（≤30字）作为证据；若未覆盖，evidence 填 null。
3. 有嵌套子项的评分点，按子项逐一判断，最终得分为命中子项分值之和，不超过该点满分。
4. 「陷阱」要点（is_trap=true）须更严格：模型需主动、明确点出陷阱的本质，一笔带过不得分。
5. 仅使用离散分：无子项时用 0 / 半分（向下取整）/ 满分；有子项时按子项累加。
   半分仅适用于满分 ≥ 3 的要点（覆盖但论述不充分）。
6. 不因格式、篇幅、语气影响评分；参考答案仅供理解要点含义，不要求输出与之一致。
7. 输出严格遵守指定 JSON 格式，不添加任何 JSON 之外的文字。
```

---

## 阶段二：User Prompt 模板

```
## 评分任务

### 题目背景
{{QUESTION}}

### 参考答案（仅供理解评分点含义，不要求模型输出与之一致）
{{REFERENCE}}

### 模型输出（待评分）
{{OUTPUT}}

---

## 评分点列表（JSON）

以下是从本题评分标准解析出的所有评分点，请逐一判断模型输出是否覆盖。

{{SCORING_ITEMS_JSON}}

---

## 评分规则

**普通要点**（sub_items 为空列表）：
- `full`：明确、具体地覆盖了该要点 → 得 max_score 分
- `partial`：有所涉及但论述不充分（仅适用于 max_score ≥ 3 的要点）→ 得 floor(max_score / 2) 分
- `none`：未覆盖，或覆盖内容与要点矛盾 → 得 0 分

**有嵌套子项的要点**（sub_items 非空）：
- 对每个子项独立判断 covered（true/false）
- score = 命中子项分值之和，不超过 max_score

**陷阱要点**（is_trap = true）：
- 模型需主动点出陷阱的本质，才可得分；仅在行文中附带提及视为 none

---

## 输出格式（严格 JSON，不得有额外文字）

{
  "item_scores": [
    {
      "description": "<原评分点描述>",
      "max_score": <整数>,
      "is_trap": <true|false>,
      "coverage": "<full|partial|none>",
      "score": <实际得分，整数或 .5>,
      "evidence": "<引用模型输出原文，≤30字；未覆盖则为 null>",
      "reason": "<评分依据，≤50字>",
      "sub_item_results": [
        {"description": "<子项描述>", "max_score": <整数>, "covered": <true|false>}
      ]
    }
  ],
  "raw_total": <各项 score 之和>,
  "max_possible": <各项 max_score 之和>,
  "normalized_score": <raw_total / max_possible * 100，保留一位小数>,
  "trap_items_total": <is_trap=true 的项目数>,
  "trap_items_hit": <is_trap=true 且 score>0 的项目数>
}
```

---

## 完整调用代码（Python）

```python
import json, re, math
import pandas as pd
import anthropic
from collections import Counter

client = anthropic.Anthropic()

SYSTEM_PROMPT = """你是一位食品进出口合规领域的资深专家，同时担任大模型评测裁判（judge）。

你的任务是：逐一检查「模型输出」是否覆盖了给定的评分要点，并给出得分和原文证据。

【评分原则】
1. 以「是否覆盖评分点的实质内容」为唯一标准，不要求措辞与要点完全相同。
2. 每个评分点必须引用模型输出的原文片段（≤30字）作为证据；若未覆盖，evidence 填 null。
3. 有嵌套子项的评分点，按子项逐一判断，最终得分为命中子项分值之和，不超过该点满分。
4. 「陷阱」要点（is_trap=true）须更严格：模型需主动、明确点出陷阱的本质，一笔带过不得分。
5. 仅使用离散分：无子项时用 0 / 半分（向下取整）/ 满分；有子项时按子项累加。
   半分仅适用于满分 ≥ 3 的要点（覆盖但论述不充分）。
6. 不因格式、篇幅、语气影响评分；参考答案仅供理解要点含义，不要求输出与之一致。
7. 输出严格遵守指定 JSON 格式，不添加任何 JSON 之外的文字。"""

USER_TEMPLATE = """## 评分任务

### 题目背景
{question}

### 参考答案（仅供理解评分点含义，不要求模型输出与之一致）
{reference}

### 模型输出（待评分）
{output}

---

## 评分点列表（JSON）

以下是从本题评分标准解析出的所有评分点，请逐一判断模型输出是否覆盖。

{scoring_items_json}

---

## 评分规则

**普通要点**（sub_items 为空列表）：
- `full`：明确、具体地覆盖了该要点 → 得 max_score 分
- `partial`：有所涉及但论述不充分（仅适用于 max_score ≥ 3 的要点）→ 得 floor(max_score / 2) 分
- `none`：未覆盖，或覆盖内容与要点矛盾 → 得 0 分

**有嵌套子项的要点**（sub_items 非空）：
- 对每个子项独立判断 covered（true/false）
- score = 命中子项分值之和，不超过 max_score

**陷阱要点**（is_trap = true）：
- 模型需主动点出陷阱的本质，才可得分；仅在行文中附带提及视为 none

## 输出格式（严格 JSON，不得有额外文字）

{{
  "item_scores": [
    {{
      "description": "<原评分点描述>",
      "max_score": <整数>,
      "is_trap": <true|false>,
      "coverage": "<full|partial|none>",
      "score": <实际得分>,
      "evidence": "<引用模型输出原文，≤30字；未覆盖则为 null>",
      "reason": "<评分依据，≤50字>",
      "sub_item_results": []
    }}
  ],
  "raw_total": <各项 score 之和>,
  "max_possible": <各项 max_score 之和>,
  "normalized_score": <raw_total / max_possible * 100，保留一位小数>,
  "trap_items_total": <is_trap=true 的项目数>,
  "trap_items_hit": <is_trap=true 且 score>0 的项目数>
}}"""


def judge_single(question: str, reference: str, output: str,
                 scoring_items: list[dict]) -> dict:
    items_json = json.dumps(scoring_items, ensure_ascii=False, indent=2)
    user_prompt = USER_TEMPLATE.format(
        question=question,
        reference=reference,
        output=output,
        scoring_items_json=items_json,
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


def batch_judge(df: pd.DataFrame, n_repeat: int = 2) -> pd.DataFrame:
    """
    对每条样本运行 n_repeat 次取众数，降低 judge 随机方差。
    """
    results = []
    for _, row in df.iterrows():
        std = row["STANDARD"]
        items = parse_standard(std)
        max_possible = get_total_score(std, items)

        run_results = []
        for _ in range(n_repeat):
            try:
                res = judge_single(
                    question=row["QUESTION"],
                    reference=row["REFERENCE"],
                    output=row["OUTPUT"],
                    scoring_items=items,
                )
                run_results.append(res)
            except Exception as e:
                print(f"Judge error: {e}")

        if not run_results:
            continue

        # 取 raw_total 的众数
        raw_totals = [r["raw_total"] for r in run_results]
        best_raw = Counter(raw_totals).most_common(1)[0][0]

        # 取与众数 raw_total 最接近的那次结果作为代表（保留详细 item_scores）
        best_run = min(run_results, key=lambda r: abs(r["raw_total"] - best_raw))

        results.append({
            "row_index": row.name,
            "raw_total": best_raw,
            "max_possible": max_possible,
            "normalized_score": round(best_raw / max_possible * 100, 1) if max_possible > 0 else 0,
            "trap_total": best_run.get("trap_items_total", 0),
            "trap_hit": best_run.get("trap_items_hit", 0),
            "detail": best_run,
        })

    result_df = pd.DataFrame(results)
    print("=== 汇总统计 ===")
    print(f"平均标准化得分: {result_df['normalized_score'].mean():.1f}")
    print(f"陷阱命中率: {result_df['trap_hit'].sum() / max(result_df['trap_total'].sum(), 1) * 100:.1f}%")
    return result_df
```

---

## 评分汇总指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| 标准化均分 | `mean(raw/max_possible × 100)` | 跨题可比的主指标 |
| 陷阱命中率 | `Σtrap_hit / Σtrap_total` | 模型识别隐含冲突的能力 |
| ≥70分通过率 | `normalized_score ≥ 70` 的样本比例 | 类 acc. 辅助指标 |
| 法规引用率 | item 含「引用具体法规」且得分>0 的比例 | 格式规范性指标 |

---
---

# 任务二：标准对齐（规则更新）

## 数据集特征说明

| 特征 | 说明 |
|------|------|
| 样本量 | 200 条 |
| GT 结构 | 两种字段组合（见下） |
| 语言 | 指令和输出均为英文 |

**GT 字段组合：**

| 类型 | 字段 | 占比 |
|------|------|------|
| 类型 A（规则更新类） | `[Legal Status]` + `[Proposed Effective Date]` + `[Target Details]` | 80%（160条）|
| 类型 B（法规修订类） | `[Core Fact]` + `[Effective Date]` + `[Key Content]` | 20%（40条）|

**Legal Status 取值分布：**
- `Notification of Update`（127条）
- `Proposed/Draft Stage`（33条）
- `Regulatory amendment`（仅出现在类型 B 的 Core Fact 字段）
- `Revocation of authorization/Amending use condition`（3条）

**Effective Date 典型值：**
- 精确日期：`2021-10-06`、`2024-02-05 00:00:00`
- 未知/不适用：`nan`、`Not applicable`、`To be determined`
- 模糊说明：`Amendments of the MRLs will enter into force at the date of publication`、`These proposed standards will take effect after a certain period of grace`

---

## 阶段一：GT 结构解析脚本

```python
import re

def parse_ground_truth(gt_text: str) -> dict:
    """
    解析 GT 字段为结构化字典，兼容类型 A 和类型 B。
    返回统一键名：
      - gt_type: "A" | "B"
      - legal_status: str（类型A的[Legal Status] 或 类型B的[Core Fact]）
      - date: str（类型A的[Proposed Effective Date] 或 类型B的[Effective Date]）
      - target_details: str（类型A的[Target Details] 或 类型B的[Key Content]）
    """
    fields = {}
    for m in re.finditer(r'\[([^\]]+)\]:\s*([\s\S]+?)(?=\n\[|$)', gt_text.strip()):
        fields[m.group(1).strip()] = m.group(2).strip()

    if 'Legal Status' in fields:
        return {
            "gt_type": "A",
            "legal_status": fields.get("Legal Status", "").rstrip('.'),
            "date": _normalize_date(fields.get("Proposed Effective Date", "")),
            "target_details": fields.get("Target Details", ""),
        }
    else:
        return {
            "gt_type": "B",
            "legal_status": fields.get("Core Fact", "").rstrip('.'),
            "date": _normalize_date(fields.get("Effective Date", "")),
            "target_details": fields.get("Key Content", ""),
        }


def _normalize_date(raw: str) -> str:
    """统一日期格式，处理 nan / Not applicable / 模糊说明。"""
    raw = raw.strip().rstrip('.')
    if not raw or raw.lower() in ('nan', 'not applicable', 'n/a', 'to be determined'):
        return "NOT_SPECIFIED"
    # 提取 YYYY-MM-DD
    m = re.search(r'\d{4}-\d{2}-\d{2}', raw)
    if m:
        return m.group(0)
    # 模糊说明（如「after a certain period」）归为 VAGUE
    if re.search(r'period|publication|determined|upon', raw, re.I):
        return "VAGUE"
    return raw
```

---

## 阶段二：System Prompt

```
You are an expert evaluator specializing in international food safety regulations and regulatory document analysis.

Your task is to score a model's response against a ground-truth answer across three dimensions: Legal Status, Effective Date, and Target Details/Key Content.

[Scoring Principles]
1. Judge correctness based on substance, not exact wording. Synonymous expressions count as correct.
2. For Legal Status: recognize that "Notification of Update", "proposed regulation", "final rule", and "regulatory amendment" are distinct categories — do not treat them as interchangeable.
3. For Effective Date: "NOT_SPECIFIED" (nan/not applicable/to be determined) is a valid GT value. If GT is NOT_SPECIFIED but the model provides a specific date, deduct points. If the model correctly states the date is unknown/unspecified, award full marks.
4. For Target Details: focus on whether the model identifies the correct regulated substance/entity and the nature of the change (addition/amendment/revocation). Extra elaboration does not earn bonus points, but core omissions lose points.
5. Cite the specific model output phrase (≤30 words) that supports each score. If absent, set evidence to null.
6. Output strictly in the specified JSON format — no text outside the JSON.
```

---

## 阶段二：User Prompt 模板

```
## Scoring Task

### Instruction (given to the model)
{{INSTRUCTION}}

### Ground-Truth Answer
- GT Type: {{GT_TYPE}}  (A = regulatory update type | B = regulatory amendment type)
- Legal Status / Core Fact: {{GT_LEGAL_STATUS}}
- Effective Date: {{GT_DATE}}
- Target Details / Key Content:
{{GT_TARGET_DETAILS}}

### Model Output (to be scored)
{{MODEL_OUTPUT}}

---

## Scoring Rubric

### F1 — Legal Status / Document Type (3 points)

Assess whether the model correctly identifies the document type.

Reference categories:
- "Notification of Update": an informational notice that an existing rule has been updated or amended, with no new rulemaking process
- "Proposed/Draft Stage": a proposed rule, open for public comment, not yet in effect
- "Regulatory amendment" (Type B only): a final rule that has already been enacted

| Score | Criterion |
|-------|-----------|
| 3 | Correct category AND explains the textual/contextual basis (e.g., "the title contains 'Proposed'", "this is a final rule as indicated by...") |
| 1 | Correct category but no supporting basis given; OR correct direction (proposed vs. enacted) but wrong sub-category label |
| 0 | Wrong category (e.g., classifies a proposal as immediately effective, or a final rule as proposed) |

---

### F2 — Target Substance / Key Content (4 points)

Assess whether the model correctly identifies the regulated subject and the nature of the change.

| Score | Criterion |
|-------|-----------|
| 4 | Correct substance/entity name AND correct change type (new listing / amendment / revocation / exemption) |
| 2 | Substance name correct but change type vague or missing; OR change type correct but substance too generic (e.g., "a food additive" instead of the specific name) |
| 0 | Substance name incorrect, or answer is so vague it provides no actionable information |

---

### F3 — Effective Date (3 points)

Assess whether the model correctly extracts or characterizes the effective date.

Ground-truth date value: {{GT_DATE}}

| GT Value | Full marks (3 pts) | Partial (1 pt) | Zero (0 pts) |
|----------|--------------------|----------------|--------------|
| Exact date (e.g., 2021-10-06) | Model gives the exact date or correct year-month | Correct year only | Wrong date or absent |
| NOT_SPECIFIED (nan/not applicable) | Model explicitly states date is unspecified/not applicable/TBD | States date is unclear but still guesses a specific date | States a specific wrong date; or ignores the date entirely |
| VAGUE (e.g., "after a certain period") | Model accurately characterizes the vague condition (e.g., "takes effect upon publication") | Acknowledges uncertainty but doesn't capture the condition | States a specific date or ignores entirely |

---

## Output Format (strict JSON, no text outside)

{
  "F1": {
    "score": <0|1|3>,
    "evidence": "<model output phrase ≤30 words, or null>",
    "reason": "<scoring rationale ≤50 words>"
  },
  "F2": {
    "score": <0|2|4>,
    "substance_identified": "<substance/entity name the model gave, or null>",
    "change_type_identified": "<change type the model gave, or null>",
    "evidence": "<model output phrase ≤30 words, or null>",
    "reason": "<scoring rationale ≤50 words>"
  },
  "F3": {
    "score": <0|1|3>,
    "date_extracted": "<date/characterization the model gave, or null>",
    "gt_date": "{{GT_DATE}}",
    "evidence": "<model output phrase ≤30 words, or null>",
    "reason": "<scoring rationale ≤50 words>"
  },
  "total": <F1+F2+F3>,
  "normalized_score": <total/10*100，保留一位小数>
}
```

---

## 完整调用代码（Python）

```python
import json, re
import pandas as pd
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are an expert evaluator specializing in international food safety regulations and regulatory document analysis.

Your task is to score a model's response against a ground-truth answer across three dimensions: Legal Status, Effective Date, and Target Details/Key Content.

[Scoring Principles]
1. Judge correctness based on substance, not exact wording. Synonymous expressions count as correct.
2. For Legal Status: recognize that "Notification of Update", "proposed regulation", "final rule", and "regulatory amendment" are distinct categories — do not treat them as interchangeable.
3. For Effective Date: "NOT_SPECIFIED" (nan/not applicable/to be determined) is a valid GT value. If GT is NOT_SPECIFIED but the model provides a specific date, deduct points. If the model correctly states the date is unknown/unspecified, award full marks.
4. For Target Details: focus on whether the model identifies the correct regulated substance/entity and the nature of the change. Extra elaboration does not earn bonus points, but core omissions lose points.
5. Cite the specific model output phrase (≤30 words) that supports each score. If absent, set evidence to null.
6. Output strictly in the specified JSON format — no text outside the JSON."""

USER_TEMPLATE = """## Scoring Task

### Instruction (given to the model)
{instruction}

### Ground-Truth Answer
- GT Type: {gt_type}  (A = regulatory update type | B = regulatory amendment type)
- Legal Status / Core Fact: {gt_legal_status}
- Effective Date: {gt_date}
- Target Details / Key Content:
{gt_target_details}

### Model Output (to be scored)
{model_output}

---

## Scoring Rubric

### F1 — Legal Status / Document Type (3 points)

Reference categories:
- "Notification of Update": an informational notice that an existing rule has been updated, no new rulemaking
- "Proposed/Draft Stage": a proposed rule, open for public comment, not yet in effect
- "Regulatory amendment" (Type B only): a final rule already enacted

| Score | Criterion |
|-------|-----------|
| 3 | Correct category AND explains the basis |
| 1 | Correct category but no basis; OR correct direction but wrong sub-label |
| 0 | Wrong category |

### F2 — Target Substance / Key Content (4 points)

| Score | Criterion |
|-------|-----------|
| 4 | Correct substance/entity AND correct change type |
| 2 | One of the two correct, the other missing or too vague |
| 0 | Substance incorrect or answer too vague to be actionable |

### F3 — Effective Date (3 points)

GT Date value: {gt_date}

| GT Value | 3 pts | 1 pt | 0 pts |
|----------|-------|------|-------|
| Exact date | Exact date or correct year-month | Correct year only | Wrong or absent |
| NOT_SPECIFIED | Explicitly states unspecified/not applicable | Acknowledges unclear but guesses | States specific wrong date or ignores |
| VAGUE | Accurately characterizes the condition | Acknowledges uncertainty only | States specific date or ignores |

## Output Format (strict JSON only)

{{
  "F1": {{
    "score": <0|1|3>,
    "evidence": "<phrase ≤30 words or null>",
    "reason": "<≤50 words>"
  }},
  "F2": {{
    "score": <0|2|4>,
    "substance_identified": "<or null>",
    "change_type_identified": "<or null>",
    "evidence": "<phrase ≤30 words or null>",
    "reason": "<≤50 words>"
  }},
  "F3": {{
    "score": <0|1|3>,
    "date_extracted": "<or null>",
    "gt_date": "{gt_date}",
    "evidence": "<phrase ≤30 words or null>",
    "reason": "<≤50 words>"
  }},
  "total": <F1+F2+F3>,
  "normalized_score": <total/10*100, one decimal>
}}"""


def judge_single_temporal(instruction: str, gt: dict, model_output: str) -> dict:
    user_prompt = USER_TEMPLATE.format(
        instruction=instruction,
        gt_type=gt["gt_type"],
        gt_legal_status=gt["legal_status"],
        gt_date=gt["date"],
        gt_target_details=gt["target_details"],
        model_output=model_output,
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


def batch_judge_temporal(df: pd.DataFrame, n_repeat: int = 2) -> pd.DataFrame:
    from collections import Counter
    results = []
    for _, row in df.iterrows():
        gt = parse_ground_truth(row["ground_truth"])
        run_results = []
        for _ in range(n_repeat):
            try:
                res = judge_single_temporal(
                    instruction=row["instruction"],
                    gt=gt,
                    model_output=row["model_output"],
                )
                run_results.append(res)
            except Exception as e:
                print(f"Judge error at index {row.get('index', '?')}: {e}")

        if not run_results:
            continue

        # 取 total 的众数
        totals = [r["total"] for r in run_results]
        best_total = Counter(totals).most_common(1)[0][0]
        best_run = min(run_results, key=lambda r: abs(r["total"] - best_total))

        results.append({
            "index": row.get("index"),
            "gt_type": gt["gt_type"],
            "F1": best_run["F1"]["score"],
            "F2": best_run["F2"]["score"],
            "F3": best_run["F3"]["score"],
            "total": best_total,
            "normalized_score": round(best_total / 10 * 100, 1),
            "detail": best_run,
        })

    result_df = pd.DataFrame(results)

    print("=== 汇总统计 ===")
    print(f"整体标准化均分: {result_df['normalized_score'].mean():.1f}")
    print(f"F1 均分: {result_df['F1'].mean():.2f} / 3")
    print(f"F2 均分: {result_df['F2'].mean():.2f} / 4")
    print(f"F3 均分: {result_df['F3'].mean():.2f} / 3")
    print()
    for gt_type in ['A', 'B']:
        sub = result_df[result_df['gt_type'] == gt_type]
        if len(sub):
            print(f"GT 类型 {gt_type}（{len(sub)}条）: 均分 {sub['normalized_score'].mean():.1f}")

    return result_df
```

---

## 评分汇总指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| 整体标准化均分 | `mean(total/10 × 100)` | 主指标，跨样本可比 |
| F1 均分 | `mean(F1_score)` | 文件类型判断能力 |
| F2 均分 | `mean(F2_score)` | 目标物质 / 内容理解能力（最难） |
| F3 均分 | `mean(F3_score)` | 日期抽取能力（可半自动化） |
| 强通过率 | `total ≥ 8` 的样本比例 | 类 acc. 主指标 |
| GT 类型分拆 | 按 A/B 分别统计均分 | 诊断模型对不同文档类型的表现差异 |

---

## F3 半自动化建议

F3（日期抽取）逻辑明确，可以用规则脚本先打初稿分，再让 judge 只处理无法规则判断的边界情况（如模糊说明），可节省约 30% 的 judge token 用量：

```python
import re

def rule_score_f3(model_output: str, gt_date: str) -> tuple[int | None, str]:
    """
    规则打初稿分。返回 (score, reason)，score=None 表示需交给 judge 处理。
    """
    # 从模型输出提取日期
    date_match = re.search(r'\b(\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}|'
                           r'(?:January|February|March|April|May|June|July|August|'
                           r'September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
                           model_output, re.I)
    extracted = date_match.group(0) if date_match else None

    # GT 为 NOT_SPECIFIED
    if gt_date == "NOT_SPECIFIED":
        not_specified_phrases = re.compile(
            r'not\s+(?:specified|applicable|available|stated|provided|determined)|'
            r'no\s+(?:specific\s+)?(?:date|effective date)|'
            r'(?:date\s+is\s+)?(?:unknown|unspecified|tbd|n/a)',
            re.I
        )
        if not_specified_phrases.search(model_output) and not extracted:
            return 3, "Model correctly states date is unspecified"
        if extracted:
            return 0, f"GT is NOT_SPECIFIED but model gave date: {extracted}"
        return None, "Ambiguous — needs judge"

    # GT 为精确日期
    if re.match(r'\d{4}-\d{2}-\d{2}', gt_date):
        if extracted:
            norm_extracted = re.sub(r'[/.]', '-', extracted)
            if norm_extracted[:10] == gt_date[:10]:
                return 3, "Exact date match"
            if norm_extracted[:7] == gt_date[:7]:
                return 1, "Correct year-month, wrong day"
            if norm_extracted[:4] == gt_date[:4]:
                return 1, "Correct year only"
            return 0, f"Wrong date: {extracted} vs GT {gt_date}"
        return 0, "No date extracted, GT has exact date"

    # GT 为 VAGUE — 交给 judge
    return None, "VAGUE date — needs judge"
```
