# 模型调用能力

通过 OpenRouter 统一接口调用任意 LLM，处理批量任务的通用模式。

---

## API 配置

```python
from openai import OpenAI
client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key='sk-or-v1-...'   # 从环境变量读取更安全
)
```

常用模型 ID（OpenRouter 格式）：
- `anthropic/claude-sonnet-4-5`
- `google/gemini-2.0-flash-001`
- `deepseek/deepseek-chat`
- `openai/gpt-4o`

---

## 标准调用模式

```python
def call_api(messages, max_tokens=300, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,        # 评测任务用 0，保证可复现
                max_tokens=max_tokens
            )
            return (resp.choices[0].message.content or '').strip()
        except Exception as e:
            wait = 5 * (attempt + 1)
            if attempt < retries - 1:
                time.sleep(wait)
    return ''
```

**max_tokens 参考值：**
| 输出类型 | 建议值 |
|---------|--------|
| 单行结论（如"违规"） | 100–150 |
| 简短分析 | 300–500 |
| 带 CoT 的推理链 | 800–1200 |
| 详细专业回答 | 1500–2000 |

---

## 断点续传模式

大批量任务必须支持中断恢复，核心思路：**先记录已处理的 key，跳过重复**。

```python
import csv
from pathlib import Path

def init_csv(path, headers):
    """初始化输出文件，返回已处理 key 集合。"""
    done = set()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(headers)
    else:
        with open(path, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                done.add(r.get(headers[0], ''))
    return done

def append_row(path, row):
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(row)
```

使用方式：
```python
done = init_csv(out_path, ['ID', 'INPUT', 'OUTPUT'])
for item in dataset:
    if item['id'] in done:
        continue   # 跳过已处理
    output = call_api(...)
    append_row(out_path, [item['id'], item['input'], output])
```

---

## 输出清洗

模型经常输出 Markdown 格式，需要清洗：

```python
import re

def strip_md(text):
    text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
```

---

## 速率控制

OpenRouter 对不同模型有不同限速，安全做法：

```python
import random, time
time.sleep(random.uniform(0.8, 1.5))   # 每次调用后随机等待
```

遇到 429（Rate Limit）时指数退避：
```python
except Exception as e:
    if '429' in str(e):
        time.sleep(2 ** attempt * 5)
```

---

## 固定子集抽样（策略对比时关键）

当需要对同一个数据集用多种策略（baseline/RAG/CoT）对比时，
必须保证每种策略处理的是**完全相同的子集**：

```python
import random

def sample_rows(rows, pct=0.2, seed=99):
    rng = random.Random(seed)   # 独立 RNG，不影响全局状态
    n = max(1, round(len(rows) * pct))
    return rng.sample(rows, min(n, len(rows)))
```

所有策略使用相同的 `pct` 和 `seed`，即可保证子集一致。

---

## LLM-as-Judge 调用

当模型输出是开放式文本（无法用规则打分）时，用另一个 LLM 来打分。
这本质上就是一次普通的 `call_api`，但有几个特殊要求：

```python
def llm_judge(system_prompt, user_prompt, judge_model='google/gemini-2.0-flash-001', retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_prompt}
                ],
                temperature=0,
                max_tokens=150        # judge 只输出 JSON，不需要长文本
            )
            raw = r.choices[0].message.content or ''
            m = re.search(r'\{.*\}', raw, re.S)
            if m:
                return json.loads(m.group())
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {}
```

**关键设计要求：**
- **用不同模型当 judge**：不要用被评测的同一个模型（自评偏高）
- **要求 JSON 输出**：`{"score": 1, "reason": "brief"}` 方便解析
- **评分标准要具体**：写进 system prompt，避免歧义
- **截断过长输入**：问题前 500 字、参考答案前 700 字、模型输出前 1000 字
- **规则能解决的不要用 judge**：有确定 GT 的任务直接字符串比对更准确
