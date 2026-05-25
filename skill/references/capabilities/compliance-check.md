# 合规判断能力

对食品档案进行合规性审查：判断污染物是否超标、给出违规结论。

---

## 核心逻辑

合规判断的本质是**将食品的实测值与目标国限量标准做比较**：

```
实测值 > 限量           → 违规
实测值 ∈ [85%限量, 限量] → 存在违规风险（边界区间）
实测值 < 85%限量        → 合规 / 建议放行
```

85% 阈值的意义：考虑检测误差，实测值接近限量时即视为有风险。

---

## 各国限量参考值

```python
LIMITS = {
    '日本': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '韩国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '法国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},   # EU 标准
    '德国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},   # EU 标准
    '中国': {'铅': 0.2,  '镉': 0.1,  '农药': 0.01},
    '美国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '澳大利亚': {'铅': 0.1, '镉': 0.05, '农药': 0.01},
}
RISK_RATIO = 0.85
```

> 注：以上为通用基准值，特定食品品类（如大米、婴儿食品）可能有更严格的限量，
> 需要根据具体食品类别核查目标国法规原文。

---

## 从食品档案提取污染物实测值

```python
import re

POLLUTANT_PAT = {
    '铅':   re.compile(r'铅[^\d\n]{0,30}[：:]\s*([\d.]+)\s*mg/kg', re.I),
    '镉':   re.compile(r'镉[^\d\n]{0,30}[：:]\s*([\d.]+)\s*mg/kg', re.I),
    '农药': re.compile(r'(?:综合农药残留|农药残留)[^\d\n]{0,40}[：:]\s*([\d.]+)\s*mg/kg', re.I),
}

def extract_pollutants(profile: str) -> dict:
    return {k: float(p.search(profile).group(1))
            for k, p in POLLUTANT_PAT.items() if p.search(profile)}
```

**只有包含实测值的食品档案才能推导 GT**（通常是初级农产品数据集，
如澳洲原始农产品）。加工食品档案一般没有精确数值，无法规则推导 GT。

---

## 推导 Ground Truth

```python
def derive_gt(poll: dict, dest: str):
    lims = LIMITS.get(dest)
    if not poll or not lims:
        return None   # 无法计算 GT
    over = [n for n, v in poll.items() if lims.get(n) and v > lims[n]]
    risk = [n for n, v in poll.items()
            if lims.get(n) and not v > lims[n] and v >= lims[n] * RISK_RATIO]
    if over:  return '违规'
    if risk:  return '存在违规风险'
    return '合规'
```

---

## Prompt 设计要点

合规审查 prompt 需要特别注意**输出格式的严格性**：

```
你现在是一位资深的跨国食品合规与通关专家。
现有一批食品计划从【{origin}】出口到【{dest}】。
请严格根据【{dest}】的食品安全国家标准审查以下食品档案。

【拟进口食品档案】：
{profile}

输出格式（严格遵守，仅输出一行）：
审查结论：违规  或  审查结论：存在违规风险  或  审查结论：建议放行
```

**关键约束：**
- 结论只能三选一，不允许其他表述
- 要求"仅输出一行"减少无关内容
- 加入 RAG 时，在档案前插入法规参考块

---

## 从模型输出提取结论

### 普通模式（非 CoT）

取第一个"审查结论："：

```python
def extract_verdict(text, is_cot=False):
    all_matches = list(re.finditer(r'审查结论[：:]\s*(.{2,20})', text))
    if all_matches:
        m = all_matches[-1] if is_cot else all_matches[0]
        v = m.group(1).strip('。，\n ')
        if '存在违规风险' in v: return '存在违规风险'
        if '违规' in v and '风险' not in v and '存在' not in v: return '违规'
        if '合规' in v or '放行' in v: return '合规'
    # 兜底：从首/尾行找关键词
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    search = list(reversed(lines[-5:])) if is_cot else lines[:3]
    for line in search:
        if '存在违规风险' in line: return '存在违规风险'
        if '违规' in line and '风险' not in line: return '违规'
        if '合规' in line or '放行' in line: return '合规'
    return '无法识别'
```

### CoT 模式（重要！）

CoT 输出包含推理链，结论在**最后**。必须：
1. 搜索全文中**所有** "审查结论：" 的出现位置
2. 取**最后一个**（`all_matches[-1]`）

CoT prompt 末尾必须加这段话，否则模型很可能不输出标准格式：
```
完成以上步骤后，最后必须另起一行，严格输出以下格式之一（不得附加任何文字）：
审查结论：违规
审查结论：存在违规风险
审查结论：建议放行
```

---

## 注意事项

- **RLHF 偏好问题**：部分模型（如 grok、gpt-4）倾向于输出"存在违规风险"而非"违规"，
  即使污染物超标 99 倍也不会明确说"违规"。这是模型的对齐偏见，不是评测错误。
- **CoT 提取失败率**：实测 Claude Sonnet CoT 输出中约 50% 未包含标准格式的"审查结论："，
  导致大量"无法识别"。需要在 prompt 中强化格式要求。
- **GT 可计算范围**：只有含污染物实测数据的档案才能算 GT，其他档案无法做规则评测。
