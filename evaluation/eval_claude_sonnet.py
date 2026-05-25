#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Sonnet 评测脚本（自包含）
三个任务全部评测，结果写入 results/claude-sonnet_eval.json
并追加到 results/final_summary.json
用法：
  set OPENROUTER_API_KEY=sk-or-...
  python eval_claude_sonnet.py
  python eval_claude_sonnet.py --no-llm   # 仅合规审查（跳过LLM judge）
"""

import csv, json, os, re, sys, time
from pathlib import Path
from openai import OpenAI

# ════════════════════════════════════════════════════════════════
#  配置
# ════════════════════════════════════════════════════════════════
API_KEY     = os.environ.get('OPENROUTER_API_KEY',
              os.environ.get('OPENROUTER_API_KEY', ''))
JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'google/gemini-2.0-flash-001')
NO_LLM      = '--no-llm' in sys.argv

EVAL  = Path('E:/论文/跨境对齐/评测阶段')
OUT   = EVAL / 'results'
OUT.mkdir(exist_ok=True)

# 数据路径
COMP_DIR  = EVAL / '合规审查/claude-sonnet'
TEMP_CSV  = EVAL / '规则更新/temporal_claude-sonnet.csv'
ALIGN_DIR = EVAL / '标准对齐/claude-sonnet'

client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=API_KEY)


def read_csv(path: Path) -> list[dict]:
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin-1'):
        try:
            with open(path, encoding=enc) as f:
                rows = list(csv.DictReader(f))
            if rows:
                return rows
        except Exception:
            continue
    return []


def section(title: str):
    print(f'\n{"─"*60}')
    print(f'  {title}')
    print(f'{"─"*60}')


# ════════════════════════════════════════════════════════════════
#  任务 1：合规审查（规则打分）
# ════════════════════════════════════════════════════════════════
LIMITS = {
    '日本': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '韩国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '法国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '中国': {'铅': 0.2,  '镉': 0.1,  '农药': 0.01},
    '美国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '德国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
}
RISK_RATIO = 0.85

POLLUTANT_PAT = {
    '铅':   re.compile(r'铅[^\d\n]{0,30}[：:]\s*([\d.]+)\s*mg/kg', re.I),
    '镉':   re.compile(r'镉[^\d\n]{0,30}[：:]\s*([\d.]+)\s*mg/kg', re.I),
    '农药': re.compile(r'(?:综合农药残留|农药残留)[^\d\n]{0,40}[：:]\s*([\d.]+)\s*mg/kg', re.I),
}


def extract_pollutants(profile: str) -> dict:
    return {k: float(p.search(profile).group(1))
            for k, p in POLLUTANT_PAT.items() if p.search(profile)}


def derive_gt(poll: dict, dest: str):
    lims = LIMITS.get(dest)
    if not poll or not lims:
        return None
    over = [n for n, v in poll.items() if lims.get(n) and v > lims[n]]
    risk = [n for n, v in poll.items()
            if lims.get(n) and not v > lims[n] and v >= lims[n] * RISK_RATIO]
    if over:  return '违规'
    if risk:  return '存在违规风险'
    return '合规'


def extract_verdict(text: str) -> str:
    m = re.search(r'审查结论[：:]\s*(.{2,15})', text)
    if m:
        v = m.group(1).strip('。，\n ')
        if '存在违规风险' in v or ('存在' in v and '风险' in v): return '存在违规风险'
        if '违规' in v and '风险' not in v and '存在' not in v:  return '违规'
        if '合规' in v or '放行' in v:                           return '合规'
        return v[:10]
    first = text.split('\n')[0][:40]
    if '存在违规风险' in first: return '存在违规风险'
    if '违规' in first and '风险' not in first: return '违规'
    if '合规' in first or '放行' in first:      return '合规'
    return '无法识别'


# ── LLM-judge：EU/USA 添加剂与成分合规评测 ──────────────────────
ADDITIVE_JUDGE_SYS = """\
你是跨境食品进口合规专家，专注于添加剂和成分合规审查。
你的任务是：根据食品档案中的添加剂、配料信息，判断该食品进口到目标国的合规结论是否正确。

各目标国关键标准：
【中国】GB 2760-2014 食品添加剂使用标准（正面清单）；GB 7718 标签；
        未列入清单的添加剂视为禁用；苯甲酸及其钠盐（E211）在饮料中≤0.2g/kg。
【日本】食品添加剂正面清单（厚生劳动省告示）；未收录即禁用；
        奎拉亚提取物（E999/Quillaja）未获批准；合成着色剂需逐一审核。
【韩国】MFDS 食品添加剂标准（식품첨가물공전）；正面清单制度；
        苯甲酸钠（E211）在碳酸饮料中≤0.6g/kg；部分EU添加剂未获批准。

判断逻辑：
- 违规：有添加剂明确禁用，或用量超标
- 存在违规风险：添加剂在目标国合规状态不明确，或限量值接近上限
- 建议放行：所有可见添加剂和成分符合目标国规定

输出 JSON（仅输出 JSON，不附加其他文字）：
{"gt_verdict": "违规/存在违规风险/建议放行", "model_correct": 0或1, "reason": "简短说明"}
"""

ADDITIVE_JUDGE_TMPL = """\
【目标进口国】：{dest}
【食品档案】：
{profile}

【模型输出的审查结论】：{verdict}

请判断模型结论是否正确，并给出标准答案（gt_verdict）。
"""

def judge_additive_compliance(profile: str, dest: str, verdict: str,
                               retries: int = 3) -> tuple[str, int]:
    """
    用 LLM-judge 评测 EU/USA 档案的添加剂/成分合规结论。
    返回 (gt_verdict, is_correct: 0/1)；judge 失败返回 ('无法判断', -1)。
    """
    if NO_LLM:
        return '无法判断', -1
    prompt = ADDITIVE_JUDGE_TMPL.format(
        dest=dest,
        profile=profile[:1200],   # 截取前1200字，避免超token
        verdict=verdict,
    )
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role': 'system', 'content': ADDITIVE_JUDGE_SYS},
                          {'role': 'user',   'content': prompt}],
                temperature=0, max_tokens=150,
            )
            raw = r.choices[0].message.content or ''
            m = re.search(r'\{.*\}', raw, re.S)
            if m:
                obj = json.loads(m.group())
                gt  = obj.get('gt_verdict', '').strip()
                ok  = int(obj.get('model_correct', 0))
                return gt, ok
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return '无法判断', -1


def eval_compliance() -> dict:
    section('合规审查评测')
    if not COMP_DIR.is_dir():
        print('  目录不存在，跳过。')
        return {}

    # AUS：规则推导 GT；EU/USA：LLM-judge 推导 GT
    aus_match, eu_usa_match = [], []
    file_summaries = []

    for fp in sorted(COMP_DIR.iterdir()):
        if fp.suffix != '.csv':
            continue
        rows = read_csv(fp)
        if not rows:
            continue

        # 用文件名前缀区分数据来源（更可靠）
        fname = fp.name.upper()
        is_aus = fname.startswith('AUS')
        data_type = 'AUS' if is_aus else 'EU_USA'

        gt_computable, correct = 0, 0
        verdict_dist: dict = {}
        judge_failures = 0

        for i, r in enumerate(rows, 1):
            dest    = r.get('DESTINATION', '').strip()
            profile = r.get('FOOD_PROFILE', '')
            mv      = extract_verdict(r.get('REVIEW_OUTPUT', ''))
            verdict_dist[mv] = verdict_dist.get(mv, 0) + 1

            if is_aus:
                # ── AUS：规则推导 GT ──────────────────────────────
                poll = extract_pollutants(profile)
                gt   = derive_gt(poll, dest)
                if gt is not None:
                    gt_computable += 1
                    hit = (mv == gt)
                    correct += int(hit)
                    aus_match.append(hit)

            else:
                # ── EU/USA：LLM-judge 评测添加剂/成分合规 ──────────
                if NO_LLM:
                    continue
                gt, ok = judge_additive_compliance(profile, dest, mv)
                if ok == -1:
                    judge_failures += 1
                    continue
                gt_computable += 1
                correct += ok
                eu_usa_match.append(bool(ok))
                print(f'  [{fp.name}] [{i:>3}/{len(rows)}] '
                      f'GT={gt}  模型={mv}  {"✓" if ok else "✗"}', end='\r')
                time.sleep(0.3)   # 避免 judge 限速

        acc = correct / gt_computable * 100 if gt_computable else None
        summary_entry = {
            'file': fp.name, 'data_type': data_type,
            'total': len(rows), 'verdict_dist': verdict_dist,
            'gt_computable': gt_computable, 'correct': correct,
            'accuracy_pct': round(acc, 2) if acc is not None else None,
        }
        if not is_aus and judge_failures:
            summary_entry['judge_failures'] = judge_failures
        file_summaries.append(summary_entry)

        status = f'{correct}/{gt_computable} = {acc:.1f}%' if acc is not None else '跳过（--no-llm）'
        tag    = '[规则]' if is_aus else '[judge]'
        print(f'  {tag} {fp.name:<38} {status}')

    # ── 汇总 ────────────────────────────────────────────────────
    all_match  = aus_match + eu_usa_match
    if not all_match:
        print('  无可计算准确率的样本。')
        return {}

    aus_acc    = sum(aus_match)    / len(aus_match)    * 100 if aus_match    else None
    eu_usa_acc = sum(eu_usa_match) / len(eu_usa_match) * 100 if eu_usa_match else None
    overall    = sum(all_match)    / len(all_match)    * 100

    print(f'\n  AUS  规则准确率: '
          f'{aus_acc:.1f}%  ({sum(aus_match)}/{len(aus_match)})' if aus_acc is not None
          else '\n  AUS  无数据')
    if eu_usa_acc is not None:
        print(f'  EU/USA judge准确率: '
              f'{eu_usa_acc:.1f}%  ({sum(eu_usa_match)}/{len(eu_usa_match)})')
    print(f'  总体准确率: {overall:.1f}%  ({sum(all_match)}/{len(all_match)})')

    return {
        'accuracy_pct':       round(overall, 2),
        'aus_accuracy_pct':   round(aus_acc, 2)    if aus_acc    is not None else None,
        'eu_usa_accuracy_pct':round(eu_usa_acc, 2) if eu_usa_acc is not None else None,
        'total':   len(all_match),
        'correct': sum(all_match),
        'file_summaries': file_summaries,
    }


# ════════════════════════════════════════════════════════════════
#  任务 2：规则更新（F1/F3规则 + F2 LLM-as-Judge）
# ════════════════════════════════════════════════════════════════
_PROPOSAL_KW = ['proposed', 'draft', 'under consideration', 'open for comment',
                'consultation', 'not yet', 'to be enacted', 'seeking comment']
_ENACTED_KW  = ['in effect', 'effective', 'enacted', 'adopted', 'promulgated',
                'notification of update', 'immediately', 'officially', 'has been']
_YEAR_PAT    = re.compile(r'\b(20\d{2})\b')
_MONTH_PAT   = re.compile(
    r'\b(january|february|march|april|may|june|july|august'
    r'|september|october|november|december)\b', re.I)
_VAGUE_GT    = re.compile(
    r'after a certain period|not specified|vague|upon|transition|once finalized'
    r'|pending|to be determined|tbd|certain period', re.I)
_NA_GT       = re.compile(r'^(nan|not applicable|n/a|na|none)\s*\.?\s*$', re.I)


def parse_gt(gt_text: str) -> dict:
    def _get(tag):
        m = re.search(rf'\[{tag}\]:\s*(.+?)(?=\n\[|\Z)', gt_text, re.S)
        return m.group(1).strip() if m else ''
    return {'legal_status':   _get('Legal Status'),
            'effective_date': _get('Proposed Effective Date'),
            'target_details': _get('Target Details')}


def score_f1(gt: dict, out: str) -> int:
    status = gt['legal_status'].lower(); o = out.lower()
    is_prop = any(k in status for k in ['proposed', 'draft'])
    is_enac = any(k in status for k in ['notification of update',
                                         'immediately effective', 'enacted'])
    if is_prop: return 1 if any(k in o for k in _PROPOSAL_KW) else 0
    if is_enac: return 1 if any(k in o for k in _ENACTED_KW)  else 0
    return 1 if any(k in o for k in _ENACTED_KW + _PROPOSAL_KW) else 0


def score_f3(gt: dict, out: str) -> int:
    gd = gt['effective_date'].strip(); o = out.lower()
    if _NA_GT.match(gd):
        return 0 if (_YEAR_PAT.search(out) or _MONTH_PAT.search(out)) else 1
    if _VAGUE_GT.search(gd):
        return 1 if re.search(
            r'after|period|certain|transition|once|following|upon|unclear|uncertain|pending', o) else 0
    gy = set(_YEAR_PAT.findall(gd))
    if gy: return 1 if gy & set(_YEAR_PAT.findall(out)) else 0
    gm = set(_MONTH_PAT.findall(gd.lower()))
    return 1 if (gm and gm & set(_MONTH_PAT.findall(o))) else 0


F2_SYS  = ('You are a food regulation expert reviewer. Judge whether the model '
            'correctly identified the target substance and change type. '
            'Output JSON only: {"score": 0 or 1, "reason": "brief"}')
F2_TMPL = ('Question: {q}\nKey info: {details}\nModel answer: {ans}\n'
           'Score 1 if substance + change type correct, else 0. JSON only.')


def llm_f2(q: str, gt: dict, out: str, retries: int = 3) -> int:
    if NO_LLM:
        return 0
    prompt = F2_TMPL.format(q=q[:600], details=gt['target_details'][:300], ans=out[:800])
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role': 'system', 'content': F2_SYS},
                           {'role': 'user',   'content': prompt}],
                temperature=0, max_tokens=100)
            raw = r.choices[0].message.content or ''
            m   = re.search(r'\{.*\}', raw, re.S)
            if m:
                return int(json.loads(m.group()).get('score', 0))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return 0


def eval_temporal() -> dict:
    section('规则更新评测（F1/F3规则 + F2 LLM-as-Judge）')
    if not TEMP_CSV.exists():
        print('  文件不存在，跳过。')
        return {}

    rows = read_csv(TEMP_CSV)
    if not rows:
        print('  文件为空，跳过。')
        return {}

    print(f'  共 {len(rows)} 条，开始评测...')
    f1s, f2s, f3s = [], [], []

    for i, row in enumerate(rows, 1):
        gt  = parse_gt(row.get('ground_truth', ''))
        q   = row.get('instruction', '')
        out = row.get('model_output', '')
        f1  = score_f1(gt, out)
        f2  = llm_f2(q, gt, out)
        f3  = score_f3(gt, out)
        f1s.append(f1); f2s.append(f2); f3s.append(f3)
        print(f'  [{i:>3}/{len(rows)}] F1={f1} F2={f2} F3={f3}', end='\r')

    n  = len(rows)
    a1 = sum(f1s) / n; a2 = sum(f2s) / n; a3 = sum(f3s) / n
    norm = (a1 + a2 + a3) / 3 * 100
    print(f'\n  F1={a1*100:.1f}%  F2={a2*100:.1f}%  F3={a3*100:.1f}%  综合={norm:.1f}%')
    return {'n': n,
            'F1_avg': round(a1, 4), 'F2_avg': round(a2, 4), 'F3_avg': round(a3, 4),
            'F1_pct': round(a1*100, 2), 'F2_pct': round(a2*100, 2), 'F3_pct': round(a3*100, 2),
            'normalized_pct': round(norm, 2)}


# ════════════════════════════════════════════════════════════════
#  任务 3：标准对齐（D1/D2/D3 LLM-as-Judge）
# ════════════════════════════════════════════════════════════════
ALIGN_SYS  = '你是食品法规专业评审专家，请严格按评分标准输出JSON，不要添加解释。'
ALIGN_TMPL = """\
## 任务类型：{task_type}
## 题目：{question}
## 参考答案：{reference}
## 模型回答：{output}

D1 问题识别（0/1）：核心合规问题是否识别正确
D2 法规依据（0/2）：法规引用是否正确且具体（2=完整，1=部分，0=无/错）
D3 方案覆盖（0/1）：解决方案是否涵盖参考答案主要行动项

输出：{{"D1":<0或1>,"D2":<0/1/2>,"D3":<0或1>}}"""

TASK_KW = [('HSCODE', 'HSCODE'), ('冲突判断', '冲突判断'), ('限量对齐', '限量对齐'),
           ('配料准入', '配料准入'), ('准入程序', '准入程序'), ('标签对齐', '标签对齐'),
           ('多国流通', '多国流通'), ('俗名映射', '俗名映射')]


def infer_task(fname: str) -> str:
    for kw, name in TASK_KW:
        if kw in fname:
            return name
    return '未知'


def llm_d123(task: str, q: str, ref: str, out: str, retries: int = 3):
    if NO_LLM:
        return 0, 0, 0
    prompt = ALIGN_TMPL.format(task_type=task, question=q[:500],
                                reference=ref[:700], output=out[:1000])
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role': 'system', 'content': ALIGN_SYS},
                           {'role': 'user',   'content': prompt}],
                temperature=0, max_tokens=100)
            raw = r.choices[0].message.content or ''
            m   = re.search(r'\{.*\}', raw, re.S)
            if m:
                obj = json.loads(m.group())
                return (min(max(int(obj.get('D1', 0)), 0), 1),
                        min(max(int(obj.get('D2', 0)), 0), 2),
                        min(max(int(obj.get('D3', 0)), 0), 1))
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return 0, 0, 0


def eval_alignment() -> dict:
    section('标准对齐评测（D1/D2/D3 LLM-as-Judge）')
    if not ALIGN_DIR.is_dir():
        print('  目录不存在，跳过。')
        return {}

    d1s, d2s, d3s = [], [], []
    by_task: dict = {}

    for fp in sorted(ALIGN_DIR.iterdir()):
        if fp.suffix != '.csv':
            continue
        rows = read_csv(fp)
        if not rows or 'OUTPUT' not in rows[0]:
            continue
        task = infer_task(fp.name)
        td1, td2, td3 = [], [], []

        for i, row in enumerate(rows, 1):
            q   = row.get('QUESTION',  '')
            ref = row.get('REFERENCE', '')
            out = row.get('OUTPUT',    '')
            d1, d2, d3 = llm_d123(task, q, ref, out)
            d1s.append(d1); d2s.append(d2); d3s.append(d3)
            td1.append(d1); td2.append(d2); td3.append(d3)
            print(f'  [{fp.name}] [{i:>2}/{len(rows)}] D1={d1} D2={d2} D3={d3}', end='\r')

        tn = len(rows)
        by_task[task] = {
            'n': tn,
            'D1': round(sum(td1)/tn, 4), 'D2': round(sum(td2)/tn, 4),
            'D3': round(sum(td3)/tn, 4),
            'avg_score': round((sum(td1)+sum(td2)+sum(td3))/tn, 4),
        }
        print(f'\n  {task:<12} D1={sum(td1)/tn*100:.1f}% '
              f'D2={sum(td2)/tn/2*100:.1f}% D3={sum(td3)/tn*100:.1f}%')

    if not d1s:
        print('  无有效数据，跳过。')
        return {}

    n  = len(d1s)
    a1 = sum(d1s) / n; a2 = sum(d2s) / n; a3 = sum(d3s) / n
    norm = (a1 + a2 + a3) / 4 * 100   # 满分4分归一化
    print(f'\n  D1={a1*100:.1f}%  D2_norm={a2/2*100:.1f}%  D3={a3*100:.1f}%  综合={norm:.1f}%')
    return {'n': n,
            'D1_avg': round(a1, 4), 'D2_avg': round(a2, 4), 'D3_avg': round(a3, 4),
            'D1_pct': round(a1*100, 2), 'D2_pct': round(a2/2*100, 2), 'D3_pct': round(a3*100, 2),
            'normalized_pct': round(norm, 2),
            'by_task': by_task}


# ════════════════════════════════════════════════════════════════
#  汇总 & 写入
# ════════════════════════════════════════════════════════════════
def main():
    print('╔══════════════════════════════════════════════════════════╗')
    print('║         Claude Sonnet 全流程评测                         ║')
    print(f'║  Judge: {JUDGE_MODEL:<49}║')
    print(f'║  LLM评分: {"已跳过" if NO_LLM else "启用"}' + ' '*51 + '║')
    print('╚══════════════════════════════════════════════════════════╝')

    comp_res  = eval_compliance()
    temp_res  = eval_temporal()   if not NO_LLM else {}
    align_res = eval_alignment()  if not NO_LLM else {}

    # ── 合规审查贡献（满分3分）
    comp_acc  = comp_res.get('accuracy_pct', 0)
    comp_cont = round(comp_acc / 100 * 3, 4)

    # ── 规则更新贡献（满分3分）
    temp_norm = temp_res.get('normalized_pct', 0)
    temp_cont = round(temp_norm / 100 * 3, 4)

    # ── 标准对齐贡献（满分4分）
    align_norm = align_res.get('normalized_pct', 0)
    align_cont = round(align_norm / 100 * 4, 4)

    total = round(comp_cont + temp_cont + align_cont, 4)

    # ── 打印汇总表 ──────────────────────────────────────────────
    print(f'\n{"═"*60}')
    print('  Claude Sonnet 评测汇总')
    print(f'{"═"*60}')
    print(f'  合规审查   {comp_acc:>6.1f}%  → {comp_cont:.3f}/3.0 分')
    print(f'  规则更新   {temp_norm:>6.1f}%  → {temp_cont:.3f}/3.0 分' +
          (' (跳过)' if NO_LLM else ''))
    print(f'  标准对齐   {align_norm:>6.1f}%  → {align_cont:.3f}/4.0 分' +
          (' (跳过)' if NO_LLM else ''))
    print(f'  {"─"*44}')
    print(f'  总分       {"":>8}     {total:.3f}/10.0 分')
    if comp_res.get('aus_accuracy_pct') is not None:
        print(f'\n  合规审查细分: AUS规则={comp_res["aus_accuracy_pct"]}%  '
              f'EU/USA judge={comp_res.get("eu_usa_accuracy_pct", "N/A")}%')
    if temp_res:
        print(f'  规则更新细分: F1={temp_res["F1_pct"]}%  '
              f'F2={temp_res["F2_pct"]}%  F3={temp_res["F3_pct"]}%')
    if align_res:
        print(f'  标准对齐细分: D1={align_res["D1_pct"]}%  '
              f'D2={align_res["D2_pct"]}%  D3={align_res["D3_pct"]}%')
    print(f'{"═"*60}')

    # ── 保存个人评测 JSON ────────────────────────────────────────
    eval_out = {
        'model': 'claude-sonnet',
        'compliance':  comp_res,
        'temporal':    temp_res,
        'alignment':   align_res,
        'summary': {
            'compliance_pct':  comp_acc,
            'temporal_pct':    temp_norm,
            'alignment_pct':   align_norm,
            'compliance_cont': comp_cont,
            'temporal_cont':   temp_cont,
            'alignment_cont':  align_cont,
            'total_10':        total,
        }
    }
    eval_path = OUT / 'claude-sonnet_eval.json'
    eval_path.write_text(json.dumps(eval_out, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    print(f'\n  详细评测已保存: {eval_path}')

    # ── 追加到 final_summary.json ────────────────────────────────
    fs_path = OUT / 'final_summary.json'
    if fs_path.exists():
        summary = json.loads(fs_path.read_text(encoding='utf-8'))
    else:
        summary = []

    # 移除旧的 claude-sonnet 条目（如果有）
    summary = [s for s in summary if s.get('model') != 'claude-sonnet']

    new_entry = {
        'model': 'claude-sonnet',
        '规则更新_得分率':  f'{temp_norm:.1f}%',
        '规则更新_贡献':    temp_cont,
        '标准对齐_得分率':  f'{align_norm:.1f}%',
        '标准对齐_贡献':    align_cont,
        '合规审查_准确率':  f'{comp_acc:.1f}%',
        '合规审查_贡献':    comp_cont,
        '总分_10分制':      total,
        '_temporal':  {'F1_pct': temp_res.get('F1_pct', '-'),
                       'F2_pct': temp_res.get('F2_pct', '-'),
                       'F3_pct': temp_res.get('F3_pct', '-'),
                       'normalized_pct': temp_norm},
        '_alignment': {'D1_pct': align_res.get('D1_pct', '-'),
                       'D2_pct': align_res.get('D2_pct', '-'),
                       'D3_pct': align_res.get('D3_pct', '-'),
                       'normalized_pct': align_norm},
        '_compliance': {'accuracy_pct': comp_acc,
                        'total': comp_res.get('total', 0),
                        'correct': comp_res.get('correct', 0)},
    }
    summary.append(new_entry)
    # 按总分排序
    summary.sort(key=lambda x: x.get('总分_10分制', 0), reverse=True)
    fs_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                       encoding='utf-8')
    print(f'  final_summary.json 已更新（当前共 {len(summary)} 个模型）')

    # ── 打印最新排名 ─────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print(f'  {"模型":<16} {"规则更新":>8} {"标准对齐":>8} {"合规审查":>8} {"总分":>7}')
    print(f'  {"─"*54}')
    for s in summary:
        marker = ' ◀' if s['model'] == 'claude-sonnet' else ''
        print(f'  {s["model"]:<16} {s["规则更新_得分率"]:>8} '
              f'{s["标准对齐_得分率"]:>8} {s["合规审查_准确率"]:>8} '
              f'{s["总分_10分制"]:>6.2f}{marker}')
    print(f'{"─"*60}')


if __name__ == '__main__':
    main()
