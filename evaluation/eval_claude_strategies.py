#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Sonnet RAG/CoT 策略评测脚本
对比 baseline / cot / rag / rag_cot 四种策略的表现

关键设计：
  - CoT 输出含推理链，需从全文找到"审查结论："的【最后一次出现】
  - 为保证公平对比，baseline 只评测与策略子集相同的那 20% 行
  - 规则更新 F2 用 LLM-as-Judge；标准对齐 D1/D2/D3 同样 LLM-as-Judge

用法：
  python eval_claude_strategies.py
  python eval_claude_strategies.py --no-llm   # 跳过 LLM judge（仅合规规则打分）
"""

import csv, json, os, random, re, sys, time
from pathlib import Path
from openai import OpenAI

# ════════════════════════════════════════════════════════════════
#  配置
# ════════════════════════════════════════════════════════════════
API_KEY     = os.environ.get('OPENROUTER_API_KEY',
              os.environ.get('OPENROUTER_API_KEY', ''))
JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'google/gemini-2.0-flash-001')
NO_LLM      = '--no-llm' in sys.argv

EVAL       = Path('E:/论文/跨境对齐/评测阶段')
OUT        = EVAL / 'results'
OUT.mkdir(exist_ok=True)

STRATEGIES = ['baseline', 'cot', 'rag', 'rag_cot']
SAMPLE_PCT = 0.2
SEED       = 99

client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=API_KEY)


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════
def read_csv(path: Path) -> list:
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin-1'):
        try:
            with open(path, encoding=enc) as f:
                rows = list(csv.DictReader(f))
            if rows:
                return rows
        except Exception:
            continue
    return []


def sample_rows(rows: list, pct: float = SAMPLE_PCT, seed: int = SEED) -> list:
    """与 run_claude_strategies.py 完全一致的抽样，保证三策略子集相同。"""
    rng = random.Random(seed)
    n = max(1, round(len(rows) * pct))
    return rng.sample(rows, min(n, len(rows)))


def section(title: str):
    print(f'\n{"═"*65}')
    print(f'  {title}')
    print(f'{"═"*65}')


# ════════════════════════════════════════════════════════════════
#  合规审查工具
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


def extract_verdict(text: str, is_cot: bool = False) -> str:
    """
    增强版结论提取：
    - CoT 模式：取全文中最后一个"审查结论："（推理链在前，结论行在最后）
    - 非 CoT 模式：取第一个"审查结论："
    """
    all_matches = list(re.finditer(r'审查结论[：:]\s*(.{2,20})', text))

    if all_matches:
        m = all_matches[-1] if is_cot else all_matches[0]
        v = m.group(1).strip('。，\n ')
        if '存在违规风险' in v or ('存在' in v and '风险' in v): return '存在违规风险'
        if '违规' in v and '风险' not in v and '存在' not in v:  return '违规'
        if '合规' in v or '放行' in v:                           return '合规'
        return v[:10]

    # 没有"审查结论："标记时，从末尾/首部行中寻找关键词
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    search_lines = list(reversed(lines[-5:])) if is_cot else lines[:3]
    for line in search_lines:
        if '存在违规风险' in line: return '存在违规风险'
        if '违规' in line and '风险' not in line and '存在' not in line: return '违规'
        if '合规' in line or '放行' in line: return '合规'

    return '无法识别'


# ════════════════════════════════════════════════════════════════
#  合规审查评测
# ════════════════════════════════════════════════════════════════
def eval_compliance_strategy(strategy: str) -> dict:
    comp_dir = (EVAL / '合规审查/claude-sonnet' if strategy == 'baseline'
                else EVAL / f'合规审查/claude-sonnet-{strategy}')
    is_cot = 'cot' in strategy

    if not comp_dir.is_dir():
        return {'accuracy_pct': None, 'total': 0, 'correct': 0, 'note': '目录不存在'}

    match_all, file_results = [], []

    for fp in sorted(comp_dir.iterdir()):
        if fp.suffix != '.csv':
            continue
        rows = read_csv(fp)
        if not rows:
            continue

        # baseline 抽取相同 20% 子集（与策略文件行数一致），保证可比性
        if strategy == 'baseline':
            rows = sample_rows(rows)

        # 跳过无实测污染物值的非 AUS_PRIMARY 文件
        if not extract_pollutants(rows[0].get('FOOD_PROFILE', '')):
            continue

        correct, gt_count = 0, 0
        verdict_dist: dict = {}

        for r in rows:
            dest    = r.get('DESTINATION', '').strip()
            poll    = extract_pollutants(r.get('FOOD_PROFILE', ''))
            gt      = derive_gt(poll, dest)
            mv      = extract_verdict(r.get('REVIEW_OUTPUT', ''), is_cot)
            verdict_dist[mv] = verdict_dist.get(mv, 0) + 1
            if gt is not None:
                gt_count += 1
                hit = (mv == gt)
                match_all.append(hit)
                if hit:
                    correct += 1

        origin = rows[0].get('ORIGIN', '?') if rows else '?'
        dest_l  = rows[0].get('DESTINATION', '?') if rows else '?'
        file_results.append({
            'file': fp.name, 'route': f'{origin}→{dest_l}',
            'total': len(rows), 'gt_count': gt_count,
            'correct': correct, 'verdict_dist': verdict_dist,
        })

    if not match_all:
        return {'accuracy_pct': None, 'total': 0, 'correct': 0,
                'note': '无可计算 GT 的样本', 'file_results': []}

    acc = sum(match_all) / len(match_all) * 100
    return {
        'accuracy_pct': round(acc, 2),
        'total': len(match_all),
        'correct': int(sum(match_all)),
        'file_results': file_results,
    }


def eval_compliance_all() -> dict:
    section('任务 1：合规审查（四策略对比）')
    results = {}
    for strategy in STRATEGIES:
        res = eval_compliance_strategy(strategy)
        results[strategy] = res
        acc_str = f"{res['accuracy_pct']:.1f}%" if res['accuracy_pct'] is not None else '无数据'
        print(f'  [{strategy:<10}] 准确率 = {acc_str:>8}  ({res["correct"]}/{res["total"]})')
    return results


# ════════════════════════════════════════════════════════════════
#  规则更新评测
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
            r'after|period|certain|transition|once|following|upon|unclear|uncertain|pending',
            o) else 0
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


def _get_temporal_subset_indices() -> set:
    """从任意一个策略文件读取已处理的 index 集合（三策略子集相同）。"""
    for strategy in ['cot', 'rag', 'rag_cot']:
        p = EVAL / f'规则更新/temporal_claude-sonnet-{strategy}.csv'
        if p.exists():
            rows = read_csv(p)
            if rows:
                return {r.get('index', '') for r in rows}
    return set()


def eval_temporal_strategy(strategy: str, subset_indices: set) -> dict:
    csv_path = (EVAL / '规则更新/temporal_claude-sonnet.csv' if strategy == 'baseline'
                else EVAL / f'规则更新/temporal_claude-sonnet-{strategy}.csv')

    if not csv_path.exists():
        return {'F1_pct': None, 'F2_pct': None, 'F3_pct': None,
                'normalized_pct': None, 'n': 0}

    rows = read_csv(csv_path)
    if not rows:
        return {'F1_pct': None, 'F2_pct': None, 'F3_pct': None,
                'normalized_pct': None, 'n': 0}

    # baseline 只取与策略相同的子集
    if strategy == 'baseline' and subset_indices:
        rows = [r for r in rows if r.get('index', '') in subset_indices]

    if not rows:
        return {'F1_pct': None, 'F2_pct': None, 'F3_pct': None,
                'normalized_pct': None, 'n': 0}

    f1s, f2s, f3s = [], [], []
    for i, row in enumerate(rows, 1):
        gt  = parse_gt(row.get('ground_truth', ''))
        q   = row.get('instruction', '')
        out = row.get('model_output', '')
        f1 = score_f1(gt, out)
        f2 = llm_f2(q, gt, out)
        f3 = score_f3(gt, out)
        f1s.append(f1); f2s.append(f2); f3s.append(f3)
        print(f'    [{strategy}] [{i:>3}/{len(rows)}] F1={f1} F2={f2} F3={f3}', end='\r')
    print()

    n = len(rows)
    a1 = sum(f1s)/n; a2 = sum(f2s)/n; a3 = sum(f3s)/n
    norm = (a1 + a2 + a3) / 3 * 100
    return {
        'n': n,
        'F1_pct': round(a1*100, 2), 'F2_pct': round(a2*100, 2), 'F3_pct': round(a3*100, 2),
        'normalized_pct': round(norm, 2),
    }


def eval_temporal_all() -> dict:
    section('任务 2：规则更新（四策略对比）')
    subset_indices = _get_temporal_subset_indices()
    print(f'  策略子集 index 数: {len(subset_indices)}')

    results = {}
    for strategy in STRATEGIES:
        print(f'\n  ── 策略: {strategy} ──')
        res = eval_temporal_strategy(strategy, subset_indices)
        results[strategy] = res
        if res['normalized_pct'] is not None:
            print(f'  [{strategy:<10}] '
                  f'F1={res["F1_pct"]:>5.1f}%  '
                  f'F2={res["F2_pct"]:>5.1f}%  '
                  f'F3={res["F3_pct"]:>5.1f}%  '
                  f'综合={res["normalized_pct"]:>5.1f}%  (n={res["n"]})')
        else:
            print(f'  [{strategy:<10}] 无数据')
    return results


# ════════════════════════════════════════════════════════════════
#  标准对齐评测
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


def _get_alignment_subset_questions() -> set:
    """从任意一个策略目录读取已处理的 QUESTION 集合。"""
    for strategy in ['cot', 'rag', 'rag_cot']:
        d = EVAL / f'标准对齐/claude-sonnet-{strategy}'
        if not d.is_dir():
            continue
        qs = set()
        for fp in d.iterdir():
            if fp.suffix == '.csv':
                for r in read_csv(fp):
                    qs.add(r.get('QUESTION', ''))
        if qs:
            return qs
    return set()


def eval_alignment_strategy(strategy: str, subset_questions: set) -> dict:
    align_dir = (EVAL / '标准对齐/claude-sonnet' if strategy == 'baseline'
                 else EVAL / f'标准对齐/claude-sonnet-{strategy}')

    if not align_dir.is_dir():
        return {'D1_pct': None, 'D2_pct': None, 'D3_pct': None,
                'normalized_pct': None, 'n': 0}

    d1s, d2s, d3s = [], [], []

    for fp in sorted(align_dir.iterdir()):
        if fp.suffix != '.csv':
            continue
        rows = read_csv(fp)
        if not rows or 'OUTPUT' not in rows[0]:
            continue

        task = infer_task(fp.name)

        # baseline 只取策略子集题目
        if strategy == 'baseline' and subset_questions:
            rows = [r for r in rows if r.get('QUESTION', '') in subset_questions]

        for i, row in enumerate(rows, 1):
            q   = row.get('QUESTION',  '')
            ref = row.get('REFERENCE', '')
            out = row.get('OUTPUT',    '')
            d1, d2, d3 = llm_d123(task, q, ref, out)
            d1s.append(d1); d2s.append(d2); d3s.append(d3)
            print(f'    [{strategy}/{task}] [{i:>2}/{len(rows)}] '
                  f'D1={d1} D2={d2} D3={d3}', end='\r')
    print()

    if not d1s:
        return {'D1_pct': None, 'D2_pct': None, 'D3_pct': None,
                'normalized_pct': None, 'n': 0}

    n = len(d1s)
    a1 = sum(d1s)/n; a2 = sum(d2s)/n; a3 = sum(d3s)/n
    norm = (a1 + a2 + a3) / 4 * 100   # 满分 4 分归一化
    return {
        'n': n,
        'D1_pct': round(a1*100, 2),
        'D2_pct': round(a2/2*100, 2),
        'D3_pct': round(a3*100, 2),
        'normalized_pct': round(norm, 2),
    }


def eval_alignment_all() -> dict:
    section('任务 3：标准对齐（四策略对比）')
    subset_questions = _get_alignment_subset_questions()
    print(f'  策略子集题目数: {len(subset_questions)}')

    results = {}
    for strategy in STRATEGIES:
        print(f'\n  ── 策略: {strategy} ──')
        res = eval_alignment_strategy(strategy, subset_questions)
        results[strategy] = res
        if res['normalized_pct'] is not None:
            print(f'  [{strategy:<10}] '
                  f'D1={res["D1_pct"]:>5.1f}%  '
                  f'D2={res["D2_pct"]:>5.1f}%  '
                  f'D3={res["D3_pct"]:>5.1f}%  '
                  f'综合={res["normalized_pct"]:>5.1f}%  (n={res["n"]})')
        else:
            print(f'  [{strategy:<10}] 无数据')
    return results


# ════════════════════════════════════════════════════════════════
#  汇总输出
# ════════════════════════════════════════════════════════════════
def _fmt(val, fmt='.1f', suffix='%'):
    return f'{val:{fmt}}{suffix}' if val is not None else '-'


def print_summary(comp_all: dict, temp_all: dict, align_all: dict):
    section('四策略综合对比汇总')
    print(f'\n  {"策略":<12} {"合规准确率":>12} {"规则综合":>12} {"对齐综合":>12} {"综合得分":>10}')
    print(f'  {"─"*60}')

    for s in STRATEGIES:
        c_acc  = comp_all.get(s, {}).get('accuracy_pct')
        t_norm = temp_all.get(s, {}).get('normalized_pct')
        a_norm = align_all.get(s, {}).get('normalized_pct')

        if all(v is not None for v in [c_acc, t_norm, a_norm]):
            total = round(c_acc/100*3 + t_norm/100*3 + a_norm/100*4, 3)
            total_str = f'{total:.3f}'
        else:
            total_str = '-'

        print(f'  {s:<12} '
              f'{_fmt(c_acc):>12} '
              f'{_fmt(t_norm):>12} '
              f'{_fmt(a_norm):>12} '
              f'{total_str:>10}')

    # 规则更新细分
    print(f'\n  规则更新细分（F1=法律状态 / F2=物质识别 / F3=日期准确）:')
    print(f'  {"策略":<12} {"F1":>8} {"F2":>8} {"F3":>8}')
    print(f'  {"─"*36}')
    for s in STRATEGIES:
        t = temp_all.get(s, {})
        print(f'  {s:<12} '
              f'{_fmt(t.get("F1_pct")):>8} '
              f'{_fmt(t.get("F2_pct")):>8} '
              f'{_fmt(t.get("F3_pct")):>8}')

    # 标准对齐细分
    print(f'\n  标准对齐细分（D1=问题识别 / D2=法规引用 / D3=方案覆盖）:')
    print(f'  {"策略":<12} {"D1":>8} {"D2":>8} {"D3":>8}')
    print(f'  {"─"*36}')
    for s in STRATEGIES:
        a = align_all.get(s, {})
        print(f'  {s:<12} '
              f'{_fmt(a.get("D1_pct")):>8} '
              f'{_fmt(a.get("D2_pct")):>8} '
              f'{_fmt(a.get("D3_pct")):>8}')

    # CoT 合规 verdict 分布（帮助诊断提取是否正常）
    print(f'\n  CoT 合规结论分布（诊断）:')
    for s in ['baseline', 'cot']:
        fr = comp_all.get(s, {}).get('file_results', [])
        if fr:
            merged: dict = {}
            for f in fr:
                for k, v in f.get('verdict_dist', {}).items():
                    merged[k] = merged.get(k, 0) + v
            print(f'  [{s}] {merged}')


# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════
def main():
    print('╔══════════════════════════════════════════════════════════════╗')
    print('║     Claude Sonnet  RAG/CoT 策略对比评测                      ║')
    print(f'║  策略: baseline / cot / rag / rag_cot  抽样: {SAMPLE_PCT*100:.0f}%          ║')
    print(f'║  Judge: {JUDGE_MODEL:<53}║')
    print(f'║  LLM评分: {"已跳过（--no-llm）" if NO_LLM else "启用"}' + ' '*44 + '║')
    print('╚══════════════════════════════════════════════════════════════╝')

    comp_all  = eval_compliance_all()
    temp_all  = (eval_temporal_all()  if not NO_LLM
                 else {s: {'F1_pct': None, 'F2_pct': None, 'F3_pct': None,
                            'normalized_pct': None, 'n': 0}
                       for s in STRATEGIES})
    align_all = (eval_alignment_all() if not NO_LLM
                 else {s: {'D1_pct': None, 'D2_pct': None, 'D3_pct': None,
                            'normalized_pct': None, 'n': 0}
                       for s in STRATEGIES})

    print_summary(comp_all, temp_all, align_all)

    # ── 保存详细结果 JSON ────────────────────────────────────────
    out_data = {
        'model':      'claude-sonnet',
        'sample_pct': SAMPLE_PCT,
        'seed':       SEED,
        'no_llm':     NO_LLM,
        'compliance': comp_all,
        'temporal':   temp_all,
        'alignment':  align_all,
    }
    out_path = OUT / 'claude-sonnet_strategy_eval.json'
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    print(f'\n  详细结果已保存: {out_path}')

    # ── 追加到 rag_cot_summary.json ─────────────────────────────
    rag_cot_path = OUT / 'rag_cot_summary.json'
    rag_cot = {}
    if rag_cot_path.exists():
        try:
            rag_cot = json.loads(rag_cot_path.read_text(encoding='utf-8'))
        except Exception:
            pass

    rag_cot['claude-sonnet'] = {
        'compliance': {s: comp_all.get(s, {}).get('accuracy_pct')  for s in STRATEGIES},
        'temporal':   {s: temp_all.get(s, {}).get('normalized_pct') for s in STRATEGIES},
        'alignment':  {s: align_all.get(s, {}).get('normalized_pct') for s in STRATEGIES},
    }
    rag_cot_path.write_text(json.dumps(rag_cot, ensure_ascii=False, indent=2),
                             encoding='utf-8')
    print(f'  rag_cot_summary.json 已更新')
    print('\n完成！')


if __name__ == '__main__':
    main()
