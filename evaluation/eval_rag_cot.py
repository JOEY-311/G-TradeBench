"""
RAG/CoT 实验评测汇总脚本
读取三个生成脚本产生的输出，复用原有三个 eval_*.py 的评测逻辑，
输出各策略在三个任务上的得分对比表。

前置：
  - request_compliance_rag_cot.py 已生成 合规审查/{model}-{strategy}/
  - request_temporal_rag_cot.py   已生成 规则更新/temporal_{model}_{strategy}.csv
  - request_alignment_rag_cot.py  已生成 标准对齐/{model}-{strategy}/

用法:
  set OPENROUTER_API_KEY=sk-or-...
  python eval_rag_cot.py              # 全部
  python eval_rag_cot.py gemini       # 只评 gemini 的各策略
  python eval_rag_cot.py --no-llm     # 跳过 LLM 评测（只看合规审查）
"""

import csv, json, os, re, sys, time
from pathlib import Path
from collections import Counter, defaultdict
from openai import OpenAI

# ── 配置 ─────────────────────────────────────────────────────────────────────
BASE_EVAL = Path('E:/论文/跨境对齐/评测阶段')
OUT       = BASE_EVAL / 'results'
OUT.mkdir(exist_ok=True)

JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'google/gemini-2.0-flash-001')
client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.environ.get('OPENROUTER_API_KEY', ''),
)

MODELS     = ['gemini']
STRATEGIES = ['baseline', 'cot', 'rag', 'rag_cot']

NO_LLM = '--no-llm' in sys.argv
filter_model = next((a for a in sys.argv[1:] if not a.startswith('--')), None)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 合规审查（规则打分，无需 LLM）
# ─────────────────────────────────────────────────────────────────────────────
LIMITS = {
    '日本':  {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '韩国':  {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '法国':  {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '中国':  {'铅': 0.2,  '镉': 0.1,  '农药': 0.01},
    '美国':  {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
}
RISK_RATIO = 0.85
POLLUTANT_PATTERNS = {
    '铅':   re.compile(r'铅[^\d\n]{0,30}[：:]\s*([\d.]+)\s*mg/kg', re.I),
    '镉':   re.compile(r'镉[^\d\n]{0,30}[：:]\s*([\d.]+)\s*mg/kg', re.I),
    '农药': re.compile(r'(?:综合农药残留|农药残留)[^\d\n]{0,40}[：:]\s*([\d.]+)\s*mg/kg', re.I),
}

def extract_pollutants(profile):
    return {k: float(p.search(profile).group(1))
            for k, p in POLLUTANT_PATTERNS.items() if p.search(profile)}

def derive_gt(poll, dest):
    if not poll: return None
    lims = LIMITS.get(dest)
    if not lims: return None
    over, risk = [], []
    for name, val in poll.items():
        lim = lims.get(name)
        if lim is None: continue
        if val > lim: over.append(name)
        elif val >= lim * RISK_RATIO: risk.append(name)
    if over:  return '违规'
    if risk:  return '存在违规风险'
    return '合规'

def extract_verdict(text):
    m = re.search(r'审查结论[：:]\s*(.{2,15})', text)
    if m:
        v = m.group(1).strip('。，\n ')
        if '存在违规风险' in v or ('存在' in v and '风险' in v): return '存在违规风险'
        if '违规' in v and '风险' not in v and '存在' not in v: return '违规'
        if '合规' in v or '放行' in v: return '合规'
        return v[:10]
    first = text.split('\n')[0][:40]
    if '存在违规风险' in first: return '存在违规风险'
    if '违规' in first and '风险' not in first: return '违规'
    if '合规' in first or '放行' in first: return '合规'
    return '无法识别'

def read_csv_file(path):
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin-1'):
        try:
            with open(path, encoding=enc) as f:
                rows = list(csv.DictReader(f))
            if rows: return rows
        except Exception: continue
    return []

def eval_compliance(model, strategy):
    tag    = f'{model}-{strategy}'
    d      = BASE_EVAL / '合规审查' / tag
    if not d.is_dir():
        return None
    match_list = []
    for fp in sorted(d.iterdir()):
        if fp.suffix != '.csv': continue
        rows = read_csv_file(fp)
        if not rows: continue
        sample_poll = extract_pollutants(rows[0].get('FOOD_PROFILE', ''))
        if not sample_poll: continue          # 非 AUS_PRIMARY，跳过
        for r in rows:
            dest = r.get('DESTINATION', '').strip()
            poll = extract_pollutants(r.get('FOOD_PROFILE', ''))
            gt   = derive_gt(poll, dest) if poll else None
            mv   = extract_verdict(r.get('REVIEW_OUTPUT', ''))
            if gt is not None:
                match_list.append(mv == gt)
    if not match_list: return None
    acc = sum(match_list) / len(match_list)
    print(f'  [合规] {tag}: {acc*100:.1f}%  ({sum(match_list)}/{len(match_list)})')
    return {'model': model, 'strategy': strategy,
            'n': len(match_list), 'correct': sum(match_list),
            'accuracy': round(acc, 4), 'accuracy_pct': round(acc*100, 2)}

# ─────────────────────────────────────────────────────────────────────────────
# 2. 规则更新（F1/F3 规则 + F2 LLM）
# ─────────────────────────────────────────────────────────────────────────────
_PROPOSAL_KW = ['proposed','draft','under consideration','open for comment',
                'consultation','not yet','to be enacted','seeking comment']
_ENACTED_KW  = ['in effect','effective','enacted','adopted','promulgated',
                'notification of update','immediately','officially','has been']

def parse_gt(gt_text):
    def _get(tag):
        m = re.search(rf'\[{tag}\]:\s*(.+?)(?=\n\[|\Z)', gt_text, re.S)
        return m.group(1).strip() if m else ''
    return {'legal_status': _get('Legal Status'),
            'effective_date': _get('Proposed Effective Date'),
            'target_details': _get('Target Details')}

def score_f1(gt, out):
    status = gt['legal_status'].lower(); o = out.lower()
    is_prop = any(k in status for k in ['proposed','draft'])
    is_enac = any(k in status for k in ['notification of update','immediately effective','enacted'])
    if is_prop:
        return 1 if any(k in o for k in _PROPOSAL_KW) else 0
    if is_enac:
        return 1 if any(k in o for k in _ENACTED_KW) else 0
    return 1 if any(k in o for k in _ENACTED_KW + _PROPOSAL_KW) else 0

_YEAR_PAT  = re.compile(r'\b(20\d{2})\b')
_MONTH_PAT = re.compile(r'\b(january|february|march|april|may|june|july|august'
                         r'|september|october|november|december)\b', re.I)
_VAGUE_GT  = re.compile(r'after a certain period|not specified|vague|upon|transition'
                         r'|once finalized|pending|to be determined|tbd|certain period', re.I)
_NA_GT     = re.compile(r'^(nan|not applicable|n/a|na|none)\s*\.?\s*$', re.I)

def score_f3(gt, out):
    gd = gt['effective_date'].strip(); o = out.lower()
    if _NA_GT.match(gd):
        return 0 if (_YEAR_PAT.search(out) or _MONTH_PAT.search(out)) else 1
    if _VAGUE_GT.search(gd):
        return 1 if re.search(r'after|period|certain|transition|once|following|upon|unclear|uncertain|pending', o) else 0
    gy = set(_YEAR_PAT.findall(gd))
    if gy: return 1 if gy & set(_YEAR_PAT.findall(out)) else 0
    gm = set(_MONTH_PAT.findall(gd.lower()))
    return 1 if (gm and gm & set(_MONTH_PAT.findall(o))) else 0

F2_SYSTEM = ("You are a food regulation expert reviewer. Judge whether the model "
             "correctly identified the target substance and change type. "
             "Output JSON only: {\"score\": 0 or 1, \"reason\": \"brief\"}")
F2_TMPL   = ("Question: {q}\nKey info: {details}\nModel answer: {ans}\n"
             "Score 1 if substance + change type correct, else 0. JSON only.")

def llm_f2(q, gt, out, retries=3):
    if NO_LLM: return 0
    prompt = F2_TMPL.format(q=q[:600], details=gt['target_details'][:300], ans=out[:800])
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role':'system','content':F2_SYSTEM},
                           {'role':'user','content':prompt}],
                temperature=0, max_tokens=100)
            raw = r.choices[0].message.content or ''
            m = re.search(r'\{.*\}', raw, re.S)
            if m: return int(json.loads(m.group()).get('score', 0))
        except Exception:
            if attempt < retries - 1: time.sleep(2**attempt)
    return 0

def eval_temporal(model, strategy):
    tag  = f'{model}-{strategy}'
    path = BASE_EVAL / '规则更新' / f'temporal_{tag}.csv'
    if not path.exists(): return None
    rows = read_csv_file(path)
    if not rows: return None
    f1s, f2s, f3s = [], [], []
    for row in rows:
        gt  = parse_gt(row.get('ground_truth', ''))
        q   = row.get('instruction', '')
        out = row.get('model_output', '')
        f1s.append(score_f1(gt, out))
        f2s.append(llm_f2(q, gt, out))
        f3s.append(score_f3(gt, out))
        print(f'  [temporal {tag}] F1={f1s[-1]} F2={f2s[-1]} F3={f3s[-1]}', end='\r')
    n = len(rows)
    f1 = sum(f1s)/n; f2 = sum(f2s)/n; f3 = sum(f3s)/n
    total = (f1+f2+f3)/3*100
    print(f'\n  [规则更新] {tag}: F1={f1*100:.1f}% F2={f2*100:.1f}% F3={f3*100:.1f}%  总={total:.1f}%')
    return {'model': model, 'strategy': strategy, 'n': n,
            'F1_pct': round(f1*100,2), 'F2_pct': round(f2*100,2), 'F3_pct': round(f3*100,2),
            'normalized_pct': round(total, 2)}

# ─────────────────────────────────────────────────────────────────────────────
# 3. 标准对齐（D1/D2/D3 LLM）
# ─────────────────────────────────────────────────────────────────────────────
ALIGN_SYSTEM = ("你是食品法规专业评审专家，请严格按评分标准输出JSON，不要添加解释。")
ALIGN_TMPL   = """## 任务类型：{task_type}
## 题目：{question}
## 参考答案：{reference}
## 模型回答：{output}

D1 问题识别（0/1）：核心合规问题是否识别正确
D2 法规依据（0/2）：法规引用是否正确且具体（2=完整，1=部分，0=无/错）
D3 方案覆盖（0/1）：解决方案是否涵盖参考答案主要行动项

输出：{{"D1":<0或1>,"D2":<0/1/2>,"D3":<0或1>}}"""

TASK_KW = [('冲突判断','冲突判断'),('限量对齐','限量对齐'),('配料准入','配料准入'),
           ('准入程序','准入程序'),('标签对齐','标签对齐'),('多国流通','多国流通'),('俗名映射','俗名映射')]

def infer_task(fname):
    for kw, name in TASK_KW:
        if kw in fname: return name
    return '未知'

def llm_d123(task, q, ref, out, retries=3):
    if NO_LLM: return 0, 0, 0
    prompt = ALIGN_TMPL.format(task_type=task, question=q[:500],
                                reference=ref[:700], output=out[:1000])
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role':'system','content':ALIGN_SYSTEM},
                           {'role':'user','content':prompt}],
                temperature=0, max_tokens=100)
            raw = r.choices[0].message.content or ''
            m = re.search(r'\{.*\}', raw, re.S)
            if m:
                obj = json.loads(m.group())
                return (min(max(int(obj.get('D1',0)),0),1),
                        min(max(int(obj.get('D2',0)),0),2),
                        min(max(int(obj.get('D3',0)),0),1))
        except Exception:
            if attempt < retries - 1: time.sleep(2**attempt)
    return 0, 0, 0

def eval_alignment(model, strategy):
    tag = f'{model}-{strategy}'
    d   = BASE_EVAL / '标准对齐' / tag
    if not d.is_dir(): return None
    d1s, d2s, d3s = [], [], []
    for fp in sorted(d.iterdir()):
        if fp.suffix != '.csv': continue
        rows = read_csv_file(fp)
        if not rows or 'OUTPUT' not in rows[0]: continue
        task = infer_task(fp.name)
        for row in rows:
            q   = row.get('QUESTION', '')
            ref = row.get('REFERENCE', '')
            out = row.get('OUTPUT', '')
            d1, d2, d3 = llm_d123(task, q, ref, out)
            d1s.append(d1); d2s.append(d2); d3s.append(d3)
            print(f'  [align {tag}] {task} D1={d1} D2={d2} D3={d3}', end='\r')
    if not d1s: return None
    n = len(d1s)
    a1=sum(d1s)/n; a2=sum(d2s)/n; a3=sum(d3s)/n
    total = (a1 + a2/2 + a3)/3*100   # D2 归一到 0-1 后平均
    avg4  = (a1+a2+a3)/n*n/n         # 直接按4分制均分
    norm  = (a1+a2+a3)/(4)*100
    print(f'\n  [标准对齐] {tag}: D1={a1*100:.1f}% D2={a2/2*100:.1f}% D3={a3*100:.1f}%  标准化={norm:.1f}%')
    return {'model': model, 'strategy': strategy, 'n': n,
            'D1_pct': round(a1*100,2), 'D2_pct': round(a2/2*100,2), 'D3_pct': round(a3*100,2),
            'normalized_pct': round(norm, 2)}

# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    target_models = [m for m in MODELS if not filter_model or filter_model in m]

    compliance_stats, temporal_stats, alignment_stats = [], [], []

    print('\n── 合规审查评测（规则打分）──────────────────────────')
    for model in target_models:
        for strategy in STRATEGIES:
            s = eval_compliance(model, strategy)
            if s: compliance_stats.append(s)

    if not NO_LLM:
        print('\n── 规则更新评测（F1/F3规则 + F2 LLM）──────────────')
        for model in target_models:
            for strategy in STRATEGIES:
                s = eval_temporal(model, strategy)
                if s: temporal_stats.append(s)

        print('\n── 标准对齐评测（D1/D2/D3 LLM）────────────────────')
        for model in target_models:
            for strategy in STRATEGIES:
                s = eval_alignment(model, strategy)
                if s: alignment_stats.append(s)

    # ── 打印对比表 ────────────────────────────────────────────────────────────
    print(f'\n\n{"="*65}')
    print('  RAG/CoT 策略对比  —  合规审查（准确率 %）')
    print(f'{"="*65}')
    print(f'  {"模型-策略":<22} {"样本":>6} {"正确":>6} {"准确率":>8}')
    print(f'  {"-"*46}')
    for s in compliance_stats:
        print(f'  {s["model"]+"-"+s["strategy"]:<22} {s["n"]:>6} {s["correct"]:>6} {s["accuracy_pct"]:>7.1f}%')

    if temporal_stats:
        print(f'\n{"="*65}')
        print('  RAG/CoT 策略对比  —  规则更新（标准化 %）')
        print(f'{"="*65}')
        print(f'  {"模型-策略":<22} {"F1%":>7} {"F2%":>7} {"F3%":>7} {"总%":>8}')
        print(f'  {"-"*54}')
        for s in temporal_stats:
            print(f'  {s["model"]+"-"+s["strategy"]:<22} '
                  f'{s["F1_pct"]:>6.1f}% {s["F2_pct"]:>6.1f}% '
                  f'{s["F3_pct"]:>6.1f}% {s["normalized_pct"]:>7.1f}%')

    if alignment_stats:
        print(f'\n{"="*65}')
        print('  RAG/CoT 策略对比  —  标准对齐（标准化 %）')
        print(f'{"="*65}')
        print(f'  {"模型-策略":<22} {"D1%":>7} {"D2%":>7} {"D3%":>7} {"总%":>8}')
        print(f'  {"-"*54}')
        for s in alignment_stats:
            print(f'  {s["model"]+"-"+s["strategy"]:<22} '
                  f'{s["D1_pct"]:>6.1f}% {s["D2_pct"]:>6.1f}% '
                  f'{s["D3_pct"]:>6.1f}% {s["normalized_pct"]:>7.1f}%')

    # ── 保存 ──────────────────────────────────────────────────────────────────
    out = {'compliance': compliance_stats,
           'temporal': temporal_stats,
           'alignment': alignment_stats}
    p = OUT / 'rag_cot_summary.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  汇总已保存: {p}')

if __name__ == '__main__':
    main()
