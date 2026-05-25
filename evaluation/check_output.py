import csv, re
from pathlib import Path

LIMITS = {
    '日本': {'铅': 0.1, '镉': 0.05, '农药': 0.01},
    '韩国': {'铅': 0.1, '镉': 0.05, '农药': 0.01},
    '法国': {'铅': 0.1, '镉': 0.05, '农药': 0.01},
    '中国': {'铅': 0.2, '镉': 0.1,  '农药': 0.01},
}
PAT = {
    '铅':   re.compile(r'铅[^\d\n]{0,30}[：:]\s*([\d.]+)\s*mg/kg', re.I),
    '镉':   re.compile(r'镉[^\d\n]{0,30}[：:]\s*([\d.]+)\s*mg/kg', re.I),
    '农药': re.compile(r'(?:综合农药残留|农药残留)[^\d\n]{0,40}[：:]\s*([\d.]+)\s*mg/kg', re.I),
}

def extract(t):
    return {k: float(p.search(t).group(1)) for k, p in PAT.items() if p.search(t)}

def gt(poll, dest):
    lims = LIMITS.get(dest)
    if not poll or not lims: return None
    over = [n for n, v in poll.items() if lims.get(n) and v > lims[n]]
    risk = [n for n, v in poll.items() if lims.get(n) and not v > lims[n] and v >= lims[n] * 0.85]
    return '违规' if over else ('存在违规风险' if risk else '合规')

for model in ['grok', 'gpt']:
    base = Path(f'E:/论文/跨境对齐/评测阶段/合规审查/{model}')
    for fp in sorted(base.iterdir()):
        if '澳日' not in fp.name:
            continue
        rows = []
        for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin-1'):
            try:
                with open(fp, encoding=enc) as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    break
            except Exception:
                pass

        print(f'\n{"="*60}')
        print(f'{model} 澳日 (前5条)')
        print('='*60)
        for i, r in enumerate(rows[:5]):
            dest = r.get('DESTINATION', '').strip()
            poll = extract(r.get('FOOD_PROFILE', ''))
            g = gt(poll, dest)
            out = r.get('REVIEW_OUTPUT', '').strip()

            # 找审查结论行
            verdict_line = ''
            for line in out.split('\n'):
                if '审查结论' in line:
                    verdict_line = line.strip()
                    break

            print(f'\n[{i+1}] GT={g}')
            print(f'     污染物: {poll}')
            if verdict_line:
                print(f'     结论行: {verdict_line[:80]}')
            else:
                print(f'     【无审查结论行】')
                print(f'     输出前150字: {out[:150]}')
