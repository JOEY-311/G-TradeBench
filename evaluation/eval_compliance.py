"""
合规审查任务评测脚本
指标体系（满分3分·模型级加权）：
  ACC. (0/1 per question) — 规则打分：模型能否正确判断进口食品合规性结论
  最终贡献 = accuracy × 3

GT推导方式：从 FOOD_PROFILE 中提取实测污染物数值（铅/镉/农药），
            对照目标国限量标准，推导出 违规 / 存在违规风险 / 合规 三类标签。
仅含实测数值的 AUS_PRIMARY 文件可计算准确率；EU/US 配料表型文件仅统计分布。

用法：
  python eval_compliance.py            # 跑全部模型
  python eval_compliance.py deepseek   # 单模型
"""

import csv, re, json, sys
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path('E:/论文/跨境对齐/评测阶段/合规审查')
OUT  = Path('E:/论文/跨境对齐/评测阶段/results')
OUT.mkdir(exist_ok=True)

MODEL_DIRS = {
    'deepseek': BASE / 'deepseek',
    'gemini':   BASE / 'gemini',
    'gpt':      BASE / 'gpt',
    'grok':     BASE / 'grok',
    'qwen':     BASE / 'qwen',
    'gpt-5.5':     BASE / 'gpt-5.5',
    'deepseek-v4': BASE / 'deepseekv4',
}

# ── 各目标国限量标准（基于公开法规） ─────────────────────────────────────────
LIMITS = {
    '日本': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '韩国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '法国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '中国': {'铅': 0.2,  '镉': 0.1,  '农药': 0.01},
    '美国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
}
RISK_RATIO = 0.85  # 实测值 ≥ 限量×85% 视为"存在违规风险"

POLLUTANT_PATTERNS = {
    '铅':   re.compile(r'铅[^\d\n]{0,30}[：:]\s*([\d.]+)\s*mg/kg', re.I),
    '镉':   re.compile(r'镉[^\d\n]{0,30}[：:]\s*([\d.]+)\s*mg/kg', re.I),
    '农药': re.compile(r'(?:综合农药残留|农药残留)[^\d\n]{0,40}[：:]\s*([\d.]+)\s*mg/kg', re.I),
}

# ── 工具函数 ──────────────────────────────────────────────────────────────────
def read_csv(path: Path) -> list[dict]:
    for enc in ('utf-8-sig', 'gbk', 'utf-8'):
        try:
            with open(path, encoding=enc) as f:
                rows = list(csv.DictReader(f))
            if rows and 'REVIEW_OUTPUT' in rows[0]:
                return rows
        except Exception:
            continue
    return []

def extract_pollutants(profile: str) -> dict:
    return {
        k: float(p.search(profile).group(1))
        for k, p in POLLUTANT_PATTERNS.items()
        if p.search(profile)
    }

def derive_gt(pollutants: dict, destination: str) -> str | None:
    if not pollutants:
        return None
    limits = LIMITS.get(destination)
    if not limits:
        return None
    over, risk = [], []
    for name, val in pollutants.items():
        lim = limits.get(name)
        if lim is None:
            continue
        if val > lim:
            over.append(name)
        elif val >= lim * RISK_RATIO:
            risk.append(name)
    if over:
        return '违规'
    if risk:
        return '存在违规风险'
    return '合规'

def extract_verdict(text: str) -> str:
    t = text.strip()
    m = re.search(r'审查结论[：:]\s*(.{2,15})', t)
    if m:
        v = m.group(1).strip('。，\n ')
        if '存在违规风险' in v or ('存在' in v and '违规' in v and '风险' in v):
            return '存在违规风险'
        if '违规' in v and '风险' not in v and '存在' not in v:
            return '违规'
        if '合规' in v or '放行' in v or '建议放行' in v:
            return '合规'
        return v[:10]
    first = t.split('\n')[0][:40]
    if '存在违规风险' in first:
        return '存在违规风险'
    if '违规' in first and '风险' not in first:
        return '违规'
    if '合规' in first or '放行' in first:
        return '合规'
    return '无法识别'

# ── 评测单模型 ────────────────────────────────────────────────────────────────
def evaluate_model(model_key: str) -> dict:
    model_dir = MODEL_DIRS.get(model_key)
    if not model_dir or not model_dir.is_dir():
        print(f'  [跳过] 目录不存在: {model_dir}')
        return {}

    all_rows, file_summaries = [], []

    for fp in sorted(model_dir.iterdir()):
        if not fp.is_file() or fp.suffix != '.csv':
            continue
        rows = read_csv(fp)
        if not rows:
            continue

        # 判断是否为 AUS_PRIMARY（含实测数值）
        sample_poll = extract_pollutants(rows[0].get('FOOD_PROFILE', ''))
        is_aus      = bool(sample_poll)

        verdicts, gt_list, match_list = [], [], []

        for r in rows:
            dest    = r.get('DESTINATION', '').strip()
            profile = r.get('FOOD_PROFILE', '')
            output  = r.get('REVIEW_OUTPUT', '')

            model_v = extract_verdict(output)
            poll    = extract_pollutants(profile) if is_aus else {}
            gt_v    = derive_gt(poll, dest) if poll else None

            match   = (model_v == gt_v) if gt_v is not None else None
            all_rows.append({
                'file': fp.name, 'ID': r.get('ID'),
                'destination': dest,
                'model_verdict': model_v,
                'gt_verdict': gt_v,
                'pollutants': poll,
                'acc': 1 if match is True else (0 if match is False else None),
            })
            verdicts.append(model_v)
            if gt_v is not None:
                gt_list.append(gt_v)
                match_list.append(model_v == gt_v)

        acc = sum(match_list) / len(match_list) * 100 if match_list else None
        file_summaries.append({
            'file': fp.name,
            'data_type': 'AUS_PRIMARY' if is_aus else 'EU/US',
            'total': len(rows),
            'verdict_dist': dict(Counter(verdicts)),
            'gt_dist': dict(Counter(gt_list)),
            'gt_computable': len(match_list),
            'correct': sum(match_list) if match_list else 0,
            'accuracy_pct': acc,
        })

    return {
        'model': model_key,
        'file_summaries': file_summaries,
        'rows': all_rows,
    }

# ── 打印单模型报告 ────────────────────────────────────────────────────────────
def print_report(model_key: str, result: dict) -> dict | None:
    summaries = result.get('file_summaries', [])
    rows      = result.get('rows', [])
    if not rows:
        return None

    aus_files = [s for s in summaries if s['accuracy_pct'] is not None]
    eu_files  = [s for s in summaries if s['accuracy_pct'] is None]

    aus_total   = sum(s['gt_computable'] for s in aus_files)
    aus_correct = sum(s['correct'] for s in aus_files)
    accuracy    = aus_correct / aus_total if aus_total > 0 else 0

    print(f'\n{"="*55}')
    print(f'  {model_key.upper()}  |  {len(rows)} 条总样本  |  AUS可计算: {aus_total} 条')
    print(f'{"="*55}')

    if aus_files:
        print(f'  [AUS_PRIMARY 文件]')
        for s in aus_files:
            print(f'    {s["file"]}:  准确率 {s["accuracy_pct"]:.1f}%  '
                  f'({s["correct"]}/{s["gt_computable"]})')
        print(f'  ── AUS综合准确率: {accuracy*100:.1f}%  ({aus_correct}/{aus_total})')

    if eu_files:
        print(f'\n  [EU/US 配料表型 - 仅统计分布]')
        for s in eu_files:
            vd = s['verdict_dist']
            print(f'    {s["file"]}({s["total"]}条): ' +
                  '  '.join(f'{k}={v}' for k, v in sorted(vd.items())))

    contrib = accuracy * 3
    print(f'\n  ── 最终贡献(准确率×3权重): {contrib:.3f}/3.0')

    # 按目标国分布
    by_dest = defaultdict(list)
    for r in rows:
        by_dest[r['destination']].append(r['model_verdict'])
    print(f'\n  按目标国:')
    for dest, vs in sorted(by_dest.items()):
        dc = Counter(vs)
        print(f'    {dest}({len(vs)}): '
              f'违规={dc.get("违规",0)}  '
              f'风险={dc.get("存在违规风险",0)}  '
              f'合规={dc.get("合规",0)+dc.get("建议放行",0)}  '
              f'未识别={dc.get("无法识别",0)}')

    return {
        'model': model_key,
        'aus_total': aus_total,
        'aus_correct': aus_correct,
        'accuracy': round(accuracy, 4),
        'accuracy_pct': round(accuracy * 100, 2),
        'contribution_of_3': round(contrib, 4),
        'file_summaries': summaries,
    }

# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    keys = [k for k in MODEL_DIRS if not args or any(a.lower() in k for a in args)]
    if not keys:
        print('无匹配模型，可用：', list(MODEL_DIRS.keys()))
        return

    all_stats = []
    for key in keys:
        print(f'\n[{key}] 开始评测 ...')
        result = evaluate_model(key)
        if result:
            stats = print_report(key, result)
            if stats:
                all_stats.append(stats)
                # 保存每模型明细
                out_path = OUT / f'compliance_{key}_eval.json'
                out_path.write_text(
                    json.dumps({'file_summaries': result['file_summaries'],
                                'rows': result['rows']},
                               ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )

    if len(all_stats) > 1:
        print(f'\n\n{"="*55}')
        print('  全模型横向对比（合规审查 ACC.）')
        print(f'{"="*55}')
        print(f'  {"模型":<12} {"AUS样本":>8} {"正确":>6} {"准确率":>8} {"贡献/3":>8}')
        print(f'  {"-"*46}')
        for s in sorted(all_stats, key=lambda x: -x['accuracy']):
            print(f'  {s["model"]:<12} {s["aus_total"]:>8} {s["aus_correct"]:>6} '
                  f'{s["accuracy_pct"]:>7.1f}% {s["contribution_of_3"]:>7.3f}')

    summary_path = OUT / 'compliance_summary.json'
    summary_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  汇总已保存: {summary_path}')

if __name__ == '__main__':
    main()
