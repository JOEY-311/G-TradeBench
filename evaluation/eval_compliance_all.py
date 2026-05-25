#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_compliance_all.py
对合规审查目录下所有模型/策略的输出文件统一评测，输出汇总表。

评测逻辑：
  - 能提取到铅/镉/农药实测值 → 规则推导 GT，精确比对（AUS 类）
  - 提取不到污染物值        → LLM-judge 评测添加剂/成分合规（EU/USA 类）

用法：
  python eval_compliance_all.py
  python eval_compliance_all.py --no-llm   # 只跑规则部分，跳过 LLM judge
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

COMP_DIR = Path('E:/论文/跨境对齐/评测阶段/合规审查')
OUT_DIR  = Path('E:/论文/跨境对齐/评测阶段/results')
OUT_DIR.mkdir(exist_ok=True)
CACHE_PATH = OUT_DIR / 'compliance_all_cache.json'   # 断点续传缓存

client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=API_KEY)


# ════════════════════════════════════════════════════════════════
#  通用工具
# ════════════════════════════════════════════════════════════════
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

def get_col(row: dict, *keys: str) -> str:
    """大小写不敏感地从 row 中取第一个存在的列。"""
    row_lower = {k.lower(): v for k, v in row.items()}
    for k in keys:
        v = row_lower.get(k.lower(), '')
        if v:
            return v
    return ''


# ════════════════════════════════════════════════════════════════
#  AUS 规则评测
# ════════════════════════════════════════════════════════════════
LIMITS = {
    '日本': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '韩国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '法国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '中国': {'铅': 0.2,  '镉': 0.1,  '农药': 0.01},
    '美国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
    '德国': {'铅': 0.1,  '镉': 0.05, '农药': 0.01},
}

# 目标国名称规范化（处理英文/缩写/中文混用）
DEST_NORM = {
    'japan': '日本', 'jp': '日本', '日本': '日本',
    'china': '中国', 'cn': '中国', '中国': '中国',
    'korea': '韩国', 'kr': '韩国', '韩国': '韩国', 'south korea': '韩国',
    'usa':   '美国', 'us': '美国', '美国': '美国', 'united states': '美国',
    'france':'法国', 'fr': '法国', '法国': '法国',
    'germany':'德国','de': '德国', '德国': '德国', 'gm': '德国',
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

def norm_dest(raw: str) -> str:
    return DEST_NORM.get(raw.strip().lower(), raw.strip())

def derive_gt(poll: dict, dest: str) -> str | None:
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
    # 先找"审查结论："标记（CoT 取最后一个）
    all_m = list(re.finditer(r'审查结论[：:]\s*(.{2,15})', text))
    if all_m:
        v = all_m[-1].group(1).strip('。，\n ')
        if '存在违规风险' in v or ('存在' in v and '风险' in v): return '存在违规风险'
        if '违规' in v and '风险' not in v and '存在' not in v:  return '违规'
        if '合规' in v or '放行' in v:                           return '合规'
        return v[:10]
    # 前两行关键词兜底
    for line in text.split('\n')[:2]:
        if '存在违规风险' in line: return '存在违规风险'
        if '违规' in line and '风险' not in line: return '违规'
        if '合规' in line or '放行' in line:      return '合规'
    return '无法识别'


# ════════════════════════════════════════════════════════════════
#  EU/USA LLM-judge 评测
# ════════════════════════════════════════════════════════════════
ADDITIVE_JUDGE_SYS = """\
你是跨境食品进口合规专家，专注于添加剂和成分合规审查。
根据食品档案中的添加剂、配料信息，判断模型的合规结论是否正确。

各目标国关键标准：
【中国】GB 2760-2014 食品添加剂使用标准（正面清单）；未列入即视为禁用；
        苯甲酸及钠盐（E211）在饮料中≤0.2g/kg；亚硫酸盐（E220-228）需标注。
【日本】厚生劳动省食品添加剂正面清单；未收录即禁用；
        奎拉亚提取物（E999）未获批准；合成着色剂需逐一审核。
【韩国】MFDS 식품첨가물공전正面清单；苯甲酸钠（E211）碳酸饮料≤0.6g/kg；
        部分EU添加剂在韩国未批准。
【美国】FDA GRAS 认定或食品添加剂法规（21 CFR）；总体比EU宽松。
【德国/法国】遵循EU统一法规（EU Reg 1333/2008 食品添加剂法规）。

判断逻辑：
- 违规：有添加剂明确禁用或用量超标
- 存在违规风险：添加剂合规状态不明确，或限量接近上限
- 建议放行：所有可见添加剂和成分符合目标国规定

输出 JSON（仅输出 JSON，不附加任何文字）：
{"gt_verdict": "违规/存在违规风险/建议放行", "model_correct": 0或1, "reason": "简短说明"}
"""

ADDITIVE_JUDGE_TMPL = """\
【目标进口国】：{dest}
【食品档案】：
{profile}

【模型输出的审查结论】：{verdict}

请判断模型结论是否正确，给出标准答案（gt_verdict）和是否正确（model_correct）。
"""

def judge_additive(profile: str, dest: str, verdict: str, retries: int = 3) -> tuple[str, int]:
    """返回 (gt_verdict, is_correct: 0/1)；失败返回 ('error', -1)。"""
    if NO_LLM:
        return 'skip', -1
    prompt = ADDITIVE_JUDGE_TMPL.format(
        dest=dest, profile=profile[:1200], verdict=verdict)
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role': 'system', 'content': ADDITIVE_JUDGE_SYS},
                           {'role': 'user',   'content': prompt}],
                temperature=0, max_tokens=150)
            raw = r.choices[0].message.content or ''
            m = re.search(r'\{.*?\}', raw, re.S)
            if m:
                obj = json.loads(m.group())
                return obj.get('gt_verdict', '').strip(), int(obj.get('model_correct', 0))
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return 'error', -1


# ════════════════════════════════════════════════════════════════
#  单个文件评测
# ════════════════════════════════════════════════════════════════
def eval_file(fp: Path, cache: dict) -> dict:
    """
    评测单个 CSV 文件，返回统计结果。
    cache 用于断点续传：key = f'{fp}:{row_idx}'，value = (gt, is_correct)
    """
    rows = read_csv(fp)
    if not rows:
        return {'file': fp.name, 'total': 0, 'type': 'empty'}

    # 用前3行检测数据类型
    sample = rows[:3]
    has_pollutant = any(
        extract_pollutants(get_col(r, 'FOOD_PROFILE', 'food_profile'))
        for r in sample
    )
    data_type = 'AUS' if has_pollutant else 'EU_USA'

    total = correct = gt_computable = judge_fail = 0
    verdict_dist = {}

    for i, row in enumerate(rows):
        profile = get_col(row, 'FOOD_PROFILE', 'food_profile')
        dest_raw= get_col(row, 'DESTINATION', 'destination')
        dest    = norm_dest(dest_raw)
        output  = get_col(row, 'REVIEW_OUTPUT', 'review_output', 'output', 'model_output')
        mv      = extract_verdict(output)
        verdict_dist[mv] = verdict_dist.get(mv, 0) + 1
        total += 1

        cache_key = f'{fp}:{i}'

        if data_type == 'AUS':
            poll = extract_pollutants(profile)
            gt   = derive_gt(poll, dest)
            if gt is None:
                continue
            gt_computable += 1
            hit = (mv == gt)
            correct += int(hit)

        else:  # EU/USA → LLM judge
            if cache_key in cache:
                gt, ok = cache[cache_key]
            else:
                gt, ok = judge_additive(profile, dest, mv)
                cache[cache_key] = (gt, ok)
                time.sleep(0.4)

            if ok == -1:
                judge_fail += 1
                continue
            gt_computable += 1
            correct += ok

    acc = correct / gt_computable * 100 if gt_computable else None
    return {
        'file':          fp.name,
        'type':          data_type,
        'total':         total,
        'gt_computable': gt_computable,
        'correct':       correct,
        'acc':           round(acc, 1) if acc is not None else None,
        'verdict_dist':  verdict_dist,
        'judge_fail':    judge_fail,
    }


# ════════════════════════════════════════════════════════════════
#  单个模型目录评测
# ════════════════════════════════════════════════════════════════
def eval_model_dir(model_dir: Path, cache: dict) -> dict:
    aus_correct = aus_total = 0
    eu_usa_correct = eu_usa_total = 0
    file_results = []

    csv_files = sorted(model_dir.glob('*.csv'))
    print(f'\n  {model_dir.name} ({len(csv_files)} 文件)')

    for fp in csv_files:
        res = eval_file(fp, cache)
        file_results.append(res)

        if res.get('gt_computable', 0) == 0:
            print(f'    {fp.name:<45} 无可评测样本')
            continue

        tag = '[规则]' if res['type'] == 'AUS' else '[judge]'
        acc_str = f'{res["acc"]:.1f}%' if res["acc"] is not None else 'N/A'
        print(f'    {tag} {fp.name:<42} {res["correct"]}/{res["gt_computable"]} = {acc_str}')

        if res['type'] == 'AUS':
            aus_correct += res['correct']
            aus_total   += res['gt_computable']
        else:
            eu_usa_correct += res['correct']
            eu_usa_total   += res['gt_computable']

    aus_acc    = aus_correct    / aus_total    * 100 if aus_total    else None
    eu_usa_acc = eu_usa_correct / eu_usa_total * 100 if eu_usa_total else None
    all_correct= aus_correct + eu_usa_correct
    all_total  = aus_total   + eu_usa_total
    overall    = all_correct / all_total * 100 if all_total else None

    return {
        'model':          model_dir.name,
        'aus_acc':        round(aus_acc,    1) if aus_acc    is not None else None,
        'aus_n':          aus_total,
        'eu_usa_acc':     round(eu_usa_acc, 1) if eu_usa_acc is not None else None,
        'eu_usa_n':       eu_usa_total,
        'overall_acc':    round(overall,    1) if overall    is not None else None,
        'overall_n':      all_total,
        'file_results':   file_results,
    }


# ════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════
def main():
    print('╔══════════════════════════════════════════════════════════╗')
    print('║       合规审查全模型评测（规则 + LLM-judge）              ║')
    print(f'║  Judge: {JUDGE_MODEL:<49}║')
    print(f'║  LLM评分: {"已跳过" if NO_LLM else "启用":<52}║')
    print('╚══════════════════════════════════════════════════════════╝')

    # 加载断点缓存
    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding='utf-8'))
            print(f'\n  已加载缓存：{len(cache)} 条 judge 结果')
        except Exception:
            pass

    model_dirs = sorted(
        d for d in COMP_DIR.iterdir()
        if d.is_dir() and list(d.glob('*.csv'))
    )
    print(f'\n  发现 {len(model_dirs)} 个模型目录')

    all_results = []
    try:
        for model_dir in model_dirs:
            res = eval_model_dir(model_dir, cache)
            all_results.append(res)
            # 每个模型完成后保存缓存
            CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                   encoding='utf-8')
    except KeyboardInterrupt:
        print('\n\n  ⚠ 用户中断，已保存进度，下次运行将续跑。')

    # ── 保存详细结果 ─────────────────────────────────────────────
    out_path = OUT_DIR / 'compliance_all_results.json'
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    print(f'\n  详细结果已保存: {out_path}')

    # ── 打印汇总表 ───────────────────────────────────────────────
    print(f'\n{"═"*78}')
    print(f'  {"模型/策略":<28} {"AUS准确率":>10} {"EU/USA准确率":>12} {"综合准确率":>10} {"样本数":>6}')
    print(f'  {"─"*72}')

    # 分组：baseline 模型 / claude 策略 / 其他
    baselines = [r for r in all_results if '-' not in r['model'] or
                 r['model'] in ('claude-sonnet', 'gemini-baseline', 'qwen-baseline')]
    strategies = [r for r in all_results if r not in baselines]

    def fmt_row(r):
        aus    = f'{r["aus_acc"]:.1f}% ({r["aus_n"]})' if r['aus_acc'] is not None else '—'
        eu_usa = f'{r["eu_usa_acc"]:.1f}% ({r["eu_usa_n"]})' if r['eu_usa_acc'] is not None else '—'
        overall= f'{r["overall_acc"]:.1f}%' if r['overall_acc'] is not None else '—'
        n      = str(r['overall_n'])
        print(f'  {r["model"]:<28} {aus:>15} {eu_usa:>15} {overall:>10} {n:>6}')

    print('  【Baseline 模型】')
    for r in baselines:
        fmt_row(r)

    if strategies:
        print(f'\n  【策略变体】')
        for r in strategies:
            fmt_row(r)

    print(f'{"═"*78}')

    # ── 保存 CSV 表格 ─────────────────────────────────────────────
    csv_path = OUT_DIR / 'compliance_all_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['模型', 'AUS准确率(%)', 'AUS样本数',
                    'EU_USA准确率(%)', 'EU_USA样本数',
                    '综合准确率(%)', '总样本数'])
        for r in all_results:
            w.writerow([
                r['model'],
                r['aus_acc']    if r['aus_acc']    is not None else '',
                r['aus_n'],
                r['eu_usa_acc'] if r['eu_usa_acc'] is not None else '',
                r['eu_usa_n'],
                r['overall_acc']if r['overall_acc']is not None else '',
                r['overall_n'],
            ])
    print(f'  汇总表已保存: {csv_path}')


if __name__ == '__main__':
    main()
