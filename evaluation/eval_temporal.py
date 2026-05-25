"""
规则更新任务评测脚本
指标体系（满分3分/题）：
  F1 状态判断 (0/1) — 规则打分：模型能否正确判断法规的阶段（已生效/草案/废止）
  F2 物质识别 (0/1) — LLM-as-a-judge：模型能否识别目标物质及变更类型
  F3 日期抽取 (0/1) — 规则打分：模型能否准确抽取生效日期或正确处理模糊/无日期情况

用法：
  set OPENROUTER_API_KEY=sk-or-...
  python eval_temporal.py                      # 跑全部模型
  python eval_temporal.py deepseek             # 单模型（文件名含关键字）
  python eval_temporal.py deepseek --resume    # 断点续跑
"""

import csv, re, json, os, sys, time
from pathlib import Path
from openai import OpenAI

# ── 配置 ─────────────────────────────────────────────────────────────────────
BASE   = Path('E:/论文/跨境对齐/评测阶段/规则更新')
OUT    = Path('E:/论文/跨境对齐/评测阶段/results')
OUT.mkdir(exist_ok=True)

JUDGE_MODEL  = os.environ.get('JUDGE_MODEL', 'google/gemini-2.0-flash-001')
SAVE_EVERY   = 10   # 每N条保存一次检查点

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.environ.get('OPENROUTER_API_KEY', ''),
)

# ── 模型文件映射 ──────────────────────────────────────────────────────────────
MODEL_FILES = {
    'deepseek':    BASE / 'temporal_ds.csv',
    'gemini':      BASE / 'temporal_gemini.csv',
    'gpt':         BASE / 'temporal_gpt.csv',
    'grok':        BASE / 'temporal_grok.csv',
    'qwen':        BASE / 'temporal_qwen.csv',
    'gpt-5.5':     BASE / 'temporal_gpt-5.5.csv',
    'deepseek-v4': BASE / 'temporal_test_results_deepseekv4.csv',
}

# ── 工具函数 ──────────────────────────────────────────────────────────────────
def read_csv(path: Path) -> list[dict]:
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            with open(path, encoding=enc) as f:
                rows = list(csv.DictReader(f))
            if rows:
                return rows
        except Exception:
            continue
    return []

def parse_gt(gt_text: str) -> dict:
    """从 ground_truth 字段提取 legal_status / effective_date / target_details"""
    def _get(tag):
        m = re.search(rf'\[{tag}\]:\s*(.+?)(?=\n\[|\Z)', gt_text, re.S)
        return m.group(1).strip() if m else ''

    return {
        'legal_status':   _get('Legal Status'),
        'effective_date': _get('Proposed Effective Date'),
        'target_details': _get('Target Details'),
    }

# ── F1 规则打分：法规状态判断 ─────────────────────────────────────────────────
_PROPOSAL_KW = ['proposed', 'draft', 'under consideration', 'open for comment',
                'consultation', 'not yet', 'to be enacted', 'seeking comment']
_ENACTED_KW  = ['in effect', 'effective', 'enacted', 'adopted', 'promulgated',
                'notification of update', 'immediately', 'officially', 'has been']

def score_f1(gt: dict, model_output: str) -> tuple[int, str]:
    status = gt['legal_status'].lower()
    out    = model_output.lower()

    is_proposal = any(kw in status for kw in ['proposed', 'draft'])
    is_enacted  = any(kw in status for kw in ['notification of update', 'immediately effective', 'enacted'])

    if is_proposal:
        if any(kw in out for kw in _PROPOSAL_KW):
            return 1, f'GT=提案，模型正确识别草案/提案状态'
        return 0, f'GT=提案，但模型未识别出草案状态（GT: {gt["legal_status"][:60]}）'
    elif is_enacted:
        if any(kw in out for kw in _ENACTED_KW):
            return 1, f'GT=已生效，模型正确识别生效状态'
        return 0, f'GT=已生效，但模型未识别出已生效状态（GT: {gt["legal_status"][:60]}）'
    else:
        # 未知类型：宽松处理，检查模型是否有明确表态
        if any(kw in out for kw in _ENACTED_KW + _PROPOSAL_KW):
            return 1, f'GT状态未分类，模型有明确表态'
        return 0, f'GT状态未分类且模型无明确表态（GT: {gt["legal_status"][:60]}）'

# ── F3 规则打分：生效日期抽取 ─────────────────────────────────────────────────
_YEAR_PAT  = re.compile(r'\b(20\d{2})\b')
_MONTH_PAT = re.compile(
    r'\b(january|february|march|april|may|june|july|august'
    r'|september|october|november|december)\b', re.I)
_VAGUE_GT  = re.compile(
    r'after a certain period|not specified|vague|upon|transition|once finalized'
    r'|pending|to be determined|tbd|certain period', re.I)
_NA_GT     = re.compile(r'^(nan|not applicable|n/a|na|none)\s*\.?\s*$', re.I)

def score_f3(gt: dict, model_output: str) -> tuple[int, str]:
    gt_date = gt['effective_date'].strip()
    out     = model_output.lower()

    # 情况1：GT无日期（nan/N/A）→ 模型不应发明具体日期
    if _NA_GT.match(gt_date):
        invented = bool(_YEAR_PAT.search(model_output) or _MONTH_PAT.search(model_output))
        if invented:
            return 0, 'GT无日期，但模型发明了具体日期'
        return 1, 'GT无日期，模型未发明日期（正确）'

    # 情况2：GT日期模糊 → 模型应承认不确定性
    if _VAGUE_GT.search(gt_date):
        vague_in_out = bool(re.search(
            r'after|period|certain|transition|once|following|upon|unclear|uncertain|pending', out))
        if vague_in_out:
            return 1, 'GT日期模糊，模型正确承认不确定性'
        return 0, 'GT日期模糊，但模型声称有确切日期'

    # 情况3：GT有具体年份 → 模型须包含该年份
    gt_years = set(_YEAR_PAT.findall(gt_date))
    if gt_years:
        out_years = set(_YEAR_PAT.findall(model_output))
        matched   = gt_years & out_years
        if matched:
            return 1, f'GT年份 {gt_years}，模型正确抽取 {matched}'
        return 0, f'GT年份 {gt_years}，模型未抽取（模型年份: {out_years}）'

    # 情况4：GT有月份但无年份
    gt_months = set(_MONTH_PAT.findall(gt_date.lower()))
    out_months = set(_MONTH_PAT.findall(out))
    if gt_months and gt_months & out_months:
        return 1, f'GT月份 {gt_months}，模型正确匹配'

    return 0, f'GT日期无法匹配（GT: {gt_date[:80]}）'

# ── F2 LLM评判：物质识别 ──────────────────────────────────────────────────────
F2_SYSTEM = """你是一名食品法规专家评审，负责评判AI模型的回答质量。
你的任务是判断模型是否正确识别出法规变更所涉及的目标物质和变更类型。
请只输出JSON，格式：{"score": 0或1, "reason": "简短说明"}"""

F2_USER_TMPL = """【题目】
{question}

【参考答案中的关键信息】
{target_details}

【模型回答】
{model_output}

评分标准（0/1）：
- 1分：模型正确识别了目标物质（如具体农药名/添加剂/污染物名称）且基本正确描述了变更类型（如MRL修订/新增/撤销/修改规格等）
- 0分：物质名称错误/过于笼统，或变更类型理解有误

仅输出JSON：{{"score": <0或1>, "reason": "<20字以内的简短说明>"}}"""

def llm_f2(question: str, gt: dict, model_output: str, retries: int = 3) -> tuple[int, str]:
    prompt = F2_USER_TMPL.format(
        question=question[:800],
        target_details=gt['target_details'][:400],
        model_output=model_output[:1000],
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {'role': 'system', 'content': F2_SYSTEM},
                    {'role': 'user',   'content': prompt},
                ],
                temperature=0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            m   = re.search(r'\{.*\}', raw, re.S)
            if m:
                obj = json.loads(m.group())
                return int(obj.get('score', 0)), str(obj.get('reason', ''))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return 0, f'API错误: {e}'
    return 0, 'API调用失败'

# ── 单行评分 ──────────────────────────────────────────────────────────────────
def score_row(row: dict) -> dict:
    gt     = parse_gt(row.get('ground_truth', ''))
    q      = row.get('instruction', '')
    output = row.get('model_output', '')
    idx    = row.get('index', '?')

    f1_score, f1_reason = score_f1(gt, output)
    f2_score, f2_reason = llm_f2(q, gt, output)
    f3_score, f3_reason = score_f3(gt, output)

    total = f1_score + f2_score + f3_score
    return {
        'index': idx,
        'gt_status':  gt['legal_status'][:80],
        'gt_date':    gt['effective_date'][:80],
        'F1': {'score': f1_score, 'reason': f1_reason},
        'F2': {'score': f2_score, 'reason': f2_reason},
        'F3': {'score': f3_score, 'reason': f3_reason},
        'total': total,
        'normalized': round(total / 3 * 100, 1),
    }

# ── 评测单模型 ────────────────────────────────────────────────────────────────
def evaluate_model(model_key: str, resume: bool = False) -> list[dict]:
    path = MODEL_FILES.get(model_key)
    if not path or not path.exists():
        print(f'  [跳过] 找不到文件: {path}')
        return []

    out_path = OUT / f'temporal_{model_key}_eval.json'
    done_idx = set()
    results  = []

    if resume and out_path.exists():
        results  = json.loads(out_path.read_text(encoding='utf-8'))
        done_idx = {str(r['index']) for r in results}
        print(f'  [续跑] 已完成 {len(results)} 条')

    rows = read_csv(path)
    pending = [r for r in rows if str(r.get('index', '')) not in done_idx]
    print(f'  待评测 {len(pending)} 条（共 {len(rows)} 条）')

    for i, row in enumerate(pending):
        result = score_row(row)
        results.append(result)
        f1, f2, f3 = result['F1']['score'], result['F2']['score'], result['F3']['score']
        print(f'  [{result["index"]}] F1={f1} F2={f2} F3={f3} 总={result["total"]}/3')

        if (i + 1) % SAVE_EVERY == 0:
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    return results

# ── 打印单模型报告 ────────────────────────────────────────────────────────────
def print_report(model_key: str, results: list[dict]):
    n = len(results)
    if n == 0:
        return

    f1_avg = sum(r['F1']['score'] for r in results) / n
    f2_avg = sum(r['F2']['score'] for r in results) / n
    f3_avg = sum(r['F3']['score'] for r in results) / n
    total_avg = sum(r['total'] for r in results) / n

    print(f'\n{"="*55}')
    print(f'  {model_key.upper()}  |  {n} 条  |  满分3分/题')
    print(f'{"="*55}')
    print(f'  F1 状态判断: {f1_avg:.3f}/1  得分率 {f1_avg*100:.1f}%')
    print(f'  F2 物质识别: {f2_avg:.3f}/1  得分率 {f2_avg*100:.1f}%')
    print(f'  F3 日期抽取: {f3_avg:.3f}/1  得分率 {f3_avg*100:.1f}%')
    print(f'  ── 平均总分: {total_avg:.3f}/3  标准化: {total_avg/3*100:.1f}%')
    # 最终贡献（指标集合权重：规则更新占10分中的3分）
    contrib = total_avg / 3 * 3
    print(f'  ── 最终贡献(×3权重): {contrib:.3f}/3.0')

    return {
        'model': model_key,
        'n': n,
        'F1_avg': round(f1_avg, 4),
        'F2_avg': round(f2_avg, 4),
        'F3_avg': round(f3_avg, 4),
        'total_avg': round(total_avg, 4),
        'normalized_pct': round(total_avg / 3 * 100, 2),
        'contribution_of_3': round(contrib, 4),
    }

# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    args   = [a for a in sys.argv[1:] if not a.startswith('--')]
    resume = '--resume' in sys.argv

    keys = [k for k in MODEL_FILES if not args or any(a.lower() in k for a in args)]
    if not keys:
        print('无匹配模型，可用：', list(MODEL_FILES.keys()))
        return

    all_stats = []
    for key in keys:
        print(f'\n[{key}] 开始评测 ...')
        results = evaluate_model(key, resume=resume)
        if results:
            stats = print_report(key, results)
            if stats:
                all_stats.append(stats)

    if len(all_stats) > 1:
        print(f'\n\n{"="*55}')
        print('  全模型横向对比（规则更新 F1/F2/F3）')
        print(f'{"="*55}')
        print(f'  {"模型":<12} {"F1%":>7} {"F2%":>7} {"F3%":>7} {"总得分率":>9} {"贡献/3":>8}')
        print(f'  {"-"*52}')
        for s in sorted(all_stats, key=lambda x: -x['normalized_pct']):
            print(f'  {s["model"]:<12} {s["F1_avg"]*100:>6.1f}% {s["F2_avg"]*100:>6.1f}% '
                  f'{s["F3_avg"]*100:>6.1f}% {s["normalized_pct"]:>8.1f}% {s["contribution_of_3"]:>7.3f}')

    summary_path = OUT / 'temporal_summary.json'
    summary_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  汇总已保存: {summary_path}')

if __name__ == '__main__':
    main()
