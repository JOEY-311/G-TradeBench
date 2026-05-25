"""
标准对齐任务评测脚本
指标体系（满分4分/题）：
  D1 问题识别 (0/1) — LLM-as-a-judge：模型是否准确定位关键合规问题
  D2 法规依据 (0/2) — LLM-as-a-judge：是否引用了正确且具体的法规/标准
  D3 方案覆盖 (0/1) — LLM-as-a-judge：解决方案是否涵盖参考答案的主要行动项

支持7种任务类型：冲突判断/限量对齐/配料准入/准入程序/标签对齐/多国流通/俗名映射

用法：
  set OPENROUTER_API_KEY=sk-or-...
  python eval_alignment.py                    # 跑全部模型
  python eval_alignment.py deepseek           # 单模型
  python eval_alignment.py deepseek --resume  # 断点续跑
"""

import csv, re, json, os, sys, time
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

# ── 配置 ─────────────────────────────────────────────────────────────────────
BASE = Path('E:/论文/跨境对齐/评测阶段/标准对齐')
OUT  = Path('E:/论文/跨境对齐/评测阶段/results')
OUT.mkdir(exist_ok=True)

JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'google/gemini-2.0-flash-001')
SAVE_EVERY  = 10

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.environ.get('OPENROUTER_API_KEY', ''),
)

# ── 模型目录映射 ──────────────────────────────────────────────────────────────
MODEL_DIRS = {
    'deepseek':    BASE / 'deepseek-V3.2',
    'gemini':      BASE / 'gemini-3.1pro',
    'gpt':         BASE / 'gpt-5.4',
    'grok':        BASE / 'grok-4.20',
    'qwen':        BASE / 'qwen-3.6plus',
    'deepseek-v4': BASE / 'deepseek-V4',
    'gpt-5.5':     BASE / 'gpt-5.5',
}

# 任务类型关键字 → 名称映射
TASK_KW = [
    ('冲突判断', '冲突判断'),
    ('限量对齐', '限量对齐'),
    ('配料准入', '配料准入'),
    ('准入程序', '准入程序'),
    ('标签对齐', '标签对齐'),
    ('多国流通', '多国流通'),
    ('俗名映射', '俗名映射'),
    ('HSCODE','HSCODE')
]

SKIP_PATTERNS = ['NOCHINA', 'HSCODE', 'MODEL_LIST']

# ── 工具函数 ──────────────────────────────────────────────────────────────────
def read_file(path: Path) -> list[dict]:
    """支持真实 xlsx（openpyxl）和 .csv（多编码探测）"""
    if path.suffix == '.xlsx':
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            ws = wb.active
            rows_iter = ws.iter_rows(values_only=True)
            headers = [str(h) if h is not None else '' for h in next(rows_iter)]
            if 'OUTPUT' in headers:
                result = [dict(zip(headers, (str(v) if v is not None else '' for v in row)))
                          for row in rows_iter]
                wb.close()
                return result
            wb.close()
        except Exception:
            pass
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin-1'):
        try:
            with open(path, encoding=enc) as f:
                rows = list(csv.DictReader(f))
            if rows and 'OUTPUT' in rows[0]:
                return rows
        except Exception:
            continue
    return []

def infer_task_type(filename: str) -> str:
    fn = filename.upper()
    for kw, name in TASK_KW:
        if kw in filename:
            return name
    return '未知'

def should_skip(filename: str) -> bool:
    return any(p in filename.upper() for p in SKIP_PATTERNS)

# ── LLM 评判（三维度合并调用，减少API次数） ──────────────────────────────────
JUDGE_SYSTEM = """你是食品法规领域的专业评审专家，负责对AI模型的合规咨询回答进行客观打分。
请严格按照评分标准输出JSON，不要添加任何解释性文字。"""

JUDGE_USER_TMPL = """## 评测任务
任务类型：{task_type}

## 题目（场景描述）
{question}

## 参考答案
{reference}

## 模型回答
{model_output}

## 评分标准

D1 问题识别（0或1分）：
- 1分：模型准确识别出场景中的核心合规问题（与参考答案的核心判断一致）
- 0分：未识别出核心问题，或判断方向与参考答案相反

D2 法规依据（0、1或2分）：
- 2分：引用了正确且具体的法规/标准（如GB编号、EU条例编号、FDA规则等），涵盖参考答案中的主要法规依据
- 1分：有引用相关法规但不够具体（如只说"相关标准"），或遗漏了参考答案中的关键法规
- 0分：无实质性法规引用，或引用明显错误

D3 方案覆盖（0或1分）：
- 1分：解决方案涵盖了参考答案中的主要行动项（不要求完全一致，关键步骤覆盖即可）
- 0分：方案缺失、过于笼统，或与参考答案主要行动项无实质交集

请输出：{{"D1": <0或1>, "D2": <0、1或2>, "D3": <0或1>, "reasons": {{"D1": "<15字内>", "D2": "<20字内>", "D3": "<15字内>"}}}}"""

def llm_judge(task_type: str, question: str, reference: str,
              model_output: str, retries: int = 3) -> dict:
    prompt = JUDGE_USER_TMPL.format(
        task_type=task_type,
        question=question[:600],
        reference=reference[:800],
        model_output=model_output[:1200],
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {'role': 'system', 'content': JUDGE_SYSTEM},
                    {'role': 'user',   'content': prompt},
                ],
                temperature=0,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', raw, re.S)
            if m:
                obj = json.loads(m.group())
                d1 = min(max(int(obj.get('D1', 0)), 0), 1)
                d2 = min(max(int(obj.get('D2', 0)), 0), 2)
                d3 = min(max(int(obj.get('D3', 0)), 0), 1)
                reasons = obj.get('reasons', {})
                return {
                    'D1': d1, 'D1_reason': str(reasons.get('D1', '')),
                    'D2': d2, 'D2_reason': str(reasons.get('D2', '')),
                    'D3': d3, 'D3_reason': str(reasons.get('D3', '')),
                }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {'D1': 0, 'D1_reason': f'API错误:{e}',
                        'D2': 0, 'D2_reason': '', 'D3': 0, 'D3_reason': ''}
    return {'D1': 0, 'D1_reason': '调用失败', 'D2': 0, 'D2_reason': '', 'D3': 0, 'D3_reason': ''}

# ── 评测单模型 ────────────────────────────────────────────────────────────────
def evaluate_model(model_key: str, resume: bool = False) -> list[dict]:
    model_dir = MODEL_DIRS.get(model_key)
    if not model_dir or not model_dir.is_dir():
        print(f'  [跳过] 目录不存在: {model_dir}')
        return []

    out_path = OUT / f'alignment_{model_key}_eval.json'
    done_keys = set()
    results   = []

    if resume and out_path.exists():
        results   = json.loads(out_path.read_text(encoding='utf-8'))
        done_keys = {(r['file'], r['row_index']) for r in results}
        print(f'  [续跑] 已完成 {len(results)} 条')

    # 收集待评测文件
    scorable_files = []
    for fp in sorted(model_dir.iterdir()):
        if not fp.is_file():
            continue
        if should_skip(fp.name):
            continue
        if fp.suffix not in ('.csv', '.xlsx'):
            continue
        rows = read_file(fp)
        if not rows:
            continue
        task_type = infer_task_type(fp.name)
        scorable_files.append((fp, rows, task_type))

    total_pending = sum(
        sum(1 for i, _ in enumerate(rows) if (fp.name, i) not in done_keys)
        for fp, rows, _ in scorable_files
    )
    print(f'  待评测 {total_pending} 条，共 {len(scorable_files)} 个文件')

    for fp, rows, task_type in scorable_files:
        for i, row in enumerate(rows):
            key = (fp.name, i)
            if key in done_keys:
                continue

            q   = row.get('QUESTION', '')
            ref = row.get('REFERENCE', '')
            out = row.get('OUTPUT', '')

            scores = llm_judge(task_type, q, ref, out)
            total  = scores['D1'] + scores['D2'] + scores['D3']

            result = {
                'file': fp.name,
                'task_type': task_type,
                'row_index': i,
                'D1': scores['D1'], 'D1_reason': scores['D1_reason'],
                'D2': scores['D2'], 'D2_reason': scores['D2_reason'],
                'D3': scores['D3'], 'D3_reason': scores['D3_reason'],
                'total': total,
                'normalized': round(total / 4 * 100, 1),
            }
            results.append(result)

            d1, d2, d3 = scores['D1'], scores['D2'], scores['D3']
            print(f'  [{fp.name[:20]} #{i}] D1={d1} D2={d2} D3={d3} 总={total}/4')

            if len(results) % SAVE_EVERY == 0:
                out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  已保存: {out_path.name}')
    return results

# ── 打印单模型报告 ────────────────────────────────────────────────────────────
def print_report(model_key: str, results: list[dict]) -> dict | None:
    n = len(results)
    if n == 0:
        return None

    d1_avg    = sum(r['D1'] for r in results) / n
    d2_avg    = sum(r['D2'] for r in results) / n
    d3_avg    = sum(r['D3'] for r in results) / n
    total_avg = sum(r['total'] for r in results) / n

    print(f'\n{"="*55}')
    print(f'  {model_key.upper()}  |  {n} 条  |  满分4分/题')
    print(f'{"="*55}')
    print(f'  D1 问题识别: {d1_avg:.3f}/1  得分率 {d1_avg*100:.1f}%')
    print(f'  D2 法规依据: {d2_avg:.3f}/2  得分率 {d2_avg/2*100:.1f}%')
    print(f'  D3 方案覆盖: {d3_avg:.3f}/1  得分率 {d3_avg*100:.1f}%')
    print(f'  ── 平均总分: {total_avg:.3f}/4  标准化: {total_avg/4*100:.1f}%')
    contrib = total_avg / 4 * 4
    print(f'  ── 最终贡献(×4权重): {contrib:.3f}/4.0')

    # 按任务类型分组
    by_task = defaultdict(list)
    for r in results:
        by_task[r['task_type']].append(r['total'])

    print(f'\n  按任务类型:')
    for task, tots in sorted(by_task.items()):
        avg = sum(tots) / len(tots)
        print(f'    {task:<8} {len(tots)}条  均分={avg:.2f}/4  {avg/4*100:.1f}%')

    return {
        'model': model_key,
        'n': n,
        'D1_avg': round(d1_avg, 4),
        'D2_avg': round(d2_avg, 4),
        'D3_avg': round(d3_avg, 4),
        'total_avg': round(total_avg, 4),
        'normalized_pct': round(total_avg / 4 * 100, 2),
        'contribution_of_4': round(contrib, 4),
        'by_task': {t: {'n': len(v), 'avg': round(sum(v)/len(v), 4)} for t, v in by_task.items()},
    }

# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    args   = [a for a in sys.argv[1:] if not a.startswith('--')]
    resume = '--resume' in sys.argv

    keys = [k for k in MODEL_DIRS if not args or any(a.lower() in k for a in args)]
    if not keys:
        print('无匹配模型，可用：', list(MODEL_DIRS.keys()))
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
        print('  全模型横向对比（标准对齐 D1/D2/D3）')
        print(f'{"="*55}')
        print(f'  {"模型":<12} {"D1%":>7} {"D2%":>7} {"D3%":>7} {"总得分率":>9} {"贡献/4":>8}')
        print(f'  {"-"*52}')
        for s in sorted(all_stats, key=lambda x: -x['normalized_pct']):
            print(f'  {s["model"]:<12} {s["D1_avg"]*100:>6.1f}% '
                  f'{s["D2_avg"]/2*100:>6.1f}% '
                  f'{s["D3_avg"]*100:>6.1f}% '
                  f'{s["normalized_pct"]:>8.1f}% '
                  f'{s["contribution_of_4"]:>7.3f}')

    summary_path = OUT / 'alignment_summary.json'
    summary_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  汇总已保存: {summary_path}')

if __name__ == '__main__':
    main()
