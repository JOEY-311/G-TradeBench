"""
三任务综合汇总脚本
读取 results/ 目录下三个任务的 summary JSON，
按照指标集合权重合并为10分制最终得分：
  规则更新（F1+F2+F3）→ 贡献 3分
  标准对齐（D1+D2+D3）→ 贡献 4分
  合规审查（ACC.）    → 贡献 3分
  总分上限 = 10分

用法：
  python eval_summary.py
  （需先分别运行 eval_temporal.py / eval_alignment.py / eval_compliance.py）
"""

import json
from pathlib import Path

RESULTS = Path('E:/论文/跨境对齐/评测阶段/results')

WEIGHT = {'temporal': 3, 'alignment': 4, 'compliance': 3}   # 三任务权重，合计10

def load(fname: str) -> list[dict]:
    p = RESULTS / fname
    if not p.exists():
        print(f'  [警告] 找不到 {fname}，跳过该任务')
        return []
    return json.loads(p.read_text(encoding='utf-8'))

def main():
    t_data = {s['model']: s for s in load('temporal_summary.json')}
    a_data = {s['model']: s for s in load('alignment_summary.json')}
    c_data = {s['model']: s for s in load('compliance_summary.json')}

    all_models = sorted(set(t_data) | set(a_data) | set(c_data))

    combined = []
    for model in all_models:
        t = t_data.get(model, {})
        a = a_data.get(model, {})
        c = c_data.get(model, {})

        # 各任务归一化得分率（0~1）
        t_rate = t.get('normalized_pct', 0) / 100   # 规则更新
        a_rate = a.get('normalized_pct', 0) / 100   # 标准对齐
        c_rate = c.get('accuracy', 0)                # 合规审查

        # 加权得分
        t_pts = t_rate * WEIGHT['temporal']     # 最高3分
        a_pts = a_rate * WEIGHT['alignment']    # 最高4分
        c_pts = c_rate * WEIGHT['compliance']   # 最高3分
        total = t_pts + a_pts + c_pts           # 最高10分

        combined.append({
            'model':       model,
            '规则更新_得分率': f'{t_rate*100:.1f}%',
            '规则更新_贡献':  round(t_pts, 3),
            '标准对齐_得分率': f'{a_rate*100:.1f}%',
            '标准对齐_贡献':  round(a_pts, 3),
            '合规审查_准确率': f'{c_rate*100:.1f}%',
            '合规审查_贡献':  round(c_pts, 3),
            '总分_10分制':   round(total, 3),
            # 细项备查
            '_temporal': {k: t.get(k) for k in ['F1_avg','F2_avg','F3_avg','normalized_pct']},
            '_alignment': {k: a.get(k) for k in ['D1_avg','D2_avg','D3_avg','normalized_pct']},
            '_compliance': {k: c.get(k) for k in ['accuracy_pct','aus_total','aus_correct']},
        })

    combined.sort(key=lambda x: -x['总分_10分制'])

    # ── 打印汇总表 ─────────────────────────────────────────────────────────
    print(f'\n{"="*75}')
    print('  三任务综合评测结果（满分10分）')
    print(f'{"="*75}')
    print(f'  {"模型":<14} {"规则更新":>8} {"标准对齐":>8} {"合规审查":>8} {"总分/10":>8}  {"排名"}')
    print(f'  {"权重":>14} {"(×3)":>8} {"(×4)":>8} {"(×3)":>8}')
    print(f'  {"-"*65}')
    for rank, row in enumerate(combined, 1):
        print(f'  {row["model"]:<14} '
              f'{row["规则更新_贡献"]:>7.3f}  '
              f'{row["标准对齐_贡献"]:>7.3f}  '
              f'{row["合规审查_贡献"]:>7.3f}  '
              f'{row["总分_10分制"]:>7.3f}   #{rank}')

    print(f'\n  细项得分率:')
    print(f'  {"模型":<14} {"F1%":>6} {"F2%":>6} {"F3%":>6}  {"D1%":>6} {"D2%":>6} {"D3%":>6}  {"ACC%":>6}')
    print(f'  {"-"*65}')
    for row in combined:
        t = row['_temporal']
        a = row['_alignment']
        c = row['_compliance']
        f1 = f'{t.get("F1_avg",0)*100:.0f}%' if t.get("F1_avg") is not None else 'N/A'
        f2 = f'{t.get("F2_avg",0)*100:.0f}%' if t.get("F2_avg") is not None else 'N/A'
        f3 = f'{t.get("F3_avg",0)*100:.0f}%' if t.get("F3_avg") is not None else 'N/A'
        d1 = f'{a.get("D1_avg",0)*100:.0f}%' if a.get("D1_avg") is not None else 'N/A'
        d2 = f'{a.get("D2_avg",0)/2*100:.0f}%' if a.get("D2_avg") is not None else 'N/A'
        d3 = f'{a.get("D3_avg",0)*100:.0f}%' if a.get("D3_avg") is not None else 'N/A'
        acc = f'{c.get("accuracy_pct",0):.0f}%' if c.get("accuracy_pct") is not None else 'N/A'
        print(f'  {row["model"]:<14} {f1:>6} {f2:>6} {f3:>6}  {d1:>6} {d2:>6} {d3:>6}  {acc:>6}')

    out_path = RESULTS / 'final_summary.json'
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  最终汇总已保存: {out_path}')

if __name__ == '__main__':
    main()
