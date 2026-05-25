"""
对已有结果文件中的 alignment 条目用新 judge prompt 重新打分，
不重新调用主模型，只花几十秒。

用法：
    python rescore_alignment.py
    python rescore_alignment.py --results results/skill_openrouter_eval/results_xxx.json
"""
import json, re, argparse
from pathlib import Path
from openai import OpenAI

# ── 配置（与 eval_skill_openrouter.py 保持一致） ──────────────────────────
API_KEY     = os.environ.get('OPENROUTER_API_KEY', '')
JUDGE_MODEL = "anthropic/claude-haiku-4-5"
BASE_URL    = "https://openrouter.ai/api/v1"
OUT_DIR     = Path(__file__).parent / "results" / "skill_openrouter_eval"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

ALIGNMENT_JUDGE_SYSTEM = """\
你是严格的评测员，对下列回答按 0–10 分评分。严格执行以下规则，不得因答案"听起来合理"而放宽标准。

【评分细则】

1. 精确数值/编码（3分）
   - HS编码：必须精确到8位数字，仅给出4位章节号得1分，未给出得0分
   - 限量值：必须给出精确数字+单位（如"0.5 g/kg"），只说"符合限量"或"在允许范围内"得0分
   - 合规结论：必须依托具体检测值与标准值对比，笼统下结论得0分

2. 法规引用精确性（4分）
   - 引用标准必须含编号+年份（如"GB 2760-2014"），只说"GB标准"或"国家标准"得1分
   - 同时引用具体条款/附录（如"附录A 表A.2"，"第X条"）再得1分
   - 共可得2分上限；如引用了不存在的标准编号或条款，该项得0分并总分扣1分
   - 参考答案涉及的每个关键法规若均正确引用，额外得2分

3. 关键问题覆盖（2分）
   - 对照参考答案，每遗漏一个核心要点（如陷阱识别、补救措施、分类依据）扣1分
   - 最低得0分

4. 准确性惩罚（-1分）
   - 若出现明显错误事实（如错误的限量数值、错误的标准归属），总分扣1分

仅输出一个整数（0–10），不附加任何解释。
"""

def judge(question: str, reference: str, standard: str, prediction: str) -> int:
    prompt = (
        f"【问题】{question[:300]}\n\n"
        f"【参考答案摘要】{reference[:300]}\n\n"
        f"【评分标准】{standard[:300]}\n\n"
        f"【模型回答】{prediction[:400]}\n\n"
        "评分（0-10）："
    )
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": ALIGNMENT_JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\d+", text)
        return min(10, max(0, int(m.group()))) if m else 0
    except Exception as e:
        print(f"  [judge error] {e}")
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        help="结果 JSON 文件路径（默认自动查找最新）")
    args = parser.parse_args()

    if args.results:
        results_path = args.results
    else:
        # 自动找最新的结果文件
        files = sorted(OUT_DIR.glob("results_*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            print("未找到结果文件，请先运行 eval_skill_openrouter.py")
            return
        results_path = files[-1]

    print(f"读取结果文件：{results_path}")
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    alignment_rows = [r for r in results if r.get("task") == "alignment"]
    print(f"找到 {len(alignment_rows)} 条 alignment 结果，开始重新打分...\n")

    old_scores, new_scores = [], []
    by_subtask: dict[str, list] = {}

    for i, row in enumerate(alignment_rows):
        old = row.get("score_10", 0)
        new = judge(
            row.get("question", ""),
            row.get("reference", ""),
            row.get("standard", ""),
            row.get("model_answer", ""),
        )
        row["score_10_old"] = old
        row["score_10"]     = new
        old_scores.append(old)
        new_scores.append(new)
        subtask = row.get("subtask", "其他")
        by_subtask.setdefault(subtask, []).append(new)
        print(f"  [{i+1}/{len(alignment_rows)}] {row.get('id','')}  旧={old}  新={new}")

    # 保存更新后的结果
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 更新 summary 文件
    summary_path = results_path.parent / results_path.name.replace("results_", "summary_")
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        summary["alignment"]["avg_score_10"] = round(sum(new_scores)/len(new_scores), 1)
        summary["alignment"]["by_subtask"] = {
            k: round(sum(v)/len(v), 1) for k, v in by_subtask.items()
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nsummary 已更新：{summary_path}")

    print("\n" + "="*50)
    print(f"旧平均分：{sum(old_scores)/len(old_scores):.1f}")
    print(f"新平均分：{sum(new_scores)/len(new_scores):.1f}")
    print("\n按子任务：")
    for k, v in by_subtask.items():
        print(f"  {k}: {sum(v)/len(v):.1f}")

if __name__ == "__main__":
    main()
