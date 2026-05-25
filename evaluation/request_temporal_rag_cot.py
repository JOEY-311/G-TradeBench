"""
规则更新任务 - RAG / CoT 对照实验生成脚本
模型: gemini (google/gemini-2.5-pro) / qwen (qwen/qwen3-235b-a22b)
策略: baseline / cot / rag / rag_cot

输入：temporal_eval_en.jsonl（英文，与原实验相同数据源）
输出格式与 eval_temporal.py 兼容（index, instruction, ground_truth, model_output）

用法:
  python request_temporal_rag_cot.py --model gemini --strategy rag_cot
  python request_temporal_rag_cot.py --model qwen   --strategy cot --sample 200
"""

import argparse, csv, json, os, random, re, time
from pathlib import Path
from openai import OpenAI

# ── 配置 ─────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get('OPENROUTER_API_KEY',
          os.environ.get('OPENROUTER_API_KEY', ''))

MODEL_IDS = {
    'gemini': 'google/gemini-3.1-pro-preview',
    'qwen':   'qwen/qwen3.6-plus',
}

INPUT_FILE   = Path('E:/论文/跨境对齐/评测阶段/规则更新/temporal_eval_en.jsonl')
BASE_OUT     = Path('E:/论文/跨境对齐/评测阶段/规则更新')
DEFAULT_SAMPLE = 200

SYSTEM_PROMPT = ("You are a professional assistant specializing in food regulatory "
                 "analysis and temporal reasoning. Please provide accurate, concise, "
                 "and logically consistent answers based on the provided information.")

# ── RAG：各主要监管机构法规更新流程参考 ──────────────────────────────────────
RAG_CONTEXT = """\
[Regulatory Update Framework Reference]

FDA (United States):
- Proposed Rules are published in the Federal Register as "NPRM" (Notice of Proposed Rulemaking)
  with a public comment period; status keywords: "proposed", "seeking comment", "open for comment"
- Final Rules are published as "Final Rule" with a specific effective date; status: "effective",
  "enacted", "in effect", "adopted"
- Immediate Final Rules may be issued for urgent public health actions (effective immediately
  upon publication)
- 21 CFR Part 180 governs pesticide tolerances; amendments follow NPRM → Final Rule process

EFSA / European Commission (EU):
- EU food law changes begin with an EFSA Scientific Opinion, followed by a Commission Regulation
- Commission Regulations are published in the Official Journal of the EU (EUR-Lex)
- Entry into force is typically 20 days after publication, with transitional periods commonly
  stated (e.g., "shall apply from [date]")
- Status keywords for enacted: "is hereby amended", "shall apply", "enter into force"
- EU MRL changes for pesticides follow Regulation (EC) No 396/2005

Japan (MHLW — Ministry of Health, Labour and Welfare):
- Food Sanitation Act (食品衛生法) amendments issued as Ministry Notifications (厚生労働省告示)
- Positive List MRL revisions published in Official Gazette (官報); effective on gazette date
  or after a stated transition period
- "Notification" or "immediate" keywords usually indicate enacted status

Codex Alimentarius Commission (CAC):
- Standards adopted by CAC Plenary sessions; published in Codex Alimentarius database
- Draft standards circulate as "Proposed Draft" (Step 3) or "Draft Standard" (Step 5/8)
- Adopted standards are not legally binding per se, but widely used as reference

China (SAMR / NHC):
- National food safety standards (国家食品安全标准, GB series) published in official gazette
- Effective dates stated explicitly; typically 6–24 months after adoption notice
- Proposed revisions circulated for public consultation (公开征求意见)

Key temporal classification guidelines:
- "Proposed" / "Draft" / "NPRM" / "under consideration" / "open for comment" → PROPOSED (not yet law)
- "Final Rule" / "in effect" / "effective" / "adopted" / "enacted" / "notification" / "immediately" → ENACTED
- "Transitional period" / "will take effect on [future date]" → ENACTED but not yet applicable
- Date missing or "once finalized" / "pending" / "upon completion" → DATE NOT SPECIFIED
"""

# ── CoT 步骤指引（英文，与任务语言匹配） ─────────────────────────────────────
COT_STEPS = """\

Please analyze the question step by step before giving your final answer:
Step 1 — Legal Status: Determine whether the regulation is Proposed/Draft, Enacted/In Effect,
  or Revoked/Withdrawn. Look for status keywords in the instruction.
Step 2 — Target Substance & Change Type: Identify the specific food substance(s), chemical(s),
  or additive(s) involved, and describe the type of regulatory change
  (e.g., new MRL, revised limit, revocation, new additive permission, labeling change).
Step 3 — Effective Date: Extract the exact effective date if stated; if vague (e.g., "after a
  certain period"), acknowledge the uncertainty; if not applicable (N/A), do not invent a date.
Step 4 — Final Answer: Provide a clear, well-structured answer based on Steps 1–3."""

# ── Prompt 构造 ───────────────────────────────────────────────────────────────
USER_PROMPT_TMPL = """\
{rag_block}Please provide a detailed answer in English:
{cot_block}
{instruction}"""


def build_prompt(instruction: str, strategy: str) -> str:
    use_rag = 'rag' in strategy
    use_cot = 'cot' in strategy

    rag_block = (RAG_CONTEXT + '\n') if use_rag else ''
    cot_block = COT_STEPS if use_cot else ''

    return USER_PROMPT_TMPL.format(
        rag_block=rag_block,
        cot_block=cot_block,
        instruction=instruction,
    )


# ── 工具 ──────────────────────────────────────────────────────────────────────
def strip_markdown(text: str) -> str:
    if not text:
        return ''
    text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    return text.strip()


def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    required=True, choices=['gemini', 'qwen'])
    parser.add_argument('--strategy', required=True,
                        choices=['baseline', 'cot', 'rag', 'rag_cot'])
    parser.add_argument('--sample',   type=int, default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    random.seed(42)   # 固定种子，确保各策略采样相同题目
    model_tag = f'{args.model}-{args.strategy}'
    out_path  = BASE_OUT / f'temporal_{model_tag}.csv'

    client   = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=API_KEY)
    model_id = MODEL_IDS[args.model]

    print(f'模型: {model_id}  策略: {args.strategy}  输出: {out_path}')

    all_data = load_jsonl(INPUT_FILE)
    print(f'数据集共 {len(all_data)} 条')

    # 断点续跑
    done_idx: set[int] = set()
    if not out_path.exists():
        with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(
                ['index', 'instruction', 'ground_truth', 'model_output'])
    else:
        with open(out_path, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                try:
                    done_idx.add(int(row.get('index', -1)))
                except ValueError:
                    pass
        print(f'[续跑] 已完成 {len(done_idx)} 条')

    pending = [item for i, item in enumerate(all_data, 1)
               if i not in done_idx]

    sample_n = min(args.sample, len(pending))
    sample   = random.sample(pending, sample_n)
    # 恢复全局序号以便断点续跑
    item_to_idx = {id(item): i for i, item in enumerate(all_data, 1)}

    print(f'待处理 {len(sample)} 条')

    for i, item in enumerate(sample, 1):
        instruction = item.get('instruction', '')
        ground_truth = item.get('answer', item.get('ground_truth', ''))
        global_idx  = item_to_idx.get(id(item), i)

        prompt = build_prompt(instruction, args.strategy)

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            output = strip_markdown(resp.choices[0].message.content)
        except Exception as e:
            print(f'  [{i}/{len(sample)}] API错误: {e}')
            time.sleep(5)
            continue

        with open(out_path, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(
                [global_idx, instruction, ground_truth, output])

        print(f'  [{i}/{len(sample)}] #{global_idx} → {output[:60].strip()}...')
        time.sleep(0.8)

    print(f'\n规则更新生成完毕 → {out_path}')


if __name__ == '__main__':
    main()
