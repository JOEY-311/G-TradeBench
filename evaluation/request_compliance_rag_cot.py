"""
合规审查任务 - RAG / CoT 对照实验生成脚本
模型: gemini (google/gemini-2.5-pro) / qwen (qwen/qwen3-235b-a22b)
策略: baseline / cot / rag / rag_cot

输出格式与 eval_compliance.py 兼容（ID, ORIGIN, DESTINATION, FOOD_PROFILE, REVIEW_OUTPUT）
仅处理 AUS_PRIMARY 文件（含实测污染物数值，可参与准确率评测）

用法:
  python request_compliance_rag_cot.py --model gemini --strategy rag_cot
  python request_compliance_rag_cot.py --model qwen   --strategy cot
  python request_compliance_rag_cot.py --model gemini --strategy baseline --sample 50
"""

import argparse, csv, os, random, re, time
from pathlib import Path
from openai import OpenAI

# ── 配置 ─────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get('OPENROUTER_API_KEY',
          os.environ.get('OPENROUTER_API_KEY', ''))

MODEL_IDS = {
    'gemini': 'google/gemini-3.1-pro-preview',
    'qwen':   'qwen/qwen3.6-plus',
}

BASE_DATA = Path('E:/论文/跨境对齐/准备阶段/商品数据集/问题集')
BASE_OUT  = Path('E:/论文/跨境对齐/评测阶段/合规审查')

# origin-destination 批次配置（与原实验保持对齐，使用 AUS_PRIMARY 可评测数据）
# (input_csv, origin, destination, sample_n, out_name_tpl)
BATCHES = [
    (BASE_DATA / 'formatted_AUS_profiles.csv', '澳大利亚', '日本',  50, 'AUS_{model_tag}澳日.csv'),
    (BASE_DATA / 'formatted_AUS_profiles.csv', '澳大利亚', '韩国',  50, 'AUS_{model_tag}澳韩.csv'),
    (BASE_DATA / 'formatted_AUS_profiles.csv', '澳大利亚', '中国',  40, 'AUS_{model_tag}澳中.csv'),
    (BASE_DATA / 'formatted_AUS_profiles.csv', '澳大利亚', '法国',  40, 'AUS_{model_tag}澳法.csv'),
]

# ── RAG：各目标国监管参考文本 ────────────────────────────────────────────────
COUNTRY_RAG = {
    '日本': """\
【参考法规 - 日本】
法规依据：《食品卫生法》（食品衛生法）第11条及相关厚生劳动省省令/告示
主要污染物限量（农产品类）：
  · 铅（Pb）：一般食品 ≤0.1 mg/kg；茶叶 ≤5.0 mg/kg
  · 镉（Cd）：一般食品 ≤0.05 mg/kg；大米 ≤0.4 mg/kg；小麦 ≤0.2 mg/kg
  · 农药残留：肯定列表制度（Positive List System, 平成17年厚生劳动省告示第498号）
              未在正面清单列明的物质，一律基准（一律基準）为 0.01 mg/kg
关键判断阈值：超过限量值即为不合格（違反）；在限量85%~100%区间属于边界风险
""",
    '韩国': """\
【参考法规 - 韩国】
法规依据：《食品安全基本法》（식품안전기본법）及《食品法典》（식품공전）
主要污染物限量（农产品/加工食品类）：
  · 铅（Pb）：一般加工食品 ≤0.1 mg/kg；谷物 ≤0.2 mg/kg
  · 镉（Cd）：一般食品 ≤0.05 mg/kg；大米 ≤0.2 mg/kg
  · 农药残留：农药残留肯定列表制度（PLS，잔류농약허용기준강화），
              未登记物质 ≤0.01 mg/kg（一律基准）
关键判断阈值：超过限量值即为不合格；在限量85%~100%区间属于边界风险
""",
    '法国': """\
【参考法规 - 法国（适用EU法规）】
法规依据：EU Regulation (EU) 2023/915（食品中污染物，取代原 Regulation (EC) No 1881/2006）
         EU Regulation (EC) No 396/2005（农药最大残留限量MRL）
主要污染物限量（肉类/农产品类）：
  · 铅（Pb）：肉类（禽畜） ≤0.1 mg/kg；蔬菜 ≤0.1 mg/kg；谷物 ≤0.2 mg/kg
  · 镉（Cd）：肉类（禽畜） ≤0.05 mg/kg；谷物 ≤0.1 mg/kg；蔬菜 ≤0.05 mg/kg
  · 农药残留：各物质参照 EU MRL 数据库（https://ec.europa.eu/food/plant/pesticides/eu-pesticides-database）
              未设定MRL的物质默认限量为 0.01 mg/kg
关键判断阈值：超过MRL/限量值即为违规；在限量85%~100%区间属于边界风险
""",
    '中国': """\
【参考法规 - 中国】
法规依据：《食品安全法》及国家食品安全标准
  · 铅（Pb）：GB 2762-2022，畜肉 ≤0.2 mg/kg；禽肉 ≤0.2 mg/kg；谷物 ≤0.2 mg/kg
  · 镉（Cd）：GB 2762-2022，畜肉 ≤0.1 mg/kg；谷物（大米除外） ≤0.1 mg/kg；大米 ≤0.2 mg/kg
  · 农药残留：GB 2763-2021（最大残留限量），
              无MRL规定的物质参考 GB 2763 附录，默认不得检出或参照 0.01 mg/kg
关键判断阈值：超过限量值即为违规；在限量85%~100%区间属于边界风险
""",
    '美国': """\
【参考法规 - 美国】
法规依据：US FDA 21 CFR（联邦法规汇编）及 EPA 农药容许量（Tolerance）
  · 铅（Pb）：FDA 行动水平（Action Level）：果汁/蔬菜汁 ≤0.05 mg/kg；糖果 ≤0.1 mg/kg；
              肉类参照 EPA 指导值（通常 ≤0.1 mg/kg）
  · 镉（Cd）：无联邦统一限量，参照 EPA 健康指导值（一般 ≤0.05 mg/kg 作为风险基准）
  · 农药残留：EPA 容许量数据库（Tolerance Database, 40 CFR Part 180），
              无容许量的物质不得检出（零容许量）
关键判断阈值：超过 Action Level / Tolerance 即为违规；在阈值85%~100%属于边界风险
""",
}

# ── CoT 步骤指引（附加在基础 prompt 末尾） ────────────────────────────────────
COT_STEPS = """\

请严格按以下步骤进行系统性分析，在最终结论前依次完成每个步骤：
步骤1：识别食品类别（农产品原料/加工食品/etc.）及主要合规风险点
步骤2：逐项核查关键污染物指标——铅(Pb)、镉(Cd)、农药残留——与【目标国】现行限量标准的差距（注明实测值与限量值）
步骤3：评估各指标状态：合规（<85%限量）/ 边界风险（≥85%且≤100%限量）/ 超标违规（>限量）
步骤4：若存在其他合规风险（添加剂、标签等），简要指出
步骤5：综合以上分析，给出最终审查结论"""

# ── Prompt 构造 ───────────────────────────────────────────────────────────────
BASE_PROMPT = """\
你现在是一位资深的跨国食品合规与通关专家，精通多国的食品进出口法规。
现有一批食品计划从【{origin}】出口到【{destination}】。

请严格根据【{destination}】的食品安全国家标准与相关进口法规，审查以下食品档案，判断其是否可以合法通关进口。

【拟进口食品档案】：
{profile}
{rag_block}{cot_block}
输出格式要求（全程纯文本，禁止 Markdown）：
{output_fmt}"""

OUTPUT_FMT_COT = """\
先完成上述步骤1~5的逐步分析，最后另起一行输出：
审查结论：[从"建议放行"/"存在违规风险"/"违规"中三选一，不附加任何其他说明]"""

OUTPUT_FMT_BASE = """\
仅输出一行，格式严格如下（不得附加任何说明）：
审查结论：[从"建议放行"/"存在违规风险"/"违规"中三选一]"""


def build_prompt(origin: str, destination: str, profile: str,
                 strategy: str) -> str:
    use_rag = 'rag' in strategy
    use_cot = 'cot' in strategy

    rag_block = ''
    if use_rag:
        ctx = COUNTRY_RAG.get(destination, '')
        if ctx:
            rag_block = f'\n{ctx}\n'

    cot_block = COT_STEPS if use_cot else ''
    output_fmt = OUTPUT_FMT_COT if use_cot else OUTPUT_FMT_BASE

    return BASE_PROMPT.format(
        origin=origin, destination=destination, profile=profile,
        rag_block=rag_block, cot_block=cot_block, output_fmt=output_fmt,
    )


# ── 工具 ──────────────────────────────────────────────────────────────────────
def strip_markdown(text: str) -> str:
    text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def read_csv_profiles(path: Path) -> list[dict]:
    for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            with open(path, encoding=enc) as f:
                rows = list(csv.DictReader(f))
            if rows:
                return rows
        except Exception:
            continue
    return []


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    required=True, choices=['gemini', 'qwen'])
    parser.add_argument('--strategy', required=True,
                        choices=['baseline', 'cot', 'rag', 'rag_cot'])
    parser.add_argument('--sample',   type=int, default=None,
                        help='每批覆盖采样数（默认使用批次配置值）')
    args = parser.parse_args()

    random.seed(42)   # 固定种子，确保各策略采样相同题目
    model_tag = f'{args.model}-{args.strategy}'
    out_dir   = BASE_OUT / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=API_KEY)
    model_id = MODEL_IDS[args.model]

    print(f'模型: {model_id}  策略: {args.strategy}  输出目录: {out_dir}')

    for (input_csv, origin, destination, default_n, out_tpl) in BATCHES:
        sample_n  = args.sample or default_n
        out_name  = out_tpl.format(model_tag=model_tag)
        out_path  = out_dir / out_name

        all_rows = read_csv_profiles(input_csv)
        if not all_rows:
            print(f'  [跳过] 读取失败: {input_csv}')
            continue

        # 断点续跑：读取已处理ID
        processed_ids: set[str] = set()
        if not out_path.exists():
            with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(['ID', 'ORIGIN', 'DESTINATION',
                                        'FOOD_PROFILE', 'REVIEW_OUTPUT'])
        else:
            with open(out_path, encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    processed_ids.add(row.get('ID', ''))

        # 采样（排除已处理）
        pending = [r for r in all_rows
                   if r.get('id', '') not in processed_ids]
        sample  = random.sample(pending, min(sample_n, len(pending)))

        print(f'\n  [{origin}→{destination}] 计划{sample_n}条，待处理{len(sample)}条 → {out_name}')

        for i, row in enumerate(sample):
            data_id = str(row.get('id', '')).strip()
            profile = str(row.get('food_profile', '')).strip()

            if not data_id or not profile:
                continue

            prompt = build_prompt(origin, destination, profile, args.strategy)

            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0,
                    max_tokens=2000,
                )
                output = strip_markdown(resp.choices[0].message.content)
            except Exception as e:
                print(f'  [{i+1}/{len(sample)}] API错误: {e}')
                time.sleep(5)
                continue

            with open(out_path, 'a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(
                    [data_id, origin, destination, profile, output])

            print(f'  [{i+1}/{len(sample)}] {data_id} → {output[:60].strip()}')
            time.sleep(random.uniform(1.5, 2.5))

    print('\n合规审查生成完毕。')


if __name__ == '__main__':
    main()
