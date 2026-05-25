"""
标准对齐任务 - RAG / CoT 对照实验生成脚本
模型: gemini (google/gemini-2.5-pro) / qwen (qwen/qwen3-235b-a22b)
策略: baseline / cot / rag / rag_cot

输入：NOCHINA_*.xlsx（标准对齐全任务类型，跳过 HSCODE）
输出格式与 eval_alignment.py 兼容（QUESTION, REFERENCE, OUTPUT, STANDARD）

用法:
  python request_alignment_rag_cot.py --model gemini --strategy rag_cot
  python request_alignment_rag_cot.py --model qwen   --strategy cot --sample 30
"""

import argparse, csv, os, random, re, time
from pathlib import Path
from openai import OpenAI

try:
    import openpyxl
except ImportError:
    raise SystemExit('请先安装: pip install openpyxl')

# ── 配置 ─────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get('OPENROUTER_API_KEY',
          os.environ.get('OPENROUTER_API_KEY', ''))

MODEL_IDS = {
    'gemini': 'google/gemini-3.1-pro-preview',
    'qwen':   'qwen/qwen3.6-plus',
}

# 源数据目录（NOCHINA_*.xlsx 文件）
SRC_DIR = Path('E:/论文/跨境对齐/准备阶段/数据收集/标准对齐全流程')
BASE_OUT = Path('E:/论文/跨境对齐/评测阶段/标准对齐')

# 源文件列 → 输出列映射
COL_QUESTION  = '业务场景案例题'
COL_REFERENCE = '标准参考答案与解析'
COL_RUBRIC    = '评分参考标准'
COL_TYPE      = '题型'

SKIP_KEYWORDS = ['HSCODE', 'HS Code']

# 每个任务文件采样数（与原实验 50 题/文件对齐）
DEFAULT_SAMPLE = 50

# ── RAG：各任务类型监管参考知识 ──────────────────────────────────────────────
TASK_RAG = {
    '冲突判断': """\
【参考框架 - 冲突判断】
跨境食品合规涉及双重体系对接：出口国法规（如德/法执行 EU 法规）与进口国法规（如中国 GB 标准）。
核心参考法规：
· 中国进口端：GB 2762（食品中污染物限量）、GB 2763（农药最大残留限量）、GB 2760（食品添加剂）、GB 7718（标签通则）、海关总署注册要求
· EU 出口端：Regulation (EU) 2023/915（污染物）、Regulation (EC) No 396/2005（农药MRL）、Regulation (EC) No 1333/2008（添加剂）
冲突判断思路：识别两国标准差异点 → 确定较严格一方 → 评估实际可行性 → 给出整改建议
""",
    '限量对齐': """\
【参考法规 - 限量对齐】
主要对比框架（污染物/农药残留/添加剂限量）：
· 中国：GB 2762-2022（污染物）、GB 2763-2021（农药），未规定者参考 Codex 或不得检出
· EU：Regulation (EU) 2023/915（污染物）、EC No 396/2005（农药 MRL 数据库）
· 美国：FDA Action Levels、EPA Tolerance Database（40 CFR Part 180）
· 日本：厚生劳动省告示（肯定列表制度，一律基准 0.01 mg/kg）
· 韩国：식품공전（食品法典）、PLS 制度
· Codex：CAC/RCP 1-1969、GSFA（通用食品添加剂标准）
""",
    '配料准入': """\
【参考框架 - 配料准入】
食品添加剂准入以目标国正面清单为准：
· 中国：GB 2760（附录 A/B/C/D），FAO/WHO 联合专家委员会（JECFA）评估结果可参考
· EU：E-number 体系，Annex II of Regulation (EC) No 1333/2008（使用范围和最大用量）
· 美国：GRAS（Generally Recognized As Safe）列表（21 CFR Part 182/184）；FDA 批准的食品添加剂（21 CFR Part 172-178）
· 日本：厚生劳动省食品添加物公定书；非指定添加物不得使用
· 韩国：식품첨가물공전（食品添加剂公典）
""",
    '准入程序': """\
【参考框架 - 准入程序】
各主要市场进口注册与检验要求：
· 中国进口：海关总署（GACC）境外生产企业注册，入境口岸 CIQ 检验检疫，需提供卫生证书、原产地证等
· EU 进口：普通食品无需事前注册；TRACES NT 系统申报；部分高风险食品需额外证明文件（Regulation (EU) 2019/1793）
· 美国进口：FDA 事先通报（Prior Notice，21 CFR Part 1 Subpart I）；食品设施注册（21 CFR Part 1 Subpart H）；FSVP（外国供应商验证计划）
· 日本进口：口岸检疫所（植物/动物/食品）检查；厚生劳动省监控检查体系
""",
    '标签对齐': """\
【参考法规 - 标签对齐】
各主要市场预包装食品标签法规：
· 中国：GB 7718-2011（预包装食品标签通则）、GB 28050-2011（营养标签通则）；必须中文标注
· EU：Regulation (EU) No 1169/2011（食品信息标示）；需官方语言；营养声明强制要求
· 美国：FDA NLEA（Nutrition Labeling and Education Act）；21 CFR Part 101；成分清单、营养事实表
· 日本：食品標示法（平成25年法律第70号）；日语标注；过敏原 7+20 种
· 韩国：식품 등의 표시·광고에 관한 법률；韩语标注
""",
    '多国流通': """\
【参考框架 - 多国流通】
设计可同时满足多国合规要求的产品策略：
1. 以 Codex Alimentarius 标准为基准基线
2. 叠加各目标国额外要求（取最严格值）
3. 重点关注：污染物限量差异（尤其铅/镉/农药 MRL）、添加剂许可状态差异、标签语言/格式差异、进口注册程序差异
4. 建议采用"最小公倍数"合规设计，同时维护多套合规文档体系
""",
    '俗名映射': """\
【参考框架 - 俗名映射】
食品成分俗名与国际标准名称对照：
· Codex GSFA：英文通用名称（INS 编号体系）
· EU E-number：欧洲添加剂编号与中文名对照（如 E330 = 柠檬酸）
· FDA CFR：英文通用名称（如 Citric acid, Sodium benzoate）
· GB 2760 附录：中文名称与功能类别
· 日本食品添加物公定书：日语名称
常见对照资源：Codex Alimentarius Database、EU Additives DB、FDA GRAS Inventory
""",
    '未知': '',  # 未识别任务类型不提供 RAG
}

# ── CoT 步骤指引 ──────────────────────────────────────────────────────────────
COT_STEPS = """\

请严格按以下步骤进行逻辑分析，在最终回答前完成每个步骤：
步骤1：准确识别场景中的核心合规问题（是什么问题，涉及哪些国家/法规体系）
步骤2：引用具体适用的法规或标准（含编号，如 GB 2762-2022、EU Regulation No 1333/2008等）
步骤3：基于上述法规，逐步推导出解决方案中的关键行动项（可操作性优先）
步骤4：给出最终专业建议（可包含替代方案或注意事项）"""

# ── Prompt 构造 ───────────────────────────────────────────────────────────────
BASE_PROMPT = """\
你现在是一位资深的跨国食品合规与通关专家，精通多国的食品进出口法规。
请以专业顾问的口吻，解答以下业务场景案例题目。
{rag_block}{cot_block}
【业务场景案例题】：
{question}

注意：回答需逻辑严密，术语专业，避免空话套话。"""


def detect_task_type(question: str, rubric: str) -> str:
    combined = (question or '') + (rubric or '')
    for kw in ['冲突判断', '限量对齐', '配料准入', '准入程序',
                '标签对齐', '多国流通', '俗名映射']:
        if kw in combined:
            return kw
    return '未知'


def build_prompt(question: str, task_type: str, strategy: str) -> str:
    use_rag = 'rag' in strategy
    use_cot = 'cot' in strategy

    rag_block = ''
    if use_rag:
        ctx = TASK_RAG.get(task_type, '')
        if ctx:
            rag_block = f'\n{ctx}\n'

    cot_block = COT_STEPS if use_cot else ''

    return BASE_PROMPT.format(
        rag_block=rag_block,
        cot_block=cot_block,
        question=question,
    )


# ── xlsx 读取 ─────────────────────────────────────────────────────────────────
def read_xlsx(path: Path) -> list[dict]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return []
    headers = [str(h) if h is not None else '' for h in rows[0]]
    return [dict(zip(headers, (str(v) if v is not None else '' for v in row)))
            for row in rows[1:]]


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
    out_dir   = BASE_OUT / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    client   = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=API_KEY)
    model_id = MODEL_IDS[args.model]

    print(f'模型: {model_id}  策略: {args.strategy}  输出目录: {out_dir}')

    # 找所有 NOCHINA_*.xlsx 源文件（跳过 HSCODE）
    src_files = sorted(SRC_DIR.glob('NOCHINA_*.xlsx'))
    src_files = [f for f in src_files
                 if not any(kw in f.name for kw in SKIP_KEYWORDS)]

    if not src_files:
        print(f'未找到源文件，请检查路径: {SRC_DIR}')
        return

    for src_path in src_files:
        task_kw  = src_path.stem.replace('NOCHINA_', '')
        out_name = f'{model_tag}_{task_kw}.csv'
        out_path = out_dir / out_name

        all_rows = read_xlsx(src_path)
        if not all_rows:
            print(f'  [跳过] 无法读取: {src_path.name}')
            continue

        # 断点续跑
        processed_qs: set[str] = set()
        if not out_path.exists():
            with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(['QUESTION', 'REFERENCE', 'OUTPUT', 'STANDARD'])
        else:
            with open(out_path, encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    processed_qs.add(row.get('QUESTION', ''))

        pending = [r for r in all_rows
                   if r.get(COL_QUESTION, '') not in processed_qs
                   and r.get(COL_QUESTION, '').strip()]
        sample  = random.sample(pending, min(args.sample, len(pending)))

        print(f'\n  [{src_path.name}] 待处理{len(sample)}条 → {out_name}')

        for i, row in enumerate(sample):
            question  = str(row.get(COL_QUESTION, '')).strip()
            reference = str(row.get(COL_REFERENCE, '')).strip()
            rubric    = str(row.get(COL_RUBRIC, '')).strip()
            row_type  = str(row.get(COL_TYPE, '')).strip()

            task_type = row_type if row_type else detect_task_type(question, rubric)
            prompt    = build_prompt(question, task_type, args.strategy)

            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0,
                    max_tokens=2000,
                )
                output = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f'  [{i+1}/{len(sample)}] API错误: {e}')
                time.sleep(5)
                continue

            with open(out_path, 'a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow([question, reference, output, rubric])

            print(f'  [{i+1}/{len(sample)}] {task_type} → {output[:50].strip()}...')
            time.sleep(random.uniform(1.5, 2.5))

    print('\n标准对齐生成完毕。')


if __name__ == '__main__':
    main()
