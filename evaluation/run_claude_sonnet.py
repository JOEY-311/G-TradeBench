#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Sonnet 全流程评测脚本
任务：合规审查（9条路线）/ 规则更新（200条）/ 标准对齐（8类题型×50题）
特性：断点续跑、实时写盘、统一进度展示
用法：直接在 VS Code 终端运行 `python run_claude_sonnet.py`
"""

import csv, json, os, random, re, time
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ════════════════════════════════════════════════════════════════
#  ▶ 配置区（运行前确认这里）
# ════════════════════════════════════════════════════════════════
API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
MODEL   = "anthropic/claude-sonnet-4-5"   # OpenRouter 模型 ID

COMPLIANCE_SAMPLE = 50   # 每条合规路线采样数
TEMPORAL_SAMPLE   = 200  # 规则更新采样数
ALIGNMENT_SAMPLE  = 50   # 标准对齐每文件采样数

SEED = 42
random.seed(SEED)

# ── 路径 ──────────────────────────────────────────────────────────
BASE = Path("E:/论文/跨境对齐")
EVAL = BASE / "评测阶段"
DATA = BASE / "准备阶段"

AUS_CSV  = DATA / "商品数据集/问题集/formatted_AUS_profiles.csv"
EU_CSV   = DATA / "商品数据集/问题集/formatted_EU_profiles.csv"
USA_CSV  = DATA / "商品数据集/问题集/formatted_USA_profiles.csv"

TEMPORAL_IN   = EVAL / "规则更新/temporal_eval_en.jsonl"
TEMPORAL_OUT  = EVAL / "规则更新/temporal_claude-sonnet.csv"

ALIGNMENT_SRC = DATA / "数据收集/标准对齐全流程"
ALIGNMENT_OUT = EVAL / "标准对齐/claude-sonnet"

COMPLIANCE_OUT = EVAL / "合规审查/claude-sonnet"

# ── 合规审查：9条路线配置 ─────────────────────────────────────────
# (输入CSV, 出口国, 目的国, 输出文件名)
COMPLIANCE_BATCHES = [
    (USA_CSV, "美国",    "日本", "USA_claude-sonnet_美日.csv"),
    (USA_CSV, "美国",    "法国", "USA_claude-sonnet_美法.csv"),
    (USA_CSV, "美国",    "中国", "USA_claude-sonnet_美中.csv"),
    (AUS_CSV, "澳大利亚", "法国", "AUS_claude-sonnet_澳法.csv"),
    (AUS_CSV, "澳大利亚", "韩国", "AUS_claude-sonnet_澳韩.csv"),
    (AUS_CSV, "澳大利亚", "美国", "AUS_claude-sonnet_澳美.csv"),
    (EU_CSV,  "德国",    "中国", "EU_claude-sonnet_德中.csv"),
    (EU_CSV,  "德国",    "美国", "EU_claude-sonnet_德美.csv"),
    (EU_CSV,  "德国",    "韩国", "EU_claude-sonnet_德韩.csv"),
]

# ── 标准对齐：8类题型 ─────────────────────────────────────────────
ALIGNMENT_FILES = [
    "NOCHINA_HSCODE.xlsx",
    "NOCHINA_冲突判断.xlsx",
    "NOCHINA_限量对齐.xlsx",
    "NOCHINA_配料准入.xlsx",
    "NOCHINA_准入程序.xlsx",
    "NOCHINA_标签对齐.xlsx",
    "NOCHINA_多国流通.xlsx",
    "NOCHINA_俗名映射.xlsx",
]

# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)


def strip_md(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def call_api(messages: list, max_tokens: int = 1500,
             temperature: float = 0, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"    ⚠ API错误 ({attempt+1}/{retries}): {e}  等待{wait}s...")
            if attempt < retries - 1:
                time.sleep(wait)
    return ""


def section(title: str):
    bar = "═" * 58
    print(f"\n╔{bar}╗")
    print(f"║  {title:<56}║")
    print(f"╚{bar}╝")


def load_csv_safe(path: Path, usecols: list) -> pd.DataFrame:
    """读取 CSV，支持大文件，编码自动检测。"""
    print(f"    读取: {path.name}  ({path.stat().st_size/1e6:.0f} MB) ...", end="", flush=True)
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, usecols=usecols, encoding=enc, low_memory=False)
            print(f" {len(df)} 行")
            return df
        except Exception:
            continue
    print(" 读取失败")
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════
#  任务 1：合规审查
# ════════════════════════════════════════════════════════════════
COMPLIANCE_PROMPT = """\
你现在是一位资深的跨国食品合规与通关专家，精通多国的食品进出口法规。
现有一批食品计划从【{origin}】出口到【{dest}】。

请严格根据【{dest}】的食品安全国家标准与相关进口法规，审查以下食品档案，判断其是否可以合法通关进口。

【拟进口食品档案】：
{profile}

【输出格式（严格遵守，禁止 Markdown，禁止附加任何说明）】：
· 若可以通关：审查结论：建议放行
· 若存在风险：审查结论：存在违规风险
· 若明确违规：审查结论：违规"""


def run_compliance():
    section("任务 1 / 3：合规审查（9条路线）")
    COMPLIANCE_OUT.mkdir(parents=True, exist_ok=True)

    # 按来源文件分组，避免重复读取大 CSV
    from collections import defaultdict
    batches_by_src = defaultdict(list)
    for (csv_path, origin, dest, out_name) in COMPLIANCE_BATCHES:
        batches_by_src[csv_path].append((origin, dest, out_name))

    for csv_path, batch_list in batches_by_src.items():
        df_src = load_csv_safe(csv_path, usecols=['id', 'food_profile'])
        if df_src.empty:
            continue
        df_src['id'] = df_src['id'].astype(str)

        for (origin, dest, out_name) in batch_list:
            out_path = COMPLIANCE_OUT / out_name
            print(f"\n  ▶ {origin} → {dest}  目标{COMPLIANCE_SAMPLE}条")

            # 断点续跑
            done_ids: set = set()
            if not out_path.exists():
                with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow(
                        ['ID', 'ORIGIN', 'DESTINATION', 'FOOD_PROFILE', 'REVIEW_OUTPUT'])
            else:
                with open(out_path, encoding='utf-8-sig') as f:
                    for row in csv.DictReader(f):
                        done_ids.add(str(row.get('ID', '')))
                print(f"    [续跑] 已完成 {len(done_ids)} 条")

            pending  = df_src[~df_src['id'].isin(done_ids)]
            sample_n = min(COMPLIANCE_SAMPLE, len(pending))
            if sample_n == 0:
                print("    已全部完成，跳过。")
                continue

            sample = pending.sample(n=sample_n, random_state=SEED)
            t0 = time.time()

            for i, (_, row) in enumerate(sample.iterrows(), 1):
                data_id = row['id'].strip()
                profile = str(row['food_profile']).strip()
                prompt  = COMPLIANCE_PROMPT.format(
                    origin=origin, dest=dest, profile=profile)

                output = strip_md(call_api(
                    [{"role": "user", "content": prompt}], max_tokens=200))

                with open(out_path, 'a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow(
                        [data_id, origin, dest, profile, output])

                elapsed = time.time() - t0
                eta = elapsed / i * (sample_n - i)
                verdict = output[:30].replace('\n', ' ')
                print(f"    [{i:>3}/{sample_n}] {data_id:<12} {verdict:<28}"
                      f"  ETA {eta/60:.1f}min")

                time.sleep(random.uniform(0.8, 1.5))

    print("\n  ✓ 合规审查全部完成。")


# ════════════════════════════════════════════════════════════════
#  任务 2：规则更新
# ════════════════════════════════════════════════════════════════
TEMPORAL_SYSTEM = (
    "You are a professional assistant specializing in food regulatory "
    "analysis and temporal reasoning. Please provide accurate, concise, "
    "and logically consistent answers based on the provided information."
)
TEMPORAL_USER_TPL = "Please provide a detailed answer in English:\n\n{instruction}"


def run_temporal():
    section("任务 2 / 3：规则更新")

    all_data = []
    with open(TEMPORAL_IN, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    print(f"  数据集共 {len(all_data)} 条")

    # 断点续跑
    done_idx: set = set()
    if not TEMPORAL_OUT.exists():
        with open(TEMPORAL_OUT, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(
                ['index', 'instruction', 'ground_truth', 'model_output'])
    else:
        with open(TEMPORAL_OUT, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                try:
                    done_idx.add(int(row.get('index', -1)))
                except ValueError:
                    pass
        print(f"  [续跑] 已完成 {len(done_idx)} 条")

    # 保留全局序号
    indexed = [(i + 1, item) for i, item in enumerate(all_data)
               if (i + 1) not in done_idx]
    sample_n = min(TEMPORAL_SAMPLE, len(indexed))
    sample   = random.sample(indexed, sample_n)
    print(f"  待处理 {sample_n} 条\n")

    t0 = time.time()
    for i, (idx, item) in enumerate(sample, 1):
        instruction  = item.get('instruction', '')
        ground_truth = item.get('answer', item.get('ground_truth', ''))

        output = strip_md(call_api([
            {"role": "system", "content": TEMPORAL_SYSTEM},
            {"role": "user",   "content": TEMPORAL_USER_TPL.format(
                instruction=instruction)},
        ], max_tokens=800, temperature=0.1))

        with open(TEMPORAL_OUT, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow([idx, instruction, ground_truth, output])

        elapsed = time.time() - t0
        eta = elapsed / i * (sample_n - i)
        print(f"  [{i:>3}/{sample_n}] #{idx:<4} {output[:60].strip():<60}"
              f"  ETA {eta/60:.1f}min")

        time.sleep(random.uniform(0.5, 1.0))

    print(f"\n  ✓ 规则更新完成 → {TEMPORAL_OUT}")


# ════════════════════════════════════════════════════════════════
#  任务 3：标准对齐
# ════════════════════════════════════════════════════════════════
ALIGNMENT_PROMPT = """\
你现在是一位资深的跨国食品合规与通关专家，精通多国的食品进出口法规。
请以专业顾问的口吻，解答以下业务场景案例题目。

【业务场景案例题】：
{question}

注意：回答需逻辑严密，术语专业，避免空话套话。"""

COL_Q   = '业务场景案例题'
COL_REF = '标准参考答案与解析'
COL_RUB = '评分参考标准'


def run_alignment():
    section("任务 3 / 3：标准对齐（8类题型）")
    ALIGNMENT_OUT.mkdir(parents=True, exist_ok=True)

    for fname in ALIGNMENT_FILES:
        src_path = ALIGNMENT_SRC / fname
        task_kw  = src_path.stem.replace('NOCHINA_', '')
        out_path = ALIGNMENT_OUT / f"claude-sonnet_{task_kw}.csv"

        if not src_path.exists():
            print(f"\n  [跳过] 文件不存在: {fname}")
            continue

        print(f"\n  ▶ 题型: {task_kw}")
        try:
            df = pd.read_excel(src_path)
        except Exception as e:
            print(f"    读取失败: {e}")
            continue

        df.drop_duplicates(subset=[COL_Q], inplace=True)
        print(f"    共 {len(df)} 题")

        # 断点续跑
        done_qs: set = set()
        if not out_path.exists():
            with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(
                    ['QUESTION', 'REFERENCE', 'OUTPUT', 'STANDARD'])
        else:
            with open(out_path, encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    done_qs.add(row.get('QUESTION', ''))
            print(f"    [续跑] 已完成 {len(done_qs)} 题")

        pending  = df[~df[COL_Q].astype(str).isin(done_qs)]
        sample_n = min(ALIGNMENT_SAMPLE, len(pending))
        if sample_n == 0:
            print("    已全部完成，跳过。")
            continue

        sample = pending.sample(n=sample_n, random_state=SEED)
        t0 = time.time()

        for i, (_, row) in enumerate(sample.iterrows(), 1):
            question  = str(row.get(COL_Q,   '')).strip()
            reference = str(row.get(COL_REF, '')).strip()
            rubric    = str(row.get(COL_RUB, '')).strip()

            if not question or question == 'nan':
                continue

            output = call_api(
                [{"role": "user", "content": ALIGNMENT_PROMPT.format(
                    question=question)}],
                max_tokens=2000)

            with open(out_path, 'a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow([question, reference, output, rubric])

            elapsed = time.time() - t0
            eta = elapsed / i * (sample_n - i)
            print(f"    [{i:>3}/{sample_n}] {question[:25]:<25}  "
                  f"ETA {eta/60:.1f}min")

            time.sleep(random.uniform(1.0, 1.8))

    print("\n  ✓ 标准对齐全部完成。")


# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Claude Sonnet 跨境食品合规全流程评测               ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  模型  : {MODEL:<49}║")
    print(f"║  任务  : 合规审查×9 + 规则更新×200 + 标准对齐×8×50    ║")
    print(f"║  输出  : {str(EVAL):<49}║")
    print("╚══════════════════════════════════════════════════════════╝")

    run_compliance()
    run_temporal()
    run_alignment()

    elapsed = time.time() - t_start
    h, m = divmod(elapsed, 3600)
    m, s = divmod(m, 60)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print(f"║  全部完成！耗时 {int(h)}h {int(m)}m {int(s)}s"
          + " " * (38 - len(f"{int(h)}h {int(m)}m {int(s)}s")) + "║")
    print(f"║  合规审查 → {str(COMPLIANCE_OUT):<46}║")
    print(f"║  规则更新 → {str(TEMPORAL_OUT):<46}║")
    print(f"║  标准对齐 → {str(ALIGNMENT_OUT):<46}║")
    print("╚══════════════════════════════════════════════════════════╝")
