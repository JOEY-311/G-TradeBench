"""
knowledge_base/batch_build_kg.py
批量从 full.md 法规文件构建知识图谱

用法：
  cd E:\\论文\\跨境对齐\\贡献\\compliance_skill
  set OPENROUTER_API_KEY=sk-or-v1-...       # Windows CMD
  $env:OPENROUTER_API_KEY="sk-or-v1-..."    # PowerShell
  python knowledge_base/batch_build_kg.py

支持中断后续跑：进度保存在 knowledge_base/batch_progress.json
"""
import os
import sys
import json
import time
from pathlib import Path

# ── 填入 API Key（或通过环境变量注入）───────────────────────────────
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')   # ← 在此填入，或留空并用环境变量

# ── 模型与延迟配置 ────────────────────────────────────────────────
MODEL        = "meta-llama/llama-3.1-8b-instruct"
DELAY_SEC    = 2.0   # 每个文本块调用后的等待秒数
CHUNK_SIZE   = 1500  # 传给 build_kg_from_markdown 内部分块大小（字符）

# ── 路径配置 ──────────────────────────────────────────────────────
REGS_DIR      = Path(r"E:\论文\跨境对齐\各国法规和部分数据集")
SKILL_DIR     = Path(__file__).resolve().parent.parent
PROGRESS_FILE = SKILL_DIR / "knowledge_base" / "batch_progress.json"

# ══════════════════════════════════════════════════════════════════
#  初始化：注入环境变量，必须在导入项目模块前完成
# ══════════════════════════════════════════════════════════════════
_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
if not _key:
    print("❌  请设置 OPENROUTER_API_KEY 环境变量，或在脚本顶部填入 Key")
    sys.exit(1)

os.environ["OPENROUTER_API_KEY"] = _key
os.environ["COMPLIANCE_MODEL"]   = MODEL

sys.path.insert(0, str(SKILL_DIR))

from knowledge_base.kg_builder import build_kg_from_markdown  # noqa: E402

# ══════════════════════════════════════════════════════════════════
#  进度管理（支持中断续跑）
# ══════════════════════════════════════════════════════════════════
def _load_done() -> set:
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text(encoding="utf-8")))
    return set()

def _save_done(done: set) -> None:
    PROGRESS_FILE.write_text(
        json.dumps(sorted(done), ensure_ascii=False, indent=2), encoding="utf-8"
    )

# ══════════════════════════════════════════════════════════════════
#  辅助：从路径提取国家名
# ══════════════════════════════════════════════════════════════════
def _country(md_path: Path) -> str:
    parts = md_path.parts
    try:
        idx = next(i for i, p in enumerate(parts) if "各国法规和部分数据集" in p)
        return parts[idx + 1]
    except (StopIteration, IndexError):
        return "未知"

# ══════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    all_files = sorted(REGS_DIR.rglob("full.md"))
    if not all_files:
        print(f"⚠  在 {REGS_DIR} 下未找到任何 full.md 文件，请检查路径")
        return

    done      = _load_done()
    remaining = [f for f in all_files if str(f) not in done]

    print(f"全部 full.md：{len(all_files)} 个")
    print(f"已处理：{len(done)} 个  |  待处理：{len(remaining)} 个")

    if not remaining:
        print("✅  所有文件已处理完毕")
        return

    # 粗估时间：假设平均文件 ~20,000 字符
    avg_chunks   = max(1, 20_000 // (CHUNK_SIZE - 200))
    est_sec      = len(remaining) * avg_chunks * (DELAY_SEC + 3)
    print(f"粗估剩余时间：~{est_sec/60:.0f} 分钟（{est_sec/3600:.1f} 小时）")
    print(f"使用模型：{MODEL}")
    print("─" * 60)

    total_added = 0
    t0 = time.time()

    for i, md_path in enumerate(remaining, 1):
        country = _country(md_path)
        label   = f"{country} / {md_path.parent.name}"
        print(f"\n[{i}/{len(remaining)}] {label}")

        try:
            added = build_kg_from_markdown(md_path, delay=DELAY_SEC)
            total_added += added
            elapsed = (time.time() - t0) / 60
            print(f"  ✓ 新增 {added} 条三元组  |  已用时 {elapsed:.1f} min")
        except KeyboardInterrupt:
            print("\n⚠  用户中断，进度已保存，下次运行将从此处继续")
            _save_done(done)
            break
        except Exception as e:
            print(f"  ✗ 失败（{e}），已跳过")

        done.add(str(md_path))
        _save_done(done)

    print("\n" + "═" * 60)
    print(f"本次新增三元组：{total_added} 条")
    print(f"总进度：{len(done)}/{len(all_files)} 个文件")
    print(f"总用时：{(time.time() - t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
