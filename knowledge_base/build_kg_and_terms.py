"""
build_kg_and_terms.py
一次性脚本：将 run_claude_skills.py 中已有的结构化数据转换为
  - knowledge_base/term_mapping.json  (术语词典)
  - knowledge_base/regulation_kg.json (SPO 三元组知识图谱)
运行：python knowledge_base/build_kg_and_terms.py
"""
import json
from pathlib import Path

BASE = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════
#  1. 术语映射词典（外文/缩写 → 国标名）
# ══════════════════════════════════════════════════════════════════════
TERM_MAP = {
    # ── 重金属 ──────────────────────────────────────────────────────
    "lead":              {"国标名": "铅",     "CAS号": "7439-92-1",   "备注": "GB 2762-2022 重金属"},
    "pb":                {"国标名": "铅",     "CAS号": "7439-92-1",   "备注": "元素符号"},
    "铅(pb)":            {"国标名": "铅",     "CAS号": "7439-92-1",   "备注": ""},
    "plumbum":           {"国标名": "铅",     "CAS号": "7439-92-1",   "备注": "拉丁文"},
    "cadmium":           {"国标名": "镉",     "CAS号": "7440-43-9",   "备注": "GB 2762-2022 重金属"},
    "cd":                {"国标名": "镉",     "CAS号": "7440-43-9",   "备注": "元素符号"},
    "镉(cd)":            {"国标名": "镉",     "CAS号": "7440-43-9",   "备注": ""},
    "cadmio":            {"国标名": "镉",     "CAS号": "7440-43-9",   "备注": "西班牙/意大利文"},
    "mercury":           {"国标名": "汞",     "CAS号": "7439-97-6",   "备注": "GB 2762-2022"},
    "hg":                {"国标名": "汞",     "CAS号": "7439-97-6",   "备注": "元素符号"},
    "methyl mercury":    {"国标名": "甲基汞", "CAS号": "22967-92-6",  "备注": "有机汞，适用于水产品"},
    "arsenic":           {"国标名": "砷",     "CAS号": "7440-38-2",   "备注": "GB 2762-2022"},
    "as":                {"国标名": "砷",     "CAS号": "7440-38-2",   "备注": "元素符号"},
    "inorganic arsenic": {"国标名": "无机砷", "CAS号": "7440-38-2",   "备注": "水产品专项"},
    # ── 农药/农残 ──────────────────────────────────────────────────
    "pesticide":         {"国标名": "农药",           "CAS号": None, "备注": "GB 2763-2021 / 肯定リスト制度"},
    "pesticides":        {"国标名": "农药",           "CAS号": None, "备注": ""},
    "mrl":               {"国标名": "农药最大残留限量", "CAS号": None, "备注": "Maximum Residue Level"},
    "农残":              {"国标名": "农药",           "CAS号": None, "备注": ""},
    "综合农药残留":      {"国标名": "农药",           "CAS号": None, "备注": ""},
    "glyphosate":        {"国标名": "草甘膦", "CAS号": "1071-83-6",   "备注": "除草剂；日本燕麦 MRL 30 mg/kg"},
    "acrylamide":        {"国标名": "丙烯酰胺","CAS号": "79-06-1",    "备注": "EU Reg 2023/194 基准值"},
    # ── 食品添加剂 ─────────────────────────────────────────────────
    "sodium benzoate":      {"国标名": "苯甲酸钠",     "CAS号": "532-32-1",    "备注": "GB 2760-2014 防腐剂"},
    "natriumbenzoat":       {"国标名": "苯甲酸钠",     "CAS号": "532-32-1",    "备注": "德文名"},
    "benzoate de sodium":   {"国标名": "苯甲酸钠",     "CAS号": "532-32-1",    "备注": "法文名"},
    "benzoic acid":         {"国标名": "苯甲酸",       "CAS号": "65-85-0",     "备注": "GB 2760-2014 防腐剂"},
    "titanium dioxide":     {"国标名": "二氧化钛",     "CAS号": "13463-67-7",  "备注": "EU Reg 2022/63 已禁用；GB 2760 尚允许"},
    "e171":                 {"国标名": "二氧化钛",     "CAS号": "13463-67-7",  "备注": "EU 食品添加剂编号"},
    "titandioxid":          {"国标名": "二氧化钛",     "CAS号": "13463-67-7",  "备注": "德文名"},
    "dioxyde de titane":    {"国标名": "二氧化钛",     "CAS号": "13463-67-7",  "备注": "法文名"},
    "monosodium glutamate": {"国标名": "谷氨酸钠",     "CAS号": "142-47-2",    "备注": "MSG；GB 2760 允许"},
    "msg":                  {"国标名": "谷氨酸钠",     "CAS号": "142-47-2",    "备注": ""},
    "glutamat de sodium":   {"国标名": "谷氨酸钠",     "CAS号": "142-47-2",    "备注": "法文名"},
    "natriumglutamat":      {"国标名": "谷氨酸钠",     "CAS号": "142-47-2",    "备注": "德文名"},
    "ascorbic acid":        {"国标名": "抗坏血酸",     "CAS号": "50-81-7",     "备注": "Vitamin C；GB 2760 抗氧化剂"},
    "vitamin c":            {"国标名": "抗坏血酸",     "CAS号": "50-81-7",     "备注": ""},
    "acesulfame":           {"国标名": "乙酰磺胺酸钾", "CAS号": "55589-62-3",  "备注": "Ace-K 甜味剂"},
    "acesulfame-k":         {"国标名": "乙酰磺胺酸钾", "CAS号": "55589-62-3",  "备注": ""},
    "e950":                 {"国标名": "乙酰磺胺酸钾", "CAS号": "55589-62-3",  "备注": "EU 编号"},
    "aspartame":            {"国标名": "阿斯巴甜",     "CAS号": "22839-47-0",  "备注": "甜味剂；GB 2760"},
    "aspartam":             {"国标名": "阿斯巴甜",     "CAS号": "22839-47-0",  "备注": "德文名"},
    "e951":                 {"国标名": "阿斯巴甜",     "CAS号": "22839-47-0",  "备注": "EU 编号"},
    "potassium sorbate":    {"国标名": "山梨酸钾",     "CAS号": "24634-61-5",  "备注": "防腐剂 GB 2760"},
    "sorbate de potassium": {"国标名": "山梨酸钾",     "CAS号": "24634-61-5",  "备注": "法文名"},
    "kaliumsorbat":         {"国标名": "山梨酸钾",     "CAS号": "24634-61-5",  "备注": "德文名"},
    "tartrazine":           {"国标名": "柠檬黄",       "CAS号": "1934-21-0",   "备注": "E102 着色剂"},
    "e102":                 {"国标名": "柠檬黄",       "CAS号": "1934-21-0",   "备注": ""},
    "sunset yellow":        {"国标名": "日落黄",       "CAS号": "2783-94-0",   "备注": "E110"},
    "e110":                 {"国标名": "日落黄",       "CAS号": "2783-94-0",   "备注": ""},
    "carrageenan":          {"国标名": "卡拉胶",       "CAS号": "9000-07-1",   "备注": "增稠剂 GB 2760"},
    "carragheen":           {"国标名": "卡拉胶",       "CAS号": "9000-07-1",   "备注": ""},
    "caramel":              {"国标名": "焦糖色",       "CAS号": "8028-89-5",   "备注": "E150 着色剂"},
    "e150":                 {"国标名": "焦糖色",       "CAS号": "8028-89-5",   "备注": ""},
    "lecithin":             {"国标名": "卵磷脂",       "CAS号": "8002-43-5",   "备注": "乳化剂 E322"},
    "soja-lecithin":        {"国标名": "大豆卵磷脂",   "CAS号": "8002-43-5",   "备注": "德文名"},
    # ── 整体食品产品（direct_pass 场景，供后续检索） ─────────────────
    "apple":          {"国标名": "苹果",       "CAS号": None, "备注": "水果及其制品"},
    "mutton":         {"国标名": "羊肉",       "CAS号": None, "备注": "畜禽肉及其制品"},
    "lamb":           {"国标名": "羊肉",       "CAS号": None, "备注": "畜禽肉及其制品"},
    "beef":           {"国标名": "牛肉",       "CAS号": None, "备注": "畜禽肉及其制品"},
    "pork":           {"国标名": "猪肉",       "CAS号": None, "备注": "畜禽肉及其制品"},
    "chicken":        {"国标名": "鸡肉",       "CAS号": None, "备注": "畜禽肉及其制品"},
    "salmon":         {"国标名": "三文鱼",     "CAS号": None, "备注": "水产品"},
    "wheat":          {"国标名": "小麦",       "CAS号": None, "备注": "谷物及其制品"},
    "oat":            {"国标名": "燕麦",       "CAS号": None, "备注": "谷物及其制品"},
    "oats":           {"国标名": "燕麦",       "CAS号": None, "备注": "谷物及其制品"},
    "milk":           {"国标名": "牛奶",       "CAS号": None, "备注": "乳及乳制品"},
    "cheese":         {"国标名": "奶酪",       "CAS号": None, "备注": "乳及乳制品"},
    "sunflower oil":  {"国标名": "葵花籽油",   "CAS号": None, "备注": "油脂及其制品"},
    "olive oil":      {"国标名": "橄榄油",     "CAS号": None, "备注": "油脂及其制品"},
    "mayonnaise":     {"国标名": "蛋黄酱",     "CAS号": None, "备注": "调味品/油脂制品"},
    "riso mandorla":  {"国标名": "大米杏仁饮料","CAS号": None,"备注": "植物基饮料"},
    "rice drink":     {"国标名": "大米饮料",   "CAS号": None, "备注": "植物基饮料"},
    "chocolate":      {"国标名": "巧克力",     "CAS号": None, "备注": "糖果类"},
    "cocoa":          {"国标名": "可可",       "CAS号": None, "备注": "糖果类"},
    "orange juice":   {"国标名": "橙汁",       "CAS号": None, "备注": "果汁"},
    # ── 中文直接映射 ─────────────────────────────────────────────
    "苯甲酸钠": {"国标名": "苯甲酸钠", "CAS号": "532-32-1",    "备注": "GB 2760-2014 防腐剂"},
    "二氧化钛": {"国标名": "二氧化钛", "CAS号": "13463-67-7",  "备注": "EU Reg 2022/63 已禁用"},
    "草甘膦":   {"国标名": "草甘膦",   "CAS号": "1071-83-6",   "备注": "除草剂"},
    "丙烯酰胺": {"国标名": "丙烯酰胺", "CAS号": "79-06-1",     "备注": "EU 基准值法规"},
    "铅":       {"国标名": "铅",       "CAS号": "7439-92-1",   "备注": "GB 2762-2022"},
    "镉":       {"国标名": "镉",       "CAS号": "7440-43-9",   "备注": "GB 2762-2022"},
    "汞":       {"国标名": "汞",       "CAS号": "7439-97-6",   "备注": "GB 2762-2022"},
    "砷":       {"国标名": "砷",       "CAS号": "7440-38-2",   "备注": "GB 2762-2022"},
    "农药":     {"国标名": "农药",     "CAS号": None,           "备注": "GB 2763-2021"},
}

# ══════════════════════════════════════════════════════════════════════
#  2. 限量数据（来自 run_claude_skills.py LIMITS 字典）
# ══════════════════════════════════════════════════════════════════════
LIMITS = {
    "中国": {
        "畜禽肉及其制品": {"铅": (0.2, "GB 2762-2022"), "镉": (0.1, "GB 2762-2022"), "农药": (0.05, "GB 2763-2021")},
        "水产品":        {"铅": (0.5, "GB 2762-2022"), "镉": (0.1, "GB 2762-2022"), "农药": (0.05, "GB 2763-2021")},
        "谷物及其制品":  {"铅": (0.2, "GB 2762-2022"), "镉": (0.1, "GB 2762-2022"), "农药": (0.05, "GB 2763-2021")},
        "蔬菜及其制品":  {"铅": (0.3, "GB 2762-2022"), "镉": (0.2, "GB 2762-2022"), "农药": (0.05, "GB 2763-2021")},
        "水果及其制品":  {"铅": (0.1, "GB 2762-2022"), "镉": (0.05,"GB 2762-2022"), "农药": (0.05, "GB 2763-2021")},
        "乳及乳制品":    {"铅": (0.05,"GB 2762-2022"), "镉": (0.01,"GB 2762-2022"), "农药": (0.05, "GB 2763-2021")},
        "油脂及其制品":  {"铅": (0.1, "GB 2762-2022"), "镉": (0.1, "GB 2762-2022"), "农药": (0.05, "GB 2763-2021")},
    },
    "日本": {
        "畜禽肉及其制品": {"铅": (0.1, "食品衛生法告示"), "镉": (0.05,"食品衛生法告示"), "农药": (0.01, "肯定リスト制度")},
        "水产品":        {"铅": (0.5, "食品衛生法告示"), "镉": (0.1, "食品衛生法告示"), "农药": (0.01, "肯定リスト制度")},
        "谷物及其制品":  {"铅": (0.2, "食品衛生法告示"), "镉": (0.4, "食品衛生法告示"), "农药": (0.01, "肯定リスト制度")},
        "蔬菜及其制品":  {"铅": (0.1, "食品衛生法告示"), "镉": (0.05,"食品衛生法告示"), "农药": (0.01, "肯定リスト制度")},
        "水果及其制品":  {"铅": (0.1, "食品衛生法告示"), "镉": (0.05,"食品衛生法告示"), "农药": (0.01, "肯定リスト制度")},
        "乳及乳制品":    {"铅": (0.02,"食品衛生法告示"), "镉": (0.01,"食品衛生法告示"), "农药": (0.01, "肯定リスト制度")},
        "油脂及其制品":  {"铅": (0.1, "食品衛生法告示"), "镉": (0.05,"食品衛生法告示"), "农药": (0.01, "肯定リスト制度")},
    },
    "韩国": {
        "畜禽肉及其制品": {"铅": (0.1, "KFDA 식품공전"), "镉": (0.05,"KFDA 식품공전"), "农药": (0.01, "KFDA MRL")},
        "水产品":        {"铅": (0.5, "KFDA 식품공전"), "镉": (0.1, "KFDA 식품공전"), "农药": (0.01, "KFDA MRL")},
        "谷物及其制品":  {"铅": (0.2, "KFDA 식품공전"), "镉": (0.2, "KFDA 식품공전"), "农药": (0.01, "KFDA MRL")},
        "蔬菜及其制品":  {"铅": (0.1, "KFDA 식품공전"), "镉": (0.05,"KFDA 식품공전"), "农药": (0.01, "KFDA MRL")},
        "水果及其制品":  {"铅": (0.1, "KFDA 식품공전"), "镉": (0.05,"KFDA 식품공전"), "农药": (0.01, "KFDA MRL")},
        "乳及乳制品":    {"铅": (0.02,"KFDA 식품공전"), "镉": (0.01,"KFDA 식품공전"), "农药": (0.01, "KFDA MRL")},
        "油脂及其制品":  {"铅": (0.1, "KFDA 식품공전"), "镉": (0.05,"KFDA 식품공전"), "农药": (0.01, "KFDA MRL")},
    },
    "美国": {
        "畜禽肉及其制品": {"铅": (None,"FDA 无通用限量"), "镉": (None,"FDA 无通用限量"), "农药": (0.1, "EPA CFR 40")},
        "水产品":        {"铅": (None,"FDA 无通用限量"), "镉": (None,"FDA 无通用限量"), "农药": (0.1, "EPA CFR 40")},
        "谷物及其制品":  {"铅": (None,"FDA 无通用限量"), "镉": (None,"FDA 无通用限量"), "农药": (0.1, "EPA CFR 40")},
        "蔬菜及其制品":  {"铅": (None,"FDA 无通用限量"), "镉": (None,"FDA 无通用限量"), "农药": (0.1, "EPA CFR 40")},
        "水果及其制品":  {"铅": (None,"FDA 无通用限量"), "镉": (None,"FDA 无通用限量"), "农药": (0.1, "EPA CFR 40")},
        "乳及乳制品":    {"铅": (None,"FDA 无通用限量"), "镉": (None,"FDA 无通用限量"), "农药": (0.1, "EPA CFR 40")},
        "油脂及其制品":  {"铅": (None,"FDA 无通用限量"), "镉": (None,"FDA 无通用限量"), "农药": (0.1, "EPA CFR 40")},
    },
    "欧盟": {
        "畜禽肉及其制品": {"铅": (0.1, "EU Reg 2023/915"), "镉": (0.05,"EU Reg 2023/915"), "农药": (0.01, "EU Reg 396/2005")},
        "水产品":        {"铅": (0.3, "EU Reg 2023/915"), "镉": (0.05,"EU Reg 2023/915"), "农药": (0.01, "EU Reg 396/2005")},
        "谷物及其制品":  {"铅": (0.2, "EU Reg 2023/915"), "镉": (0.1, "EU Reg 2023/915"), "农药": (0.01, "EU Reg 396/2005")},
        "蔬菜及其制品":  {"铅": (0.1, "EU Reg 2023/915"), "镉": (0.05,"EU Reg 2023/915"), "农药": (0.01, "EU Reg 396/2005")},
        "水果及其制品":  {"铅": (0.1, "EU Reg 2023/915"), "镉": (0.05,"EU Reg 2023/915"), "农药": (0.01, "EU Reg 396/2005")},
        "乳及乳制品":    {"铅": (0.02,"EU Reg 2023/915"), "镉": (0.01,"EU Reg 2023/915"), "农药": (0.01, "EU Reg 396/2005")},
        "油脂及其制品":  {"铅": (0.1, "EU Reg 2023/915"), "镉": (0.05,"EU Reg 2023/915"), "农药": (0.01, "EU Reg 396/2005")},
    },
}
# 德法直接引用欧盟标准
LIMITS["德国"] = LIMITS["法国"] = LIMITS["欧盟"]


# ══════════════════════════════════════════════════════════════════════
#  3. 将 LIMITS 转为 SPO 三元组
# ══════════════════════════════════════════════════════════════════════
COUNTRY_DATE = {
    "中国":  "2023-06", "日本": "2018-12", "韩国": "2020-01",
    "美国":  "2017-09", "欧盟": "2023-07", "德国": "2023-07", "法国": "2023-07",
}
COUNTRY_SOURCE_REF = {
    "中国": "GB 2762-2022 / GB 2763-2021",
    "日本": "食品衛生法告示 / 肯定リスト制度",
    "韩国": "KFDA 식품공전",
    "美国": "EPA CFR 40 / FDA",
    "欧盟": "EU Reg 2023/915 / EU Reg 396/2005",
    "德国": "EU Reg 2023/915 / EU Reg 396/2005",
    "法国": "EU Reg 2023/915 / EU Reg 396/2005",
}

def limits_to_triples(limits_dict):
    triples = []
    for country, cats in limits_dict.items():
        if country in ("德国", "法国"):  # 已在欧盟中覆盖，避免重复
            continue
        for cat, substances in cats.items():
            for sub, (val, source) in substances.items():
                limit_str = f"{val} mg/kg" if val is not None else "无通用限量"
                triples.append({
                    "subject":    f"{cat}（{country}进口）",
                    "predicate":  f"{sub}限量",
                    "object":     limit_str,
                    "country":    country,
                    "source":     source,
                    "effective_date": COUNTRY_DATE.get(country, "N/A"),
                    "legal_status":   "Immediately Effective",
                    "confidence": 0.90,
                    "tags":       [country, cat, sub, "MRL"],
                })
    return triples


# ══════════════════════════════════════════════════════════════════════
#  4. 静态 KB（来自 run_claude_skills.py _STATIC_KB，含更多条目）
# ══════════════════════════════════════════════════════════════════════
STATIC_KB_TRIPLES = [
    {
        "subject":    "二氧化钛 E171",
        "predicate":  "欧盟食品添加剂准入状态",
        "object":     "已禁用（2022年8月起）",
        "country":    "欧盟",
        "source":     "EU Reg 2022/63",
        "effective_date": "2022-08",
        "legal_status":   "Immediately Effective",
        "confidence": 0.95,
        "tags":       ["欧盟", "食品添加剂", "二氧化钛", "E171", "禁用"],
    },
    {
        "subject":    "丙烯酰胺",
        "predicate":  "欧盟基准水平",
        "object":     "EU Reg 2023/194 更新基准值，适用薯片/饼干/咖啡等",
        "country":    "欧盟",
        "source":     "EU Reg 2023/194",
        "effective_date": "2023-01",
        "legal_status":   "Immediately Effective",
        "confidence": 0.92,
        "tags":       ["欧盟", "丙烯酰胺", "基准值"],
    },
    {
        "subject":    "铅（所有食品类别）",
        "predicate":  "欧盟限量收严",
        "object":     "EU Reg 2023/915 全面收严铅限量",
        "country":    "欧盟",
        "source":     "EU Reg 2023/915",
        "effective_date": "2023-07",
        "legal_status":   "Immediately Effective",
        "confidence": 0.95,
        "tags":       ["欧盟", "铅", "污染物"],
    },
    {
        "subject":    "可可/巧克力镉限量",
        "predicate":  "欧盟拟议收严",
        "object":     "拟进一步降低可可和巧克力产品镉限量（2025年起）",
        "country":    "欧盟",
        "source":     "EU Reg 拟议",
        "effective_date": "2025-01",
        "legal_status":   "Notification of Update",
        "confidence": 0.80,
        "tags":       ["欧盟", "镉", "巧克力", "可可"],
    },
    {
        "subject":    "燕麦草甘膦",
        "predicate":  "日本MRL",
        "object":     "日本将燕麦草甘膦MRL从0.2提高至30 mg/kg",
        "country":    "日本",
        "source":     "肯定リスト制度 2018修订",
        "effective_date": "2018-12",
        "legal_status":   "Immediately Effective",
        "confidence": 0.92,
        "tags":       ["日本", "草甘膦", "燕麦", "MRL"],
    },
    {
        "subject":    "未注册农药（日本）",
        "predicate":  "肯定リスト制度默认限量",
        "object":     "一律基准 0.01 mg/kg（未制定MRL的农药统一适用）",
        "country":    "日本",
        "source":     "食品衛生法 第11条第3项",
        "effective_date": "2006-05",
        "legal_status":   "Immediately Effective",
        "confidence": 0.95,
        "tags":       ["日本", "农药", "肯定リスト", "一律基准"],
    },
    {
        "subject":    "GB 2762-2022",
        "predicate":  "适用范围",
        "object":     "中国食品污染物（铅/镉/汞/砷）国家标准，2023年6月实施",
        "country":    "中国",
        "source":     "GB 2762-2022",
        "effective_date": "2023-06",
        "legal_status":   "Immediately Effective",
        "confidence": 0.95,
        "tags":       ["中国", "污染物", "铅", "镉", "汞", "砷"],
    },
    {
        "subject":    "GB 2763-2021",
        "predicate":  "适用范围",
        "object":     "中国食品农药最大残留限量国家标准，2021年9月实施",
        "country":    "中国",
        "source":     "GB 2763-2021",
        "effective_date": "2021-09",
        "legal_status":   "Immediately Effective",
        "confidence": 0.95,
        "tags":       ["中国", "农药", "MRL", "GB2763"],
    },
    {
        "subject":    "境外食品企业注册",
        "predicate":  "中国GACC要求",
        "object":     "GACC 248/249令：境外食品生产企业须向GACC注册，2022年1月实施",
        "country":    "中国",
        "source":     "GACC Decree 248 / 249",
        "effective_date": "2022-01",
        "legal_status":   "Immediately Effective",
        "confidence": 0.92,
        "tags":       ["中国", "GACC", "注册", "进口", "程序"],
    },
    {
        "subject":    "GB 2760-2014",
        "predicate":  "食品添加剂使用标准",
        "object":     "中国食品添加剂使用标准，规定各类添加剂允许范围和最大使用量",
        "country":    "中国",
        "source":     "GB 2760-2014",
        "effective_date": "2015-05",
        "legal_status":   "Immediately Effective",
        "confidence": 0.92,
        "tags":       ["中国", "食品添加剂", "防腐剂", "甜味剂", "着色剂"],
    },
    {
        "subject":    "FDA FSMA",
        "predicate":  "美国食品安全现代化法",
        "object":     "FSMA预防控制规则全面生效，适用进口食品设施注册",
        "country":    "美国",
        "source":     "FDA FSMA 2011",
        "effective_date": "2017-09",
        "legal_status":   "Immediately Effective",
        "confidence": 0.92,
        "tags":       ["美国", "FSMA", "FDA", "进口", "食品安全"],
    },
    {
        "subject":    "PFAS 食品接触材料",
        "predicate":  "FDA 行动水平（拟议）",
        "object":     "FDA 拟议PFAS行动水平，最终规则待定",
        "country":    "美国",
        "source":     "FDA PFAS Action Plan",
        "effective_date": "TBD",
        "legal_status":   "Proposed",
        "confidence": 0.70,
        "tags":       ["美国", "PFAS", "食品接触", "拟议"],
    },
    {
        "subject":    "韩国进口食品申报",
        "predicate":  "MFDS 程序",
        "object":     "韩国MFDS修订进口食品申报及检验规定，2020年1月实施",
        "country":    "韩国",
        "source":     "MFDS 2020",
        "effective_date": "2020-01",
        "legal_status":   "Immediately Effective",
        "confidence": 0.88,
        "tags":       ["韩国", "MFDS", "进口", "申报", "检验"],
    },
    {
        "subject":    "EU Reg 1169/2011",
        "predicate":  "欧盟食品标签法规",
        "object":     "强制标注过敏原（花生/坚果/麸质/牛奶/蛋/甲壳类等14种），2014年实施",
        "country":    "欧盟",
        "source":     "EU Reg 1169/2011",
        "effective_date": "2014-12",
        "legal_status":   "Immediately Effective",
        "confidence": 0.92,
        "tags":       ["欧盟", "标签", "过敏原", "1169/2011"],
    },
    {
        "subject":    "EU Reg 1924/2006",
        "predicate":  "营养和健康声称法规",
        "object":     "欧盟食品营养及健康声称须经EFSA批准并列于正面清单",
        "country":    "欧盟",
        "source":     "EU Reg 1924/2006",
        "effective_date": "2007-07",
        "legal_status":   "Immediately Effective",
        "confidence": 0.90,
        "tags":       ["欧盟", "标签", "健康声称", "营养声称"],
    },
    {
        "subject":    "中国预包装食品标签",
        "predicate":  "GB 7718-2011 要求",
        "object":     "须标注食品名称/配料/净含量/保质期/生产商/产地等，中文标注强制",
        "country":    "中国",
        "source":     "GB 7718-2011",
        "effective_date": "2012-04",
        "legal_status":   "Immediately Effective",
        "confidence": 0.92,
        "tags":       ["中国", "标签", "预包装", "GB7718"],
    },
    {
        "subject":    "日本食品标签法",
        "predicate":  "过敏原标注",
        "object":     "日本强制标注7大过敏原（小麦/荞麦/蛋/奶/花生/虾/蟹），建议标注20种",
        "country":    "日本",
        "source":     "食品表示法（2015）",
        "effective_date": "2015-04",
        "legal_status":   "Immediately Effective",
        "confidence": 0.90,
        "tags":       ["日本", "标签", "过敏原", "食品表示法"],
    },
    {
        "subject":    "Australia FSANZ",
        "predicate":  "澳大利亚食品标准法典",
        "object":     "Food Standards Australia New Zealand (FSANZ) 制定澳新食品标准法典，规定进口食品限量/标签/添加剂",
        "country":    "澳大利亚",
        "source":     "FSANZ Food Standards Code",
        "effective_date": "2002-01",
        "legal_status":   "Immediately Effective",
        "confidence": 0.88,
        "tags":       ["澳大利亚", "FSANZ", "进口", "标准法典"],
    },
    {
        "subject":    "苯甲酸钠（饮料）",
        "predicate":  "中国使用限量",
        "object":     "GB 2760-2014 规定碳酸饮料中苯甲酸钠最大使用量 0.2 g/kg",
        "country":    "中国",
        "source":     "GB 2760-2014 表A.1",
        "effective_date": "2015-05",
        "legal_status":   "Immediately Effective",
        "confidence": 0.92,
        "tags":       ["中国", "苯甲酸钠", "防腐剂", "饮料", "GB2760"],
    },
    {
        "subject":    "苯甲酸钠（饮料）",
        "predicate":  "欧盟使用限量",
        "object":     "EU Reg 1333/2008 规定软饮料中苯甲酸（E211）最大使用量 150 mg/kg",
        "country":    "欧盟",
        "source":     "EU Reg 1333/2008 Annex II",
        "effective_date": "2011-01",
        "legal_status":   "Immediately Effective",
        "confidence": 0.92,
        "tags":       ["欧盟", "苯甲酸钠", "E211", "防腐剂", "饮料"],
    },
    {
        "subject":    "亚硝酸钠",
        "predicate":  "中国腌制肉制品限量",
        "object":     "GB 2760-2014 规定腌制肉制品中亚硝酸钠最大使用量 0.15 g/kg，残留量 ≤ 30 mg/kg",
        "country":    "中国",
        "source":     "GB 2760-2014",
        "effective_date": "2015-05",
        "legal_status":   "Immediately Effective",
        "confidence": 0.92,
        "tags":       ["中国", "亚硝酸钠", "肉制品", "添加剂"],
    },
]

# ══════════════════════════════════════════════════════════════════════
#  5. 生成并写出 JSON 文件
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # term_mapping.json
    tm_path = BASE / "term_mapping.json"
    with open(tm_path, "w", encoding="utf-8") as f:
        json.dump(TERM_MAP, f, ensure_ascii=False, indent=2)
    print(f"term_mapping.json  : {len(TERM_MAP)} 条术语映射")

    # regulation_kg.json
    limit_triples = limits_to_triples(LIMITS)
    all_triples   = limit_triples + STATIC_KB_TRIPLES
    kg_path = BASE / "regulation_kg.json"
    with open(kg_path, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)
    print(f"regulation_kg.json : {len(all_triples)} 条三元组")
    print(f"  其中限量三元组 {len(limit_triples)} 条")
    print(f"  静态KB三元组   {len(STATIC_KB_TRIPLES)} 条")
