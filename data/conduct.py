import pandas as pd
import random
import re

# 设置随机种子
random.seed(42)

def mock_ingredients_content(ingredients_str, additives_tags=""):
    """
    核心功能：将逗号分隔的配料字符串，转换为带有合理随机数值的成分明细列表。
    逻辑：前面的配料按 g/100g 递减，后面的（或明确是添加剂的）给 mg/kg。
    """
    if pd.isna(ingredients_str) or str(ingredients_str).strip() == "":
        return "未提供详细配料表"
        
    # 简单处理掉括号内的逗号，防止如 "Flour (wheat, barley), sugar" 被错误切分
    safe_str = re.sub(r'\([^)]*\)', lambda x: x.group(0).replace(',', '，'), str(ingredients_str))
    items = [i.strip() for i in safe_str.split(',') if i.strip()]
    
    result = []
    # 模拟主要配料的初始重量占比（比如占比 30g ~ 60g）
    current_g = random.uniform(30.0, 60.0) 
    
    for idx, item in enumerate(items):
        item = item.replace('，', ',') # 恢复逗号
        
        # 如果配料表已经读到后半段，或者包含某些添加剂特征词，判定为微量成分
        is_minor = idx >= (len(items) * 0.7) or any(keyword in item.lower() for keyword in ['acid', 'color', 'flavor', 'preservative', 'extract', 'gum'])
        
        if is_minor:
            # 微量成分/添加剂，采用 mg/kg，随机参考范围 10 - 2500
            val_mg = random.randint(10, 2500)
            result.append(f"  - {item}: {val_mg} mg/kg")
        else:
            # 主要成分，采用 g/100g
            result.append(f"  - {item}: {current_g:.1f} g/100g")
            # 下一个成分的含量递减 (乘以 0.4 ~ 0.8)
            current_g = current_g * random.uniform(0.4, 0.8)

    # 针对欧盟数据，如果有明确提取出的添加剂E编号，单独列出
    if pd.notna(additives_tags) and str(additives_tags).strip() and str(additives_tags) != 'nan':
        adds = [a.strip() for a in str(additives_tags).split(',') if a.strip()]
        result.append("  [特定添加剂/E编号专项检测]:")
        for a in adds:
            val_mg = random.randint(20, 1500)
            result.append(f"  - {a}: {val_mg} mg/kg")
            
    return "\n".join(result)

def format_us_profile(df_us):
    profiles = []
    # 只要有产品名和配料表的行就保留
    df_valid = df_us.dropna(subset=['description', 'ingredients'])
    
    print(f"  - 正在处理美国数据 ({len(df_valid)} 行)...")
    for idx, row in df_valid.iterrows():
        ing_text = mock_ingredients_content(row['ingredients'])
        
        profile_text = (
            f"【食品基本信息】\n"
            f"- 数据来源：USA FDC\n"
            f"- 产品名称：{row.get('description', 'N/A')}\n"
            f"- 品牌所有者：{row.get('brand_owner', 'N/A')}\n"
            f"- 产品类别：{row.get('branded_food_category', 'N/A')}\n"
            f"- 建议食用量：{row.get('serving_size', 'N/A')} {row.get('serving_size_unit', '')}\n\n"
            f"【营养成分基础数据】\n"
            f"- 总能量：{row.get('能量_Energy(kcal)', 'N/A')} kcal\n"
            f"- 蛋白质：{row.get('蛋白质_Protein(g)', 'N/A')} g\n"
            f"- 总脂肪：{row.get('总脂肪_Fat(g)', 'N/A')} g (其中反式脂肪: {row.get('反式脂肪_TransFat(g)', 'N/A')} g)\n"
            f"- 碳水化合物：{row.get('碳水化合物_Carbs(g)', 'N/A')} g (其中总糖: {row.get('总糖_Sugars(g)', 'N/A')} g)\n"
            f"- 钠：{row.get('钠_Sodium(mg)', 'N/A')} mg\n"
            f"- 钙：{row.get('钙_Calcium(mg)', 'N/A')} mg\n\n"
            f"【详细配料及含量信息】\n"
            f"{ing_text}"
        )
        profiles.append({"id": f"US_DATA_{idx}", "food_profile": profile_text})
    return profiles

def format_eu_profile(df_eu):
    profiles = []
    df_valid = df_eu.dropna(subset=['product_name', 'ingredients_text'])
    
    print(f"  - 正在处理欧盟数据 ({len(df_valid)} 行)...")
    for idx, row in df_valid.iterrows():
        ing_text = mock_ingredients_content(row['ingredients_text'], row.get('additives_tags', ''))
        
        profile_text = (
            f"【食品基本信息】\n"
            f"- 数据来源：EU OpenFoodFacts\n"
            f"- 产品名称：{row.get('product_name', 'N/A')}\n"
            f"- 品牌：{row.get('brands', 'N/A')}\n"
            f"- 销售国家：{row.get('countries_en', 'N/A')}\n"
            f"- 食品大类：{row.get('main_category_en', 'N/A')}\n"
            f"- 营养等级(法国)：{row.get('nutrition_grade_fr', 'N/A')}\n\n"
            f"【标示过敏原】\n"
            f"- {row.get('allergens', '无声明或未检测到')}\n\n"
            f"【每100g营养数据】\n"
            f"- 能量：{row.get('energy_100g', 'N/A')} kJ/kcal\n"
            f"- 蛋白质：{row.get('proteins_100g', 'N/A')} g\n"
            f"- 脂肪：{row.get('fat_100g', 'N/A')} g\n"
            f"- 碳水化合物：{row.get('carbohydrates_100g', 'N/A')} g (含糖: {row.get('sugars_100g', 'N/A')} g)\n"
            f"- 钠：{row.get('sodium_100g', 'N/A')} g\n\n"
            f"【详细配料及含量信息】\n"
            f"{ing_text}"
        )
        profiles.append({"id": f"EU_DATA_{idx}", "food_profile": profile_text})
    return profiles

def format_aus_profile(df_aus):
    profiles = []
    df_valid = df_aus.dropna(subset=['Food Name'])
    
    print(f"  - 正在处理澳洲数据 ({len(df_valid)} 行)...")
    for idx, row in df_valid.iterrows():
        # 澳洲是初级农产品，通常没有长串的工业配料表，我们为其生成“实验室理化/污染物分析报告”格式
        pb_val = round(random.uniform(0.01, 0.5), 3) # 铅 mg/kg
        cd_val = round(random.uniform(0.01, 0.2), 3) # 镉 mg/kg
        pest_val = round(random.uniform(0.0, 2.0), 2) # 农残 mg/kg
        
        profile_text = (
            f"【农产品基本信息】\n"
            f"- 数据来源：Australia Primary Food\n"
            f"- 产品/样本名称：{row.get('Food Name', 'N/A')}\n"
            f"- 详细描述：{row.get('Food Description', 'N/A')}\n"
            f"- 分类归属：{row.get('Classification Name', 'N/A')} (编号: {row.get('Classification', 'N/A')})\n"
            f"- 采样详情：{row.get('Sampling Details', 'N/A')}\n"
            f"- 公共食物键码：{row.get('Public Food Key', 'N/A')}\n\n"
            f"【理化特性指标】\n"
            f"- 氮换算系数 (Nitrogen Factor)：{row.get('Nitrogen Factor', 'N/A')}\n"
            f"- 脂肪换算系数 (Fat Factor)：{row.get('Fat Factor', 'N/A')}\n"
            f"- 比重 (Specific Gravity)：{row.get('Specific Gravity', 'N/A')}\n"
            f"- 分析部位：{row.get('Analysed Portion', 'N/A')}\n"
            f"- 未分析部位(废弃率)：{row.get('Unanalysed Portion', 'N/A')}\n\n"
            f"【实验室污染物随机测定数据明细】\n"
            f"  - 铅 (Lead, Pb): {pb_val} mg/kg\n"
            f"  - 镉 (Cadmium, Cd): {cd_val} mg/kg\n"
            f"  - 综合农药残留 (Pesticide Residues): {pest_val} mg/kg\n"
            f"  - 水分 (Moisture): {round(random.uniform(10.0, 85.0), 1)} g/100g"
        )
        profiles.append({"id": f"AUS_DATA_{idx}", "food_profile": profile_text})
    return profiles

def main():
    # ===============
    # 请修改为你的实际文件名
    file_us = r"E:\论文\跨境对齐\准备阶段\商品数据集\可用\fooddata_central_USA\Final_Compliance_Dataset.csv"
    file_eu = r"E:\论文\跨境对齐\准备阶段\商品数据集\可用\OPENFOODFACT\Cleaned_European_Foods.csv"
    file_aus = r"E:\论文\跨境对齐\准备阶段\商品数据集\可用\澳洲初级农产品\Food Composition.csv"
    # ===============

    try:
        df_us = pd.read_csv(file_us, low_memory=False)
        us_data = format_us_profile(df_us)
        pd.DataFrame(us_data).to_csv("formatted_USA_profiles.csv", index=False, encoding="utf-8-sig")
        print("✅ 成功生成: formatted_USA_profiles.csv\n")
    except FileNotFoundError:
        print(f"❌ 找不到文件: {file_us}\n")

    try:
        df_eu = pd.read_csv(file_eu, low_memory=False)
        eu_data = format_eu_profile(df_eu)
        pd.DataFrame(eu_data).to_csv("formatted_EU_profiles.csv", index=False, encoding="utf-8-sig")
        print("✅ 成功生成: formatted_EU_profiles.csv\n")
    except FileNotFoundError:
        print(f"❌ 找不到文件: {file_eu}\n")

    try:
        df_aus = pd.read_csv(file_aus, low_memory=False)
        aus_data = format_aus_profile(df_aus)
        pd.DataFrame(aus_data).to_csv("formatted_AUS_profiles.csv", index=False, encoding="utf-8-sig")
        print("✅ 成功生成: formatted_AUS_profiles.csv\n")
    except FileNotFoundError:
        print(f"❌ 找不到文件: {file_aus}\n")

    print("全部格式化档案生成完毕！你现在可以用这些数据自由构建任何提问了。")

if __name__ == "__main__":
    main()

