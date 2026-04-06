import os
import pandas as pd
import numpy as np

# ===================== 1. 路径配置 =====================

# 基础根目录
root_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识网络分析\1125计算结果-对齐版"

# 输入1：原始能力-金额数据 (源头数据，包含 total_amount)
# 路径通常在 "能力向量与金额" 文件夹下
raw_data_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识网络分析\能力向量与金额"
raw_file_path = os.path.join(raw_data_dir, "designer_ability_vectors_all_years.csv")

# 输入2：上一步计算的生态位数值结果 (Niche Width & Overlap)
niche_data_dir = os.path.join(root_dir, "生态位数值计算结果")
niche_file_path = os.path.join(niche_data_dir, "Niche_Values_All_Years.csv")

# 输出：最终的价值网络节点表 (按年)
output_dir = os.path.join(root_dir, "价值网络构建数据")
os.makedirs(output_dir, exist_ok=True)


# ===================== 2. 核心计算函数 =====================

def calculate_value_potential(df_raw):
    print("正在计算价值位势 (Value Potential)...")

    # 1. 聚合计算总金额和总单量 (按人+年)
    # 原始数据可能包含多个能力，需要sum起来
    df_agg = df_raw.groupby(['designer', 'year']).agg({
        'total_amount': 'sum',
        'order_count': 'sum'
    }).reset_index()

    # 2. 计算价值位势 (平均单价)
    # Value_Potential = Total_Amount / Order_Count
    df_agg['Value_Potential'] = df_agg['total_amount'] / (df_agg['order_count'] + 1e-9)

    # 重命名列以匹配生态位表
    df_agg = df_agg.rename(columns={
        'designer': 'Designer_ID',
        'year': 'Year',
        'total_amount': 'Total_Revenue'  # 总营收
    })

    return df_agg[['Designer_ID', 'Year', 'Value_Potential', 'Total_Revenue']]


# ===================== 3. 主程序：合并与拆分输出 =====================

def main():
    # --- Step 1: 读取数据 ---
    if not os.path.exists(niche_file_path) or not os.path.exists(raw_file_path):
        print(f"[错误] 输入文件缺失。请检查路径。")
        return

    print("1. 读取生态位指标与原始金额数据...")
    df_niche = pd.read_csv(niche_file_path)
    df_raw = pd.read_csv(raw_file_path)

    # --- Step 2: 计算价值位势 ---
    df_value = calculate_value_potential(df_raw)

    # --- Step 3: 合并数据 (Inner Join) ---
    print("2. 正在合并数据...")

    # 类型转换，防止 merge 失败
    df_niche['Year'] = df_niche['Year'].astype(int)
    df_value['Year'] = df_value['Year'].astype(int)
    df_niche['Designer_ID'] = df_niche['Designer_ID'].astype(str)
    df_value['Designer_ID'] = df_value['Designer_ID'].astype(str)

    df_final = pd.merge(df_niche, df_value, on=['Designer_ID', 'Year'], how='inner')

    # 处理对数价值 (方便绘图)
    df_final['Log_Value'] = np.log1p(df_final['Value_Potential'])

    # --- Step 4: 按年度拆分并保存 (核心修改点) ---
    print(f"\n3. 开始按年度输出文件至: {output_dir}")

    unique_years = sorted(df_final['Year'].unique())

    for year in unique_years:
        # 筛选当年的数据
        df_year = df_final[df_final['Year'] == year].copy()

        if df_year.empty:
            continue

        # 构造文件名
        filename = f"Value_Network_Nodes_{year}.csv"
        save_path = os.path.join(output_dir, filename)

        # 保存
        df_year.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"  - [成功] {year} 年数据已保存: {filename} (包含 {len(df_year)} 位设计师)")

    # (可选) 依然保存一份总表，以此备用
    master_path = os.path.join(output_dir, "Value_Network_Nodes_Master_All_Years.csv")
    df_final.to_csv(master_path, index=False, encoding='utf-8-sig')
    print(f"\n[汇总] 所有年份的总表也已保存: Value_Network_Nodes_Master_All_Years.csv")


if __name__ == "__main__":
    main()