"""
功能:
    1. 从全量原始数据中每年筛选加权度前25%能力节点
    2. 构建全局核心能力集合（并集）
    3. 跨年追踪每个核心能力的设计师人数（缺失填0）
    4. 每年计算“核心技能平均规模”作为系统知识密度
    5. 输出年度系统知识密度 + 核心面板数据
"""
import pandas as pd
import numpy as np
import os
from collections import defaultdict

# ==================== 配置区 ====================
# 数据文件夹
folder_path = r'D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\设计师知识网络-全部\加权度排序'

# 年份范围
years = list(range(2015, 2025))

# 筛选比例
TOP_PERCENT = 0.25        # 每年加权度前25%
MIN_NODES = 30            # 最小选取节点数，防止小年份样本不足

# 输出目录
output_dir = r'D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识流动模型'
os.makedirs(output_dir, exist_ok=True)  # 自动创建
# ===============================================

print("开始计算知识密度（核心技能平均规模法）...\n")

# Step 1: 读取全量数据 + 每年筛选加权度前25%
year_data_full = {}       # year -> DataFrame (原始全量)
core_ids_per_year = {}    # year -> set of Ids

print("Step 1: 每年筛选加权度前25%能力节点...")
for year in years:
    file_path = os.path.join(folder_path, f'{year}.csv')
    if not os.path.exists(file_path):
        print(f"   [跳过] {year}.csv 不存在")
        continue

    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # 自动识别列名
    id_col = next((c for c in df.columns if 'Id' in str(c)), None)
    size_col = next((c for c in df.columns if '设计师人数' in str(c)), None)
    degree_col = next((c for c in df.columns if 'weighted degree' in str(c).lower()), None)

    if not all([id_col, size_col, degree_col]):
        print(f"   [跳过] {year}：缺少必要列")
        continue

    total_nodes = len(df)
    select_k = max(int(TOP_PERCENT * total_nodes), MIN_NODES)
    df_sorted = df.sort_values(degree_col, ascending=False).head(select_k)

    year_data_full[year] = df  # 保存原始数据用于查值
    core_ids_per_year[year] = set(df_sorted[id_col])

    print(f"   {year}: 总节点 {total_nodes} → 筛选前25% = {len(df_sorted)} 个")

# Step 2: 全局核心能力集合（并集）
global_core_ids = set()
for ids_set in core_ids_per_year.values():
    global_core_ids.update(ids_set)
print(f"\n全局核心能力节点数（并集）：{len(global_core_ids)}")

# Step 3: 构建跨年面板数据：{Id: {year: designers}}
print("\nStep 2: 构建跨年面板数据（从原始数据提取真实人数）...")
panel_data = defaultdict(dict)  # {Id: {year: designers_count}}

for node_id in global_core_ids:
    for year in years:
        if year not in year_data_full:
            panel_data[node_id][year] = 0
            continue
        df_year = year_data_full[year]
        id_c = next((c for c in df_year.columns if 'Id' in str(c)), None)
        size_c = next((c for c in df_year.columns if '设计师人数' in str(c)), None)
        if id_c is None or size_c is None:
            panel_data[node_id][year] = 0
            continue
        match = df_year[df_year[id_c] == node_id]
        if not match.empty:
            panel_data[node_id][year] = int(match[size_c].iloc[0])
        else:
            panel_data[node_id][year] = 0

# Step 4: 每年计算系统知识密度
print("\nStep 3: 计算每年系统知识密度...")
results = []

for year in years:
    # 收集该年所有核心节点的设计师人数
    sizes = [panel_data[node_id].get(year, 0) for node_id in global_core_ids]
    active_sizes = [s for s in sizes if s > 0]  # 只保留活跃节点
    n_active = len(active_sizes)

    # *** 计算逻辑修改处 ***
    if n_active > 0:
        # 核心技能平均规模 (内部视角)
        system_density = np.mean(active_sizes)
    else:
        # 如果当年没有活跃的核心节点，则密度为 NaN
        print(f"   {year}: 活跃节点 {n_active}，样本不足，密度设为 NaN")
        system_density = np.nan

    results.append({
        'year': year,
        'active_core_nodes': n_active,
        'total_core_nodes': len(global_core_ids),
        'system_knowledge_density': system_density,
    })

    status = f"NaN" if np.isnan(system_density) else f"{system_density:.4f}"
    print(f"   {year}: 活跃 {n_active}/{len(global_core_ids)}，系统密度 = {status}")

# Step 5: 保存结果
results_df = pd.DataFrame(results)
results_csv = os.path.join(output_dir, 'system_knowledge_density_简化.csv')
results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
print(f"\n系统知识密度已保存：{results_csv}")

# 保存核心面板数据（每行一个能力，每列一年）
panel_list = []
for node_id in global_core_ids:
    row = {'Id': node_id}
    row.update({f'designers_{year}': panel_data[node_id].get(year, 0) for year in years})
    panel_list.append(row)
panel_df = pd.DataFrame(panel_list)
panel_csv = os.path.join(output_dir, 'core_panel_data_简化.csv')
panel_df.to_csv(panel_csv, index=False, encoding='utf-8-sig')
print(f"核心面板数据已保存：{panel_csv}")

print(f"\n所有结果保存在目录：\n   {output_dir}")
print("""
输出文件说明：
   system_knowledge_density.csv → 年度系统知识密度（主结果）
   core_panel_data.csv         → 核心能力跨年设计师人数（含0）
""")