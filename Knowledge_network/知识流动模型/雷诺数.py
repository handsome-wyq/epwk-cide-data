import pandas as pd
import numpy as np
import os

# ==================== 配置 ====================
# 输入文件夹
data_dir = r'D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\设计师知识网络-全部\加权度排序'
# 2014年数据文件夹
data_2014_path = r'D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\设计师知识网络-全部\节点表-2014_updated.csv'

# 输出文件夹
output_dir = r'D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识流动模型'
os.makedirs(output_dir, exist_ok=True)

# 年份
years = list(range(2015, 2025))

# 读取核心面板数据（用于 v 和 L）
panel_csv = os.path.join(output_dir, 'core_panel_data_简化.csv')
density_csv = os.path.join(output_dir, 'system_knowledge_density_简化.csv')

panel_df = pd.read_csv(panel_csv)
density_df = pd.read_csv(density_csv)
global_core_ids = set(panel_df['Id'])

# 结果存储
re_results = []

print("开始计算真实 PageRank 黏度与雷诺数...\n")

for year in years:
    print(f"处理 {year} 年...")

    # 1. 读取当年网络数据（含 pageranks）
    file_path = os.path.join(data_dir, f'{year}.csv')
    if not os.path.exists(file_path):
        print(f"   [跳过] {year}.csv 不存在")
        continue

    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # 识别列名
    id_col = next((c for c in df.columns if 'Id' in str(c)), None)
    size_col = next((c for c in df.columns if '设计师人数' in str(c)), None)
    pr_col = next((c for c in df.columns if 'pageranks' in str(c).lower()), None)

    if not all([id_col, size_col, pr_col]):
        print(f"   [跳过] {year}：缺少 Id / 设计师人数 / pageranks 列")
        continue
'''
    # 2. 提取活跃核心节点的 PageRank
    active_pr = []
    for _, row in df.iterrows():
        nid = row[id_col]
        if nid in global_core_ids and row[size_col] > 0:
            active_pr.append(row[pr_col])

    if len(active_pr) == 0:
        print(f"   {year}：无活跃核心节点")
        continue

    mean_pr = np.mean(active_pr)
    mu = 1 / ( mean_pr * 100 ) if mean_pr > 0 else 1  # 黏度
    '''
    # === 在您的主循环中替换 μ 计算 ===
all_pr_for_global = []  # 全局收集，用于归一化

# 第一轮：收集所有年份的 PageRank（用于全局均值）
print("第一轮：收集全局 PageRank 均值...")
for year in years:
    file_path = os.path.join(data_dir, f'{year}.csv')
    if not os.path.exists(file_path): continue
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    id_col = next((c for c in df.columns if 'Id' in str(c)), None)
    size_col = next((c for c in df.columns if '设计师人数' in str(c)), None)
    pr_col = next((c for c in df.columns if 'pagerank' in str(c).lower()), None)
    if not pr_col: continue
    for _, row in df.iterrows():
        if row[id_col] in global_core_ids and row[size_col] > 0:
            all_pr_for_global.append(row[pr_col])

global_mean_pr = np.mean(all_pr_for_global) if all_pr_for_global else 0.01
print(f"全局 PageRank 均值 = {global_mean_pr:.6f}")

# 第二轮：正式计算
re_results = []
for year in years:
    # ...（前面代码不变）
    file_path = os.path.join(data_dir, f'{year}.csv')
    if not os.path.exists(file_path): continue
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    id_col = next((c for c in df.columns if 'Id' in str(c)), None)
    size_col = next((c for c in df.columns if '设计师人数' in str(c)), None)
    pr_col = next((c for c in df.columns if 'pagerank' in str(c).lower()), None)
    if not pr_col: continue
    for _, row in df.iterrows():
        if row[id_col] in global_core_ids and row[size_col] > 0:
            all_pr_for_global.append(row[pr_col])
    # 提取活跃 PageRank
    active_pr = [row[pr_col] for _, row in df.iterrows()
                 if row[id_col] in global_core_ids and row[size_col] > 0]

    if len(active_pr) == 0:
        mu = 1.0
    else:
        mean_pr = np.mean(active_pr)
        print(f"Year: {year}, mean_pr: {mean_pr}")
        # 全局归一化黏度
        mu = global_mean_pr / mean_pr if mean_pr > 0 else 1.0


    # 3. 其他变量
    density_row = density_df[density_df['year'] == year]
    if density_row.empty:
        continue
    rho = density_row['system_knowledge_density'].iloc[0]
    L = density_row['active_core_nodes'].iloc[0]

    # 4. 速度 v
    # 读取2014年设计师人数数据
    df_2014 = pd.read_csv(data_2014_path, encoding='utf-8-sig')
    df_2014 = df_2014[[id_col, size_col]]  # 只保留Id和设计师人数列
    df_2014.set_index(id_col, inplace=True)  # 将Id作为索引
    if year == 2015:
        v = 0  # 默认为0
        for node_id in global_core_ids:
            if node_id in df[id_col].values and node_id in df_2014.index:
                size_2014 = df_2014.loc[node_id, size_col]  # 2014年设计师人数
                size_2015 = df[df[id_col] == node_id][size_col].values[0]  # 2015年设计师人数
                v += np.abs(size_2015 - size_2014)  # 计算差异
        v = v / len(global_core_ids)  # 平均速度
    elif year > 2015:
        year_cols = [f'designers_{y}' for y in years]
        f_matrix = panel_df[year_cols].values
        t_idx = years.index(year)
        if t_idx == 0:
            v = 0
        else:
            prev = f_matrix[:, t_idx - 1]
            curr = f_matrix[:, t_idx]
            delta = np.abs(curr - prev)
            v = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0

    # 5. Re
    if rho <= 0 or L <= 0 or mu <= 0:
        Re = 0
    else:
        Re = rho * v * L / mu /100

    re_results.append({
        'year': year,
        'rho': round(rho, 6),
        'v': round(v, 1),
        'L': int(L),
        'mean_pagerank': round(mean_pr, 6),
        'mu_pagerank': round(mu, 6),
        'Re': round(Re, 1)
    })

    state = "层流" if Re < 2000 else "过渡流" if Re < 4000 else "湍流"
    print(f"   ρ={rho:.6f}, v={v:.1f}, L={L}, μ={mu:.6f}, Re={Re:.1f} → {state}")

# 保存
re_df = pd.DataFrame(re_results)
re_csv = os.path.join(output_dir, 'reynolds_number_final.csv')
re_df.to_csv(re_csv, index=False, encoding='utf-8-sig')
print(f"\n最终雷诺数结果已保存：{re_csv}")