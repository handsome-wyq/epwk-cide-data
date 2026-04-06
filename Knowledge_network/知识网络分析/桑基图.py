import os
import pandas as pd

# ================= 配置 =================
root_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识网络分析\1125计算结果-对齐版"
input_path = os.path.join(root_dir, "价值网络聚类分析结果", "Designers_Clustered_Master.csv")
output_path = os.path.join(root_dir, "价值网络聚类分析结果", "Sankey_Data_Full_Lifecycle_15-24.csv")

# 类型映射
cluster_map = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4"}


# ================= 核心逻辑 =================

def generate_full_sankey():
    print("=== 开始生成全生命周期桑基图 (含新增/流失) ===")

    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return

    df = pd.read_csv(input_path)

    # 严格过滤年份 [2015, 2024]
    all_years = sorted(df['Year'].unique())
    target_years = [y for y in all_years if 2015 <= y <= 2024]
    print(f"处理年份范围: {target_years}")

    all_flows = []

    # 循环处理相邻年份
    for i in range(len(target_years) - 1):
        year_start = target_years[i]
        year_end = target_years[i + 1]

        # 1. 获取两年的数据
        df_s = df[df['Year'] == year_start][['Designer_ID', 'Cluster_ID']]
        df_e = df[df['Year'] == year_end][['Designer_ID', 'Cluster_ID']]

        # 获取 ID 集合
        ids_s = set(df_s['Designer_ID'])
        ids_e = set(df_e['Designer_ID'])

        # ==========================================
        # Part 1: 存续 (Survivors) - 交集
        # ==========================================
        ids_survive = ids_s & ids_e
        if ids_survive:
            # 筛选出存续者
            survivors_s = df_s[df_s['Designer_ID'].isin(ids_survive)]
            survivors_e = df_e[df_e['Designer_ID'].isin(ids_survive)]

            # 合并
            merged = pd.merge(survivors_s, survivors_e, on='Designer_ID')
            flow_survive = merged.groupby(['Cluster_ID_x', 'Cluster_ID_y']).size().reset_index(name='Value')

            # 命名
            flow_survive['Source'] = flow_survive['Cluster_ID_x'].apply(lambda x: f"{year_start} {cluster_map.get(x)}")
            flow_survive['Target'] = flow_survive['Cluster_ID_y'].apply(lambda x: f"{year_end} {cluster_map.get(x)}")
            all_flows.append(flow_survive)

        # ==========================================
        # Part 2: 流失 (Exit) - 差集 (Start - End)
        # ==========================================
        ids_exit = ids_s - ids_e
        if ids_exit:
            exits = df_s[df_s['Designer_ID'].isin(ids_exit)]
            flow_exit = exits.groupby('Cluster_ID').size().reset_index(name='Value')

            # 命名：流向当年的 "流失" 节点
            # 注意：Target 命名为 "2016 [流失]"，这样它会出现在 2016 那一列的底部
            flow_exit['Source'] = flow_exit['Cluster_ID'].apply(lambda x: f"{year_start} {cluster_map.get(x)}")
            flow_exit['Target'] = f"{year_end} [流失]"
            all_flows.append(flow_exit)

        # ==========================================
        # Part 3: 新增 (Entry) - 差集 (End - Start)
        # ==========================================
        ids_new = ids_e - ids_s
        if ids_new:
            entries = df_e[df_e['Designer_ID'].isin(ids_new)]
            flow_new = entries.groupby('Cluster_ID').size().reset_index(name='Value')

            # 命名：来自前一年的 "新增" 节点
            # 注意：Source 命名为 "2015 [新增]"，这样它会出现在 2015 那一列，流向 2016
            flow_new['Source'] = f"{year_start} [新增]"
            flow_new['Target'] = flow_new['Cluster_ID'].apply(lambda x: f"{year_end} {cluster_map.get(x)}")
            all_flows.append(flow_new)

        print(f"{year_start}->{year_end}: 存续 {len(ids_survive)}, 流失 {len(ids_exit)}, 新增 {len(ids_new)}")

    # 合并保存
    if all_flows:
        df_final = pd.concat(all_flows)
        # 调整列顺序
        df_final = df_final[['Source', 'Target', 'Value']]
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[成功] 完整桑基图数据已保存: {output_path}")
    else:
        print("未生成数据。")


if __name__ == "__main__":
    generate_full_sankey()