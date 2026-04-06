import os
import pandas as pd
#2025.12.26
# ================= 配置 =================
root_dir = r"D:/Documents/Desktop/Crowd_intelligence_wyq/Value_network/1125计算结果-对齐版"
input_path = os.path.join(root_dir, "价值网络聚类分析结果", "Designers_Clustered_Master.csv")
output_path = os.path.join(root_dir, "价值网络聚类分析结果", "Sankey_Data_Survivors_Only_15-24.csv")

# 类型映射（按你的设定）
cluster_map = {1: "Type 2", 2: "Type 3", 3: "Type 4", 4: "Type 1"}

# ================= 核心逻辑 =================
def generate_survivor_sankey():
    print("=== 生成桑基图数据：仅存续者（相邻两年均存在） ===")

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

        # 取两年数据（只保留绘图必要列）
        df_s = df[df['Year'] == year_start][['Designer_ID', 'Cluster_ID']].dropna()
        df_e = df[df['Year'] == year_end][['Designer_ID', 'Cluster_ID']].dropna()

        ids_s = set(df_s['Designer_ID'])
        ids_e = set(df_e['Designer_ID'])

        # 仅保留存续者（交集）
        ids_survive = ids_s & ids_e
        print(f"{year_start}->{year_end}: 存续 {len(ids_survive)}")

        if not ids_survive:
            continue

        survivors_s = df_s[df_s['Designer_ID'].isin(ids_survive)]
        survivors_e = df_e[df_e['Designer_ID'].isin(ids_survive)]

        # 合并得到每个设计师在两年的类型
        merged = pd.merge(
            survivors_s, survivors_e,
            on='Designer_ID',
            suffixes=('_start', '_end'),
            how='inner'
        )

        # 统计流向：Cluster_ID_start -> Cluster_ID_end
        flow = (
            merged
            .groupby(['Cluster_ID_start', 'Cluster_ID_end'])
            .size()
            .reset_index(name='Value')
        )

        # 命名：如 “2018 Type 1” -> “2019 Type 3”
        flow['Source'] = flow['Cluster_ID_start'].apply(lambda x: f"{year_start} {cluster_map.get(int(x), x)}")
        flow['Target'] = flow['Cluster_ID_end'].apply(lambda x: f"{year_end} {cluster_map.get(int(x), x)}")

        all_flows.append(flow[['Source', 'Target', 'Value']])

    if all_flows:
        df_final = pd.concat(all_flows, ignore_index=True)
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[成功] 桑基图数据（仅存续者）已保存: {output_path}")
    else:
        print("未生成任何流向数据（可能因为相邻年份交集为空或年份过滤范围不匹配）。")


if __name__ == "__main__":
    generate_survivor_sankey()
