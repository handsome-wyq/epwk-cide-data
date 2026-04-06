import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ===================== 1. 配置 =====================

# 基础目录
root_dir = r"D:/Documents/Desktop/Crowd_intelligence_wyq/Value_network/1125计算结果-对齐版"

# 输入文件：之前生成的价值网络节点总表
input_path = os.path.join(root_dir, "价值网络构建数据", "Value_Network_Nodes_Master_All_Years.csv")

# 输出目录
output_dir = os.path.join(root_dir, "价值网络聚类分析结果")
os.makedirs(output_dir, exist_ok=True)

# 分年数据输出子目录
output_yearly_dir = os.path.join(output_dir, "Yearly_Clustered_Files")
os.makedirs(output_yearly_dir, exist_ok=True)

# 聚类数量 (建议 4 或 5)
N_CLUSTERS = 4


# ===================== 2. 核心逻辑 =====================

def perform_clustering():
    print("=== 开始 K-Means 聚类分析 (全量聚类 -> 按年输出) ===")

    # 1. 读取数据
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"原始数据加载成功，共 {len(df)} 条记录")

    # 2. 准备聚类特征 (Features)
    features = ['Niche_Width', 'Mean_Overlap', 'Log_Value']

    # 检查列是否存在
    for f in features:
        if f not in df.columns:
            print(f"错误：数据表中缺少列 {f}")
            return

    # 提取数据矩阵
    X = df[features].values

    # 3. 数据标准化 (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 执行 K-Means (全量数据统一聚类，保证标准一致)
    print(f"正在执行聚类 (k={N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 5. 将结果回写到 DataFrame
    df['Cluster_ID'] = labels + 1

    # 6. 生成“聚类画像解读表” (Cluster Profiling)
    profile = df.groupby('Cluster_ID')[features].mean()
    profile['Count'] = df.groupby('Cluster_ID').size()
    profile['Ratio'] = profile['Count'] / len(df)

    # 全局均值辅助判断
    global_mean = df[features].mean()

    print("\n=== 聚类结果解读 (Centroids) ===")
    print(profile.round(4))

    # 7. 自动打标建议
    def naming_helper(row):
        w = "宽" if row['Niche_Width'] > global_mean['Niche_Width'] else "窄"
        o = "高重叠" if row['Mean_Overlap'] > global_mean['Mean_Overlap'] else "低重叠"
        v = "高价" if row['Log_Value'] > global_mean['Log_Value'] else "低价"
        return f"{w} | {o} | {v}"

    profile['Auto_Tag'] = profile.apply(naming_helper, axis=1)

    # ===================== 8. 保存文件 (核心修改部分) =====================

    # A. 保存总表
    save_data_path = os.path.join(output_dir, "Designers_Clustered_Master.csv")
    df.to_csv(save_data_path, index=False, encoding='utf-8-sig')

    # B. 保存画像分析表
    save_profile_path = os.path.join(output_dir, "Cluster_Profile_Report.csv")
    profile.to_csv(save_profile_path, encoding='utf-8-sig')

    print(f"\n[成功] 总表已保存: {save_data_path}")
    print(f"[成功] 解读表已保存: {save_profile_path}")

    # C. 按年份拆分并保存
    print(f"\n正在按年份拆分数据至: {output_yearly_dir} ...")
    years = sorted(df['Year'].unique())

    for year in years:
        df_year = df[df['Year'] == year].copy()
        if df_year.empty: continue

        # 文件名示例: Value_Network_Nodes_2020_Clustered.csv
        filename = f"Value_Network_Nodes_{year}_Clustered.csv"
        path = os.path.join(output_yearly_dir, filename)

        df_year.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  - {year} 年数据已保存 (含 Cluster_ID)")

    print("\n所有操作完成！")


if __name__ == "__main__":
    perform_clustering()