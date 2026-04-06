import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ===================== 1. 配置 =====================

root_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识网络分析\1125计算结果-对齐版"
input_path = os.path.join(root_dir, "价值网络构建数据", "Value_Network_Nodes_Master.csv")
output_dir = os.path.join(root_dir, "聚类分析结果")
os.makedirs(output_dir, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 2. 核心逻辑 =====================

def plot_elbow_and_silhouette():
    print("=== 开始寻找最佳聚类数 K (手肘法) ===")

    # 1. 读取数据
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return
    df = pd.read_csv(input_path)

    # 2. 准备特征
    features = ['Niche_Width', 'Mean_Overlap', 'Log_Value']
    X = df[features].values

    # 3. 标准化 (至关重要)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 循环计算 K=1 到 K=10
    k_range = range(1, 11)
    inertias = []  # 手肘法指标：簇内误方差 (越小越好)
    silhouettes = []  # 轮廓系数指标：分离度 (越接近1越好)

    print("正在计算不同 K 值的效果...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        # 记录 Inertia
        inertias.append(kmeans.inertia_)

        # 记录 Silhouette (至少要有2类才能算)
        if k > 1:
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouettes.append(score)
        else:
            silhouettes.append(None)

    # 5. 保存评估数据到 CSV (方便去 Origin 画图)
    df_eval = pd.DataFrame({
        'K': list(k_range),
        'Inertia': inertias,
        'Silhouette': silhouettes
    })
    save_path = os.path.join(output_dir, "K_Evaluation_Metrics.csv")
    df_eval.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"评估数据已保存: {save_path}")

    # ===================== 6. 绘图 (双轴图) =====================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制手肘线 (Inertia) - 左轴
    color = 'tab:blue'
    ax1.set_xlabel('聚类数量 (k)', fontsize=12)
    ax1.set_ylabel('Inertia (簇内误差平方和)', color=color, fontsize=12)
    ax1.plot(k_range, inertias, 'o-', color=color, linewidth=2, markersize=8, label='手肘法 (Inertia)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 绘制轮廓系数 (Silhouette) - 右轴
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score (轮廓系数)', color=color, fontsize=12)
    # 注意：轮廓系数从 k=2 开始有效
    ax2.plot(k_range[1:], silhouettes[1:], 's--', color=color, linewidth=2, markersize=8, label='轮廓系数')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('聚类数量 K 的评估：手肘法与轮廓系数', fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_elbow_and_silhouette()