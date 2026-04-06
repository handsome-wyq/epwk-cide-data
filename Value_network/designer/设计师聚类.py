'''
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns  # 用于更美观的图表
import os

# --- 配置 ---
base_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\画图522"
nodes_file = os.path.join(base_path, "designer_nodes.csv")
output_clustered_file = os.path.join(base_path, "designer_nodes_clustered_cn.csv")
elbow_plot_file = os.path.join(base_path, "kmeans_elbow_plot_cn.png")
silhouette_plot_file = os.path.join(base_path, "kmeans_silhouette_plot_cn.png")  # 轮廓系数图路径
pair_plot_file = os.path.join(base_path, "kmeans_pairplot_clusters_cn.png")  # 配对图路径

# 用于聚类的特征列名
features_for_clustering = ['NicheWidth', 'MeanOverlap', 'ValuePotential']

# 测试的k值范围
k_range = range(2, 11)  # 例如，从2到10个簇

# 用户名和日期，用于图表标题
user_name = "handsome-wyq"
current_date = "2025-05-20"  # 你提供的日期


def run_clustering():
    print(f"正在从以下文件加载数据: {nodes_file}")
    try:
        df_nodes = pd.read_csv(nodes_file, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {nodes_file}")
        return
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    print(f"原始数据 (前5行):\n{df_nodes.head()}")

    # 检查必要的列是否存在
    if not all(feature in df_nodes.columns for feature in features_for_clustering):
        missing_features = [f for f in features_for_clustering if f not in df_nodes.columns]
        print(f"错误: 文件中缺少用于聚类的必要列: {missing_features}")
        return

    # 选择用于聚类的数据
    X = df_nodes[features_for_clustering].copy()

    # 处理NaN (如果之前的脚本未能处理或出现了新的NaN)
    # 你的 create_network_files.py 脚本已经将NaN替换为0。
    # 如果你确定NaN已处理，可以跳过或修改此步骤。
    if X.isnull().sum().any():
        print("在聚类特征中检测到NaN值。将使用列的平均值填充。")
        for col in features_for_clustering:
            X[col].fillna(X[col].mean(), inplace=True)  # 也可以替换为 median() 或 0

    print(f"\n用于聚类的数据 (前5行):\n{X.head()}")

    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n缩放后的数据 (前5行):\n{pd.DataFrame(X_scaled, columns=features_for_clustering).head()}")

    # --- 确定最佳k值 ---
    print("\n正在确定最佳聚类数量 (k)...")

    # 1. "肘部法则"
    inertia = []
    for k_val in k_range:
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')  # n_init='auto' 抑制警告
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), inertia, marker='o', linestyle='--')
    plt.xlabel("聚类数量 (k)")
    plt.ylabel("簇内平方和 (Inertia)")
    plt.title(f"“肘部法则”确定最佳k值 ({user_name} - {current_date})")
    plt.xticks(list(k_range))
    plt.grid(True)
    plt.savefig(elbow_plot_file)
    print(f"“肘部法则”图表已保存到: {elbow_plot_file}")
    # plt.show() # 如果希望立即显示图表，请取消注释

    # 2. 轮廓系数分析 (对于大数据集可能较慢)
    # from sklearn.metrics import silhouette_score
    # silhouette_scores = []
    # print("\n计算轮廓系数...")
    # for k_val in k_range:
    #     if k_val < 2: continue # 轮廓系数分析要求 k >= 2
    #     kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
    #     cluster_labels = kmeans.fit_predict(X_scaled)
    #     if len(set(cluster_labels)) > 1: # silhouette_score 要求多于1个唯一标签
    #         score = silhouette_score(X_scaled, cluster_labels)
    #         silhouette_scores.append(score)
    #         print(f"  k={k_val}, 轮廓系数: {score:.3f}")
    #     else:
    #         silhouette_scores.append(-1) # 无法计算
    #         print(f"  k={k_val}, 轮廓系数: 无法计算 (簇太少)")

    # if silhouette_scores: # 仅当列表不为空时绘图
    #     plt.figure(figsize=(10, 6))
    #     plt.plot([k for k in k_range if k >= 2], silhouette_scores, marker='o', linestyle='--')
    #     plt.xlabel("聚类数量 (k)")
    #     plt.ylabel("轮廓系数")
    #     plt.title(f"轮廓系数分析确定最佳k值 ({user_name} - {current_date})")
    #     plt.xticks([k for k in k_range if k >= 2])
    #     plt.grid(True)
    #     plt.savefig(silhouette_plot_file)
    #     print(f"轮廓系数分析图表已保存到: {silhouette_plot_file}")
    # else:
    #     print("未能计算轮廓系数（可能k_range太小）。")

    print("\n--- 重要提示 ---")
    print(f"请检查生成的图表 ('{os.path.basename(elbow_plot_file)}')")
    print("并确定最佳的聚类数量 'k'。")
    print("然后修改此脚本中的 'k_chosen' 变量的值，并重新运行脚本以执行最终的聚类。")

    # --- 使用选定的k值执行k-means ---
    # 在分析图表后替换此值！
    k_chosen = 5  # 示例：根据“肘部法则”分析选择k值
    print(f"\n使用 k = {k_chosen} 执行k-means聚类...")

    if k_chosen < 2:
        print("错误: 聚类数量 (k_chosen) 必须至少为2。")
        return

    final_kmeans = KMeans(n_clusters=k_chosen, random_state=42, n_init='auto')
    df_nodes['Cluster'] = final_kmeans.fit_predict(X_scaled)

    print(f"\n带有聚类标签的数据 (前5行):\n{df_nodes.head()}")
    print(f"\n各簇的分布情况:\n{df_nodes['Cluster'].value_counts().sort_index()}")

    # 保存结果
    try:
        df_nodes.to_csv(output_clustered_file, index=False, encoding='utf-8-sig')
        print(f"\n聚类结果已保存到: {output_clustered_file}")
    except Exception as e:
        print(f"保存聚类结果文件时出错: {e}")

    # 可选：按簇可视化特征的成对分布图
    try:
        # plt.figure() # 如果之前的图表已显示，确保创建一个新图形
        pair_plot = sns.pairplot(df_nodes, vars=features_for_clustering, hue='Cluster', palette='viridis',
                                 diag_kind='kde')
        pair_plot.fig.suptitle(f'{k_chosen}个簇的特征成对分布图 ({user_name} - {current_date})', y=1.02)  # y调整标题位置
        pair_plot.savefig(pair_plot_file)
        print(f"成对分布图已保存到: {pair_plot_file}")
    except Exception as e:
        print(f"创建成对分布图时出错: {e}")


if __name__ == "__main__":
    # 确保输出目录存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"创建目录: {base_path}")

    run_clustering()
    plt.show()  # 在脚本末尾显示所有生成的Matplotlib图表'''

    # plt.show()
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 配置 ---
base_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\test63\\画图64"  # 请确保此路径有效
nodes_file = os.path.join(base_path, "designer_nodes-2025.csv")
output_clustered_file = os.path.join(base_path, "designer_nodes_clustered_normalized-2025.csv")
elbow_plot_file = os.path.join(base_path, "kmeans_elbow_plot_normalized-2025.png")
pair_plot_file = os.path.join(base_path, "kmeans_pairplot_clusters_normalized-2025.png")

# 用于聚类的特征列名
features_for_clustering = ['NicheWidth', 'MeanOverlap', 'ValuePotential']

# 测试的k值范围
k_range = range(2, 11)

# 用户名和日期，用于图表标题
user_name = ""
current_date = ""  # 使用提供的日期或动态获取


def run_clustering_with_normalization():
    print(f"正在从以下文件加载数据: {nodes_file}")
    try:
        df_nodes_original = pd.read_csv(nodes_file, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {nodes_file}")
        return
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    print(f"原始数据 (前5行):\n{df_nodes_original.head()}")

    if not all(feature in df_nodes_original.columns for feature in features_for_clustering):
        missing_features = [f for f in features_for_clustering if f not in df_nodes_original.columns]
        print(f"错误: 文件中缺少用于聚类的必要列: {missing_features}")
        return

    # 仅选择用于聚类的特征列进行复制和处理
    X_features = df_nodes_original[features_for_clustering].copy()

    if X_features.isnull().sum().any():
        print("在聚类特征中检测到NaN值。将使用列的平均值填充。")
        for col in features_for_clustering:
            if X_features[col].isnull().any():  # 再次检查以防万一
                X_features[col].fillna(X_features[col].mean(), inplace=True)

    print(f"\n用于聚类的原始特征数据 (前5行):\n{X_features.head()}")

    normalizer = MinMaxScaler()
    X_normalized_values = normalizer.fit_transform(X_features)

    # 创建一个包含归一化特征的临时DataFrame，用于验证打印
    df_temp_normalized_print = pd.DataFrame(X_normalized_values, columns=features_for_clustering,
                                            index=X_features.index)
    print(f"\n归一化后的特征数据 (前5行):\n{df_temp_normalized_print.head()}")

    print("\n正在确定最佳聚类数量 (k)...")
    inertia = []
    for k_val in k_range:
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
        kmeans.fit(X_normalized_values)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), inertia, marker='o', linestyle='--')
    plt.xlabel("聚类数量 (k)")
    plt.ylabel("簇内平方和 (Inertia)")
    plt.title(f"“肘部法则”确定最佳k值 (使用归一化数据)\n({user_name} - {current_date})")
    plt.xticks(list(k_range))
    plt.grid(True)
    plt.savefig(elbow_plot_file)
    print(f"“肘部法则”图表已保存到: {elbow_plot_file}")
    # plt.show() # 可以在脚本末尾统一调用

    # 2. 轮廓系数分析 (可选, 对于大数据集可能较慢)
    from sklearn.metrics import silhouette_score
    silhouette_scores = []
    print("\n计算轮廓系数...")
    for k_val in k_range:
        if k_val < 2: continue
        kmeans_sil = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
        cluster_labels_sil = kmeans_sil.fit_predict(X_normalized_values) # 修改：使用归一化后的数据
        if len(set(cluster_labels_sil)) > 1:
            score = silhouette_score(X_normalized_values, cluster_labels_sil) # 修改：使用归一化后的数据
            silhouette_scores.append(score)
            print(f"  k={k_val}, 轮廓系数: {score:.3f}")
        else:
            silhouette_scores.append(-1)
            print(f"  k={k_val}, 轮廓系数: 无法计算 (簇太少)")

    if silhouette_scores:
        plt.figure(figsize=(10, 6))
        plt.plot([k for k in k_range if k >= 2], silhouette_scores, marker='o', linestyle='--')
        plt.xlabel("聚类数量 (k)")
        plt.ylabel("轮廓系数")
        plt.title(f"轮廓系数分析确定最佳k值 (使用归一化数据)\n({user_name} - {current_date})") # 标题更新
        plt.xticks([k for k in k_range if k >= 2])
        plt.grid(True)
        # plt.savefig(silhouette_plot_file) # 如果使用，取消注释并确保文件名正确
        # print(f"轮廓系数分析图表已保存到: {silhouette_plot_file}")
    else:
        print("未能计算轮廓系数（可能k_range太小或未启用此部分）。")

    print("\n--- 重要提示 ---")
    print(f"请检查生成的图表 ('{os.path.basename(elbow_plot_file)}')")
    print("并确定最佳的聚类数量 'k'。")
    print("然后修改此脚本中的 'k_chosen' 变量的值，并重新运行脚本以执行最终的聚类。")

    k_chosen = 5  # 示例：请在分析图表后修改此值！
    print(f"\n使用 k = {k_chosen} 执行k-means聚类 (基于归一化数据)...")

    if k_chosen < 2:
        print("错误: 聚类数量 (k_chosen) 必须至少为2。")
        return

    final_kmeans = KMeans(n_clusters=k_chosen, random_state=42, n_init='auto')
    cluster_labels = final_kmeans.fit_predict(X_normalized_values)

    # --- 创建最终输出的DataFrame ---
    # 1. 开始于原始数据中不参与聚类的列
    df_output = df_nodes_original.drop(columns=features_for_clustering, errors='ignore').copy()

    # 2. 添加归一化后的特征列
    for i, feature_col in enumerate(features_for_clustering):
        df_output[feature_col] = X_normalized_values[:, i]

    # 3. 添加聚类标签
    df_output['Cluster'] = cluster_labels

    # (可选) 调整列顺序，使标识列在前，然后是归一化特征，最后是聚类标签
    cols_order = [col for col in df_nodes_original.columns if col not in features_for_clustering]
    cols_order.extend(features_for_clustering)
    cols_order.append('Cluster')

    # 确保所有df_output中的列都在cols_order中，以防万一
    final_ordered_cols = [col for col in cols_order if col in df_output.columns]
    df_output = df_output[final_ordered_cols]

    print(f"\n包含归一化特征和聚类结果的数据 (前5行):\n{df_output.head()}")
    print(f"\n各簇的分布情况:\n{df_output['Cluster'].value_counts().sort_index()}")

    try:
        df_output.to_csv(output_clustered_file, index=False, encoding='utf-8-sig')
        print(f"\n包含归一化特征和聚类结果的数据已保存到: {output_clustered_file}")
    except Exception as e:
        print(f"保存合并数据文件时出错: {e}")

    # --- 可选：按簇可视化特征的成对分布图 ---
    # 使用原始特征值进行绘图，但根据归一化数据得到的簇标签着色
    df_for_plot = df_nodes_original.copy()  # 使用原始特征
    df_for_plot['Cluster'] = cluster_labels  # 添加由归一化数据产生的聚类标签

    try:
        pair_plot_viz = sns.pairplot(df_for_plot, vars=features_for_clustering, hue='Cluster', palette='viridis',
                                     diag_kind='kde')
        pair_plot_viz.fig.suptitle(
            f'{k_chosen}个簇的特征成对分布图 (原始特征值, 基于归一化数据聚类)\n({user_name} - {current_date})', y=1.02)
        pair_plot_viz.savefig(pair_plot_file)
        print(f"成对分布图已保存到: {pair_plot_file}")
    except Exception as e:
        print(f"创建成对分布图时出错: {e}")


if __name__ == "__main__":
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"创建目录: {base_path}")

    run_clustering_with_normalization()
    plt.show()  # 显示所有plt生成的图表

