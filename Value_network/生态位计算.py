import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

# ===================== 1. 路径配置 =====================

# 基础根目录 (请根据实际情况修改)
root_dir = r"D:/Documents/Desktop/Crowd_intelligence_wyq/Value_network/1125计算结果-对齐版"

# 输入：矩阵数据目录
matrix_dir = os.path.join(root_dir, "矩阵与知识向量_Sparse_v2")
global_idx_dir = os.path.join(matrix_dir, "global_indices")

# 输出：生态位数值结果
output_dir = os.path.join(root_dir, "生态位数值计算结果")
os.makedirs(output_dir, exist_ok=True)

# 阈值设置：计算重叠度时的噪音过滤 (只统计相似度 > 0.2 的竞争关系)
# 如果想统计所有人，设为 0.0 (但计算速度会变慢且包含无效噪音)
OVERLAP_THRESHOLD = 0.2

years = list(range(2015, 2025))

# ===================== 2. 加载全域索引 =====================

print("正在加载全域索引...")
try:
    with open(os.path.join(global_idx_dir, "all_designers.pkl"), 'rb') as f:
        all_designers = pickle.load(f)
    print(f"索引加载成功。")
except Exception as e:
    print(f"索引加载失败: {e}")
    exit()


# ===================== 3. 核心计算逻辑 =====================

def calculate_niche_values(year):
    print(f"\n=== 正在计算 {year} 年生态位数值 ===")

    # --- A. 加载矩阵 ---
    mat_path = os.path.join(matrix_dir, str(year), "Global_Aligned", f"DK_counts_global_{year}.npz")
    if not os.path.exists(mat_path):
        print(f"  [跳过] 文件不存在")
        return None

    # 加载稀疏矩阵
    mat = sp.load_npz(mat_path)

    # --- B. 筛选活跃设计师 ---
    row_sums = np.array(mat.sum(axis=1)).flatten()
    active_indices = np.where(row_sums > 0)[0]

    if len(active_indices) < 2:
        return None

    sub_mat = mat[active_indices]
    active_names = [all_designers[i] for i in active_indices]
    n_users = len(active_names)

    print(f"  活跃设计师: {n_users} 人")

    # ================= 1. 计算生态位宽度 (Niche Width) =================
    # 指标：Shannon Entropy
    print("  1. 计算生态位宽度...")
    widths = []
    for i in range(sub_mat.shape[0]):
        row_data = sub_mat[i].data
        if row_data.size == 0:
            widths.append(0)
        else:
            pi = row_data / row_data.sum()
            widths.append(entropy(pi))  # 默认 base=e

    # 初始化结果表
    df_values = pd.DataFrame({
        'Designer_ID': active_names,
        'Year': year,
        'Niche_Width': widths,  # 生态位宽度值
        'Resource_Volume': row_sums[active_indices]  # 资源总量
    })

    # ================= 2. 计算生态位重叠度 (Niche Overlap) =================
    # 指标：基于 Pianka (Cosine) 的聚合值
    print("  2. 计算生态位重叠度矩阵...")

    # 计算余弦相似度矩阵 (Dense)
    # 注意：如果人数超过 2万，这步可能会爆内存，建议分块计算
    sim_matrix = cosine_similarity(sub_mat, dense_output=True)

    # 将对角线(自己和自己)设为0，以免影响求和
    np.fill_diagonal(sim_matrix, 0)

    # 应用阈值过滤 (小于阈值的视为无有效竞争，置为0)
    sim_matrix[sim_matrix < OVERLAP_THRESHOLD] = 0

    print("  3. 聚合个体重叠度指标...")

    # --- 指标 A: 平均重叠度 (Mean Overlap) ---
    # 反映同质化程度。公式：sum(overlaps) / (N-1)
    # 这里我们只对“有重叠的对象”求平均，更能反映有效竞争圈内的相似度
    # 或者对“全员”求平均。生态学通常用 sum / (N-1)

    total_overlaps = sim_matrix.sum(axis=1)  # 按行求和

    # 竞争对手数量 (Degree): 有多少人的重叠度 > 阈值
    competitor_counts = (sim_matrix > 0).sum(axis=1)

    # 计算平均值 (避免除以0)
    # 方式1: 全局平均 (反映在整个生态中的独特性) -> total / (n_users - 1)
    mean_overlaps_global = total_overlaps / (n_users - 1)

    # 方式2: 局部平均 (反映在竞争圈内的相似度) -> total / competitor_counts
    # 这里我们输出全局平均，因为更适合做全平台对比

    # 将计算结果添加到表中
    df_values['Total_Overlap'] = total_overlaps  # 总竞争压力 (值越大，环境越恶劣)
    df_values['Mean_Overlap'] = mean_overlaps_global  # 平均同质化程度 (值越大，越大众脸)
    df_values['Competitor_Count'] = competitor_counts  # 竞争对手数量

    return df_values


# ===================== 4. 主执行流程 =====================

def main():
    all_data = []

    for year in years:
        df_res = calculate_niche_values(year)

        if df_res is not None:
            # 保存单年结果
            save_path = os.path.join(output_dir, f"Niche_Values_{year}.csv")
            df_res.to_csv(save_path, index=False, encoding='utf-8-sig')

            all_data.append(df_res)
            print(f"  [完成] {year} 年计算，已保存。")

    # 保存总表
    if all_data:
        df_all = pd.concat(all_data)
        total_path = os.path.join(output_dir, "Niche_Values_All_Years.csv")
        df_all.to_csv(total_path, index=False, encoding='utf-8-sig')

        print(f"\n全部完成！汇总文件已保存至: {total_path}")
        print(
            "包含列: [Designer_ID, Year, Niche_Width, Resource_Volume, Total_Overlap, Mean_Overlap, Competitor_Count]")


if __name__ == "__main__":
    main()