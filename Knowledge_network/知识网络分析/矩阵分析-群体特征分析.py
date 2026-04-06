'''主流定义 (The Mainstream)：
        计算群体平均向量 (Centroid)。
        提取权重最高的 Top 10 知识点，定义“当年的标准技能包”。
    离散度分析 (Dispersion)：
        计算每个设计师到平均向量的余弦距离 (Deviation)。
        看直方图：大家是紧紧抱团（同质化），还是散落在各地（多样化）？
    人才梯度 (The Gradient)：
        结合 偏差 (X轴) + 多元度 (Y轴) + 活跃度 (气泡大小)。
        识别四类人群：主流通才、主流工匠、边缘怪才、利基专家。
    能力缺口 (The Gap)：
        对比 “普通设计师（均值）” 与 “头部设计师（Top 10% 活跃度）” 的知识向量。
        头部掌握但普通人没掌握的，就是当年的能力缺口/进阶方向。'''
import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_distances

# ===================== 1. 路径配置 =====================

# 基础根目录 (请确认你的实际路径)
root_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识网络分析\1125计算结果-对齐版"

# 输入1：矩阵数据目录 (读取 Global_Aligned 版本)
matrix_dir = os.path.join(root_dir, "矩阵与知识向量_Sparse_v2")

# 输入2：全域索引目录 (用于解析 ID)
global_idx_dir = os.path.join(matrix_dir, "global_indices")

# 输入3：上一步的个体特征分析结果 (读取 Individual_Features_YYYY.csv)
individual_dir = os.path.join(root_dir, "个体特征分析结果")  # 或者是 "Origin_Data_Individual_Analysis"

# 输出：本步骤的输出目录
output_dir = os.path.join(root_dir, "群体特征分析结果")
os.makedirs(output_dir, exist_ok=True)

years = list(range(2014, 2026))

# ===================== 2. 加载全域索引 =====================

print("正在加载全域索引 (Designers & Knowledge)...")
try:
    with open(os.path.join(global_idx_dir, "all_designers.pkl"), 'rb') as f:
        all_designers = pickle.load(f)
    with open(os.path.join(global_idx_dir, "all_knowledge.pkl"), 'rb') as f:
        all_knowledge = pickle.load(f)
    print(f"索引加载成功。设计师总数: {len(all_designers)}, 知识点总数: {len(all_knowledge)}")
except Exception as e:
    print(f"索引加载失败，请检查路径: {e}")
    exit()


# ===================== 3. 辅助函数：清洗知识名称 =====================

def clean_k_name(name):
    """去掉 'knowledge:' 前缀"""
    if isinstance(name, str):
        return name.replace("knowledge:", "")
    return name


# ===================== 4. 核心计算逻辑 =====================

def analyze_group_features(year):
    print(f"\n=== 正在分析 {year} 年群体特征 ===")

    # --- A. 加载 Global Aligned 矩阵 ---
    mat_path = os.path.join(matrix_dir, str(year), "Global_Aligned", f"DK_counts_global_{year}.npz")
    if not os.path.exists(mat_path):
        print(f"  [跳过] 矩阵文件不存在: {mat_path}")
        return None, None, None

    # 加载稀疏矩阵
    mat = sp.load_npz(mat_path)

    # --- B. 筛选活跃设计师 ---
    row_sums = np.array(mat.sum(axis=1)).flatten()
    active_indices = np.where(row_sums > 0)[0]

    if len(active_indices) < 5:
        print("  [跳过] 活跃设计师过少")
        return None, None, None

    # 提取活跃子矩阵
    sub_mat = mat[active_indices]
    active_designer_names = [all_designers[i] for i in active_indices]

    # --- C. 计算主流与偏差 (Mainstream & Deviation) ---

    # 1. 计算平均向量 (Centroid)
    mean_vec = np.array(sub_mat.mean(axis=0)).flatten()

    # 2. 提取当年“市场热点” (Top 10 Knowledge)
    top_k_idx = mean_vec.argsort()[::-1][:10]
    market_hotspots = [
        {
            'Year': year,
            'Rank': i + 1,
            'Knowledge': clean_k_name(all_knowledge[idx]),  # <--- 这里清洗名称
            'Weight': mean_vec[idx]
        }
        for i, idx in enumerate(top_k_idx)
    ]

    # 3. 计算个体偏差 (Deviation)
    dists = cosine_distances(sub_mat, mean_vec.reshape(1, -1)).flatten()

    # --- D. 数据整合 ---

    # 加载全域设计师名单以匹配 ID
    # (为了性能，这里假设 active_indices 对应的是 all_designers 的原始顺序)

    df_group_temp = pd.DataFrame({
        'Designer_ID': active_designer_names,
        'Deviation': dists
    })

    # 读取上一步生成的个体特征表
    ind_csv_path = os.path.join(individual_dir, f"Individual_Features_{year}.csv")
    if os.path.exists(ind_csv_path):
        df_ind = pd.read_csv(ind_csv_path)
        df_gradient = pd.merge(df_group_temp, df_ind, on='Designer_ID', how='inner')
    else:
        print(f"  [警告] 找不到对应的个体特征表: {ind_csv_path}，仅输出偏差数据")
        df_gradient = df_group_temp

    # --- E. 能力缺口分析 (Gap Analysis) ---
    if 'Volume' in df_gradient.columns:
        threshold = df_gradient['Volume'].quantile(0.90)

        # 使用 all_designers 的映射加速查找
        name_to_idx = {name: i for i, name in enumerate(all_designers)}

        top_designers_list = df_gradient[df_gradient['Volume'] >= threshold]['Designer_ID'].tolist()

        if len(top_designers_list) > 1:
            top_global_indices = [name_to_idx[name] for name in top_designers_list if name in name_to_idx]

            if top_global_indices:
                top_mat = mat[top_global_indices]
                top_mean = np.array(top_mat.mean(axis=0)).flatten()

                # 计算缺口向量
                gap_vec = top_mean - mean_vec

                # 提取 Top 15 缺口
                gap_idx_list = gap_vec.argsort()[::-1][:15]
                capability_gaps = []
                for idx in gap_idx_list:
                    if gap_vec[idx] > 0:
                        capability_gaps.append({
                            'Year': year,
                            'Knowledge': clean_k_name(all_knowledge[idx]),  # <--- 这里清洗名称
                            'Gap_Score': gap_vec[idx],
                            'Top_Level': top_mean[idx],
                            'Avg_Level': mean_vec[idx]
                        })
            else:
                capability_gaps = []
        else:
            capability_gaps = []
    else:
        capability_gaps = []

    return df_gradient, market_hotspots, capability_gaps


# ===================== 5. 主执行流程 =====================

def main():
    all_years_gradient = []
    all_years_hotspots = []
    all_years_gaps = []

    for year in years:
        res = analyze_group_features(year)
        if res:
            df_grad, hotspots, gaps = res

            # 1. 保存当年的人才梯度表
            save_path = os.path.join(output_dir, f"Group_Gradient_{year}.csv")
            df_grad.to_csv(save_path, index=False, encoding='utf-8-sig')

            # 2. 收集汇总
            if 'Year' not in df_grad.columns:
                df_grad['Year'] = year
            all_years_gradient.append(df_grad)

            if hotspots: all_years_hotspots.extend(hotspots)
            if gaps: all_years_gaps.extend(gaps)

            top_hotspot = hotspots[0]['Knowledge'] if hotspots else 'None'
            print(f"  [完成] {year} 年分析。Top 1 热点: {top_hotspot}")

    # --- 保存汇总结果 ---

    # 1. 人才梯度汇总
    if all_years_gradient:
        pd.concat(all_years_gradient).to_csv(
            os.path.join(output_dir, "Group_Gradient_All_Years.csv"),
            index=False, encoding='utf-8-sig'
        )

    # 2. 市场热点汇总 (Knowledge 列已清洗)
    if all_years_hotspots:
        pd.DataFrame(all_years_hotspots).to_csv(
            os.path.join(output_dir, "Market_Hotspots_All_Years.csv"),
            index=False, encoding='utf-8-sig'
        )

    # 3. 能力缺口汇总 (Knowledge 列已清洗)
    if all_years_gaps:
        pd.DataFrame(all_years_gaps).to_csv(
            os.path.join(output_dir, "Capability_Gaps_All_Years.csv"),
            index=False, encoding='utf-8-sig'
        )

    print(f"\n全部计算完成！前缀已移除。结果保存在: {output_dir}")


if __name__ == "__main__":
    main()