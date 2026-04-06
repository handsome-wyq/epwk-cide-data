import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import json

# ===================== 路径配置 =====================

base_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network"
ability_vec_dir = os.path.join(base_dir, r"知识网络分析\能力向量与金额")
global_net_dir = os.path.join(base_dir, r"知识系统\总网络-节点和边")

# 输出目录
output_dir = os.path.join(base_dir, r"知识网络分析\矩阵与知识向量_Sparse_v2")
os.makedirs(output_dir, exist_ok=True)

# 索引存储目录 (全域索引)
global_index_dir = os.path.join(output_dir, "global_indices")
os.makedirs(global_index_dir, exist_ok=True)


# ===================== 工具函数 =====================

def parse_year_from_date(date_str):
    if pd.isna(date_str):
        return None
    dt = pd.to_datetime(str(date_str), errors='coerce')
    if pd.isna(dt):
        return None
    return int(dt.year)


def save_list_pkl(data_list, path):
    """保存列表到 pickle"""
    with open(path, 'wb') as f:
        pickle.dump(data_list, f)


def load_list_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_sparse_matrix(matrix, folder, filename):
    """保存稀疏矩阵为 .npz"""
    os.makedirs(folder, exist_ok=True)
    sp.save_npz(os.path.join(folder, filename), matrix)


def save_local_indices(folder, row_list, col_list, prefix):
    """保存年度内部的行列索引，方便单独读取分析"""
    os.makedirs(folder, exist_ok=True)
    # 保存为 JSON 方便人眼查看，也可以用 pickle
    indices = {
        'rows': list(row_list),
        'cols': list(col_list)
    }
    with open(os.path.join(folder, f"{prefix}_indices.json"), 'w', encoding='utf-8') as f:
        json.dump(indices, f, ensure_ascii=False)


# ===================== 1. 数据加载与全域维度构建 =====================

def load_data_and_build_universe():
    print("=== 1. 正在加载数据并构建全域维度索引 ===")

    # --- A. 加载设计师-能力数据 ---
    csv_da = os.path.join(ability_vec_dir, "designer_ability_vectors_all_years.csv")
    df_da = pd.read_csv(csv_da, encoding='utf-8-sig')
    df_da['year'] = df_da['year'].astype(int)
    df_da = df_da.dropna(subset=['designer', 'ability', 'year'])

    # --- B. 加载总网络边数据 ---
    nodes_orders = pd.read_csv(os.path.join(global_net_dir, "nodes_orders_all.csv"), encoding='utf-8-sig')
    edges_ao = pd.read_csv(os.path.join(global_net_dir, "edges_ability_order_all.csv"), encoding='utf-8-sig')
    edges_ok = pd.read_csv(os.path.join(global_net_dir, "edges_order_knowledge_all.csv"), encoding='utf-8-sig')

    # --- C. 处理订单年份 ---
    nodes_orders['year'] = nodes_orders['Publication_Time'].apply(parse_year_from_date)
    nodes_orders = nodes_orders.dropna(subset=['year'])
    nodes_orders['year'] = nodes_orders['year'].astype(int)

    # 关联 Ability -> Order -> Knowledge
    eao = edges_ao.rename(columns={'Source': 'ability', 'Target': 'order_id'})
    eao = eao.merge(nodes_orders[['ID', 'year']], left_on='order_id', right_on='ID', how='left').dropna(subset=['year'])
    eao['year'] = eao['year'].astype(int)
    eok = edges_ok.rename(columns={'Source': 'order_id', 'Target': 'knowledge'})
    df_ak = eao.merge(eok[['order_id', 'knowledge']], on='order_id', how='inner')
    df_ak = df_ak.dropna(subset=['ability', 'knowledge'])

    # --- D. 构建全域唯一列表 (Universe) ---
    all_designers = sorted(df_da['designer'].unique())
    all_abilities = sorted(set(df_da['ability'].unique()) | set(df_ak['ability'].unique()))
    all_knowledge = sorted(df_ak['knowledge'].unique())

    # --- E. 建立全域映射字典 ---
    des_map_global = {name: i for i, name in enumerate(all_designers)}
    ab_map_global = {name: i for i, name in enumerate(all_abilities)}
    know_map_global = {name: i for i, name in enumerate(all_knowledge)}

    # 保存全域索引
    save_list_pkl(all_designers, os.path.join(global_index_dir, "all_designers.pkl"))
    save_list_pkl(all_abilities, os.path.join(global_index_dir, "all_abilities.pkl"))
    save_list_pkl(all_knowledge, os.path.join(global_index_dir, "all_knowledge.pkl"))

    print(f"[维度统计] Designer: {len(all_designers)}, Ability: {len(all_abilities)}, Knowledge: {len(all_knowledge)}")

    return df_da, df_ak, des_map_global, ab_map_global, know_map_global, len(all_designers), len(all_abilities), len(
        all_knowledge)


# ===================== 2. 构建 D-A 矩阵 (双版本) =====================

def build_DA_matrices(df_da, des_map_global, ab_map_global, n_des_global, n_ab_global):
    print("=== 2. 开始构建 D-A 稀疏矩阵 (Local + Global) ===")

    years = sorted(df_da['year'].unique())
    da_mats_global = {}  # 仅返回 global 供后续 DK 计算使用（如需做全域DK）

    for year in years:
        sub = df_da[df_da['year'] == year].copy()

        # --- A. 构建【Local Compact】版本 (仅包含当年存在的) ---
        local_designers = sorted(sub['designer'].unique())
        local_abilities = sorted(sub['ability'].unique())

        # 建立当年的临时映射
        des_map_local = {name: i for i, name in enumerate(local_designers)}
        ab_map_local = {name: i for i, name in enumerate(local_abilities)}

        row_local = sub['designer'].map(des_map_local).values
        col_local = sub['ability'].map(ab_map_local).values
        data = sub['order_count'].values

        mat_local = sp.coo_matrix((data, (row_local, col_local)),
                                  shape=(len(local_designers), len(local_abilities))).tocsr()

        # 保存 Local 版
        local_dir = os.path.join(output_dir, str(year), "Local_Compact")
        save_sparse_matrix(mat_local, local_dir, f"DA_orders_local_{year}.npz")
        save_local_indices(local_dir, local_designers, local_abilities, "DA")

        # --- B. 构建【Global Aligned】版本 (全域维度) ---
        row_global = sub['designer'].map(des_map_global).values
        col_global = sub['ability'].map(ab_map_global).values
        # 注意：如果有在全域表里找不到的脏数据(极少)，需要 dropna，这里假设 map 完可能有 NaN
        mask = ~np.isnan(row_global) & ~np.isnan(col_global)

        mat_global = sp.coo_matrix((data[mask], (row_global[mask], col_global[mask])),
                                   shape=(n_des_global, n_ab_global)).tocsr()

        # 保存 Global 版
        global_dir = os.path.join(output_dir, str(year), "Global_Aligned")
        save_sparse_matrix(mat_global, global_dir, f"DA_orders_global_{year}.npz")

        da_mats_global[year] = mat_global

    return da_mats_global


# ===================== 3. 构建 A-K 矩阵 (双版本) =====================

def build_AK_matrices(df_ak, ab_map_global, know_map_global, n_ab_global, n_know_global):
    print("=== 3. 开始构建 A-K 稀疏矩阵 (Local + Global) ===")

    grouped = df_ak.groupby(['year', 'ability', 'knowledge']).size().reset_index(name='count')
    years = sorted(grouped['year'].unique())
    ak_mats_global = {}

    for year in years:
        sub = grouped[grouped['year'] == year].copy()

        # --- A. 构建【Local Compact】版本 ---
        local_abilities = sorted(sub['ability'].unique())
        local_knowledge = sorted(sub['knowledge'].unique())

        ab_map_local = {name: i for i, name in enumerate(local_abilities)}
        know_map_local = {name: i for i, name in enumerate(local_knowledge)}

        row_local = sub['ability'].map(ab_map_local).values
        col_local = sub['knowledge'].map(know_map_local).values
        data = sub['count'].values

        mat_counts_local = sp.coo_matrix((data, (row_local, col_local)),
                                         shape=(len(local_abilities), len(local_knowledge))).tocsr()

        # 计算 Local 的行归一化
        row_sums = np.array(mat_counts_local.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        mat_norm_local = sp.diags(1.0 / row_sums).dot(mat_counts_local)

        # 保存 Local 版
        local_dir = os.path.join(output_dir, str(year), "Local_Compact")
        save_sparse_matrix(mat_norm_local, local_dir, f"AK_row_norm_local_{year}.npz")
        save_local_indices(local_dir, local_abilities, local_knowledge, "AK")

        # --- B. 构建【Global Aligned】版本 ---
        row_global = sub['ability'].map(ab_map_global).values
        col_global = sub['knowledge'].map(know_map_global).values
        mask = ~np.isnan(row_global) & ~np.isnan(col_global)

        mat_counts_global = sp.coo_matrix((data[mask], (row_global[mask], col_global[mask])),
                                          shape=(n_ab_global, n_know_global)).tocsr()

        # 计算 Global 的行归一化
        row_sums_g = np.array(mat_counts_global.sum(axis=1)).flatten()
        row_sums_g[row_sums_g == 0] = 1
        mat_norm_global = sp.diags(1.0 / row_sums_g).dot(mat_counts_global)

        global_dir = os.path.join(output_dir, str(year), "Global_Aligned")
        save_sparse_matrix(mat_norm_global, global_dir, f"AK_row_norm_global_{year}.npz")

        ak_mats_global[year] = mat_norm_global

    return ak_mats_global


# ===================== 4. 构建 D-K 矩阵 (基于 Global 计算，Local 可选) =====================

def build_DK_matrices(da_mats_global, ak_mats_global):
    """
    这里演示基于 Global 矩阵计算 DK，因为这样代码最简单（直接点乘）。
    计算完后的结果也是 Global Aligned 的。
    如果需要 Local Compact 的 DK，可以对结果去掉全零列，或者单独用 Local DA * Local AK 计算。
    """
    print("=== 4. 开始计算 D-K 稀疏矩阵 (Global Aligned) ===")

    common_years = sorted(set(da_mats_global.keys()) & set(ak_mats_global.keys()))

    for year in common_years:
        M_da = da_mats_global[year]
        M_ak = ak_mats_global[year]

        # 计算 D-K
        M_dk = M_da.dot(M_ak)

        # 保存 Global 版
        global_dir = os.path.join(output_dir, str(year), "Global_Aligned")
        save_sparse_matrix(M_dk, global_dir, f"DK_counts_global_{year}.npz")

        # --- 额外：如果需要 Local 版的 DK (仅包含当年的有效数据) ---
        # 由于我们已经有了 Global 结果，提取非零行列即可，不用重算
        # 但为了方便你使用，这里通过“裁剪”Global矩阵来生成 Local 版

        # 1. 找出非零的行 (有数据的设计师)
        row_mask = np.diff(M_dk.indptr) != 0
        # 2. 找出非零的列 (有数据的知识) - CSR格式取列稍微麻烦点，转CSC
        col_mask = np.diff(M_dk.tocsc().indptr) != 0

        M_dk_local = M_dk[row_mask][:, col_mask]

        # 获取对应的真实 ID
        all_designers = load_list_pkl(os.path.join(global_index_dir, "all_designers.pkl"))
        all_knowledge = load_list_pkl(os.path.join(global_index_dir, "all_knowledge.pkl"))

        local_designers = [all_designers[i] for i in np.where(row_mask)[0]]
        local_knowledge = [all_knowledge[i] for i in np.where(col_mask)[0]]

        local_dir = os.path.join(output_dir, str(year), "Local_Compact")
        save_sparse_matrix(M_dk_local, local_dir, f"DK_counts_local_{year}.npz")
        save_local_indices(local_dir, local_designers, local_knowledge, "DK")

        print(f"  [D-K] {year} 完成 (Global Shape: {M_dk.shape}, Local Shape: {M_dk_local.shape})")


# ===================== 主流程 =====================

def main():
    # 1. 准备全域数据
    df_da, df_ak, des_map, ab_map, know_map, n_des, n_ab, n_know = load_data_and_build_universe()

    # 2. 计算 D-A (Local + Global)
    da_mats = build_DA_matrices(df_da, des_map, ab_map, n_des, n_ab)

    # 3. 计算 A-K (Local + Global)
    ak_mats = build_AK_matrices(df_ak, ab_map, know_map, n_ab, n_know)

    # 4. 计算 D-K
    build_DK_matrices(da_mats, ak_mats)

    print("\n=== 全部完成。结果已保存为 .npz 格式 (分 Local 和 Global 目录) ===")


if __name__ == "__main__":
    main()