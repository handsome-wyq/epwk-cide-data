#输出文件在“个体特征分析结果”使用origin画图，画分布直方图，kd曲线拟合，文件在test1125-1

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import entropy

# ===================== 1. 路径配置 =====================
# 请修改为你实际的输出路径
base_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识网络分析\1125计算结果-对齐版/矩阵与知识向量_Sparse_v2"

# 输出结果的文件夹 (专门存放用于 Origin 绘图的 CSV)
result_dir = os.path.join(base_dir, "个体特征分析结果")
os.makedirs(result_dir, exist_ok=True)

# 年份范围
years = list(range(2014, 2026))


# ===================== 2. 核心算法函数 =====================

def gini_coefficient_nonzero(x):
    """
    计算基尼系数 (Internal Gini)
    只针对非零元素计算：衡量设计师在他【已掌握】的技能中，是否存在极度偏科。
    0.0 = 技能点完全平均分布
    1.0 = 技能点全部集中在某一项上
    """
    x = np.array(x, dtype=np.float64)
    if x.size == 0: return 0.0
    if x.size == 1: return 1.0  # 只有一项技能，视为绝对专注 (或者0，视定义而定，通常1代表极端集中)

    # 确保非负
    if np.amin(x) < 0: x -= np.amin(x)
    x += 1e-9  # 防止除零

    x = np.sort(x)
    index = np.arange(1, x.shape[0] + 1)
    n = x.shape[0]

    return ((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))


def calculate_metrics_for_year(year):
    """
    计算单一年份的所有指标
    """
    # 构造路径 (使用 Local Compact 版本，计算个体特征更准且快)
    local_dir = os.path.join(base_dir, str(year), "Local_Compact")
    mat_path = os.path.join(local_dir, f"DK_counts_local_{year}.npz")
    idx_path = os.path.join(local_dir, "DK_indices.json")

    if not os.path.exists(mat_path):
        print(f"[跳过] {year} 年数据不存在: {mat_path}")
        return None

    # 加载稀疏矩阵和索引
    try:
        mat = sp.load_npz(mat_path)
        with open(idx_path, 'r', encoding='utf-8') as f:
            indices = json.load(f)
            designers = indices['rows']
    except Exception as e:
        print(f"[错误] {year} 年数据加载失败: {e}")
        return None

    print(f"正在计算 {year} 年... (设计师数: {len(designers)})")

    results = []

    # 遍历每个设计师 (每一行)
    for i in range(mat.shape[0]):
        # 获取第 i 行的所有非零数据 (counts)
        # mat[i] 是 csr_matrix，.data 直接给出一维非零数组
        row_data = mat[i].data

        # 1. 活跃度 (Volume): 知识点/订单总频次
        vol = row_data.sum()

        if vol == 0:
            # 理论上 Local 矩阵不应有全空行，但防万一
            continue

        # 2. 多元度 (Entropy): 香农熵
        # 先归一化为概率分布
        probs = row_data / vol
        # base=2, 单位为 bit
        ent = entropy(probs, base=2)

        # 3. 专注度 (Gini): 基尼系数
        gin = gini_coefficient_nonzero(row_data)

        results.append({
            'Designer_ID': designers[i],
            'Year': year,
            'Volume': vol,  # 活跃度/总量
            'Entropy': ent,  # 多元度/广度
            'Gini': gin  # 专注度/偏科度
        })

    return pd.DataFrame(results)


# ===================== 3. 主执行流程 =====================

def main():
    print("=== 开始计算个体特征 (Volume, Entropy, Gini) ===")

    all_years_data = []

    for year in years:
        df_year = calculate_metrics_for_year(year)

        if df_year is not None and not df_year.empty:
            # 1. 保存单年文件 (方便 Origin 导入单个分析)
            filename = f"Individual_Features_{year}.csv"
            save_path = os.path.join(result_dir, filename)
            df_year.to_csv(save_path, index=False, encoding='utf-8-sig')

            # 添加到总表
            all_years_data.append(df_year)

    # 2. 保存合并的总表 (方便 Origin 做跨年趋势分析)
    if all_years_data:
        df_all = pd.concat(all_years_data, ignore_index=True)
        total_path = os.path.join(result_dir, "Individual_Features_All_Years_Merged.csv")
        df_all.to_csv(total_path, index=False, encoding='utf-8-sig')
        print(f"\n[完成] 所有计算结束。")
        print(f"单年数据已保存至: {result_dir}")
        print(f"汇总数据已保存至: {total_path}")
    else:
        print("未生成任何数据，请检查输入路径。")


if __name__ == "__main__":
    main()