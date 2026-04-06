import os
import re
import pandas as pd
import numpy as np


# ===================== 路径配置 =====================

# 1）能力向量目录（前面“能力向量与金额”脚本的输出）
ability_vec_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识网络分析\能力向量与金额"

# 2）总网络目录（“总网络-节点和边”脚本的输出）
global_net_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识系统\总网络-节点和边"

# 3）本脚本输出目录：矩阵与知识向量（放在“知识网络分析”下面）
output_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识网络分析\矩阵与知识向量"
os.makedirs(output_dir, exist_ok=True)

years = list(range(2014, 2026))

# ===================== 工具函数 =====================

def parse_year_from_date(date_str):
    """日期字符串转年份（int），失败返回 None。"""
    if pd.isna(date_str):
        return None
    dt = pd.to_datetime(str(date_str), errors='coerce')
    if pd.isna(dt):
        return None
    return int(dt.year)


# ===================== 1. 读取基础数据 =====================

def load_designer_ability_vectors():
    """
    读取前面输出的 designer_ability_vectors_all_years.csv
    列应包含：designer, year, ability, order_count, total_amount
    """
    csv_path = os.path.join(ability_vec_dir, "designer_ability_vectors_all_years.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到能力向量文件: {csv_path}")

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # 清理类型
    df['year'] = df['year'].astype(int)
    # 保守处理：只要有 designer、ability、year 就保留
    df = df.dropna(subset=['designer', 'ability', 'year'])
    return df


def load_global_edges_and_nodes():
    """
    读取总网络中的：
      - 节点：nodes_orders_all.csv
      - 边：edges_能力到订单, 边：订单到知识
    """
    nodes_orders_path = os.path.join(global_net_dir, "nodes_orders_all.csv")
    edges_ao_path = os.path.join(global_net_dir, "edges_ability_order_all.csv")
    edges_ok_path = os.path.join(global_net_dir, "edges_order_knowledge_all.csv")

    if not os.path.exists(nodes_orders_path):
        raise FileNotFoundError(f"找不到订单节点文件: {nodes_orders_path}")
    if not os.path.exists(edges_ao_path):
        raise FileNotFoundError(f"找不到能力-订单边文件: {edges_ao_path}")
    if not os.path.exists(edges_ok_path):
        raise FileNotFoundError(f"找不到订单-知识边文件: {edges_ok_path}")

    nodes_orders = pd.read_csv(nodes_orders_path, encoding='utf-8-sig')
    edges_ao = pd.read_csv(edges_ao_path, encoding='utf-8-sig')
    edges_ok = pd.read_csv(edges_ok_path, encoding='utf-8-sig')

    return nodes_orders, edges_ao, edges_ok


# ===================== 2. 构建 D–A 矩阵 =====================

def build_DA_matrices(df_vec):
    """
    输入：designer_ability_vectors_all_years（长表）
    输出：
      1）按年在 output_dir/year/ 下保存：
         - DA_orders_yearYYYY.csv (designer × ability, 订单数)
         - DA_amount_yearYYYY.csv (designer × ability, 金额)
      2）整体矩阵（all_years）也保存一份
    返回：
      一个 dict: {year: {'orders': df, 'amount': df}, 'all': {...}}
    """
    result = {}

    # 逐年处理矩阵
    for year in sorted(df_vec['year'].unique()):
        year_df = df_vec[df_vec['year'] == year].copy()

        # 订单数矩阵
        mat_orders = year_df.pivot_table(
            index='designer',
            columns='ability',
            values='order_count',
            aggfunc='sum',
            fill_value=0
        )

        # 金额矩阵
        mat_amount = year_df.pivot_table(
            index='designer',
            columns='ability',
            values='total_amount',
            aggfunc='sum',
            fill_value=0.0
        )

        year_out_dir = os.path.join(output_dir, str(year))
        os.makedirs(year_out_dir, exist_ok=True)

        mat_orders.to_csv(os.path.join(year_out_dir, f"DA_orders_year{year}.csv"),
                          encoding='utf-8-sig')
        mat_amount.to_csv(os.path.join(year_out_dir, f"DA_amount_year{year}.csv"),
                          encoding='utf-8-sig')

        result[year] = {'orders': mat_orders, 'amount': mat_amount}
        print(f"[D-A] 年 {year}: designer×ability 矩阵已输出。")

    # 构建整体矩阵（不区分年）
    all_df = df_vec.copy()
    mat_orders_all = all_df.pivot_table(
        index='designer',
        columns='ability',
        values='order_count',
        aggfunc='sum',
        fill_value=0
    )
    mat_amount_all = all_df.pivot_table(
        index='designer',
        columns='ability',
        values='total_amount',
        aggfunc='sum',
        fill_value=0.0
    )

    mat_orders_all.to_csv(os.path.join(output_dir, "DA_orders_all_years.csv"),
                          encoding='utf-8-sig')
    mat_amount_all.to_csv(os.path.join(output_dir, "DA_amount_all_years.csv"),
                          encoding='utf-8-sig')

    result['all'] = {'orders': mat_orders_all, 'amount': mat_amount_all}

    print("[D-A] 全部年份的 designer×ability 矩阵已输出。")
    return result


# ===================== 3. 构建 A–K 矩阵 =====================

def build_AK_matrices(nodes_orders, edges_ao, edges_ok):
    """
    使用：
      - edges_ability_order_all:   Source=能力, Target=订单ID
      - edges_order_knowledge_all: Source=订单ID, Target=知识ID
      - nodes_orders_all:          ID=订单ID, Publication_Time=日期
    构建 能力–知识 共现矩阵：
      对于每个 (year, ability, knowledge)，统计共现订单数。

    输出：
      - AK_counts_yearYYYY.csv       (ability × knowledge，共现次数)
      - AK_row_norm_yearYYYY.csv     (能力层行归一化后的“概率矩阵”)
      - AK_counts_all_years.csv, AK_row_norm_all_years.csv
    """

    # ========== 1）为订单节点打上年份 ==========
    nodes_orders = nodes_orders.copy()
    nodes_orders['year'] = nodes_orders['Publication_Time'].apply(parse_year_from_date)

    # 保留有年份的订单
    nodes_orders = nodes_orders.dropna(subset=['year'])
    nodes_orders['year'] = nodes_orders['year'].astype(int)

    # ========== 2）能力-订单边 加上年份 ==========
    # edges_ao: Source=能力, Target=订单ID
    eao = edges_ao.copy()
    eao = eao.rename(columns={'Source': 'ability', 'Target': 'order_id'})
    # 连接订单以获得年份
    eao = eao.merge(
        nodes_orders[['ID', 'year']],
        left_on='order_id',
        right_on='ID',
        how='left'
    )
    eao = eao.drop(columns=['ID'])
    eao = eao.dropna(subset=['year'])
    eao['year'] = eao['year'].astype(int)

    # ========== 3）订单-知识边 ==========
    eok = edges_ok.copy()
    eok = eok.rename(columns={'Source': 'order_id', 'Target': 'knowledge'})

    # ========== 4）合并 得到 (year, ability, knowledge) ==========
    merged = eao.merge(
        eok[['order_id', 'knowledge']],
        on='order_id',
        how='inner'
    )
    # 去掉缺失
    merged = merged.dropna(subset=['ability', 'knowledge', 'year'])
    merged['year'] = merged['year'].astype(int)

    # ========== 5）统计共现次数 ==========
    grouped = (
        merged
        .groupby(['year', 'ability', 'knowledge'], as_index=False)
        .size()
        .rename(columns={'size': 'count'})
    )

    # 保存一个明细共现表，方便检查
    grouped.to_csv(
        os.path.join(output_dir, "AK_cooccurrence_triplets_all_years.csv"),
        index=False,
        encoding='utf-8-sig'
    )

    # 年度矩阵 + 总体矩阵
    result = {}

    # 各年矩阵
    for year in sorted(grouped['year'].unique()):
        year_df = grouped[grouped['year'] == year].copy()

        # ability × knowledge 频次矩阵
        mat_counts = year_df.pivot_table(
            index='ability',
            columns='knowledge',
            values='count',
            aggfunc='sum',
            fill_value=0
        )

        # 行归一化：每个能力对应的知识分布（概率意义）
        row_sum = mat_counts.sum(axis=1)
        # 避免除以 0
        row_sum[row_sum == 0] = 1
        mat_row_norm = mat_counts.div(row_sum, axis=0)

        year_out_dir = os.path.join(output_dir, str(year))
        os.makedirs(year_out_dir, exist_ok=True)

        mat_counts.to_csv(os.path.join(year_out_dir, f"AK_counts_year{year}.csv"),
                          encoding='utf-8-sig')
        mat_row_norm.to_csv(os.path.join(year_out_dir, f"AK_row_norm_year{year}.csv"),
                            encoding='utf-8-sig')

        result[year] = {'counts': mat_counts, 'row_norm': mat_row_norm}
        print(f"[A-K] 年 {year}: ability×knowledge 矩阵已输出。")

    # 全部年份汇总矩阵
    all_df = grouped.copy()
    all_df_agg = (
        all_df
        .groupby(['ability', 'knowledge'], as_index=False)['count']
        .sum()
    )
    mat_counts_all = all_df_agg.pivot_table(
        index='ability',
        columns='knowledge',
        values='count',
        aggfunc='sum',
        fill_value=0
    )
    row_sum_all = mat_counts_all.sum(axis=1)
    row_sum_all[row_sum_all == 0] = 1
    mat_row_norm_all = mat_counts_all.div(row_sum_all, axis=0)

    mat_counts_all.to_csv(os.path.join(output_dir, "AK_counts_all_years.csv"),
                          encoding='utf-8-sig')
    mat_row_norm_all.to_csv(os.path.join(output_dir, "AK_row_norm_all_years.csv"),
                            encoding='utf-8-sig')

    result['all'] = {'counts': mat_counts_all, 'row_norm': mat_row_norm_all}
    print("[A-K] 全部年份的 ability×knowledge 矩阵已输出。")

    return result


# ===================== 4. 构建 D–K 矩阵 =====================

def build_DK_matrices(DA_mats, AK_mats):
    """
    利用：
      - D–A 矩阵（订单数版）
      - A–K 矩阵（行归一化版本，即每个能力到知识的分布）
    计算：
      D–K 矩阵： V(DK) = V(DA_orders) × P(AK_row_norm)

    对每个 year 以及 all_years 输出：
      - DK_counts_yearYYYY.csv（设计师×知识，“有效订单数在知识维度上的投影”）
    """

    for key in AK_mats.keys():  # year 或 'all'
        if key == 'all':
            print("[D-K] 跳过 all 年份的整体矩阵计算（矩阵太大）")
            continue

        if key not in DA_mats:
            print(f"[D-K] 警告：{key} 在 D–A 矩阵中不存在，跳过。")
            continue

        # 设计师×能力（订单数）
        DA_orders = DA_mats[key]['orders']   # index: designer, columns: ability
        # 能力×知识（行归一化后的“权重矩阵”）
        AK_row = AK_mats[key]['row_norm']    # index: ability, columns: knowledge

        # 对齐能力维度：只用共同出现的能力
        common_abilities = sorted(set(DA_orders.columns) & set(AK_row.index))
        if not common_abilities:
            print(f"[D-K] {key}: 没有共同的能力标签，跳过。")
            continue

        DA_sub = DA_orders[common_abilities]
        AK_sub = AK_row.loc[common_abilities]

        # 矩阵乘法：D×A * A×K = D×K
        D_mat = DA_sub.values      # shape: (n_designer, n_ability)
        P_mat = AK_sub.values      # shape: (n_ability, n_knowledge)
        DK_mat = D_mat.dot(P_mat)  # shape: (n_designer, n_knowledge)

        designers = list(DA_sub.index)
        knowledges = list(AK_sub.columns)

        df_DK = pd.DataFrame(DK_mat, index=designers, columns=knowledges)

        # 输出
        if key == 'all':
            out_path = os.path.join(output_dir, "DK_counts_all_years.csv")
        else:
            year_out_dir = os.path.join(output_dir, str(key))
            os.makedirs(year_out_dir, exist_ok=True)
            out_path = os.path.join(year_out_dir, f"DK_counts_year{key}.csv")

        df_DK.to_csv(out_path, encoding='utf-8-sig')
        print(f"[D-K] {key}: 设计师×知识 矩阵已输出 -> {out_path}")


# ===================== 5. 一些简单的分析函数示例 =====================

def top_abilities_for_designer(df_vec, designer, year, topn=10):
    """
    查看某设计师在某年的“能力构成”（按订单数 & 金额排序）。
    df_vec: designer_ability_vectors_all_years
    """
    sub = df_vec[(df_vec['designer'] == designer) & (df_vec['year'] == year)]
    if sub.empty:
        print(f"[分析] {year} 年没有找到设计师 {designer} 的记录。")
        return

    print(f"\n[分析] 设计师 {designer} 在 {year} 年的能力向量 (Top {topn})")

    print("\n按订单数排序：")
    print(
        sub[['ability', 'order_count']]
        .sort_values(by='order_count', ascending=False)
        .head(topn)
        .to_string(index=False)
    )

    print("\n按金额排序：")
    print(
        sub[['ability', 'total_amount']]
        .sort_values(by='total_amount', ascending=False)
        .head(topn)
        .to_string(index=False)
    )


def top_knowledge_for_designer(DK_all, designer, key='all', topn=10):
    """
    查看某设计师的“知识构成”。
    DK_all 是前面 build_DK_matrices 输出后的字典可以不传（这里为了简单直接从 CSV 读）。
    为通用起见，这里直接读 CSV。
    key = 'all' or year(int)
    """
    if key == 'all':
        path = os.path.join(output_dir, "DK_counts_all_years.csv")
    else:
        path = os.path.join(output_dir, str(key), f"DK_counts_year{key}.csv")

    if not os.path.exists(path):
        print(f"[分析] 找不到 D–K 矩阵文件: {path}")
        return

    df_DK = pd.read_csv(path, encoding='utf-8-sig', index_col=0)
    if designer not in df_DK.index:
        print(f"[分析] 在 {key} 中找不到设计师 {designer} 的知识向量。")
        return

    row = df_DK.loc[designer]
    top = row.sort_values(ascending=False).head(topn)
    print(f"\n[分析] 设计师 {designer} 在 {key} 中的 Top {topn} 知识：")
    for k, v in top.items():
        print(f"  {k}: {v:.3f}")


def top_knowledge_for_ability(AK_mats, ability, key='all', topn=10):
    """
    查看某个能力最重要的知识（基于 A–K 行归一化矩阵）
    AK_mats: build_AK_matrices 返回的字典
    """
    if key not in AK_mats:
        print(f"[分析] {key} 不在 AK 矩阵中。")
        return

    mat = AK_mats[key]['row_norm']   # index: ability, columns: knowledge
    if ability not in mat.index:
        print(f"[分析] 在 {key} 中找不到能力 {ability}。")
        return

    row = mat.loc[ability]
    top = row.sort_values(ascending=False).head(topn)
    print(f"\n[分析] 能力 {ability} 在 {key} 中的 Top {topn} 相关知识：")
    for k, v in top.items():
        print(f"  {k}: {v:.3f}")


# ===================== 主流程 =====================

def main():
    print("=== 开始构建 能力向量 × 知识网络 的矩阵 ===")

    # 1. 读入设计师–能力向量
    df_vec = load_designer_ability_vectors()

    # 2. 构建 D–A 矩阵
    DA_mats = build_DA_matrices(df_vec)

    # 3. 读入 总网络 的订单节点和边
    nodes_orders, edges_ao, edges_ok = load_global_edges_and_nodes()

    # 4. 构建 A–K 矩阵
    AK_mats = build_AK_matrices(nodes_orders, edges_ao, edges_ok)

    # 5. 构建 D–K 矩阵
    build_DK_matrices(DA_mats, AK_mats)

    print("\n=== 矩阵构建完成，可以开始做各种分析了 ===")

    # ====== 示例：分析调用（你可以按需要换名字/年份） ======
    # 例：看一个具体设计师在 2020 年的能力分布
    # top_abilities_for_designer(df_vec, designer="B244146", year=2014, topn=10)

    # 例：看一个设计师在整体的知识偏好
    # top_knowledge_for_designer(DK_all=None, designer="B244146", key='all', topn=10)

    # 例：看某个能力（比如“LOGO设计”）最典型关联的知识
    # top_knowledge_for_ability(AK_mats, ability="LOGO设计", key='all', topn=10)


if __name__ == "__main__":
    main()
