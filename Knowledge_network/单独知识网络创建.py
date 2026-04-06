'''import os
import pandas as pd
from itertools import combinations   # 用于生成知识对组合
from collections import Counter      # 用于统计共现次数

# ========= 路径配置 =========

# 年度网络节点和边（你前面构建多层网络的输出路径）
network_base_dir =  r"D:/Documents/Desktop/Crowd_intelligence_wyq/F1_网络构建/网络数据1"

# 知识演化 + 价值分析输出目录
output_base_dir  = r"D:/Documents/Desktop/Crowd_intelligence_wyq/F1_网络构建/知识演化与价值2"
os.makedirs(output_base_dir, exist_ok=True)

years = list(range(2014, 2026))


def safe_read_csv(path):
    """安全读取 CSV，不存在则返回空表"""
    if not os.path.exists(path):
        print(f"  [警告] 文件不存在：{path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        print(f"  [警告] 读取失败 {path}: {e}")
        return pd.DataFrame()


# ========= 第一步：扫描所有年份，计算每个知识的 first_year =========

knowledge_first_year = {}  # {Knowledge_ID: first_year}

print("=== 第一步：计算每个知识节点的首次出现年份 first_year ===")

for year in years:
    year_dir = os.path.join(network_base_dir, str(year))
    nodes_knowledge_path = os.path.join(year_dir, f"nodes_knowledge_{year}.csv")

    df_k = safe_read_csv(nodes_knowledge_path)
    if df_k.empty:
        continue

    # 统一 ID 类型
    df_k["ID"] = df_k["ID"].astype(str)

    for kid in df_k["ID"].unique():
        if kid not in knowledge_first_year:
            knowledge_first_year[kid] = year

print(f"  共发现知识节点 {len(knowledge_first_year)} 个。")


# ========= 第二步：逐年计算“演化 + 价值”并输出年度节点表 =========

all_year_dfs = []  # 用来合并成一个总表

print("\n=== 第二步：逐年计算知识的演化标记 + 赚钱能力（并输出 K-K 加权知识网络） ===")

for year in years:
    print(f"\n===== 处理年份：{year} =====")

    year_dir = os.path.join(network_base_dir, str(year))
    if not os.path.isdir(year_dir):
        print(f"  [跳过] 年度目录不存在：{year_dir}")
        continue

    # 1) 节点 & 边数据
    nodes_orders_path    = os.path.join(year_dir, f"nodes_orders_{year}.csv")
    nodes_knowledge_path = os.path.join(year_dir, f"nodes_knowledge_{year}.csv")
    edges_ok_path        = os.path.join(year_dir, f"edges_order_knowledge_{year}.csv")
    edges_kk_path        = os.path.join(year_dir, f"edges_knowledge_knowledge_{year}.csv")

    df_o  = safe_read_csv(nodes_orders_path)
    df_k  = safe_read_csv(nodes_knowledge_path)
    df_ok = safe_read_csv(edges_ok_path)
    df_kk = safe_read_csv(edges_kk_path)

    if df_k.empty:
        print("  [提示] 当年知识节点表为空，跳过本年。")
        continue

    # ---------- 1. 知识度数（知识-知识网络，未加权） ----------
    degree_df = pd.DataFrame(columns=["Knowledge_ID", "degree_kk"])

    if not df_kk.empty:
        df_kk = df_kk.copy()
        df_kk["Source"] = df_kk["Source"].astype(str)
        df_kk["Target"] = df_kk["Target"].astype(str)

        deg_s = df_kk.groupby("Source").size()
        deg_t = df_kk.groupby("Target").size()
        degree_series = deg_s.add(deg_t, fill_value=0).astype(int)

        degree_df = degree_series.reset_index()
        degree_df.columns = ["Knowledge_ID", "degree_kk"]

    # ---------- 2. 知识赚钱能力（订单-知识 + 订单价格） ----------
    # 注意：这里只是用 O-K 边来统计每个知识参与的订单数和金额，
    # 但不再输出 O-K 加权边文件。

    value_df = pd.DataFrame(
        columns=[
            "Knowledge_ID", "order_count",
            "demand_sum", "trans_sum",
            "demand_sum_share", "trans_sum_share",
            "avg_trans_per_order",
        ]
    )

    # K-K 加权边输出路径（只输出这一种）
    edges_kk_weighted_path = os.path.join(
        output_base_dir, f"edges_knowledge_knowledge_weighted_{year}.csv"
    )

    if not df_o.empty and not df_ok.empty:
        df_o = df_o.copy()
        df_o["ID"] = df_o["ID"].astype(str)

        df_ok = df_ok.copy().rename(columns={"Source": "Order_ID", "Target": "Knowledge_ID"})
        df_ok["Order_ID"] = df_ok["Order_ID"].astype(str)
        df_ok["Knowledge_ID"] = df_ok["Knowledge_ID"].astype(str)

        # ===== 价格列兜底：nodes_orders_YYYY.csv 中已经是纯数字 =====
        for col in ["Demand_Price", "Transaction_Price"]:
            if col not in df_o.columns:
                df_o[col] = pd.NA

        # 只做一次简单挂 price，不做任何填补/分摊
        df_edge_price = df_ok.merge(
            df_o[["ID", "Demand_Price", "Transaction_Price"]],
            left_on="Order_ID",
            right_on="ID",
            how="left"
        ).drop(columns=["ID"])

        # 方法：整单金额全部算给知识（严格用原始 Transaction_Price）
        value_df = df_edge_price.groupby("Knowledge_ID").agg(
            order_count=("Order_ID", "nunique"),
            demand_sum=("Demand_Price", "sum"),  # 需求金额总和（参考）
            trans_sum=("Transaction_Price", "sum"),  # 成交金额总和（核心）
        ).reset_index()

        # 平均每单成交价（只基于 trans_sum）
        value_df["avg_trans_per_order"] = (
                value_df["trans_sum"] / value_df["order_count"]
        )
    else:
        print("  [提示] 订单 or 订单-知识边为空，本年度无法计算价值，只做结构和演化标记。")

        # ========== 2.x 额外输出：基于 O-K 共现统计的 K-K 加权边 ==========
        # 思路：同一个订单里的知识两两配对，共现一次 Weight_cooccur += 1

        pair_counter = Counter()

        # 用 O-K 边（df_ok）来做共现统计即可
        for order_id, g in df_ok.groupby("Order_ID"):
            ks = sorted(set(g["Knowledge_ID"]))
            if len(ks) < 2:
                continue
            for a, b in combinations(ks, 2):
                pair_counter[(a, b)] += 1

        if pair_counter:
            rows = []
            for (src, tgt), w in pair_counter.items():
                rows.append([src, tgt, "共现", w])

            df_edges_kk_weighted = pd.DataFrame(
                rows,
                columns=["Source", "Target", "Type", "Weight_cooccur"]
            )

            df_edges_kk_weighted.to_csv(
                edges_kk_weighted_path,
                index=False,
                encoding="utf-8-sig"
            )
            print(f"  [OK] 已输出基于共现的加权 K-K 边：{edges_kk_weighted_path}（{len(df_edges_kk_weighted)} 条）")
        else:
            print("  [提示] 本年度 O-K 边不足以形成任何知识共现对，未生成 K-K 加权边。")


    # ---------- 3. 合并到当年的知识节点表 ----------

    df_k = df_k.copy()
    df_k["ID"] = df_k["ID"].astype(str)
    df_k_small = df_k[["ID", "Name"]].drop_duplicates()
    df_k_small = df_k_small.rename(columns={"ID": "Knowledge_ID"})

    # 3.1 合并 degree（未加权度数）
    df_year = df_k_small.merge(degree_df, on="Knowledge_ID", how="left")

    # 3.2 合并价值指标
    df_year = df_year.merge(value_df, on="Knowledge_ID", how="left")

    # 3.3 加上年份、first_year、是否新知识
    df_year["year"] = year
    df_year["first_year"] = df_year["Knowledge_ID"].map(knowledge_first_year)
    df_year["is_new_this_year"] = (df_year["first_year"] == year).astype(int)

    # 3.4 缺失数值列填 0
    num_cols = [
        "degree_kk",
        "order_count",
        "demand_sum",
        "trans_sum",
        "avg_trans_per_order",
    ]
    for col in num_cols:
        if col in df_year.columns:
            df_year[col] = df_year[col].fillna(0)

    # 3.5 整理列顺序
    df_year = df_year[
        [
            "Knowledge_ID",
            "Name",
            "year",
            "first_year",
            "is_new_this_year",
            "degree_kk",
            "order_count",
            "demand_sum",
            "trans_sum",
            "avg_trans_per_order",
        ]
    ]

    # ---------- 4. 输出年度节点增强表 ----------
    year_out_path = os.path.join(output_base_dir, f"knowledge_nodes_evo_value_{year}.csv")
    df_year.to_csv(year_out_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 年度知识节点增强表已输出：{year_out_path}（{len(df_year)} 条）")

    # 1）当年最赚钱知识 TOP-20（按总赚的钱）
    top_value = df_year.sort_values("trans_sum", ascending=False).head(20)
    top_value_path = os.path.join(output_base_dir, f"top_value_knowledge_{year}.csv")
    top_value.to_csv(top_value_path, index=False, encoding="utf-8-sig")

    # 2）当年最常出现知识 TOP-20
    top_freq = df_year.sort_values("order_count", ascending=False).head(20)
    top_freq_path = os.path.join(output_base_dir, f"top_freq_knowledge_{year}.csv")
    top_freq.to_csv(top_freq_path, index=False, encoding="utf-8-sig")

    all_year_dfs.append(df_year)

# ========= 第三步：输出跨年总表 =========

if all_year_dfs:
    df_all = pd.concat(all_year_dfs, ignore_index=True)
    all_out_path = os.path.join(output_base_dir, "knowledge_nodes_evo_value_all_years.csv")
    df_all.to_csv(all_out_path, index=False, encoding="utf-8-sig")
    print(f"\n=== 跨年总表已输出：{all_out_path}，共 {len(df_all)} 行 ===")
else:
    print("\n[提示] 没有任何年份生成有效结果。")
'''
import os
import pandas as pd
from itertools import combinations   # 用于生成知识对组合
from collections import Counter      # 用于统计共现次数

# ========= 路径配置 =========

# 年度网络节点和边（你前面构建多层网络的输出路径）
network_base_dir =  r"D:/Desktop/CID/F1_网络构建/网络数据1"

# 知识演化 + 价值分析输出目录
output_base_dir  = r"D:/Desktop/CID/F1_网络构建/知识演化与价值4"
os.makedirs(output_base_dir, exist_ok=True)

years = list(range(2014, 2026))


def safe_read_csv(path):
    """安全读取 CSV，不存在则返回空表"""
    if not os.path.exists(path):
        print(f"  [警告] 文件不存在：{path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        print(f"  [警告] 读取失败 {path}: {e}")
        return pd.DataFrame()


def strip_knowledge_prefix(kid):
    """
    去掉 'knowledge:' 前缀，用于统一知识点的 ID 表示。
    例如：'knowledge:品牌定位' -> '品牌定位'
    """
    if pd.isna(kid):
        return kid
    s = str(kid)
    if s.startswith("knowledge:"):
        return s[len("knowledge:"):]
    return s


# ========= 第一步：扫描所有年份，计算每个知识的 first_year =========

knowledge_first_year = {}  # {Knowledge_ID_clean: first_year}

print("=== 第一步：计算每个知识节点的首次出现年份 first_year ===")

for year in years:
    year_dir = os.path.join(network_base_dir, str(year))
    nodes_knowledge_path = os.path.join(year_dir, f"nodes_knowledge_{year}.csv")

    df_k = safe_read_csv(nodes_knowledge_path)
    if df_k.empty:
        continue

    # 统一 ID 类型，并去掉 knowledge: 前缀
    df_k["ID"] = df_k["ID"].astype(str)
    df_k["Knowledge_ID_clean"] = df_k["ID"].apply(strip_knowledge_prefix)

    for kid in df_k["Knowledge_ID_clean"].unique():
        if kid not in knowledge_first_year:
            knowledge_first_year[kid] = year

print(f"  共发现知识节点 {len(knowledge_first_year)} 个。")


# ========= 第二步：逐年计算“演化 + 价值”并输出年度节点表 =========

all_year_dfs = []  # 用来合并成一个总表

print("\n=== 第二步：逐年计算知识的演化标记 + 赚钱能力（并输出 K-K 加权知识网络） ===")

for year in years:
    print(f"\n===== 处理年份：{year} =====")

    year_dir = os.path.join(network_base_dir, str(year))
    if not os.path.isdir(year_dir):
        print(f"  [跳过] 年度目录不存在：{year_dir}")
        continue

    # 1) 节点 & 边数据
    nodes_orders_path    = os.path.join(year_dir, f"nodes_orders_{year}.csv")
    nodes_knowledge_path = os.path.join(year_dir, f"nodes_knowledge_{year}.csv")
    edges_ok_path        = os.path.join(year_dir, f"edges_order_knowledge_{year}.csv")
    edges_kk_path        = os.path.join(year_dir, f"edges_knowledge_knowledge_{year}.csv")

    df_o  = safe_read_csv(nodes_orders_path)
    df_k  = safe_read_csv(nodes_knowledge_path)
    df_ok = safe_read_csv(edges_ok_path)
    df_kk = safe_read_csv(edges_kk_path)

    if df_k.empty:
        print("  [提示] 当年知识节点表为空，跳过本年。")
        continue

    # ---------- 1. 知识度数（知识-知识网络，未加权） ----------
    degree_df = pd.DataFrame(columns=["Knowledge_ID", "degree_kk"])

    if not df_kk.empty:
        df_kk = df_kk.copy()
        df_kk["Source"] = df_kk["Source"].astype(str).apply(strip_knowledge_prefix)
        df_kk["Target"] = df_kk["Target"].astype(str).apply(strip_knowledge_prefix)

        deg_s = df_kk.groupby("Source").size()
        deg_t = df_kk.groupby("Target").size()
        degree_series = deg_s.add(deg_t, fill_value=0).astype(int)

        degree_df = degree_series.reset_index()
        degree_df.columns = ["Knowledge_ID", "degree_kk"]

    # ---------- 2. 知识赚钱能力（订单-知识 + 订单价格） ----------
    # 注意：这里只是用 O-K 边来统计每个知识参与的订单数和金额，
    # 但不再输出 O-K 加权边文件。

    value_df = pd.DataFrame(
        columns=[
            "Knowledge_ID", "order_count",
            "demand_sum", "trans_sum",
            "demand_sum_share", "trans_sum_share",
            "avg_trans_per_order",
        ]
    )

    # K-K 加权边输出路径（只输出这一种）
    edges_kk_weighted_path = os.path.join(
        output_base_dir, f"edges_knowledge_knowledge_weighted_{year}.csv"
    )

    # ========== 2.1 如果有订单和 O-K 边：计算价值 ==========
    if not df_o.empty and not df_ok.empty:
        df_o = df_o.copy()
        df_o["ID"] = df_o["ID"].astype(str)

        # Source: 订单ID，Target: 知识ID（带前缀），这里统一改名并去前缀
        df_ok = df_ok.copy().rename(columns={"Source": "Order_ID", "Target": "Knowledge_ID"})
        df_ok["Order_ID"] = df_ok["Order_ID"].astype(str)
        df_ok["Knowledge_ID"] = df_ok["Knowledge_ID"].astype(str).apply(strip_knowledge_prefix)

        # ===== 价格列兜底：nodes_orders_YYYY.csv 中已经是纯数字 =====
        for col in ["Demand_Price", "Transaction_Price"]:
            if col not in df_o.columns:
                df_o[col] = pd.NA

        # 只做一次简单挂 price，不做任何填补/分摊
        df_edge_price = df_ok.merge(
            df_o[["ID", "Demand_Price", "Transaction_Price"]],
            left_on="Order_ID",
            right_on="ID",
            how="left"
        ).drop(columns=["ID"])

        # 方法：整单金额全部算给知识（严格用原始 Transaction_Price）
        value_df = df_edge_price.groupby("Knowledge_ID").agg(
            order_count=("Order_ID", "nunique"),
            demand_sum=("Demand_Price", "sum"),      # 需求金额总和（参考）
            trans_sum=("Transaction_Price", "sum"),  # 成交金额总和（核心）
        ).reset_index()

        # 平均每单成交价（只基于 trans_sum）
        value_df["avg_trans_per_order"] = (
                value_df["trans_sum"] / value_df["order_count"]
        )

    else:
        print("  [提示] 订单 or 订单-知识边为空，本年度无法计算价值，只做结构和演化标记。")
        # df_ok 可能为空，也可能只是缺其一，这种情况下就不算价值

    # ========== 2.2 基于 O-K 共现统计的 K-K 加权边（无论是否有订单，都可以算） ==========
    pair_counter = Counter()

    if not df_ok.empty:
        # 确保使用的是干净的 Knowledge_ID（如果前面没 rename，这里补一遍）
        if "Order_ID" not in df_ok.columns or "Knowledge_ID" not in df_ok.columns:
            df_ok = df_ok.copy().rename(columns={"Source": "Order_ID", "Target": "Knowledge_ID"})
        df_ok["Order_ID"] = df_ok["Order_ID"].astype(str)
        df_ok["Knowledge_ID"] = df_ok["Knowledge_ID"].astype(str).apply(strip_knowledge_prefix)

        for order_id, g in df_ok.groupby("Order_ID"):
            ks = sorted(set(g["Knowledge_ID"]))
            if len(ks) < 2:
                continue
            for a, b in combinations(ks, 2):
                pair_counter[(a, b)] += 1

    if pair_counter:
        rows = []
        for (src, tgt), w in pair_counter.items():
            rows.append([src, tgt, "共现", w])

        df_edges_kk_weighted = pd.DataFrame(
            rows,
            columns=["Source", "Target", "Type", "Weight_cooccur"]
        )

        df_edges_kk_weighted.to_csv(
            edges_kk_weighted_path,
            index=False,
            encoding="utf-8-sig"
        )
        print(f"  [OK] 已输出基于共现的加权 K-K 边：{edges_kk_weighted_path}（{len(df_edges_kk_weighted)} 条）")
    else:
        print("  [提示] 本年度 O-K 边不足以形成任何知识共现对，未生成 K-K 加权边。")

    # ---------- 3. 合并到当年的知识节点表 ----------

    df_k = df_k.copy()
    df_k["ID"] = df_k["ID"].astype(str)
    df_k["Knowledge_ID"] = df_k["ID"].apply(strip_knowledge_prefix)
    df_k_small = df_k[["Knowledge_ID", "Name"]].drop_duplicates()

    # 3.1 合并 degree（未加权度数）
    df_year = df_k_small.merge(degree_df, on="Knowledge_ID", how="left")

    # 3.2 合并价值指标
    df_year = df_year.merge(value_df, on="Knowledge_ID", how="left")

    # 3.3 加上年份、first_year、是否新知识
    df_year["year"] = year
    df_year["first_year"] = df_year["Knowledge_ID"].map(knowledge_first_year)
    df_year["is_new_this_year"] = (df_year["first_year"] == year).astype(int)

    # 3.4 缺失数值列填 0
    num_cols = [
        "degree_kk",
        "order_count",
        "demand_sum",
        "trans_sum",
        "avg_trans_per_order",
    ]
    for col in num_cols:
        if col in df_year.columns:
            df_year[col] = df_year[col].fillna(0)
    df_year = df_year.sort_values(by="is_new_this_year", ascending=True)                  
    # 3.5 整理列顺序
    df_year = df_year[
        [
            "Knowledge_ID",
            "Name",
            "year",
            "first_year",
            "is_new_this_year",
            "degree_kk",
            "order_count",
            "demand_sum",
            "trans_sum",
            "avg_trans_per_order",
        ]
    ].rename(columns={"Knowledge_ID": "id"})

    # ---------- 4. 输出年度节点增强表 ----------
    year_out_path = os.path.join(output_base_dir, f"knowledge_nodes_evo_value_{year}.csv")
    df_year.to_csv(year_out_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 年度知识节点增强表已输出：{year_out_path}（{len(df_year)} 条）")

    # 1）当年最赚钱知识 TOP-20（按总赚的钱）
    top_value = df_year.sort_values("trans_sum", ascending=False).head(20)
    top_value_path = os.path.join(output_base_dir, f"top_value_knowledge_{year}.csv")
    top_value.to_csv(top_value_path, index=False, encoding="utf-8-sig")

    # 2）当年最常出现知识 TOP-20
    top_freq = df_year.sort_values("order_count", ascending=False).head(20)
    top_freq_path = os.path.join(output_base_dir, f"top_freq_knowledge_{year}.csv")
    top_freq.to_csv(top_freq_path, index=False, encoding="utf-8-sig")

    all_year_dfs.append(df_year)

# ========= 第三步：输出跨年总表 =========

if all_year_dfs:
    df_all = pd.concat(all_year_dfs, ignore_index=True)
    all_out_path = os.path.join(output_base_dir, "knowledge_nodes_evo_value_all_years.csv")
    df_all.to_csv(all_out_path, index=False, encoding="utf-8-sig")
    print(f"\n=== 跨年总表已输出：{all_out_path}，共 {len(df_all)} 行 ===")
else:
    print("\n[提示] 没有任何年份生成有效结果。")
