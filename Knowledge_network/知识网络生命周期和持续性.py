import os
import pandas as pd
import numpy as np


# =========================================================
# 0) 配置
# =========================================================
INPUT_CSV = r"D:\Documents\Desktop\Crowd_intelligence_wyq\F1_网络构建\知识演化与价值3\knowledge_nodes_evo_value_all_years.csv"
OUTPUT_DIR = r"D:\Documents\Desktop\Crowd_intelligence_wyq\F1_网络构建\知识生命周期与持续性分析结果_精简版"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 研究期
START_YEAR = 2015
END_YEAR = 2024

# 长寿命知识定义：5年及以上
LONG_LIFE_THRESHOLD = 5

# 每个寿命长度保留多少个案例候选
TOP_CASES_PER_LIFESPAN = 10


# =========================================================
# 1) 工具函数
# =========================================================
def longest_consecutive_run(years_sorted):
    """计算最长连续活跃年数"""
    if not years_sorted:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(years_sorted)):
        if years_sorted[i] == years_sorted[i - 1] + 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def count_reactivation(years_sorted):
    """
    回潮次数：
    若活跃年份由多个连续块构成，则回潮次数 = 连续块数 - 1
    """
    if not years_sorted:
        return 0
    blocks = 1
    for i in range(1, len(years_sorted)):
        if years_sorted[i] != years_sorted[i - 1] + 1:
            blocks += 1
    return max(0, blocks - 1)


def safe_save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已输出: {path}")


def make_distribution_table(df, group_col, count_name="知识数量", sort_col=None):
    out = (
        df.groupby(group_col, observed=False)
        .size()
        .reset_index(name=count_name)
    )
    if sort_col is not None:
        out = out.sort_values(sort_col)
    return out


# =========================================================
# 2) 读取与清洗数据
# =========================================================
def load_and_clean_data(input_csv):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    required_cols = ["id", "Name", "year"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要字段: {col}")

    df = df.copy()
    df["id"] = df["id"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    print(f"[INFO] 原始数据年份范围: {df['year'].min()} - {df['year'].max()}")

    # 仅保留研究期内数据
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    print(f"[INFO] 研究期筛选后年份范围: {df['year'].min()} - {df['year'].max()}")

    # 只保留真正活跃记录
    if "order_count" in df.columns:
        df = df[df["order_count"].fillna(0) > 0].copy()

    print(f"[INFO] 研究期内活跃记录数: {len(df)}")
    print(f"[INFO] 研究期内知识数量(id): {df['id'].nunique()}")

    return df


# =========================================================
# 3) 构建生命周期与持续性明细
# =========================================================
def build_persistence_detail(df):
    results = []

    for kid, g in df.groupby("id"):
        name = g["Name"].iloc[0]
        years_list = sorted(g["year"].dropna().unique().tolist())
        if not years_list:
            continue

        first_year = min(years_list)
        last_year = max(years_list)
        lifespan_years = last_year - first_year + 1
        active_years = len(years_list)

        persistence_index = active_years / lifespan_years if lifespan_years > 0 else np.nan
        max_consecutive_active_years = longest_consecutive_run(years_list)
        continuity_index = (
            max_consecutive_active_years / lifespan_years if lifespan_years > 0 else np.nan
        )
        reactivation_count = count_reactivation(years_list)

        total_order_count = g["order_count"].sum() if "order_count" in g.columns else np.nan
        total_trans_sum = g["trans_sum"].sum() if "trans_sum" in g.columns else np.nan

        results.append({
            "知识ID": kid,
            "知识名称": name,
            "首次出现年份": first_year,
            "最后出现年份": last_year,
            "生命周期长度": lifespan_years,
            "活跃年数": active_years,
            "持续性指数": persistence_index,
            "最长连续活跃年数": max_consecutive_active_years,
            "连续性指数": continuity_index,
            "回潮次数": reactivation_count,
            "总订单数": total_order_count,
            "总交易额": total_trans_sum,
            "活跃年份序列": ",".join(map(str, years_list))
        })

    return pd.DataFrame(results)


# =========================================================
# 4) 长寿命知识年度轨迹表
# =========================================================
def build_longlife_year_matrix(df_active, df_detail, threshold):
    df_long = df_detail[df_detail["生命周期长度"] >= threshold][["知识ID", "知识名称", "生命周期长度"]].copy()
    long_ids = set(df_long["知识ID"])

    df_case = df_active[df_active["id"].isin(long_ids)].copy()

    year_matrix = df_case.pivot_table(
        index=["id", "Name"],
        columns="year",
        values="order_count",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    year_matrix = year_matrix.rename(columns={"id": "知识ID", "Name": "知识名称"})
    year_matrix = year_matrix.merge(
        df_long,
        on=["知识ID", "知识名称"],
        how="left"
    )

    year_cols = sorted([c for c in year_matrix.columns if isinstance(c, int)])
    ordered_cols = ["知识ID", "知识名称", "生命周期长度"] + year_cols
    year_matrix = year_matrix[ordered_cols]

    return year_matrix


# =========================================================
# 5) 输出必要结果
# =========================================================
def export_outputs(df_active, df_detail, output_dir, threshold):
    # 5.1 生命周期分布（全样本）
    lifespan_dist = make_distribution_table(
        df_detail, "生命周期长度", count_name="知识数量", sort_col="生命周期长度"
    )
    safe_save_csv(
        lifespan_dist,
        os.path.join(output_dir, "知识生命周期分布.csv")
    )

    # 5.2 长寿命知识筛选
    df_long = df_detail[df_detail["生命周期长度"] >= threshold].copy()

    if df_long.empty:
        print(f"[WARN] 没有生命周期长度大于等于 {threshold} 年的知识。")
        return

    # 5.3 长寿命知识按寿命逐年分组比较
    longlife_summary = (
        df_long.groupby("生命周期长度", observed=False)
        .agg(
            知识数量=("知识ID", "count"),
            平均持续性指数=("持续性指数", "mean"),
            中位持续性指数=("持续性指数", "median"),
            平均最长连续活跃年数=("最长连续活跃年数", "mean"),
            平均连续性指数=("连续性指数", "mean"),
            平均回潮次数=("回潮次数", "mean"),
            平均总订单数=("总订单数", "mean"),
            平均总交易额=("总交易额", "mean")
        )
        .reset_index()
        .sort_values("生命周期长度")
    )
    safe_save_csv(
        longlife_summary,
        os.path.join(output_dir, "长寿命知识持续性比较表.csv")
    )

    # 5.4 长寿命知识回潮分布
    reactivation_dist = make_distribution_table(
        df_long, "回潮次数", count_name="知识数量", sort_col="回潮次数"
    )
    safe_save_csv(
        reactivation_dist,
        os.path.join(output_dir, "长寿命知识回潮分布.csv")
    )

    # 5.5 长寿命知识年度活跃轨迹表
    year_matrix = build_longlife_year_matrix(df_active, df_detail, threshold)
    safe_save_csv(
        year_matrix,
        os.path.join(output_dir, "长寿命知识年度活跃轨迹表.csv")
    )

    # 5.6 长寿命知识案例候选
    case_rows = []
    for life, g in df_long.groupby("生命周期长度"):
        g_sorted = g.sort_values(
            ["总交易额", "持续性指数", "总订单数"],
            ascending=[False, False, False]
        ).head(TOP_CASES_PER_LIFESPAN)
        case_rows.append(g_sorted)

    if case_rows:
        df_cases = pd.concat(case_rows, ignore_index=True)
        safe_save_csv(
            df_cases,
            os.path.join(output_dir, "长寿命知识案例候选.csv")
        )


# =========================================================
# 6) 主程序
# =========================================================
def main():
    df_active = load_and_clean_data(INPUT_CSV)
    df_detail = build_persistence_detail(df_active)
    export_outputs(df_active, df_detail, OUTPUT_DIR, LONG_LIFE_THRESHOLD)

    print("\n=== 分析完成 ===")
    print(df_detail.head())


if __name__ == "__main__":
    main()
