import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_csv_auto(path):
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSV 文件读取失败，请检查编码格式。")

def extract_year_and_type(series: pd.Series):
    """
    兼容以下常见格式：
    2015 Type 1
    2015_Type1
    2015-Type 1
    2015Type1
    """
    parsed = series.astype(str).str.extract(r"(?P<year>\d{4}).*?Type\s*(?P<type>\d+)", expand=True)

    # 若上面没匹配上，再尝试更宽松一点的规则
    missing_mask = parsed["year"].isna() | parsed["type"].isna()
    if missing_mask.any():
        parsed2 = series[missing_mask].astype(str).str.extract(r"(?P<year>\d{4}).*?(?P<type>\d+)", expand=True)
        parsed.loc[missing_mask, "year"] = parsed2["year"]
        parsed.loc[missing_mask, "type"] = parsed2["type"]

    if parsed["year"].isna().any() or parsed["type"].isna().any():
        bad_rows = series[parsed["year"].isna() | parsed["type"].isna()].head(10).tolist()
        raise ValueError(f"部分 Source/Target 无法解析年份和类别，请检查这些值：{bad_rows}")

    parsed["year"] = parsed["year"].astype(int)
    parsed["type"] = parsed["type"].astype(int)
    return parsed

def build_transition_matrix(df_window, type_order):
    count_mat = df_window.pivot_table(
        index="src_type",
        columns="tgt_type",
        values="Value",
        aggfunc="sum",
        fill_value=0
    )
    count_mat = count_mat.reindex(index=type_order, columns=type_order, fill_value=0)

    row_sums = count_mat.sum(axis=1)
    prob_mat = count_mat.div(row_sums.replace(0, pd.NA), axis=0).fillna(0)

    return count_mat, prob_mat

def main():
    input_path = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Value_network\1125计算结果-对齐版\价值网络聚类分析结果\Sankey_Data_Survivors_Only_15-24.csv"
    output_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Value_network\1125计算结果-对齐版\价值网络聚类分析结果"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    os.makedirs(output_dir, exist_ok=True)

    df = read_csv_auto(input_path)

    required_cols = ["Source", "Target", "Value"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列：{col}")

    # 提取 Source / Target 中的年份和类别
    src_info = extract_year_and_type(df["Source"])
    tgt_info = extract_year_and_type(df["Target"])

    df["src_year"] = src_info["year"]
    df["src_type"] = src_info["type"]
    df["tgt_year"] = tgt_info["year"]
    df["tgt_type"] = tgt_info["type"]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0)

    # 仅保留合法的 t -> t+1 转移
    df = df[df["tgt_year"] == df["src_year"] + 1].copy()

    # 代表性转移窗口
    representative_windows = [
        (2015, 2016),
        (2018, 2019),
        (2021, 2022),
        (2023, 2024),
    ]

    # 自动识别类别数，通常应为 1~4
    all_types = sorted(set(df["src_type"].dropna().astype(int).tolist()) | set(df["tgt_type"].dropna().astype(int).tolist()))
    type_order = all_types
    type_labels = [f"Type {t}" for t in type_order]

    if len(type_order) == 0:
        raise ValueError("未识别到任何有效的 Type 类别。")

    # 画图
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (src_year, tgt_year) in zip(axes, representative_windows):
        df_window = df[(df["src_year"] == src_year) & (df["tgt_year"] == tgt_year)].copy()

        if df_window.empty:
            # 如果该窗口没有数据，也输出空矩阵
            count_mat = pd.DataFrame(0, index=type_order, columns=type_order)
            prob_mat = pd.DataFrame(0.0, index=type_order, columns=type_order)
        else:
            count_mat, prob_mat = build_transition_matrix(df_window, type_order)

        # 保存 count matrix 和 probability matrix
        count_save_path = os.path.join(output_dir, f"Transition_{src_year}_{tgt_year}_count_matrix.csv")
        prob_save_path = os.path.join(output_dir, f"Transition_{src_year}_{tgt_year}_probability_matrix.csv")
        count_mat.to_csv(count_save_path, encoding="utf-8-sig")
        prob_mat.to_csv(prob_save_path, encoding="utf-8-sig")

        sns.heatmap(
            prob_mat,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar=True,
            ax=ax
        )

        ax.set_title(f"{src_year}-{tgt_year}", fontsize=22)
        ax.set_xlabel("Target state at t+1", fontsize=20)
        ax.set_ylabel("Source state at t", fontsize=20)
        ax.set_xticklabels(type_labels, rotation=0)
        ax.set_yticklabels(type_labels, rotation=0)

    plt.tight_layout()

    fig_save_path = os.path.join(output_dir, "Figure13_Representative_Transition_Heatmaps.png")
    plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("处理完成，输出文件已保存到：")
    print(output_dir)
    print(f"热图文件：{fig_save_path}")

if __name__ == "__main__":
    main()