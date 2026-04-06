import os
import re
import itertools as it
from collections import Counter
import pandas as pd

# 输入目录
BASE_DIR = r"D:\Documents\Desktop\一品威客数据采集4月\test63"
FILENAME_TEMPLATE = "设计师宽度单一重叠度-{year}.csv"
YEARS = range(2014, 2015)  # 2015-2025

# 输出目录
OUTPUT_DIR = r"D:\Documents\Desktop\一品威客数据采集4月\设计师知识网络"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_csv_with_guess(path: str) -> pd.DataFrame:
    last_err = None
    # 尝试几种常见编码
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def find_col(cols, keyword: str):
    """模糊找到包含关键字的列名"""
    for c in cols:
        if keyword in str(c):
            return c
    return None

SEP_PATTERN = re.compile(r"[，,、]+")

def split_tasks(text) -> set:
    if pd.isna(text):
        return set()
    parts = SEP_PATTERN.split(str(text))
    return {p.strip() for p in parts if p and p.strip()}

def build_year_graph(df: pd.DataFrame, year: int, designer_col: str, task_col: str):
    # 仅保留相关列并清洗
    df = df[[designer_col, task_col]].dropna(subset=[designer_col, task_col]).copy()
    df[designer_col] = df[designer_col].astype(str).str.strip()
    df[task_col] = df[task_col].astype(str)

    # 聚合同一设计师的所有任务类型（若同一设计师多行）
    grouped = df.groupby(designer_col)[task_col].apply(lambda s: set().union(*[split_tasks(x) for x in s]))

    # 节点：任务类型 -> 设计师数
    task_to_designer_count = Counter()
    for tasks in grouped:
        for t in tasks:
            task_to_designer_count[t] += 1

    nodes = pd.DataFrame(
        [{"Id": t, "设计师人数": cnt, "年份": year} for t, cnt in task_to_designer_count.items()]
    ).sort_values("设计师人数", ascending=False, kind="mergesort")

    # 边：任务类型对 -> 共同设计师数（权重）
    pair_count = Counter()
    for tasks in grouped:
        if not tasks or len(tasks) < 2:
            continue
        sorted_tasks = sorted(tasks)
        for a, b in it.combinations(sorted_tasks, 2):
            pair_count[(a, b)] += 1

    edges = pd.DataFrame(
        [{"source": a, "target": b, "weight": w, "年份": year} for (a, b), w in pair_count.items()]
    ).sort_values("weight", ascending=False, kind="mergesort")

    return nodes, edges, len(grouped)

def main():
    summary = []
    for year in YEARS:
        file_path = os.path.join(BASE_DIR, FILENAME_TEMPLATE.format(year=year))
        if not os.path.exists(file_path):
            print(f"[跳过] {year}: 未找到文件 -> {file_path}")
            continue

        try:
            df = read_csv_with_guess(file_path)
        except Exception as e:
            print(f"[错误] {year}: 无法读取文件 {file_path}: {e}")
            continue

        # 容错匹配列名
        designer_col = "设计师" if "设计师" in df.columns else find_col(df.columns, "设计师")
        task_col = "任务类型" if "任务类型" in df.columns else find_col(df.columns, "任务类型")

        if designer_col is None or task_col is None:
            print(f"[错误] {year}: 找不到所需列。现有列: {list(df.columns)}")
            continue

        nodes, edges, designer_n = build_year_graph(df, year, designer_col, task_col)

        nodes_path = os.path.join(OUTPUT_DIR, f"节点表-{year}.csv")
        edges_path = os.path.join(OUTPUT_DIR, f"边表-{year}.csv")
        nodes.to_csv(nodes_path, index=False, encoding="utf-8-sig")
        edges.to_csv(edges_path, index=False, encoding="utf-8-sig")

        print(f"[完成] {year}: 设计师={designer_n}, 节点数={len(nodes)}, 边数={len(edges)}")
        print(f"       节点表 -> {nodes_path}")
        print(f"       边表   -> {edges_path}")
        summary.append((year, designer_n, len(nodes), len(edges)))

    if summary:
        s_df = pd.DataFrame(summary, columns=["年份", "设计师数", "节点数", "边数"])
        summary_path = os.path.join(OUTPUT_DIR, "汇总-各年规模.csv")
        s_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[汇总] 已输出各年规模到: {summary_path}")

if __name__ == "__main__":
    main()