import os
import pandas as pd

# ===================== 1. 路径配置 =====================
# 请根据你的实际路径修改
base_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network"
global_net_dir = os.path.join(base_dir, r"知识系统\总网络-节点和边")
output_dir = os.path.join(base_dir, r"知识网络分析\知识热度分析")
os.makedirs(output_dir, exist_ok=True)


# ===================== 2. 工具函数 =====================

def parse_year(date_str):
    """从日期字符串解析年份"""
    try:
        return pd.to_datetime(str(date_str)).year
    except:
        return None


def clean_knowledge_name(name):
    """清洗知识名称，去掉可能存在的 'knowledge:' 前缀"""
    if isinstance(name, str):
        return name.replace("knowledge:", "")
    return name


# ===================== 3. 核心计算逻辑 =====================

def calculate_annual_hot_words():
    print("=== 开始从原始文件统计年度热词 ===")

    # --- Step 1: 读取订单节点表 (获取 ID -> Year 映射) ---
    nodes_path = os.path.join(global_net_dir, "nodes_orders_all.csv")
    if not os.path.exists(nodes_path):
        print(f"[错误] 找不到文件: {nodes_path}")
        return

    print("1. 正在读取订单时间信息...")
    df_nodes = pd.read_csv(nodes_path, encoding='utf-8-sig')

    # 解析年份
    df_nodes['Year'] = df_nodes['Publication_Time'].apply(parse_year)
    # 只保留需要的列：ID 和 Year
    df_orders = df_nodes[['ID', 'Year']].dropna().astype({'Year': 'int'})

    # --- Step 2: 读取 订单-知识 边表 (获取 Order -> Knowledge 关系) ---
    edges_path = os.path.join(global_net_dir, "edges_order_knowledge_all.csv")
    if not os.path.exists(edges_path):
        print(f"[错误] 找不到文件: {edges_path}")
        return

    print("2. 正在读取订单-知识关联信息...")
    df_edges = pd.read_csv(edges_path, encoding='utf-8-sig')

    # 重命名列以方便合并 (假设 Source是订单, Target是知识)
    # 根据你之前的描述：Source=订单ID, Target=知识ID
    df_edges = df_edges.rename(columns={'Source': 'ID', 'Target': 'Knowledge'})

    # --- Step 3: 合并并清洗 ---
    print("3. 正在合并数据...")
    df_merged = pd.merge(df_edges, df_orders, on='ID', how='inner')

    # 清洗知识名称 (去掉前缀)
    df_merged['Knowledge'] = df_merged['Knowledge'].apply(clean_knowledge_name)

    # --- Step 4: 分组统计频次 ---
    print("4. 正在统计年度词频...")
    # 按 Year 和 Knowledge 分组，计算出现的订单数
    hot_words = df_merged.groupby(['Year', 'Knowledge']).size().reset_index(name='Frequency')

    # --- Step 5: 计算排名并输出 ---
    years = sorted(hot_words['Year'].unique())
    all_ranks = []

    print(f"5. 正在生成报表 (涵盖年份: {years})...")

    for year in years:
        # 提取当年的数据
        df_year = hot_words[hot_words['Year'] == year].copy()

        # 按频次降序排列
        df_year = df_year.sort_values(by='Frequency', ascending=False)

        # 计算排名 (Rank)
        df_year['Rank'] = range(1, len(df_year) + 1)

        # 保存单年 Top 100 热词 (方便查阅)
        top_100 = df_year.head(100)
        save_path = os.path.join(output_dir, f"Hot_Words_{year}.csv")
        top_100.to_csv(save_path, index=False, encoding='utf-8-sig')

        # 收集数据用于汇总
        all_ranks.append(df_year)

        # 打印每年的 No.1
        if not top_100.empty:
            top_word = top_100.iloc[0]['Knowledge']
            top_freq = top_100.iloc[0]['Frequency']
            print(f"  - {year}年 Top1: {top_word} (出现 {top_freq} 次)")

    # --- Step 6: 生成演化分析总表 (用于画凹凸图) ---
    if all_ranks:
        df_all = pd.concat(all_ranks)

        # 筛选出至少在某一年进入过 Top 20 的词 (避免数据量太大)
        # 逻辑：只要这个词在任意一年排名前20，就保留它所有年份的数据
        top_keywords = df_all[df_all['Rank'] <= 20]['Knowledge'].unique()
        df_evolution = df_all[df_all['Knowledge'].isin(top_keywords)]

        # 保存总表
        total_path = os.path.join(output_dir, "Hot_Words_Evolution_Top20.csv")
        df_evolution.to_csv(total_path, index=False, encoding='utf-8-sig')

        # 保存一个透视表版本 (Matrix格式，行=词，列=年份，值=排名)，方便 Origin 直接画图
        pivot_rank = df_evolution.pivot_table(index='Knowledge', columns='Year', values='Rank')
        pivot_path = os.path.join(output_dir, "Hot_Words_Rank_Matrix_Origin.csv")
        pivot_rank.to_csv(pivot_path, encoding='utf-8-sig')

        print(f"\n[完成] 结果已保存在: {output_dir}")
        print(f"  - 单年文件: Hot_Words_YYYY.csv")
        print(f"  - 演化分析表: Hot_Words_Evolution_Top20.csv (用于分析趋势)")
        print(f"  - Origin矩阵表: Hot_Words_Rank_Matrix_Origin.csv (用于画凹凸图)")


if __name__ == "__main__":
    calculate_annual_hot_words()