import pandas as pd

# --- 配置参数 ---
single_overlap_file = "D:\\Documents\\Desktop\\一品威客数据采集4月\\test63\\设计师宽度单一重叠度-2014.csv"
overlap_matrix_file = "D:\\Documents\\Desktop\\一品威客数据采集4月\\test63\\设计师重叠度矩阵-2014.csv"

# 新生成的节点和边列表CSV文件的输出路径
nodes_output_file = "D:\\Documents\\Desktop\\一品威客数据采集4月\\test63\\画图64\\designer_nodes-2014.csv"
edges_output_file = "D:\\Documents\\Desktop\\一品威客数据采集4月\\test63\\画图64\\designer_edges-2014.csv"

# 边创建阈值：只有当两个设计师的重叠度大于此值时，才在它们之间创建一条边
# 你可以根据数据的实际分布调整这个值。较高的阈值会产生较稀疏的网络。
OVERLAP_THRESHOLD =1  # 示例值，根据需要调整


# --- 主逻辑 ---
def create_network_data():
    print(f"正在从以下文件加载数据:")
    print(f"  节点属性文件: {single_overlap_file}")
    print(f"  重叠度矩阵文件: {overlap_matrix_file}")

    try:
        # 加载包含节点属性的CSV
        df_single_overlap = pd.read_csv(single_overlap_file, encoding='utf-8-sig')
        # 加载重叠度矩阵CSV，将第一列作为索引
        df_overlap_matrix = pd.read_csv(overlap_matrix_file, encoding='utf-8-sig', index_col=0)
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}. 请检查文件路径是否正确。")
        return
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    # 1. 创建节点列表 (designer_nodes.csv)
    print("\n正在创建节点列表...")
    if not {'设计师', '生态位宽度', '价值位势', '单一重叠度'}.issubset(df_single_overlap.columns):
        print("错误: '设计师宽度单一重叠度-招标雇佣3.csv' 文件中缺少必要的列。")
        print("需要列: '设计师', '生态位宽度', '价值位势', '单一重叠度'")
        return

    df_nodes = df_single_overlap[['设计师', '生态位宽度', '价值位势', '单一重叠度']].copy()
    df_nodes.rename(columns={
        '设计师': 'DesignerID',
        '生态位宽度': 'NicheWidth',
        '价值位势': 'ValuePotential',
        '单一重叠度': 'MeanOverlap'
    }, inplace=True)

    # 处理可能存在的NaN值（例如，如果某个设计师的价值位势无法计算）
    df_nodes['NicheWidth'].fillna(0, inplace=True)
    df_nodes['ValuePotential'].fillna(0, inplace=True)
    df_nodes['MeanOverlap'].fillna(0, inplace=True)

    df_nodes.to_csv(nodes_output_file, index=False, encoding='utf-8-sig')
    print(f"节点列表已保存到: {nodes_output_file}")
    print(f"共找到 {len(df_nodes)} 个节点。")
    print(df_nodes.head())

    # 2. 创建边列表 (designer_edges.csv)
    print("\n正在创建边列表...")
    edges = []
    designers = df_overlap_matrix.index.tolist()  # 获取设计师列表（行名）
    designer_columns = df_overlap_matrix.columns.tolist()  # 获取设计师列表（列名）

    # 确保行索引和列索引代表相同的设计师集合且顺序一致
    if designers != designer_columns:
        print("警告: 重叠度矩阵的行索引和列索引不完全匹配。结果可能不准确。")
        # 可以尝试基于共同的设计师进行处理，但简单起见，这里假设它们应该匹配

    num_designers = len(designers)
    for i in range(num_designers):
        for j in range(i + 1, num_designers):  # 遍历上三角矩阵，避免重复边和自环
            designer1 = designers[i]
            designer2 = designers[j]  # 列名应该与行名对应

            # 确保 designer2 在矩阵的列中
            if designer2 not in df_overlap_matrix.columns:
                # print(f"警告: 设计师 {designer2} 未在重叠度矩阵的列中找到。跳过边 ({designer1}, {designer2}).")
                continue

            overlap_score = df_overlap_matrix.loc[designer1, designer2]

            if pd.isna(overlap_score):  # 跳过NaN值
                continue

            if overlap_score > OVERLAP_THRESHOLD:
                edges.append({
                    'Source': designer1,
                    'Target': designer2,
                    'Weight_Overlap': overlap_score
                })

    df_edges = pd.DataFrame(edges)
    if not df_edges.empty:
        df_edges.to_csv(edges_output_file, index=False, encoding='utf-8-sig')
        print(f"边列表已保存到: {edges_output_file}")
        print(f"根据阈值 > {OVERLAP_THRESHOLD}，共找到 {len(df_edges)} 条边。")
        print(df_edges.head())
    else:
        print(f"根据阈值 > {OVERLAP_THRESHOLD}，未找到任何边。尝试降低阈值或检查重叠度数据。")
        # 即使没有边，也创建一个空的edges文件，列名正确
        pd.DataFrame(columns=['Source', 'Target', 'Weight_Overlap']).to_csv(edges_output_file, index=False,
                                                                            encoding='utf-8-sig')
        print(f"空的边列表文件已创建: {edges_output_file}")

    print("\n处理完成！")
    print("现在你可以使用 'designer_nodes.csv' 和 'designer_edges.csv' 文件在网络可视化工具中构建你的三维价值网络。")


if __name__ == "__main__":
    create_network_data()