import pandas as pd
import json
import os

# --- 基本目录配置 ---
base_dir = "D:/Documents/Desktop/Crowd_intelligence_wyq/Knowledge_network/抽取输出/2-添加时间后/SortedByYearNEW"
output_dir = "D:/Documents/Desktop/Crowd_intelligence_wyq/Knowledge_network/知识系统/知识层节点和边/"


# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 处理单个年份的JSON数据并生成知识节点和边
def process_year_data(year):
    print(f"--- 处理 {year} 年的数据 ---")

    # 输入文件路径
    json_file_path = os.path.join(base_dir, f"{year}.json")
    if not os.path.exists(json_file_path):
        print(f"警告: 文件 {json_file_path} 不存在，跳过此年数据处理。")
        return

    # 读取JSON数据
    with open(json_file_path, 'r', encoding='utf-8-sig') as f:
        data = [json.loads(line) for line in f.readlines()]

    knowledge_nodes = []  # 知识层节点
    knowledge_edges = []  # 知识层到订单层的边
    order_nodes = []  # 订单层节点
    knowledge_relations = []  # 知识之间的边

    # 生成订单层节点（从数据中提取任务编号、发布时间等）
    for record in data:
        task_id = record['任务编号']  # 直接使用原始的任务编号（不做任何提取）
        publication_time = pd.to_datetime(record['发布时间'], errors='coerce').strftime('%Y-%m-%d')
        order_nodes.append([task_id, publication_time, '订单', 1])  # 订单层节点：任务编号, 发布时间, 类型和[z]

        # 提取知识实体
        entities = record['提取实体']['structured_data']['entities']
        if entities:  # 确保有实体
            for entity in entities:
                entity_name = entity['id']
                # 只保留知识节点ID和Name，忽略entity_type
                knowledge_nodes.append([f"knowledge:{entity_name}", entity_name, '知识', 0])  # 添加 "knowledge:" 前缀

                # 为每个任务编号和实体创建边
                knowledge_edges.append([task_id, f"knowledge:{entity_name}", '关联'])

        # 提取关系
        relations = record['提取实体']['structured_data']['relations']
        if relations:  # 检查 relations 是否为空
            for relation in relations:
                # 确保关系有 source、target 和 relation 字段
                source_entity = relation.get('source')
                target_entity = relation.get('target')
                relation_type = relation.get('relation')

                if source_entity and target_entity and relation_type:
                    knowledge_relations.append([f"knowledge:{source_entity}", f"knowledge:{target_entity}", relation_type])
                else:
                    print(f"警告：任务编号 {task_id} 中的关系数据不完整，跳过该关系。")

        else:
            print(f"警告：任务编号 {task_id} 中没有提取到关系。")

    # 转化为 DataFrame
    order_nodes_df = pd.DataFrame(order_nodes, columns=['ID', 'Publication_Time', 'type', '[z]'])
    knowledge_nodes_df = pd.DataFrame(knowledge_nodes, columns=['ID', 'Name', 'Type', '[z]'])
    knowledge_edges_df = pd.DataFrame(knowledge_edges, columns=['Source', 'Target', 'Type'])
    knowledge_relations_df = pd.DataFrame(knowledge_relations, columns=['Source', 'Target', 'Type'])

    # 处理知识节点的去重（合并重复的节点）
    knowledge_nodes_df = knowledge_nodes_df.groupby(['ID', 'Name', 'Type'], as_index=False).agg(
        {'[z]': 'first'}  # 对于重复的知识节点，保留第一个 [z] 值（也可以根据需要进行合并）
    )

    # 创建年份子目录
    year_output_dir = os.path.join(output_dir, str(year))
    if not os.path.exists(year_output_dir):
        os.makedirs(year_output_dir)

    # 输出文件路径（按年份存储）
    order_nodes_output_path = os.path.join(year_output_dir, f"订单层-{year}.csv")
    knowledge_nodes_output_path = os.path.join(year_output_dir, f"知识层-{year}.csv")
    knowledge_edges_output_path = os.path.join(year_output_dir, f"知识层到订单层边-{year}.csv")
    knowledge_relations_output_path = os.path.join(year_output_dir, f"知识层之间的边-{year}.csv")

    # 保存为 CSV 文件
    order_nodes_df.to_csv(order_nodes_output_path, index=False, encoding='utf-8-sig')
    knowledge_nodes_df.to_csv(knowledge_nodes_output_path, index=False, encoding='utf-8-sig')
    knowledge_edges_df.to_csv(knowledge_edges_output_path, index=False, encoding='utf-8-sig')
    knowledge_relations_df.to_csv(knowledge_relations_output_path, index=False, encoding='utf-8-sig')

    print(f"--- {year} 年的数据处理完成，文件已保存 ---")


# 遍历所有年份并处理
years = [str(year) for year in range(2014, 2026)]
for year in years:
    process_year_data(year)

print("--- 所有年份的数据处理完毕 ---")


