import json
import pandas as pd
import re

# JSON 文件路径
json_file_path = r'D:/Documents/Desktop/ner/entities_extracted_test.json'

# 读取 JSON 文件
json_data = []
with open(json_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        json_data.append(json.loads(line))

# 初始化节点和边列表
nodes = []
edges = []

# 用于存储所有雇主、实体和任务类型（避免重复）
all_employers = set()
all_entities = set()
all_task_types = set()

# 定义提取任务类型的函数
def extract_task_type(task):
    # 假设任务类型从标题或其他字段中提取
    # 这里假设 JSON 中有 "标题" 字段，若无，请替换为实际字段
    title = task.get("标题", "")
    match = re.search(r'【(.*?)】', title)
    if match:
        return match.group(1)
    # 如果标题中没有【】，尝试从 category 或其他字段推导
    return task.get("category", "未知任务类型")

# 遍历 JSON 数据
for task in json_data:
    # 获取雇主信息
    employer = task.get("employer")
    # 获取实体和关系数据
    entities = task.get("提取实体", {}).get("structured_data", {}).get("entities", [])
    relations = task.get("提取实体", {}).get("structured_data", {}).get("relations", [])
    # 获取任务类型
    task_type = extract_task_type(task)

    # 添加雇主节点
    if employer and employer not in all_employers:
        nodes.append({
            "Id": employer,
            "Label": employer,
            "Layer": "employer",
            "Degree[z]": "4"
        })
        all_employers.add(employer)

    # 添加任务类型节点
    if task_type and task_type not in all_task_types:
        nodes.append({
            "Id": task_type,
            "Label": task_type,
            "Layer": "task_type",
            "Degree[z]": "2"
        })
        all_task_types.add(task_type)

    # 添加实体节点
    for entity in entities:
        entity_id = entity.get("id")
        if entity_id and entity_id not in all_entities:
            nodes.append({
                "Id": entity_id,
                "Label": entity_id,
                "Layer": "entity",
                "Degree[z]": "3"
            })
            all_entities.add(entity_id)

    # 添加雇主与实体的“雇佣”关系
    for entity in entities:
        entity_id = entity.get("id")
        if employer and entity_id:
            edges.append({
                "Source": employer,
                "Target": entity_id,
                "Type": "Directed",
                "Label": "雇佣"
            })

    # 添加实体与任务类型的“属于”关系
    for entity in entities:
        entity_id = entity.get("id")
        if entity_id and task_type:
            edges.append({
                "Source": entity_id,
                "Target": task_type,
                "Type": "Directed",
                "Label": "属于"
            })

    # 添加实体之间的关系
    for relation in relations:
        source = relation.get("source")
        target = relation.get("target")
        rel_type = relation.get("relation")
        if source and target and rel_type:
            edges.append({
                "Source": source,
                "Target": target,
                "Type": "Directed",
                "Label": rel_type
            })

# 转换为 DataFrame
nodes_df = pd.DataFrame(nodes)
edges_df = pd.DataFrame(edges)

# 保存为 CSV 文件
nodes_csv_path = r'D:\Documents\Desktop\ner\Knowledge-nodes.csv'
edges_csv_path = r'D:\Documents\Desktop\ner\Knowledge-edges.csv'
nodes_df.to_csv(nodes_csv_path, index=False, encoding="utf-8-sig")
edges_df.to_csv(edges_csv_path, index=False, encoding="utf-8-sig")

print("节点和边文件已生成：")
print(f"节点文件：{nodes_csv_path}")
print(f"边文件：{edges_csv_path}")