import json
from neo4j import GraphDatabase

# 连接 Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "wangyunqing"))

# 动态创建约束
def create_constraints(tx, unique_types):
    for label in unique_types:
        # 清理标签名，确保合法（替换非法字符，Neo4j 标签不能包含空格或特殊字符）
        clean_label = "".join(c if c.isalnum() or c == "_" else "_" for c in label)
        if clean_label:
            tx.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{clean_label}) REQUIRE n.name IS UNIQUE")
            print(f"创建约束: {clean_label}")
    print("所有约束已创建")

# 创建节点
def create_nodes(tx, task):
    if "提取实体" not in task or not task["提取实体"]:
        task_id = task.get("需求方", task.get("需求文本", "未知任务"))
        print(f"任务 {task_id} 缺少提取实体字段或为空，跳过")
        return

    if "structured_data" not in task["提取实体"] or "entities" not in task["提取实体"]["structured_data"]:
        task_id = task.get("需求方", task.get("需求文本", "未知任务"))
        print(f"任务 {task_id} 缺少 structured_data 或 entities，跳过")
        return

    entities = task["提取实体"]["structured_data"]["entities"]
    for entity in entities:
        entity_id = entity.get("id")
        entity_type = entity.get("type")

        if not entity_id or not entity_type:
            task_id = task.get("需求方", task.get("需求文本", "未知任务"))
            print(f"任务 {task_id} 的实体缺少 id 或 type，跳过: {entity}")
            continue

        # 清理标签名，确保 Neo4j 标签合法
        clean_label = "".join(c if c.isalnum() or c == "_" else "_" for c in entity_type)
        if not clean_label:
            task_id = task.get("需求方", task.get("需求文本", "未知任务"))
            print(f"任务 {task_id} 的实体类型 {entity_type} 无效，跳过")
            continue

        # 创建节点
        query = f"""
        MERGE (n:{clean_label} {{name: $name}})
        """
        tx.run(query, name=entity_id)
        print(f"创建节点: {clean_label}({entity_id})")

# 创建关系
def create_relationships(tx, task):
    if "提取实体" not in task or not task["提取实体"]:
        task_id = task.get("需求方", task.get("需求文本", "未知任务"))
        print(f"任务 {task_id} 缺少提取实体字段或为空，跳过")
        return

    if "structured_data" not in task["提取实体"] or "relations" not in task["提取实体"]["structured_data"]:
        task_id = task.get("需求方", task.get("需求文本", "未知任务"))
        print(f"任务 {task_id} 缺少 structured_data 或 relations，跳过")
        return

    relationships = task["提取实体"]["structured_data"]["relations"]
    entities = task["提取实体"]["structured_data"]["entities"]

    for rel in relationships:
        source = rel.get("source")
        target = rel.get("target")
        relation = rel.get("relation", "").upper()

        if not source or not target or not relation:
            task_id = task.get("需求方", task.get("需求文本", "未知任务"))
            print(f"任务 {task_id} 的关系缺少 source, target 或 relation，跳过: {rel}")
            continue

        # 清理关系类型，确保 Neo4j 关系类型合法
        clean_relation = "".join(c if c.isalnum() or c == "_" else "_" for c in relation)
        if not clean_relation:
            task_id = task.get("需求方", task.get("需求文本", "未知任务"))
            print(f"任务 {task_id} 的关系类型 {relation} 无效，跳过")
            continue

        # 查找源节点和目标节点的标签
        source_label = target_label = None
        for entity in entities:
            if entity.get("id") == source:
                source_label = entity.get("type")
            if entity.get("id") == target:
                target_label = entity.get("type")

        if not source_label or not target_label:
            task_id = task.get("需求方", task.get("需求文本", "未知任务"))
            print(f"任务 {task_id} 的关系 {source} -[{relation}]-> {target} 的源或目标标签未找到，跳过")
            continue

        # 清理标签名，确保 Neo4j 标签合法
        clean_source_label = "".join(c if c.isalnum() or c == "_" else "_" for c in source_label)
        clean_target_label = "".join(c if c.isalnum() or c == "_" else "_" for c in target_label)
        if not clean_source_label or not clean_target_label:
            task_id = task.get("需求方", task.get("需求文本", "未知任务"))
            print(f"任务 {task_id} 的关系 {source} -[{relation}]-> {target} 的标签无效，跳过")
            continue

        # 创建关系
        query = f"""
        MATCH (s:{clean_source_label} {{name: $source}})
        MATCH (t:{clean_target_label} {{name: $target}})
        MERGE (s)-[r:{clean_relation}]->(t)
        """
        tx.run(query, source=source, target=target)
        print(f"创建关系: {source} -[{clean_relation}]-> {target} (原始关系类型: {relation})")

# 导入数据
def import_to_neo4j(file_path):
    data = []
    unique_types = set()  # 收集所有唯一的 type 值

    # 读取 JSON 数据并收集唯一的 type
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                    # 提取实体中的 type
                    if "提取实体" in json_obj and json_obj["提取实体"] and \
                       "structured_data" in json_obj["提取实体"] and \
                       "entities" in json_obj["提取实体"]["structured_data"]:
                        for entity in json_obj["提取实体"]["structured_data"]["entities"]:
                            entity_type = entity.get("type")
                            if entity_type:
                                # 清理标签名
                                clean_label = "".join(c if c.isalnum() or c == "_" else "_" for c in entity_type)
                                if clean_label:
                                    unique_types.add(clean_label)
                except json.JSONDecodeError as e:
                    print(f"解析错误在行: {line}, 错误: {e}")

    with driver.session() as session:
        # 动态创建约束
        session.execute_write(create_constraints, unique_types)

        # 导入节点和关系
        for task in data:
            task_id = task.get("需求方", task.get("需求文本", "未知任务"))
            print(f"处理任务: {task_id}")
            session.execute_write(create_nodes, task)
            session.execute_write(create_relationships, task)
        print("数据导入完成！")

# 执行
import_to_neo4j("D:/Desktop/CID/抽取结果/2-添加时间后/SortedByYear/2015.json")
driver.close()