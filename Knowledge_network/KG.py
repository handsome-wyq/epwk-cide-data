#构建neo4j知识网络
import json
from neo4j import GraphDatabase

# 连接 Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "wangyunqing"))

# 创建约束
def create_constraints(tx):
    # 为每种实体类型创建唯一约束
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Demand) REQUIRE d.name IS UNIQUE")  # 需求本体
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (do:DesignObject) REQUIRE do.name IS UNIQUE")  # 设计对象
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE")  # 行业领域
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (tu:TargetUser) REQUIRE tu.name IS UNIQUE")  # 目标用户
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE")  # 区域限定
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (tp:ThirdParty) REQUIRE tp.name IS UNIQUE")  # 第三方融合
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tech) REQUIRE t.name IS UNIQUE")  # 工具与技术
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Material) REQUIRE m.name IS UNIQUE")  # 材质与工艺
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (tc:TimeConstraint) REQUIRE tc.name IS UNIQUE")  # 时间窗
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (pc:PhysicalConstraint) REQUIRE pc.name IS UNIQUE")  # 物理约束
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (dr:DesignRequirement) REQUIRE dr.name IS UNIQUE")  # 设计要求

# 创建节点
def create_nodes(tx, task):
    # 检查提取实体字段是否存在
    if "提取实体" not in task or not task["提取实体"]:
        print(f"任务 {task.get('标题', '未知标题')} 缺少提取实体字段或为空，跳过")
        return

    # 检查 structured_data 和 entities 是否存在
    if "structured_data" not in task["提取实体"] or "entities" not in task["提取实体"]["structured_data"]:
        print(f"任务 {task.get('标题', '未知标题')} 缺少 structured_data 或 entities，跳过")
        return

    entities = task["提取实体"]["structured_data"]["entities"]
    for entity in entities:
        entity_id = entity["id"]
        entity_type = entity["type"]

        attributes = entity.get("attributes", {})

        # 根据实体类型映射到 Neo4j 标签
        label_map = {
            "需求本体": "Demand",
            "设计功能": "DesignFunction",
            "设计对象": "DesignObject",
            "设计风格": "DesignStyle",
            "行业领域": "Industry",
            "目标用户": "TargetUser",
            "区域限定": "Region",
            "第三方融合": "ThirdParty",
            "工具与技术": "Tech",
            "材质与工艺": "Material",
            "时间窗": "TimeConstraint",
            "物理约束": "PhysicalConstraint",
            "资源约束": "SourceConstraint",
            "设计要求": "DesignRequirement"
        }
        label = label_map.get(entity_type, "Unknown")  # 默认标签为 Unknown

        # 构建 Cypher 查询，创建节点并设置属性
        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET n += $attributes
        """
        tx.run(query, name=entity_id, attributes=attributes)
        print(f"创建节点: {label}({entity_id})")

# 创建关系
def create_relationships(tx, task):
    # 检查提取实体字段是否存在
    if "提取实体" not in task or not task["提取实体"]:
        print(f"任务 {task.get('标题', '未知标题')} 缺少提取实体字段或为空，跳过")
        return

    # 检查 structured_data 和 relations 是否存在
    if "structured_data" not in task["提取实体"] or "relations" not in task["提取实体"]["structured_data"]:
        print(f"任务 {task.get('标题', '未知标题')} 缺少 structured_data 或 relations，跳过")
        return

    relationships = task["提取实体"]["structured_data"]["relations"]
    entities = task["提取实体"]["structured_data"]["entities"]

    for rel in relationships:
        source = rel["source"]
        target = rel["target"]
        relation = rel["relation"].upper()  # 关系类型大写（如 DESIGN、面向）

        # 查找源节点和目标节点的标签
        source_label = None
        target_label = None
        for entity in entities:
            if entity["id"] == source:
                source_label = {
                    "需求本体": "Demand",
                    "设计功能": "DesignFunction",
                    "设计对象": "DesignObject",
                    "设计风格": "DesignStyle",
                    "行业领域": "Industry",
                    "目标用户": "TargetUser",
                    "区域限定": "Region",
                    "第三方融合": "ThirdParty",
                    "工具与技术": "Tech",
                    "材质与工艺": "Material",
                    "时间窗": "TimeConstraint",
                    "物理约束": "PhysicalConstraint",
                    "资源约束": "SourceConstraint",
                    "设计要求": "DesignRequirement"
                }.get(entity["type"], "Unknown")
            if entity["id"] == target:
                target_label = {
                    "需求本体": "Demand",
                    "设计功能": "DesignFunction",
                    "设计对象": "DesignObject",
                    "设计风格": "DesignStyle",
                    "行业领域": "Industry",
                    "目标用户": "TargetUser",
                    "区域限定": "Region",
                    "第三方融合": "ThirdParty",
                    "工具与技术": "Tech",
                    "材质与工艺": "Material",
                    "时间窗": "TimeConstraint",
                    "物理约束": "PhysicalConstraint",
                    "资源约束": "SourceConstraint",
                    "设计要求": "DesignRequirement"
                }.get(entity["type"], "Unknown")

        if not source_label or not target_label:
            print(f"关系 {source} -[{relation}]-> {target} 的源或目标标签未找到，跳过")
            continue

        # 创建关系
        query = f"""
        MATCH (s:{source_label} {{name: $source}})
        MATCH (t:{target_label} {{name: $target}})
        MERGE (s)-[r:{relation}]->(t)
        """
        tx.run(query, source=source, target=target)
        print(f"创建关系: {source} -[{relation}]-> {target}")

# 导入数据
def import_to_neo4j(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"解析错误在行: {line}, 错误: {e}")

    with driver.session() as session:
        # 创建约束
        session.execute_write(create_constraints)

        # 导入节点和关系
        for task in data:
            if task.get("标题") is not None:  # 跳过空记录
                print(f"处理任务: {task.get('标题', '未知标题')}")
                print(f"任务数据: {json.dumps(task, ensure_ascii=False, indent=2)}")
                session.execute_write(create_nodes, task)
                session.execute_write(create_relationships, task)
        print("数据导入完成！")

# 执行
'''import_to_neo4j("D:/Documents/Desktop/ner/420test_output/420test_output.json")'''
import_to_neo4j("D:/Documents/Desktop/ner/420test_output/updated_420test_output.json")
driver.close()