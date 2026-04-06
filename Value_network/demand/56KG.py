#绘制需求方知识图谱
import json
from neo4j import GraphDatabase
import math

# 连接 Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "wangyunqing"))

# 允许的实体类型
ALLOWED_ENTITY_TYPES = {
    "设计对象": "DesignObject",
    "设计功能": "DesignFunction",
    "设计风格": "DesignStyle",
    "目标用户": "TargetUser",
    "区域限定": "Region",
    "第三方融合": "ThirdParty",
    "工具与技术": "Tech",
    "材质与工艺": "Material",
    "物理约束": "PhysicalConstraint",
    "资源约束": "SourceConstraint",
    "设计要求": "DesignRequirement",
    "技术栈实体": "Tech"  # 映射“技术栈实体”到“工具与技术”
}

# 创建约束
def create_constraints(tx):
    # 为需求方、订单和实体类型创建唯一约束
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Client) REQUIRE c.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (o:Order) REQUIRE o.task_id IS UNIQUE")
    for label in ALLOWED_ENTITY_TYPES.values():
        tx.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")

# 创建需求方和订单节点，并关联到现有 Category 节点
def create_hierarchy_nodes(tx, task):
    title = task.get("标题", "未知标题")

    # 获取字段并清理
    task_type = task.get("节点")
    client_name = task.get("需求方")
    task_id = task.get("任务编号")

    # 类型检查和 NaN 处理
    def is_valid_string(value):
        if value is None or isinstance(value, float) and math.isnan(value):
            return False
        return isinstance(value, str) and value.strip()

    # 调试日志
    print(f"任务 {title}: 节点={task_type}, 需求方={client_name}, 任务编号={task_id}")

    # 检查字段是否有效
    if not is_valid_string(task_type) or not is_valid_string(client_name) or not is_valid_string(task_id):
        print(f"任务 {title} 字段无效（节点={task_type}, 需求方={client_name}, 任务编号={task_id}），跳过")
        return None

    # 清理任务编号，去除前缀
    if "任务编号：" in task_id:
        task_id = task_id.replace("任务编号：", "").strip()

    # 查找现有 Category 节点
    result = tx.run("""
    MATCH (c:Category {name: $name})
    RETURN c
    """, name=task_type)
    category_node = result.single()
    if not category_node:
        print(f"任务 {title} 的任务类型（name='{task_type}'）在图谱中不存在，跳过")
        return None

    # 创建或合并需求方节点，并连接到 Category
    tx.run("""
    MERGE (cl:Client {name: $client_name})
    WITH cl
    MATCH (c:Category {name: $task_type})
    MERGE (c)-[:HAS_CLIENT]->(cl)
    """, client_name=client_name, task_type=task_type)

    # 创建或合并订单节点，并连接到需求方，存储订单属性
    order_attributes = {
        "title": title,
        "demand_price": task.get("需求价格", ""),
        "transaction_price": task.get("交易价格", ""),
        "status": task.get("状态", ""),
        "publish_time": task.get("发布时间", ""),
        "complete_time": task.get("完成时间", ""),
        "work_cycle": task.get("工作周期", ""),
        "designer": task.get("设计师", ""),
        "designer_location": task.get("设计师地点", "")
    }
    tx.run("""
    MERGE (o:Order {task_id: $task_id})
    SET o += $attributes
    WITH o
    MATCH (cl:Client {name: $client_name})
    MERGE (cl)-[:PLACED_ORDER]->(o)
    """, task_id=task_id, attributes=order_attributes, client_name=client_name)

    return task_id

# 创建实体节点并关联到订单
def create_entities(tx, task, task_id):
    if not task_id:
        return

    title = task.get("标题", "未知标题")

    # 检查提取实体字段
    if "提取实体" not in task or not task["提取实体"]:
        print(f"任务 {title} 缺少提取实体字段或为空，跳过实体创建")
        return

    # 检查 structured_data 和 entities
    if "structured_data" not in task["提取实体"] or "entities" not in task["提取实体"]["structured_data"]:
        print(f"任务 {title} 缺少 structured_data 或 entities，跳过实体创建")
        return

    entities = task["提取实体"]["structured_data"]["entities"]
    for entity in entities:
        entity_id = entity["id"]
        entity_type = entity["type"]

        # 跳过不在允许列表中的实体类型
        if entity_type not in ALLOWED_ENTITY_TYPES:
            print(f"任务 {title} 的实体类型 {entity_type} 不在允许列表中，跳过实体 {entity_id}")
            continue

        attributes = entity.get("attributes", {})
        label = ALLOWED_ENTITY_TYPES[entity_type]

        # 创建或合并实体节点，并连接到订单
        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET n += $attributes
        WITH n
        MATCH (o:Order {{task_id: $task_id}})
        MERGE (o)-[:CONTAINS_ENTITY]->(n)
        """
        tx.run(query, name=entity_id, attributes=attributes, task_id=task_id)
        print(f"任务 {title}: 创建实体节点 {label}({entity_id}) 并关联到订单 {task_id}")

# 创建实体之间的关系
def create_relationships(tx, task, task_id):
    if not task_id:
        return

    title = task.get("标题", "未知标题")

    # 检查提取实体字段
    if "提取实体" not in task or not task["提取实体"]:
        print(f"任务 {title} 缺少提取实体字段或为空，跳过关系创建")
        return

    # 检查 structured_data 和 relations
    if "structured_data" not in task["提取实体"] or "relations" not in task["提取实体"]["structured_data"]:
        print(f"任务 {title} 缺少 structured_data 或 relations，跳过关系创建")
        return

    relationships = task["提取实体"]["structured_data"]["relations"]
    entities = task["提取实体"]["structured_data"]["entities"]

    for rel in relationships:
        source = rel["source"]
        target = rel["target"]
        relation = rel["relation"].upper()

        # 查找源节点和目标节点的标签
        source_label = None
        target_label = None
        for entity in entities:
            if entity["id"] == source:
                source_label = ALLOWED_ENTITY_TYPES.get(entity["type"], None)
            if entity["id"] == target:
                target_label = ALLOWED_ENTITY_TYPES.get(entity["type"], None)

        if not source_label or not target_label:
            print(f"任务 {title}: 关系 {source} -[{relation}]-> {target} 的源或目标标签未找到或无效，跳过")
            continue

        # 创建关系
        query = f"""
        MATCH (s:{source_label} {{name: $source}})
        MATCH (t:{target_label} {{name: $target}})
        MERGE (s)-[r:{relation}]->(t)
        """
        tx.run(query, source=source, target=target)
        print(f"任务 {title}: 创建关系 {source} -[{relation}]-> {target}")

# 导入数据
def import_to_neo4j(file_path):
    data = []
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:  # 使用 utf-8-sig 处理 BOM
            for line in f:
                if line.strip():
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"解析错误在行: {line}, 错误: {e}")
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径")
        return
    except UnicodeDecodeError:
        print(f"文件 {file_path} 编码错误，请确保文件为 UTF-8 编码")
        return

    with driver.session() as session:
        # 创建约束
        session.execute_write(create_constraints)

        # 导入节点和关系
        for task in data:
            title = task.get("标题", "未知标题")
            if not title:
                print("跳过空记录")
                continue

            print(f"处理任务: {title}")
            # 创建需求方、订单节点并关联到现有 Category
            task_id = session.execute_write(create_hierarchy_nodes, task)
            if not task_id:
                continue

            # 创建实体节点并关联到订单
            session.execute_write(create_entities, task, task_id)
            # 创建实体之间的关系
            session.execute_write(create_relationships, task, task_id)
        print("数据导入完成！")

# 执行
import_to_neo4j("D:/Documents/Desktop/ner/56test/56.json")
driver.close()