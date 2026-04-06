#在neo4j中绘制标签图
from neo4j import GraphDatabase
import csv

# Neo4j 数据库连接配置
URI = "bolt://localhost:7687"  # 替换为你的 Neo4j 实例地址
USERNAME = "neo4j"             # 替换为你的用户名
PASSWORD = "wangyunqing"     # 替换为你的密码

# 文件路径
NODE_FILE = "capability-nodes.csv"
RELATION_FILE = "capability.csv"

# 连接到 Neo4j 数据库
class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.driver.close()

    # 批量创建节点
    def create_nodes_batch(self, batch_data):
        with self.driver.session() as session:
            session.run(
                """
                UNWIND $batch AS row
                MERGE (c:Category {name: row.name})
                FOREACH (d IN CASE WHEN row.degree = 0 THEN [1] ELSE [] END | SET c:Layer0)
                FOREACH (d IN CASE WHEN row.degree = 1 THEN [1] ELSE [] END | SET c:Layer1)
                FOREACH (d IN CASE WHEN row.degree = 2 THEN [1] ELSE [] END | SET c:Layer2)
                FOREACH (d IN CASE WHEN row.degree = 3 THEN [1] ELSE [] END | SET c:Layer3)
                """,
                batch=batch_data
            )

    # 批量创建关系
    def create_relations_batch(self, batch_data):
        with self.driver.session() as session:
            session.run(
                """
                UNWIND $batch AS row
                MATCH (parent:Category {name: row.head})
                MATCH (child:Category {name: row.tail})
                MERGE (parent)-[:HAS_SUBCATEGORY]->(child)
                """,
                batch=batch_data
            )

# 读取节点数据 (CSV)
def load_nodes_to_neo4j(node_file, conn, batch_size=100):
    batch_data = []
    try:
        with open(node_file, newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            print(f"节点 CSV 表头: {reader.fieldnames}")

            if 'ID' not in reader.fieldnames or 'degree' not in reader.fieldnames:
                raise ValueError("节点 CSV 文件必须包含 'ID' 和 'degree' 列")

            for row in reader:
                try:
                    name = row['ID'].strip()
                    degree = int(row['degree'])
                    if name and degree in [0, 1, 2, 3]:
                        batch_data.append({"name": name, "degree": degree})
                        print(f"收集节点: {name} (degree: {degree})")
                    else:
                        print(f"跳过无效节点: {row}")
                except (KeyError, ValueError) as e:
                    print(f"节点行错误: {row}. 原因: {e}")
                    continue

                if len(batch_data) >= batch_size:
                    conn.create_nodes_batch(batch_data)
                    batch_data = []

            if batch_data:
                conn.create_nodes_batch(batch_data)

    except FileNotFoundError:
        print(f"错误: 未找到节点 CSV 文件 '{node_file}'")
    except UnicodeDecodeError:
        print("错误: 节点 CSV 文件编码不是 UTF-8。请确保文件使用 UTF-8 编码保存")
    except Exception as e:
        print(f"读取节点 CSV 文件时发生错误: {e}")

# 读取关系数据 (CSV)
def load_relations_to_neo4j(relation_file, conn, batch_size=100):
    batch_data = []
    try:
        with open(relation_file, newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            print(f"关系 CSV 表头: {reader.fieldnames}")

            if 'head' not in reader.fieldnames or 'tail' not in reader.fieldnames:
                raise ValueError("关系 CSV 文件必须包含 'head' 和 'tail' 列")

            for row in reader:
                try:
                    head = row['head'].strip()
                    tail = row['tail'].strip()
                    if head and tail:
                        batch_data.append({"head": head, "tail": tail})
                        print(f"收集关系: {head} -> {tail}")
                    else:
                        print(f"跳过无效关系: {row}")
                except KeyError as e:
                    print(f"关系行错误: {row}. 缺失键: {e}")
                    continue

                if len(batch_data) >= batch_size:
                    conn.create_relations_batch(batch_data)
                    batch_data = []

            if batch_data:
                conn.create_relations_batch(batch_data)

    except FileNotFoundError:
        print(f"错误: 未找到关系 CSV 文件 '{relation_file}'")
    except UnicodeDecodeError:
        print("错误: 关系 CSV 文件编码不是 UTF-8。请确保文件使用 UTF-8 编码保存")
    except Exception as e:
        print(f"读取关系 CSV 文件时发生错误: {e}")

# 主函数
def main():
    try:
        conn = Neo4jConnection(URI, USERNAME, PASSWORD)
        try:
            # 先创建节点
            print("从 capability-nodes.csv 加载节点...")
            load_nodes_to_neo4j(NODE_FILE, conn)
            # 再创建关系
            print("\n从 capability.csv 加载关系...")
            load_relations_to_neo4j(RELATION_FILE, conn)
            print("三层知识图谱成功加载到 Neo4j。")
        finally:
            conn.close()
    except Exception as e:
        print(f"连接 Neo4j 时发生错误: {e}")

if __name__ == "__main__":
    main()