from neo4j import GraphDatabase


def clear_database(uri, username, password):
    try:
        # 建立连接
        driver = GraphDatabase.driver(uri, auth=(username, password))

        with driver.session() as session:
            # 删除所有节点和关系
            session.run("MATCH (n) DETACH DELETE n")

            # 删除所有索引和约束
            session.run("CALL apoc.schema.assert({}, {})")

        print("数据库清理完成！")

    except Exception as e:
        print(f"发生错误：{str(e)}")

    finally:
        # 关闭连接
        driver.close()


# 使用示例
uri = "neo4j://localhost:7687"
username = "neo4j"
password = "wangyunqing"

clear_database(uri, username, password)
