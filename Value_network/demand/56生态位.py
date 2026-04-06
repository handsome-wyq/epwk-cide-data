import json
import numpy as np
from scipy import sparse
from collections import defaultdict
from itertools import combinations
from joblib import Parallel, delayed
import math
import time

# 计时开始
start_time = time.time()
data = []
with open('56.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # 跳过空行
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"解析第 {f.tell()} 行失败: {e}")
                continue

# 打印解析结果
print(f"共解析 {len(data)} 个 JSON 对象")

# 统计需求方及其订单
orders_by_client = defaultdict(list)
for item in data:
    client = item.get("需求方")
    order_info = {
        "任务编号": item.get("任务编号"),
        "标题": item.get("标题"),
        "节点": item.get("节点"),
        "交易价格": item.get("交易价格"),
        "发布时间": item.get("发布时间"),
        "完成时间": item.get("完成时间"),
        "工作周期": item.get("工作周期")
    }
    orders_by_client[client].append(order_info)

# 输出需求方及其订单
print("所有需求者及其订单：")
for client, orders in orders_by_client.items():
    print(f"\n需求方: {client}")
    print(f"订单数: {len(orders)}")
    for order in orders:
        print(f"  - 任务编号: {order['任务编号']}")
        print(f"    标题: {order['标题']}")
        print(f"    节点: {order['节点']}")
        print(f"    交易价格: {order['交易价格']}")
        print(f"    发布时间: {order['发布时间']}")
        print(f"    完成时间: {order['完成时间']}")
        print(f"    工作周期: {order['工作周期']}")
# 步骤 1：预处理数据，提取具体实体
entity_ids_by_client = defaultdict(lambda: defaultdict(int))
all_entity_ids = set()
for item in data:
    client = item.get("需求方")
    entities = item.get("提取实体", {}).get("structured_data", {}).get("entities", [])
    for entity in entities:
        entity_id = entity.get("id")
        entity_ids_by_client[client][entity_id] += 1
        all_entity_ids.add(entity_id)


# 转换为列表和索引
clients = list(entity_ids_by_client.keys())
entity_id_list = list(all_entity_ids)
entity_id_index = {eid: i for i, eid in enumerate(entity_id_list)}
print("完成步骤一")
# 步骤 2：构建稀疏矩阵
rows, cols, values = [], [], []
for i, client in enumerate(clients):
    total_entities = sum(entity_ids_by_client[client].values())
    for entity_id, count in entity_ids_by_client[client].items():
        rows.append(i)
        cols.append(entity_id_index[entity_id])
        values.append(count / total_entities)  # 预计算比例 p_i
entity_matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(clients), len(entity_id_list)))
print("完成步骤2")
# 步骤 3：计算生态位宽度
niche_width = {}
for i, client in enumerate(clients):
    p_i = entity_matrix[i].data  # 非零比例
    shannon_index = -np.sum(p_i * np.log(p_i)) if p_i.size > 0 else 0
    niche_width[client] = shannon_index
print("完成步骤3")
# 步骤 4：并行计算生态位重叠度
def compute_overlap(i, j, matrix):
    p_j = matrix[i].toarray().ravel()
    p_k = matrix[j].toarray().ravel()
    numerator = np.dot(p_j, p_k)
    denominator = np.sqrt(np.sum(p_j ** 2) * np.sum(p_k ** 2))
    return (i, j, numerator / denominator if denominator != 0 else 0)

n_jobs = -1  # 使用所有可用CPU核心
overlap_results = Parallel(n_jobs=n_jobs)(
    delayed(compute_overlap)(i, j, entity_matrix)
    for i, j in combinations(range(len(clients)), 2)
)

# 整理重叠度结果
niche_overlap = {
    (clients[i], clients[j]): overlap
    for i, j, overlap in overlap_results
}
print("完成步骤4")
# 步骤 5：输出结果
print("生态位宽度（基于具体实体）：")
for client, width in sorted(niche_width.items(), key=lambda x: x[1], reverse=True):
    print(f"{client}: {width:.4f}")

print("\n生态位重叠度（基于具体实体，部分）：")
for (client_j, client_k), overlap in sorted(niche_overlap.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{client_j} 与 {client_k}: {overlap:.4f}")

# 计时结束
print(f"\n总耗时: {time.time() - start_time:.2f} 秒")