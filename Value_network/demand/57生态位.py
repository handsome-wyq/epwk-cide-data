#计算需求方的生态位
import json
import numpy as np
from scipy import sparse
from collections import defaultdict
from itertools import combinations
from joblib import Parallel, delayed
import math
import time
import pandas as pd

# 计时开始
start_time = time.time()
data = []
with open('merged_56.json', 'r', encoding='utf-8') as f:
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

# 步骤 1：统计全局实体总量和需求方实体
global_entity_counts = defaultdict(int)
entity_ids_by_client = defaultdict(lambda: defaultdict(int))
all_entity_ids = set()
for item in data:
    client = item.get("需求方")
    entities = item.get("提取实体", {}).get("structured_data", {}).get("entities", [])
    for entity in entities:
        entity_id = entity.get("id")
        global_entity_counts[entity_id] += 1
        entity_ids_by_client[client][entity_id] += 1
        all_entity_ids.add(entity_id)

# 计算全局实体总次数
total_global_entities = sum(global_entity_counts.values())

# 转换为列表和索引
# 步骤 1：统计全局实体总量和需求方实体
global_entity_counts = defaultdict(int)
entity_ids_by_client = defaultdict(lambda: defaultdict(int))
all_entity_ids = set()
for item in data:
    client = item.get("需求方")
    entities = item.get("提取实体", {}).get("structured_data", {}).get("entities", [])
    for entity in entities:
        entity_id = entity.get("id")
        global_entity_counts[entity_id] += 1
        entity_ids_by_client[client][entity_id] += 1
        all_entity_ids.add(entity_id)

# 计算全局实体总次数
total_global_entities = sum(global_entity_counts.values())

# 转换为列表和索引
clients = list(entity_ids_by_client.keys())
entity_id_list = list(all_entity_ids)
entity_id_index = {eid: i for i, eid in enumerate(entity_id_list)}

# 步骤 2：构建稀疏矩阵（存储全局利用比例 p_ij）
rows, cols, values = [], [], []
for i, client in enumerate(clients):
    for entity_id, count in entity_ids_by_client[client].items():
        p_ij = count / total_global_entities  # 全局利用比例
        rows.append(i)
        cols.append(entity_id_index[entity_id])
        values.append(p_ij)
knowledge_matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(clients), len(entity_id_list)))

# 步骤 3：计算生态位宽度
niche_width = {}
for i, client in enumerate(clients):
    p_ij = knowledge_matrix[i].data  # 非零比例
    shannon_index = -np.sum(p_ij * np.log(p_ij)) if p_ij.size > 0 else 0
    niche_width[client] = shannon_index

# 步骤 4：并行计算全局重叠度矩阵
def compute_overlap(i, j, matrix):
    p_j = matrix[i].toarray().ravel()
    p_k = matrix[j].toarray().ravel()
    numerator = np.dot(p_j, p_k)
    denominator = np.sqrt(np.sum(p_j ** 2) * np.sum(p_k ** 2))
    return (i, j, numerator / denominator if denominator != 0 else 0)

n_jobs = -1  # 使用所有可用CPU核心
overlap_results = Parallel(n_jobs=n_jobs)(
    delayed(compute_overlap)(i, j, knowledge_matrix)
    for i, j in combinations(range(len(clients)), 2)
)

# 构建重叠度矩阵
N = len(clients)
overlap_matrix = np.eye(N)  # 初始化为单位矩阵（对角线为1）
for i, j, overlap in overlap_results:
    overlap_matrix[i, j] = overlap
    overlap_matrix[j, i] = overlap  # 对称矩阵

# 步骤 5：计算单一重叠度
single_overlap = {}
for i, client in enumerate(clients):
    overlaps = np.delete(overlap_matrix[i], i)  # 移除对角线元素
    single_overlap[client] = np.mean(overlaps) if overlaps.size > 0 else 0

# 步骤 6：转换知识利用矩阵为稠密格式（用于输出和保存）
knowledge_matrix_dense = knowledge_matrix.toarray().T  # 转置为 M x N (实体 x 需求方)

# 步骤 7：输出结果
print("生态位宽度（基于具体实体，全局利用比例，降序）：")
for client, width in sorted(niche_width.items(), key=lambda x: x[1], reverse=True):
    print(f"{client}: {width:.4f}")

print("\n知识利用矩阵（部分，M x N，前10个实体和需求方）：")
print(f"实体: {entity_id_list[:10]} ...")
print(f"需求方: {clients[:10]} ...")
print(knowledge_matrix_dense[:10, :10])  # 展示 10 x 10 子矩阵

print("\n全局重叠度矩阵（部分，N x N）：")
print(f"需求方: {clients[:5]} ...")
print(overlap_matrix[:5, :5])

print("\n高重叠度需求方对（前10，降序）：")
for i, j, overlap in sorted(overlap_results, key=lambda x: x[2], reverse=True)[:10]:
    print(f"{clients[i]} 与 {clients[j]}: {overlap:.4f}")

print("\n单一重叠度（基于具体实体，全局利用比例，降序）：")
for client, overlap in sorted(single_overlap.items(), key=lambda x: x[1], reverse=True):
    print(f"{client}: {overlap:.4f}")

# 计时结束
print(f"\n总耗时: {time.time() - start_time:.2f} 秒")

# 保存结果到CSV
width_df = pd.DataFrame.from_dict(niche_width, orient="index", columns=["生态位宽度"])
width_df.to_csv("niche_width.csv")
knowledge_df = pd.DataFrame(knowledge_matrix_dense, index=entity_id_list, columns=clients)
knowledge_df.to_csv("knowledge_usage_matrix.csv")
matrix_df = pd.DataFrame(overlap_matrix, index=clients, columns=clients)
matrix_df.to_csv("overlap_matrix.csv")
single_overlap_df = pd.DataFrame.from_dict(single_overlap, orient="index", columns=["单一重叠度"])
single_overlap_df.to_csv("single_overlap.csv")