import json
import logging
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 加载 BERT 模型和分词器
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"使用设备: {device}")


# 读取 JSON 文件
def load_json_data(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            if not file_content.strip():
                logger.error("文件为空，请检查文件内容！")
                return []
            data = []
            for i, line in enumerate(file_content.splitlines(), 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"第 {i} 行解析错误: {e}，内容: {line}")
                        continue
        if not data:
            logger.error("没有成功解析任何数据，请检查文件格式！")
            return []
        logger.info(f"成功加载 JSON 文件，包含 {len(data)} 条数据")
        return data
    except FileNotFoundError:
        logger.error(f"文件 {json_path} 未找到，请检查路径！")
        return []


# 验证和提取实体 ID
def extract_entity_ids(data):
    entity_ids = []
    original_ids = {}
    entity_id_counts = Counter()
    for i, item in enumerate(data, 1):
        extract_data = item.get("提取实体", {})
        structured_data = extract_data.get("structured_data", {})
        entity_list = structured_data.get("entities", [])
        if not entity_list:
            logger.warning(f"第 {i} 条数据缺少 entities 或 entities 为空: {json.dumps(item, ensure_ascii=False)}")
            continue
        for entity in entity_list:
            if not isinstance(entity, dict) or "id" not in entity:
                logger.warning(f"第 {i} 条数据中的实体格式错误: {json.dumps(entity, ensure_ascii=False)}")
                continue
            raw_id = entity["id"]
            norm_id = raw_id.strip().lower()
            original_ids[norm_id] = raw_id
            entity_id_counts[norm_id] += 1
            entity_ids.append(norm_id)
    logger.info(f"提取到 {len(entity_ids)} 个实体 ID")
    logger.info(f"实体 ID 分布: {entity_id_counts.most_common(10)}")
    if entity_id_counts and entity_id_counts.most_common(1)[0][1] > len(entity_ids) * 0.5:
        logger.warning(f"单一实体 ID 占比过高: {entity_id_counts.most_common(1)[0]}")
    return entity_ids, original_ids


# 批量获取 BERT 嵌入
def get_bert_embeddings(texts, tokenizer, model, device, max_length=256):
    if not texts:
        logger.warning("输入文本列表为空，返回空嵌入")
        return np.zeros((0, 768))
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not valid_texts:
        logger.warning("所有输入文本无效，返回零向量")
        return np.zeros((len(texts), 768))

    logger.info(f"处理 {len(valid_texts)} 个有效文本: {valid_texts[:5]}")
    inputs = tokenizer(
        valid_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    embeddings = np.nan_to_num(embeddings, nan=0.0)

    if len(embeddings) > 1:
        std = np.std(embeddings, axis=0).mean()
        logger.info(f"嵌入向量标准差: {std:.4f}")
        if std < 1e-3:
            logger.warning("嵌入向量高度相似，可能导致聚类失效")

    result = np.zeros((len(texts), 768))
    valid_idx = 0
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            result[i] = embeddings[valid_idx]
            valid_idx += 1
        else:
            logger.warning(f"无效文本: {text}")
    return result


# 计算簇内平均余弦相似度
def compute_cluster_similarity(embeddings, cluster_ids, cos_sim):
    if len(cluster_ids) < 2:
        return 1.0
    indices = [i for i, eid in enumerate(cluster_ids)]
    cluster_sim = cos_sim[np.ix_(indices, indices)]
    np.fill_diagonal(cluster_sim, 0)
    return np.mean(cluster_sim[cluster_sim > 0]) if np.sum(cluster_sim > 0) > 0 else 0.0


# 聚类实体 ID
def cluster_entities(entity_ids, tokenizer, model, device):
    if not entity_ids:
        logger.info("实体 ID 列表为空，跳过聚类")
        return {}
    if len(entity_ids) == 1:
        logger.info(f"仅有一个实体 ID，无需聚类: {entity_ids[0]}")
        return {entity_ids[0]: entity_ids[0]}

    embeddings = get_bert_embeddings(entity_ids, tokenizer, model, device)

    if np.all(embeddings == 0):
        logger.warning(f"所有嵌入为零向量，实体 ID: {entity_ids[:5]}...")
        return {eid: eid for eid in entity_ids}

    norm = np.linalg.norm(embeddings, axis=1)
    norm[norm == 0] = 1
    cos_sim = np.dot(embeddings, embeddings.T) / (norm[:, None] * norm[None, :])
    cos_sim = np.clip(cos_sim, -1, 1)
    cos_sim = np.nan_to_num(cos_sim, nan=0.0)
    logger.info(
        f"余弦相似度范围: min={cos_sim.min():.4f}, max={cos_sim.max():.4f}, mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}")
    if cos_sim.std() < 0.01:
        logger.warning("余弦相似度分布异常，可能导致单一簇")
        return {eid: eid for eid in entity_ids}

    # 动态估计簇数
    n_clusters = max(20, len(entity_ids) // 5)  # 约 844 簇
    logger.info(f"使用 AgglomerativeClustering，预计簇数: {n_clusters}")

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.4,  # 控制簇大小
        metric="cosine",
        linkage="average"
    ).fit(embeddings)

    cluster_counts = np.bincount(clustering.labels_)
    logger.info(
        f"簇分布: {len(cluster_counts)} 个簇，簇大小: {cluster_counts.tolist()[:20]}{'...' if len(cluster_counts) > 20 else ''}")
    if len(cluster_counts) <= 10 and len(entity_ids) > 50:
        logger.warning("生成簇数过少，回退到不聚类")
        id_counts = Counter(entity_ids)
        merge_details = [f"回退（簇数过少）：保持 {len(entity_ids)} 个原始 ID，分布: {id_counts.most_common(10)}"]
        entity_mapping = {eid: eid for eid in entity_ids}
        return entity_mapping

    if len(set(clustering.labels_)) > 1:
        try:
            score = silhouette_score(embeddings, clustering.labels_, metric="cosine")
            logger.info(f"聚类轮廓系数: {score:.4f}")
        except Exception as e:
            logger.warning(f"无法计算轮廓系数: {e}")

    entity_mapping = {}
    merge_details = []
    unique_ids = {eid for eid, count in Counter(entity_ids).items() if count == 1}

    for label in set(clustering.labels_):
        cluster_ids = [eid for i, eid in enumerate(entity_ids) if clustering.labels_[i] == label]
        id_counts = Counter(cluster_ids)
        merged_id = id_counts.most_common(1)[0][0]

        # 检查簇大小
        if len(cluster_ids) > 50:
            logger.warning(f"簇 {label} 过大（{len(cluster_ids)} 个 ID），拆分为噪声点")
            for eid in set(cluster_ids):
                if eid in unique_ids or id_counts[eid] == 1:
                    entity_mapping[eid] = eid
                    merge_details.append(f"噪声点（大簇拆分）: '{eid}' (不合并)")
                else:
                    entity_mapping[eid] = merged_id
            cluster_log = cluster_ids[:10] if len(cluster_ids) > 10 else cluster_ids
            merge_details.append(
                f"簇 {label}（部分合并）: {cluster_log}{'...' if len(cluster_ids) > 10 else ''} -> merged_id: '{merged_id}'")
            logger.info(
                f"簇 {label}: {len(cluster_ids)} 个 ID，ID 分布: {id_counts.most_common(5)}, 部分合并到 '{merged_id}'")
            continue

        # 计算簇内相似度
        sim = compute_cluster_similarity(embeddings, cluster_ids, cos_sim)
        cluster_log = cluster_ids[:10] if len(cluster_ids) > 10 else cluster_ids
        merge_details.append(
            f"簇 {label}: 合并 {cluster_log}{'...' if len(cluster_ids) > 10 else ''} -> merged_id: '{merged_id}' (平均相似度: {sim:.4f})")
        logger.info(
            f"簇 {label}: {len(cluster_ids)} 个 ID，ID 分布: {id_counts.most_common(5)}, 选择 merged_id: '{merged_id}', 平均相似度: {sim:.4f}")

        if sim < 0.7:
            logger.warning(f"簇 {label} 相似度低 ({sim:.4f})，可能包含无关词")

        for eid in set(cluster_ids):
            if eid not in entity_mapping:
                entity_mapping[eid] = merged_id
            else:
                logger.warning(f"簇 {label} 中重复 ID: '{eid}'，已映射到 '{entity_mapping[eid]}'")

    logger.info("\n合并详情：")
    for detail in merge_details:
        logger.info(detail)
    logger.info(f"\n生成 {len(entity_mapping)} 个映射，映射表: {dict(list(entity_mapping.items())[:10])}...")
    return entity_mapping


# 保存 JSON 数据（JSON Lines 格式）
def save_json_data(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')


# 保存合并后的 JSON 数据
def save_merged_json(data, entity_mapping, original_ids, output_path="merged_56.json"):
    merged_data = []
    replaced_id_counts = Counter()
    for item in data:
        new_item = json.loads(json.dumps(item))
        extract_data = new_item.get("提取实体", {})
        structured_data = extract_data.get("structured_data", {})
        entity_list = structured_data.get("entities", [])
        relation_list = structured_data.get("relations", [])

        for entity in entity_list:
            if entity.get("id"):
                lower_id = entity["id"].strip().lower()
                if lower_id in entity_mapping:
                    entity["id"] = original_ids.get(entity_mapping[lower_id], entity_mapping[lower_id])
                    replaced_id_counts[entity["id"]] += 1

        for relation in relation_list:
            if relation.get("source"):
                lower_source = relation["source"].strip().lower()
                if lower_source in entity_mapping:
                    relation["source"] = original_ids.get(entity_mapping[lower_source], entity_mapping[lower_source])
            if relation.get("target"):
                lower_target = relation["target"].strip().lower()
                if lower_target in entity_mapping:
                    relation["target"] = original_ids.get(entity_mapping[lower_target], entity_mapping[lower_target])

        merged_data.append(new_item)

    save_json_data(merged_data, output_path)
    logger.info(f"合并后的数据已保存到 {output_path}（JSON Lines 格式）")
    logger.info(f"替换后实体 ID 分布: {replaced_id_counts.most_common(10)}")
    logger.info(f"输出文件前3条数据：")
    for i, item in enumerate(merged_data[:3], 1):
        logger.info(f"第 {i} 条: {json.dumps(item, ensure_ascii=False)}")


# 主函数
def main(json_path="56.json", output_path="merged_56.json"):
    data = load_json_data(json_path)
    if not data:
        return None

    logger.info("\n前3条数据结构（若存在）：")
    for i, item in enumerate(data[:3], 1):
        logger.info(f"第 {i} 条: {json.dumps(item, ensure_ascii=False)}")

    entity_ids, original_ids = extract_entity_ids(data)
    if not entity_ids:
        return None

    entity_mapping = cluster_entities(entity_ids, tokenizer, model, device)
    save_merged_json(data, entity_mapping, original_ids, output_path)

    return entity_mapping


if __name__ == "__main__":
    entity_mapping = main()