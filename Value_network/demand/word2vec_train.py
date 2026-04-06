import json
import logging
import jieba
from gensim.models import Word2Vec
from collections import Counter
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# 加载 JSON 文件
def load_json_data(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"成功加载 JSON 文件，包含 {len(data)} 条数据")
        return data
    except Exception as e:
        logger.error(f"加载 JSON 失败: {e}")
        return []


# 提取语料和实体 ID
def extract_corpus_from_json(json_path):
    data = load_json_data(json_path)
    corpus = []
    entity_ids = []
    for item in data:
        # 提取需求文本
        demand_text = item.get("需求文本", "")
        if demand_text:
            corpus.append(demand_text)
        # 提取实体 ID
        extract_data = item.get("提取实体", {})
        structured_data = extract_data.get("structured_data", {})
        entity_list = structured_data.get("entities", [])
        for entity in entity_list:
            if entity.get("id"):
                corpus.append(entity["id"])
                entity_ids.append(entity["id"])
    logger.info(f"提取到 {len(corpus)} 条语料，{len(entity_ids)} 个实体 ID")
    return corpus, entity_ids


# 分词和准备语料
def prepare_sentences(corpus, entity_ids):
    # 自定义词典
    for eid in set(entity_ids):
        jieba.add_word(eid)

    # 分词
    sentences = [jieba.lcut(text) for text in corpus]
    sentences = [s for s in sentences if len(s) > 0]

    # 增强实体 ID（重复 5 次）
    for eid in entity_ids:
        words = jieba.lcut(eid)
        for _ in range(20):
            sentences.append(words)

    # 统计词数
    total_words = sum(len(s) for s in sentences)
    logger.info(f"语料总词数: {total_words}, 句子数: {len(sentences)}")
    return sentences


# 训练 Word2Vec 模型
def train_word2vec(sentences, vector_size=100, window=5, min_count=1, workers=4, output_path="custom_word2vec.bin"):
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=1  # Set to 1 initially to control the loop manually
    )

    # Train for the remaining epochs with tqdm progress bar
    total_epochs = 500
    for _ in tqdm(range(total_epochs - 1), desc="Training Word2Vec", unit="epoch"):
        model.train(sentences, total_examples=model.corpus_count, epochs=1)

    model.wv.save_word2vec_format(output_path, binary=True)
    logger.info(f"Word2Vec 模型训练完成，保存至 {output_path}")
    return model


# 主函数（训练阶段）
def main(json_path="56.json", model_path="custom_word2vec.bin"):
    # 1. 提取语料
    corpus, entity_ids = extract_corpus_from_json(json_path)
    if not corpus:
        logger.error("未提取到语料，退出")
        return None

    # 2. 准备分词后的句子
    sentences = prepare_sentences(corpus, entity_ids)
    if not sentences:
        logger.error("未生成有效句子，退出")
        return None

    # 3. 训练模型
    model = train_word2vec(sentences, vector_size=100, window=5, min_count=1, workers=4, output_path=model_path)

    # 4. 验证模型
    logger.info("验证模型：检查常见实体相似度")
    sample_entities = entity_ids[:5] if len(entity_ids) >= 5 else entity_ids
    for entity in sample_entities:
        if entity in model.wv:
            similar = model.wv.most_similar(entity, topn=5)
            logger.info(f"实体 '{entity}' 的相似实体: {similar}")
        else:
            logger.warning(f"实体 '{entity}' 不在模型词表中")

    return model


if __name__ == "__main__":
    model = main()