import logging
import json
import pandas as pd
import asyncio
import os
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化异步 DeepSeek 客户端
client = AsyncOpenAI(
    base_url="https://chatapi.littlewheat.com/v1",
    api_key="sk-aEpQY7lI3dLk3ILoaeLHbh6VtFxW3cTFXzljkv3QN5CgHb0I"
)

def generate_prompt(row):
    """生成单行数据的提示文本"""
    title = row['标题'] if pd.notna(row['标题']) else "未知标题"
    requirements = row['需求文本'] if pd.notna(row['需求文本']) else "无任务要求"
    prompt = f"""
    你是一个专业的需求分析助 手，从设计任务文本中提取规范实体和关系，用于群智众包场景。请使用思维链（CoT）方法，逐步推理，生成结构化JSON输出。
    **任务**：
    1. 基于分类体系，提取规范实体。
    2. 生成实体对，推理关系（如"设计"、"面向"），验证三元组逻辑。
    3. 输出JSON，包含"category"、"explanation"和"structured_data"。

    **实体分类**：
    - 核心实体：包括设计对象（如"餐饮品牌"）、设计功能（如"品牌标识突出"）等
    - 附属实体：包括设计风格（如"简约"）、目标用户（如"年轻人"）、工具（如"Photoshop"）等
    - 约束实体：包括物理约束（如"1920x1080像素"）、资源约束（如"版权合规"）等

    **模板**：
    - 实体：{{"id": "{{实体名称}}", "type": "{{实体类型}}"}}
    - 关系：{{"source": "{{主语}}", "target": "{{宾语}}", "relation": "{{关系类型}}"}}

    **输入**：
    标题: {title}
    需求: {requirements}

    **输出**：
    ```json
    {{"category": "实体与关系提取","explanation":"推理过程","structured_data": {{"entities": [], "relations": []}}}}
    ```
    """
    return prompt


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def extract_entities_and_relations(row, index):
    """异步提取单行数据的实体和关系"""
    try:
        prompt = generate_prompt(row)
        logger.info(f"正在处理行: {index}")
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的需求分析助手，擅长从设计任务文本中提取规范实体和关系。请严格按照用户指定的 JSON 格式返回结果，仅返回 JSON 字符串，不添加任何多余内容。"
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )

        result = response.choices[0].message.content.strip()
        result = result.strip('```json').strip('```').strip().replace('\n', '').replace('\r', '')
        logger.debug(f"第 {index} 行 API 返回结果: {repr(result)}")

        parsed_result = json.loads(result)
        entities = {e["id"] for e in parsed_result["structured_data"]["entities"]}
        valid_relations = [
            r for r in parsed_result["structured_data"]["relations"]
            if r["source"] in entities and r["target"] in entities
        ]
        parsed_result["structured_data"]["relations"] = valid_relations
        return index, parsed_result
    except json.JSONDecodeError as e:
        logger.error(f"第 {index} 行返回结果无效 JSON: {str(e)}, 原始结果: {repr(result)}")
        return index, None
    except Exception as e:
        logger.error(f"处理行 {index} 时发生错误: {str(e)}")
        return index, None

async def process_batch(data_subset, output_file, batch_size=10):
    """异步处理数据子集，提取实体和关系"""
    if data_subset.empty:
        logger.warning("数据子集为空，无法处理！")
        return [], []

    all_results = []
    all_failed_rows = []

    for i in range(0, len(data_subset), batch_size):
        batch = data_subset.iloc[i:i + batch_size]
        logger.info(f"正在处理第 {i + 1} 到 {i + batch_size} 行...")

        # 创建异步任务
        tasks = [
            extract_entities_and_relations(row, index)
            for index, row in batch.iterrows()
        ]

        # 并行执行任务并捕获异常
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        batch_records = []
        for task_result in results:
            if isinstance(task_result, Exception):
                logger.error(f"任务失败: {str(task_result)}")
                continue

            index, result = task_result
            if result is None:
                logger.error(f"第 {index} 行提取失败")
                all_failed_rows.append((index, "提取失败"))
                continue

            try:
                record = data_subset.loc[index].to_dict()
                
                # 提取并保存需要的字段
                output_record = {
                    '需求方': record.get('需求方', ''),
                    '需求文本': record.get('需求文本', ''),
                    '任务编号': record.get('任务编号', ''),
                    '交易价格': record.get('交易价格', ''),
                    '提取实体': result
                }
                
                batch_records.append(output_record)
                all_results.append(result)
                logger.info(f"第 {index} 行处理成功")
            except Exception as e:
                logger.error(f"记录第 {index} 行结果失败: {str(e)}")
                all_failed_rows.append((index, "结果记录失败"))

        # 批量写入
        if batch_records:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'a', encoding='utf-8') as f:
                for record in batch_records:
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')

    logger.info(f"处理完成，结果数量: {len(all_results)}, 失败数量: {len(all_failed_rows)}")
    return all_results, all_failed_rows

async def async_main(data_subset, output_file):
    return await process_batch(data_subset, output_file, batch_size=10)

def process_all_files(input_root, output_root="815TEST"):
    """遍历所有子文件夹和CSV文件，处理数据并保存结果"""
    # 遍历根目录下的所有子文件夹
    for subfolder in os.listdir(input_root):
        subfolder_path = os.path.join(input_root, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        logger.info(f"处理子文件夹: {subfolder}")
        # 遍历子文件夹中的所有CSV文件
        for file_name in os.listdir(subfolder_path):
            if not file_name.endswith('.csv'):
                continue

            file_path = os.path.join(subfolder_path, file_name)
            logger.info(f"处理文件: {file_path}")

            try:
                data = pd.read_csv(file_path, encoding='utf-8')
                logger.info(f"文件 {file_name} 原始数据行数: {len(data)}")
            except FileNotFoundError:
                logger.error(f"文件 {file_path} 不存在，请检查路径！")
                continue
            except Exception as e:
                logger.error(f"读取 CSV 文件 {file_path} 失败: {str(e)}")
                continue

            # 校验必要列
            required_columns = ['标题', '需求文本']
            recommended_columns = ['需求方', '任务编号', '交易价格']
            missing_columns = [col for col in required_columns if col not in data.columns]
            missing_recommended = [col for col in recommended_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"文件 {file_name} 缺少必要列: {missing_columns}")
                continue
            if missing_recommended:
                logger.warning(f"文件 {file_name} 缺少推荐列: {missing_recommended}")

            # 设置输出文件路径
            relative_path = os.path.relpath(subfolder_path, input_root)
            output_subfolder = os.path.join(output_root, relative_path)
            output_file_name = os.path.splitext(file_name)[0] + '.json'
            output_file = os.path.join(output_subfolder, output_file_name)
            error_log_file = os.path.join(output_subfolder, os.path.splitext(file_name)[0] + '_error_log.json')

            # 保存源数据
            #source_file_path = os.path.join(output_subfolder, file_name)
            #os.makedirs(output_subfolder, exist_ok=True)
            #try:
                #data.to_csv(source_file_path, encoding='utf-8', index=False)
                #logger.info(f"源数据已保存至: {source_file_path}")
            #except Exception as e:
                #logger.error(f"保存源数据 {file_name} 失败: {str(e)}")

            # 初始化输出文件

            #if not os.path.exists(output_file):
                #with open(output_file, 'w', encoding='utf-8') as f:
                    #pass
            try:
                os.makedirs(output_subfolder, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    pass
                logger.info(f"初始化输出文件: {output_file}")
            except Exception as e:
                logger.error(f"初始化输出文件 {output_file} 失败: {str(e)}")
                continue

            # 异步处理整个数据集
            results, failed_rows = asyncio.run(async_main(data, output_file))

            # 保存错误日志
            if failed_rows:
                os.makedirs(output_subfolder, exist_ok=True)
                with open(error_log_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_rows, f, ensure_ascii=False, indent=4)
                logger.info(f"错误日志已保存至: {error_log_file}")
            else:
                logger.info(f"文件 {file_name} 无处理失败的行。")

            logger.info(f"文件 {file_name} 提取结果已增量保存至: {output_file}")

if __name__ == "__main__":
    input_root = 'D:/Documents/Desktop/Crowd_intelligence_wyq/Knowledge_network/未完成'
    output_root = '815TEST'
    process_all_files(input_root, output_root)