import os
import time
import requests
import pandas as pd
import re



# ========== 路径 & 配置 ==========
BASE_INPUT_DIR = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\直接雇佣知识抽取\原始数据"
BASE_OUTPUT_DIR = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\直接雇佣知识抽取\输出数据"

TITLE_COL = "详情标题"     # 原始句子的列名
OUTPUT_COL = "知识"      # 新增列名，用来存知识点
MODEL_NAME = "qwen3:8b"   # Ollama 里的模型名（按你实际情况改）

YEARS = range(2014, 2026)  # 2014~2025

# CSV 编码：如果 utf-8 读不了，可以改成 "gbk"
CSV_ENCODING = "utf-8-sig"      # 或 "utf-8-sig" / "gbk"

# 每条之间的休眠（防止太密集，视情况改）
SLEEP_SECONDS = 0.0

# ========== Prompt 模板 ==========
PROMPT_TEMPLATE = """
你是一个关键短语抽取工具。

任务：
- 输入是一条很短的中文需求描述（比如众包任务标题）
- 只从中抽取 1~3 个“与任务内容相关的关键知识点短语”
- 用中文输出，用英文逗号分隔
- 不要输出价格、金额（例如“500元”“1000”）
- 不要输出纯数字或序号（例如“1”“2”“一”“二”“三”）
- 不要输出与任务无关的虚词（例如“请帮忙”“急需”“在线等”）
- 不要解释，不要输出分析过程，只输出关键短语本身

示例：
输入：设计一个女包LOGO
输出：女包, logo

输入：体重管理微信公众账号开发
输出：体重管理, 微信公众号

输入：【动漫其他】500元画场景（二）
输出：动漫场景, 封面/插画设计
现在请处理下面的文本：
{text}

只输出关键短语本身，不要加“输出：”“结果：”“分析如下：”等前缀
/no_think
"""

def clean_model_output(raw: str) -> str:
    """清洗模型原始输出，只留下关键词这一行。"""
    if not raw:
        return ""

    # 1. 去掉 <think> ... </think>（包括中间的换行）
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I)

    # 2. 去掉 ``` 之类的包裹
    raw = raw.strip().strip("`").strip()

    # 3. 拆行，只保留最后一行非空内容（一般是最终答案）
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""

    line = lines[-1]  # 最后一行通常是：单本教辅, 封面设计

    # 4. 去掉“输出：”“结果：”这类前缀
    prefixes = ["输出：", "输出:", "结果：", "结果:", "回答：", "回答:", "Answer:", "答案：", "答案:"]
    for p in prefixes:
        if line.startswith(p):
            line = line[len(p):].strip()

    # 5. 彻底去掉换行和多余的引号，避免 CSV 损坏
    line = line.replace("\r", " ").replace("\n", " ")
    line = line.replace('"', "")  # 防止出现奇怪的未闭合引号

    # 6. 长度保护，防止模型乱说一大坨
    if len(line) > 100:
        line = line[:100]

    return line


def call_ollama_extract(text: str) -> str:
    """
    调用本地 Ollama，对一句话做知识点抽取。
    返回类似：'单本教辅, 封面设计'
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    prompt = PROMPT_TEMPLATE.format(text=text)
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 32  # 给一点空间即可，不要太大
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
    except Exception as e:
        print(f"[错误] 调用模型失败：{e}，文本= {text}")
        return ""

    cleaned = clean_model_output(raw)
    return cleaned

def process_one_year(year: int):
    """
    处理某一年的 CSV：A{year}.csv -> A{year}_with_knowledge.csv
    逐行处理，每处理一行就立刻追加写入输出文件。
    """
    input_filename = f"A{year}.csv"
    input_path = os.path.join(BASE_INPUT_DIR, input_filename)

    if not os.path.exists(input_path):
        print(f"[跳过] 找不到文件：{input_path}")
        return

    print(f"\n===== 处理年份 {year}，文件：{input_path} =====")

    # 读入 CSV 整体（方便遍历 & 保留原有列）
    try:
        df = pd.read_csv(input_path, encoding=CSV_ENCODING)
    except UnicodeDecodeError:
        print(f"[警告] 用编码 {CSV_ENCODING} 读取失败，你可能需要改成 'gbk' 或 'utf-8-sig'")
        raise

    if TITLE_COL not in df.columns:
        raise ValueError(f"文件 {input_filename} 中找不到列：{TITLE_COL}")

    # 确保输出目录存在
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # 输出文件名：A2014_with_knowledge.csv
    output_filename = f"A{year}_with_knowledge.csv"
    output_path = os.path.join(BASE_OUTPUT_DIR, output_filename)

    # 如果之前已经有同名输出，先删除，避免累加旧数据
    if os.path.exists(output_path):
        os.remove(output_path)

    total = len(df)
    write_header = True  # 只在第一次写入时写表头

    for i, row in df.iterrows():
        text = str(row[TITLE_COL])
        print(f"[{year}] [{i+1}/{total}] 处理：{text}")

        knowledge = call_ollama_extract(text)

        # 把当前行转换为 dict，并加上知识点列
        row_dict = row.to_dict()
        row_dict[OUTPUT_COL] = knowledge

        # 单行 DataFrame
        out_df = pd.DataFrame([row_dict])

        # 追加写入 CSV
        out_df.to_csv(
            output_path,
            mode="a",                 # 追加模式
            header=write_header,      # 第一次写入才写表头
            index=False,
            encoding=CSV_ENCODING
        )

        # 之后的写入就不再写表头
        if write_header:
            write_header = False

        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)

    print(f"[完成] 年份 {year} 已处理，输出实时写入：{output_path}")


def main():
    for year in YEARS:
        process_one_year(year)


if __name__ == "__main__":
    main()
