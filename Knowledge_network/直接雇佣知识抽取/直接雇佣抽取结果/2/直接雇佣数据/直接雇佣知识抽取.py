import os
import time
import requests
import pandas as pd
import re



# ========== 路径 & 配置 ==========
BASE_INPUT_DIR = r"/root/autodl-tmp/直接雇佣数据/原始数据"
BASE_OUTPUT_DIR = r"/root/autodl-tmp/直接雇佣数据/输出数据20"

TITLE_COL = "详情标题"     # 原始句子的列名
OUTPUT_COL = "知识"      # 新增列名，用来存知识点
MODEL_NAME = "qwen3:8b"   # Ollama 里的模型名（按你实际情况改）

YEARS = range(2020, 2021)  # 2014~2025

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


CHINESE_NUMS = "零一二三四五六七八九十百千万两"

def is_noise_token(token: str) -> bool:
    """判断一个 token 是不是我们不想要的“噪音”知识点。"""
    t = token.strip()
    if not t:
        return True

    # 1. 全是数字或数字+小数点
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return True

    # 2. 金额：包含 元, ￥, RMB, $, “块钱”等
    if any(x in t for x in ["元", "￥", "RMB", "美元", "块钱", "块"]):
        return True

    # 3. 纯中文数字（如 “一”“二”“三”“二十”）
    if all(ch in CHINESE_NUMS for ch in t):
        return True

    # 4. 非常短而且信息量极低的 token（比如单个标点）
    if len(t) == 1 and not ("\u4e00" <= t <= "\u9fff"):  # 不是汉字的单字符
        return True

    return False


def clean_model_output(raw: str) -> str:
    """清洗模型原始输出，只留下有用的关键词这一行。"""
    if not raw:
        return ""

    # 1. 去掉 <think> ... </think>
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I)

    # 2. 去掉 ``` 包裹
    raw = raw.strip().strip("`").strip()

    # 3. 拆行，只保留最后一行非空内容
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""

    line = lines[-1]

    # 4. 去掉前缀“输出：”“结果：”等
    prefixes = ["输出：", "输出:", "结果：", "结果:", "回答：", "回答:", "Answer:", "答案：", "答案:"]
    for p in prefixes:
        if line.startswith(p):
            line = line[len(p):].strip()

    # 5. 去掉回车和多余引号
    line = line.replace("\r", " ").replace("\n", " ")
    line = line.replace('"', "")

    # 6. 分词：按逗号切开做过滤
    parts = [p.strip() for p in re.split(r"[，,]", line) if p.strip()]

    # 7. 过滤噪音 token（数字、金额、“二”等）
    clean_parts = [p for p in parts if not is_noise_token(p)]

    # 8. 去重 & 保留顺序
    seen = set()
    uniq_parts = []
    for p in clean_parts:
        if p not in seen:
            seen.add(p)
            uniq_parts.append(p)

    # 9. 重新组合
    final_line = ", ".join(uniq_parts)

    # 10. 长度保护
    if len(final_line) > 100:
        final_line = final_line[:100]

    return final_line



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
        "think": False,
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
