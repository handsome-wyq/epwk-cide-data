import os
import glob
import json
import pandas as pd

# 指定目录路径
dir_path = r'D:\Documents\Desktop\Crowd_intelligence_wyq\抽取结果\测试'

# 查找所有JSON文件
json_files = glob.glob(os.path.join(dir_path, '*.json'))

# 查找所有CSV文件
csv_files = glob.glob(os.path.join(dir_path, '*.csv'))

# 创建一个字典来存储任务编号到完成时间的映射
task_to_time = {}

# 读取所有CSV文件并构建映射
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')  # 假设CSV使用UTF-8编码，可根据需要调整为'gbk'等
        if '任务编号' in df.columns and '发布时间' in df.columns:
            for _, row in df.iterrows():
                task_id = row['任务编号']
                complete_time = row['发布时间']
                if task_id not in task_to_time:  # 如果重复，只取第一个
                    task_to_time[task_id] = complete_time
        else:
            print(f"警告: CSV文件 {csv_file} 缺少'任务编号'或'发布时间'列，已跳过。")
    except Exception as e:
        print(f"错误: 读取CSV文件 {csv_file} 时出错: {e}")

# 处理每个JSON文件（JSONL格式）
for json_file in json_files:
    try:
        modified_data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())  # 逐行解析JSON
                    task_id = item.get('任务编号')
                    if task_id in task_to_time:
                        item['发布时间'] = task_to_time[task_id]
                    modified_data.append(item)
                except json.JSONDecodeError as je:
                    print(f"警告: JSON文件 {json_file} 中某行解析失败: {je}")
                    continue

        # 输出到新文件（保持JSONL格式）
        modified_file = json_file.replace('.json', '_modified.json')
        with open(modified_file, 'w', encoding='utf-8') as f:
            for item in modified_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')  # 每条数据一行

        print(f"处理完成: {json_file} -> {modified_file}")
    except Exception as e:
        print(f"错误: 处理JSON文件 {json_file} 时出错: {e}")