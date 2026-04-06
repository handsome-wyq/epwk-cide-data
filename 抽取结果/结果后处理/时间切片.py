import os
import glob
import json
from datetime import datetime

# 指定主目录路径
base_dir = r'D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\抽取输出\2-添加时间后'

# 创建输出目录（如果不存在）
output_dir = os.path.join(base_dir, 'SortedByYearNEW')
os.makedirs(output_dir, exist_ok=True)

# 初始化按年份存储数据的字典
data_by_year = {str(year): [] for year in range(2014, 2026)}

# 递归查找所有子文件夹中的JSON文件
json_files = []
for root, _, files in os.walk(base_dir):
    json_files.extend(glob.glob(os.path.join(root, '*.json')))

# 处理每个JSON文件（JSONL格式）
for json_file in json_files:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())  # 逐行解析JSON
                    complete_time = item.get('发布时间', '')
                    if complete_time:
                        # 尝试解析发布时间，假设格式如 '2023-10-01' 或其他可解析格式
                        try:
                            year = str(datetime.strptime(complete_time, '%Y.%m.%d').year)
                        except ValueError:
                            # 如果时间格式不同，尝试其他格式或提取前4位作为年份
                            year = complete_time[:4] if complete_time[:4].isdigit() else None

                        if year and year in data_by_year:
                            data_by_year[year].append(item)
                        else:
                            print(
                                f"警告: JSON文件 {json_file} 中数据 {item.get('任务编号', '未知')} 的年份 {year} 不在2015-2025范围内，已跳过。")
                    else:
                        print(
                            f"警告: JSON文件 {json_file} 中数据 {item.get('任务编号', '未知')} 缺少'发布时间'字段，已跳过。")
                except json.JSONDecodeError as je:
                    print(f"警告: JSON文件 {json_file} 中某行解析失败: {je}")
                    continue
    except Exception as e:
        print(f"错误: 处理JSON文件 {json_file} 时出错: {e}")

# 将按年份整理的数据写入新的JSONL文件
for year, data in data_by_year.items():
    if data:  # 仅对有数据的年份生成文件
        output_file = os.path.join(output_dir, f'{year}.json')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')  # 保持JSONL格式
            print(f"生成文件: {output_file}，包含 {len(data)} 条数据")
        except Exception as e:
            print(f"错误: 写入文件 {output_file} 时出错: {e}")
    else:
        print(f"提示: {year} 年没有数据，跳过生成文件。")