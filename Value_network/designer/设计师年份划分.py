import pandas as pd
import os
from datetime import datetime

# 定义主文件夹路径
main_folder = r"D:\Documents\Desktop\一品威客数据采集4月\527test-设计师处理终极版\清洗预处理20251111"

# 定义领域列表（子文件夹名称）
domains = ["设计", "开发", "文案", "视频", "装修", "营销", "生活", "AI", "企业服务"]

# 用于存储按年份分组的数据
yearly_data = {}

# 遍历每个领域文件夹
for domain in domains:
    folder_path = os.path.join(main_folder, domain)
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在，跳过")
        continue

    # 遍历文件夹中的所有CSV文件
    for file_name in os.listdir(folder_path):
        if file_name.startswith("设计师订单信息-") and file_name.endswith("-处理后.csv"):
            file_path = os.path.join(folder_path, file_name)

            # 读取CSV文件
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except Exception as e:
                print(f"无法读取文件 {file_path}，错误：{e}")
                continue

            # 确保发布时间列存在
            if '发布时间' not in df.columns:
                print(f"文件 {file_path} 中缺少'发布时间'列，跳过")
                continue

            # 将发布时间转换为年份
            for index, row in df.iterrows():
                try:
                    # 将发布时间字符串转换为日期并提取年份
                    date_str = str(row['发布时间'])
                    # 尝试两种格式：YYYY-MM-DD 和 YYYY/MM/DD
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        date_obj = datetime.strptime(date_str, '%Y/%m/%d')

                    year = date_obj.year

                    # 初始化年份数据框
                    if year not in yearly_data:
                        yearly_data[year] = []

                    # 添加数据到对应年份
                    yearly_data[year].append(row)
                except ValueError as e:
                    print(f"文件 {file_path} 中行 {index + 1} 的发布时间格式错误：{date_str}，跳过")
                    continue

# 将每个年份的数据合并并保存
output_folder = os.path.join(main_folder, "按年份合并结果")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for year, data_list in yearly_data.items():
    if data_list:
        # 将数据列表转换为DataFrame
        yearly_df = pd.DataFrame(data_list)

        # 保存到CSV文件
        output_file = os.path.join(output_folder, f"设计师订单信息-{year}.csv")
        yearly_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"已生成 {output_file}")
    else:
        print(f"年份 {year} 没有数据")

print("合并完成！")