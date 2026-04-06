import pandas as pd
import os

# --- 基础目录配置 ---
base_dir = "D:\\Documents\\Desktop\\一品威客数据采集4月\\test"
input_base_dir = os.path.join(base_dir, "数据", "AI")  # 输入文件所在目录
input_filename = "AI.csv"
input_path = os.path.join(input_base_dir, input_filename)

output_dir_base_main = os.path.join(base_dir, "清洗预处理")  # 输出根目录
output_main_subdir = os.path.join(output_dir_base_main, "AI")  # AI领域输出子目录

# 支持的任务类型
task_types = ["单人悬赏", "招标", "雇佣", "直接雇佣"]

# --- 拆分文件 ---
def split_file_by_task_type(input_path):
    print(f"\n--- 开始处理文件: {input_path} ---")

    # 确保输出目录存在
    if not os.path.exists(output_main_subdir):
        os.makedirs(output_main_subdir)
        print(f"创建目录: {output_main_subdir}")

    # 1. 读取数据
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        print(f"原始数据加载成功，共 {len(df)} 条记录。")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 - {input_path}")
        return
    except Exception as e:
        print(f"读取输入文件 '{input_path}' 时出错: {e}")
        return

    # 2. 检查任务类型列
    if '任务类型' not in df.columns:
        print("错误: 数据中缺少 '任务类型' 列，无法拆分文件。")
        return

    # 3. 根据任务类型拆分并保存
    for task_type in task_types:
        # 过滤对应任务类型的数据
        df_task = df[df['任务类型'] == task_type].copy()
        if df_task.empty:
            print(f"警告: 任务类型 '{task_type}' 无数据，跳过生成文件。")
            continue

        # 动态生成输出文件名
        output_filename = f"AI-{task_type}.csv"
        output_path = os.path.join(output_main_subdir, output_filename)

        # 保存文件
        try:
            df_task.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"任务类型 '{task_type}' 数据已保存到 {output_path}，共 {len(df_task)} 条记录。")
        except Exception as e:
            print(f"保存任务类型 '{task_type}' 文件时出错: {e}")

    print(f"--- 文件 {input_path} 拆分完成 ---")

# --- 执行处理 ---
if __name__ == "__main__":
    if not os.path.exists(input_path):
        print(f"致命错误: 输入文件不存在 - {input_path}")
        print("请确保 'input_path' 配置正确。")
    else:
        split_file_by_task_type(input_path)