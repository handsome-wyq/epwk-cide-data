import pandas as pd
import glob
import os
import time
import traceback
import numpy as np  # Added for np.nan handling if needed, though dropna handles it

# --- Configuration ---
# Input configuration (where to find CSVs to merge)
scan_folders = [
    "D:\\Documents\\Desktop\\一品威客数据采集4月\\test\\清洗预处理",  # Source folder for CSVs
]
file_pattern = "设计师订单信息-*.csv"  # Pattern to match CSV files
scan_recursive = True  # Search recursively in subfolders

# Output configuration for the merged and deduplicated file
output_folder = "D:\\Documents\\Desktop\\一品威客数据采集4月\\521计算"  # Destination folder
output_filename = "合并去重并清洗交易价格后数据.csv"  # Updated filename
deduplication_column = "任务编号"  # Column to use for deduplication
price_column_to_clean = "交易价格"  # Column to check for empty/NaN values

# --- Derived paths ---
final_output_path = os.path.join(output_folder, output_filename)


def create_merged_deduplicated_and_cleaned_file():
    prep_start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- 开始数据合并、去重与清洗流程 ---")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: 输出文件夹已创建: {output_folder}")
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: 创建输出文件夹失败: {output_folder}. 错误: {e}")
            return

    # STEP 1: 文件发现逻辑
    step_time_start = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 1: 开始搜索输入文件...")
    discovered_input_files = []
    for folder_path in scan_folders:
        if not os.path.isdir(folder_path):
            print(f"    警告：扫描文件夹不存在: {folder_path}")
            continue
        if scan_recursive:
            search_pattern = os.path.join(folder_path, '**', file_pattern)
        else:
            search_pattern = os.path.join(folder_path, file_pattern)
        matched_files = glob.glob(search_pattern, recursive=scan_recursive)
        if matched_files:
            discovered_input_files.extend(matched_files)
    discovered_input_files = sorted(list(set(discovered_input_files)))
    if not discovered_input_files:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: 未能找到任何匹配 '{file_pattern}' 的输入文件。程序终止。")
        return
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 1: 文件搜索完成。找到 {len(discovered_input_files)} 个文件。耗时: {time.time() - step_time_start:.2f} 秒")

    # STEP 2: 读取并合并文件
    step_time_start = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 2: 开始读取并合并CSV文件...")
    all_dataframes = []
    for file_path in discovered_input_files:
        try:
            temp_df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
            all_dataframes.append(temp_df)
        except Exception as e:
            print(f"    读取文件 {file_path} 时出错: {e}，已跳过。")
    if not all_dataframes:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: 未能成功读取任何已发现的输入文件。程序终止。")
        return
    df = pd.concat(all_dataframes, ignore_index=True)
    del all_dataframes
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 2: 文件读取与合并完成。合并后总行数: {len(df)}。耗时: {time.time() - step_time_start:.2f} 秒")

    # STEP 3: 根据 '任务编号' 全局去重
    step_time_start = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 3: 开始根据列 '{deduplication_column}' 进行全局去重...")
    if deduplication_column in df.columns:
        initial_rows_before_dedup = len(df)
        df[deduplication_column] = df[deduplication_column].astype(str)
        df.drop_duplicates(subset=[deduplication_column], keep='first', inplace=True)
        rows_removed_dedup = initial_rows_before_dedup - len(df)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 3: 全局去重完成。处理前行数: {initial_rows_before_dedup}, 处理后行数: {len(df)} (减少 {rows_removed_dedup} 条)。耗时: {time.time() - step_time_start:.2f} 秒")
    else:
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: 列 '{deduplication_column}' 不存在于合并后的DataFrame中，无法进行全局去重。")

    # STEP 3.5: 清洗 '交易价格' 列，去除空值行
    step_time_start = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 3.5: 开始清洗列 '{price_column_to_clean}' (去除空值行)...")
    if price_column_to_clean in df.columns:
        initial_rows_before_price_clean = len(df)

        # First, ensure the column is numeric, coercing errors to NaN
        # This helps if '交易价格' might be read as object/string due to some non-numeric entries
        df[price_column_to_clean] = pd.to_numeric(df[price_column_to_clean], errors='coerce')

        # Drop rows where '交易价格' is NaN (which includes original empty strings or non-numeric values after coercion)
        df.dropna(subset=[price_column_to_clean], inplace=True)

        # Optional: also remove rows where '交易价格' might be zero, if that's considered "empty"
        # If you want to remove zeros as well:
        # df = df[df[price_column_to_clean] != 0]

        rows_removed_price_clean = initial_rows_before_price_clean - len(df)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 3.5: '{price_column_to_clean}' 清洗完成。处理前行数: {initial_rows_before_price_clean}, 处理后行数: {len(df)} (减少 {rows_removed_price_clean} 条)。耗时: {time.time() - step_time_start:.2f} 秒")
    else:
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: 列 '{price_column_to_clean}' 不存在于DataFrame中，无法进行空值清洗。")

    # STEP 4: 保存处理后的DataFrame
    step_time_start = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 4: 开始将最终数据保存到: {final_output_path}")
    try:
        df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 4: 文件已成功保存。耗时: {time.time() - step_time_start:.2f} 秒")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: 保存文件时出错: {e}")
        traceback.print_exc()

    total_script_time = time.time() - prep_start_time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- 数据合并、去重与清洗流程结束 ---")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 总耗时: {total_script_time:.2f} 秒")


if __name__ == "__main__":
    create_merged_deduplicated_and_cleaned_file()