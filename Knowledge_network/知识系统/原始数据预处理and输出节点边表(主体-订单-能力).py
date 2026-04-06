import pandas as pd
import re
import os
from datetime import datetime

# --- 基础目录配置 ---
base_dir = "D:/Desktop/CID/Knowledge_network/知识系统"
input_base_dir = os.path.join(base_dir, "数据")  # 输入根目录
output_dir_base_main = os.path.join(base_dir, "预处理and节点边表and分年度")  # 输出根目录


# 支持的文件类型
file_types = ["单人悬赏", "招标", "雇佣", "计件", "多人悬赏"]

# --- 正则提取函数 ---
def extract_bracket_content(title):
    if pd.isna(title):
        return None
    match = re.search(r'【(.*?)】', str(title))
    return match.group(1) if match else None

# --- 交易价格清洗函数（修复多价格问题） ---
def clean_transaction_price(price):
    if pd.isna(price):
        return None
    price_str = str(price).replace('￥', '').replace('元', '').strip()
    prices = [p.strip() for p in price_str.split(',') if p.strip()]
    valid_prices = []
    for p in prices:
        p_clean = p.replace(',', '')
        match = re.search(r'(\d+\.?\d*)', p_clean)
        if match:
            try:
                valid_prices.append(float(match.group(0)))
            except ValueError:
                continue
    return valid_prices[0] if valid_prices else None

# --- 单文件处理流程 ---
def process_single_file(input_path, domain, file_type, output_main_subdir):
    input_filename_no_ext = f"{domain}-{file_type}"
    print(f"\n--- 开始处理文件: {input_path} ({domain}-{file_type}) ---")

    # 动态生成输出路径
    output_main_file_path = os.path.join(output_main_subdir, f"{input_filename_no_ext}-预处理后完整版.csv")
    subject_orders_output_path = os.path.join(output_main_subdir, f"主体订单信息-{input_filename_no_ext}-处理后.csv")

    # 1. 读取数据
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        print(f"原始数据加载成功，共 {len(df)} 条记录。")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 - {input_path}")
        return None
    except Exception as e:
        print(f"读取输入文件 '{input_path}' 时出错: {e}")
        return None

    # --- 数据清洗 ---
    print("\n--- 开始数据清洗 ---")
    df_cleaned = df.copy()

    # a. 过滤 '标题', '需求文本', '任务编号' 为空的行（不删除'需求方'为空的行）
    initial_rows = len(df_cleaned)
    df_cleaned.dropna(subset=['标题', '需求文本', '任务编号'], inplace=True)
    print(
        f"过滤'标题', '需求文本', '任务编号'空值后，剩余 {len(df_cleaned)} 条记录 (减少 {initial_rows - len(df_cleaned)} 条)。")

    # b. 处理交易价格：如果为空，复制需求价格，然后清洗
    if '交易价格' in df_cleaned.columns:
        if '需求价格' in df_cleaned.columns:
            initial_null_count = df_cleaned['交易价格'].isna().sum()
            df_cleaned['交易价格'] = df_cleaned['交易价格'].where(df_cleaned['交易价格'].notna(),
                                                                  df_cleaned['需求价格'])
            final_null_count = df_cleaned['交易价格'].isna().sum()
            print(f"交易价格处理：原始空值 {initial_null_count} 个，复制需求价格后剩余空值 {final_null_count} 个。")
        df_cleaned['交易价格'] = df_cleaned['交易价格'].apply(clean_transaction_price)
        invalid_prices = df_cleaned[df_cleaned['交易价格'].isna() | (df_cleaned['交易价格'] > 1e10)]
        if not invalid_prices.empty:
            print(f"警告：发现 {len(invalid_prices)} 个无效或异常交易价格：")
            print(invalid_prices[['任务编号', '交易价格']].head())
        print("交易价格清洗完成。")
    else:
        print("警告: 数据中缺少 '交易价格' 列。")

    # c. 处理发布时间并进行时间过滤
    if '发布时间' in df_cleaned.columns:
        df_cleaned['发布时间'] = pd.to_datetime(df_cleaned['发布时间'], format='%Y.%m.%d', errors='coerce')
        initial_rows_before_date_filter = len(df_cleaned)
        df_cleaned.dropna(subset=['发布时间'], inplace=True)
        print(f"因无效日期格式移除 {initial_rows_before_date_filter - len(df_cleaned)} 条记录。")
        initial_rows_before_year_filter = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned['发布时间'].dt.year >= 2014].copy()
        print(
            f"过滤2014年以前的数据后，剩余 {len(df_cleaned)} 条记录 (减少 {initial_rows_before_year_filter - len(df_cleaned)} 条)。")
    else:
        print("警告: 数据中缺少 '发布时间' 列，无法进行时间处理和过滤。")

    # d. 根据"任务编号"去重
    if '任务编号' in df_cleaned.columns:
        initial_rows_before_dedup = len(df_cleaned)
        df_empty_task_id = df_cleaned[df_cleaned['任务编号'].isna()].copy()
        df_non_empty_task_id = df_cleaned[df_cleaned['任务编号'].notna()].copy()
        df_non_empty_task_id.drop_duplicates(subset=['任务编号'], keep='first', inplace=True)
        df_cleaned = pd.concat([df_non_empty_task_id, df_empty_task_id], ignore_index=True)
        print(
            f"根据'任务编号'去重后，剩余 {len(df_cleaned)} 条记录 (因去重减少 {initial_rows_before_dedup - len(df_cleaned)} 条)。")
    else:
        print("警告: 数据中缺少 '任务编号' 列，无法进行去重。")
    print(f"--- 数据清洗阶段完成，最终剩余 {len(df_cleaned)} 条记录 ---")

    # --- 补充缺失主体 ---
    print("\n--- 开始补充缺失主体 ---")
    if '需求方' in df_cleaned.columns and '任务编号' in df_cleaned.columns:
        df_cleaned['需求方'] = df_cleaned.apply(
            lambda row: "A" + str(row['任务编号']) if pd.isna(row['需求方']) and not pd.isna(row['任务编号']) else row['需求方'],
            axis=1
        )
    if file_type == "多人悬赏":
        if '设计师1' in df_cleaned.columns and '设计师2' in df_cleaned.columns:
            def get_designers(row):
                designers = []
                if not pd.isna(row['设计师1']):
                    designers.append(row['设计师1'])
                if not pd.isna(row['设计师2']):
                    designers.append(row['设计师2'])
                if not designers and not pd.isna(row['任务编号']):
                    designers.append("B" + str(row['任务编号']))
                return designers
            df_cleaned['设计师_list'] = df_cleaned.apply(get_designers, axis=1)
            df_exploded = df_cleaned.explode('设计师_list')
            df_exploded['设计师'] = df_exploded['设计师_list']
            df_exploded.drop(columns=['设计师_list', '设计师1', '设计师2'], inplace=True, errors='ignore')
            df_cleaned = df_exploded
    else:
        if '设计师' in df_cleaned.columns:
            df_cleaned['设计师'] = df_cleaned.apply(
                lambda row: "B" + str(row['任务编号']) if pd.isna(row['设计师']) and not pd.isna(row['任务编号']) else row['设计师'],
                axis=1
            )
        else:
            df_cleaned['设计师'] = df_cleaned.apply(
                lambda row: "B" + str(row['任务编号']) if not pd.isna(row['任务编号']) else None,
                axis=1
            )
    print("--- 补充缺失主体完成 ---")

    # --- 提取字段和主体订单信息 ---
    print("\n--- 开始提取字段和主体订单信息 ---")
    df_cleaned['能力标签'] = df_cleaned['标题'].apply(extract_bracket_content)
    print("能力标签提取完成。")

    if all(col in df_cleaned.columns for col in ['需求方', '设计师', '任务编号', '能力标签', '交易价格', '发布时间']):
        subject_orders = df_cleaned.dropna(subset=['需求方', '设计师', '任务编号']).copy()
        subject_orders_selected = subject_orders[['需求方', '设计师', '任务编号', '能力标签', '交易价格', '发布时间']]
        try:
            subject_orders_selected.to_csv(subject_orders_output_path, index=False, encoding='utf-8-sig')
            print(f"主体订单信息已保存到 {subject_orders_output_path}")
            unique_demands = subject_orders['需求方'].nunique()
            unique_designers = subject_orders['设计师'].nunique()
            print(f"共找到 {unique_demands} 个唯一需求方和 {unique_designers} 个唯一设计师，涉及 {len(subject_orders_selected)} 条订单记录。")
        except Exception as e:
            print(f"保存主体订单信息时出错: {e}")
    else:
        print("警告：数据中缺少必要的列，无法提取主体订单信息。")
    print("--- 提取字段和主体订单信息完成 ---")

    # --- 保存清洗后完整数据 ---
    try:
        df_cleaned.to_csv(output_main_file_path, index=False, encoding='utf-8-sig')
        print(f"清洗后完整数据已保存到: {output_main_file_path}")
    except Exception as e:
        print(f"保存清洗后数据时出错: {e}")

    print(f"--- 文件 {input_path} 处理结束 ---")

    # 返回处理后的数据框，用于后续提取节点和边
    return df_cleaned

# --- 新增: 生成主体层、订单层、能力标签层的节点和边表（支持按年度） ---
def generate_nodes_and_edges(all_df, output_dir, year=None):
    prefix = f"年份{year}-" if year is not None else ""
    print(f"\n--- 开始生成节点和边表 (年份: {year if year else '全局'}) ---")

    # 去重任务编号
    if '任务编号' in all_df.columns:
        all_df = all_df.drop_duplicates(subset=['任务编号'], keep='first')

    # 1. 主体层节点：需求方和设计师的唯一列表，区分类型
    if all(col in all_df.columns for col in ['需求方', '设计师']):
        # 需求方节点
        demanders = all_df['需求方'].dropna().unique()
        demanders_df = pd.DataFrame({'ID': demanders, 'Name': demanders, 'Type': '需求方'})

        # 设计师节点
        designers = all_df['设计师'].dropna().unique()
        designers_df = pd.DataFrame({'ID': designers, 'Name': designers, 'Type': '设计师'})

        # 合并主体节点
        subjects_df = pd.concat([demanders_df, designers_df], ignore_index=True)
        subjects_path = os.path.join(output_dir, f"{prefix}nodes_subjects.csv")
        subjects_df.to_csv(subjects_path, index=False, encoding='utf-8-sig')
        print(f"主体层节点表已保存到: {subjects_path}")
    else:
        print("警告: 缺少 '需求方' 或 '设计师' 列，无法生成主体节点。")

    # 2. 订单层节点：唯一任务编号及其他属性
    if all(col in all_df.columns for col in ['任务编号', '交易价格', '发布时间']):
        orders_df = all_df[['任务编号', '交易价格', '发布时间']].dropna(subset=['任务编号']).drop_duplicates()
        orders_df.columns = ['ID', 'Transaction_Price', 'Publication_Time']
        orders_path = os.path.join(output_dir, f"{prefix}nodes_orders.csv")
        orders_df.to_csv(orders_path, index=False, encoding='utf-8-sig')
        print(f"订单层节点表已保存到: {orders_path}")
    else:
        print("警告: 缺少必要的订单列，无法生成订单节点。")

    # 3. 能力标签层节点：唯一能力标签
    if '能力标签' in all_df.columns:
        abilities = all_df['能力标签'].dropna().unique()
        abilities_df = pd.DataFrame({'ID': abilities, 'Name': abilities})
        abilities_path = os.path.join(output_dir, f"{prefix}nodes_abilities.csv")
        abilities_df.to_csv(abilities_path, index=False, encoding='utf-8-sig')
        print(f"能力标签层节点表已保存到: {abilities_path}")
    else:
        print("警告: 缺少 '能力标签' 列，无法生成能力标签节点。")

    # 4. 边表：主体到订单
    if all(col in all_df.columns for col in ['需求方', '设计师', '任务编号', '交易价格', '发布时间']):
        # 需求方到订单的边
        edges_demand_order = all_df[['需求方', '任务编号', '交易价格', '发布时间']].dropna()
        edges_demand_order['Type'] = '发布'
        edges_demand_order.columns = ['Source', 'Target', 'Transaction_Price', 'Publication_Time', 'Type']

        # 设计师到订单的边
        edges_design_order = all_df[['设计师', '任务编号', '交易价格', '发布时间']].dropna()
        edges_design_order['Type'] = '完成'
        edges_design_order.columns = ['Source', 'Target', 'Transaction_Price', 'Publication_Time', 'Type']

        # 合并主体到订单的边
        edges_subject_order = pd.concat([edges_demand_order, edges_design_order], ignore_index=True)
        edges_so_path = os.path.join(output_dir, f"{prefix}edges_subject_order.csv")
        edges_subject_order.to_csv(edges_so_path, index=False, encoding='utf-8-sig')
        print(f"主体到订单边表已保存到: {edges_so_path}")
    else:
        print("警告: 缺少必要的列，无法生成主体到订单边。")

    # 5. 边表：订单到能力标签
    if all(col in all_df.columns for col in ['任务编号', '能力标签']):
        edges_order_ability = all_df[['任务编号', '能力标签']].dropna()
        edges_order_ability.columns = ['Source', 'Target']
        edges_order_ability['Type'] = '属于'
        edges_oa_path = os.path.join(output_dir, f"{prefix}edges_order_ability.csv")
        edges_order_ability.to_csv(edges_oa_path, index=False, encoding='utf-8-sig')
        print(f"订单到能力标签边表已保存到: {edges_oa_path}")
    else:
        print("警告: 缺少必要的列，无法生成订单到能力标签边。")

    print("--- 生成节点和边表完成 ---")

# --- 自动遍历文件夹并处理所有文件 ---
def process_all_files():
    # 确保输出目录存在
    if not os.path.exists(output_dir_base_main):
        os.makedirs(output_dir_base_main)
        print(f"创建目录: {output_dir_base_main}")

    # 用于收集所有领域的数据
    all_domains_data = {}

    # 遍历输入文件夹下的领域子文件夹
    for domain in os.listdir(input_base_dir):
        domain_path = os.path.join(input_base_dir, domain)
        if not os.path.isdir(domain_path):
            continue  # 跳过非文件夹

        # 创建领域对应的输出子目录
        output_main_subdir = os.path.join(output_dir_base_main, domain)
        if not os.path.exists(output_main_subdir):
            os.makedirs(output_main_subdir)
            print(f"创建领域输出目录: {output_main_subdir}")

        # 用于存储该领域的所有清洗后数据
        domain_data = []

        # 遍历领域文件夹中的文件
        for file_type in file_types:
            input_filename = f"{domain}-{file_type}.csv"
            input_path = os.path.join(domain_path, input_filename)
            if os.path.exists(input_path):
                # 处理单个文件
                df_cleaned = process_single_file(
                    input_path,
                    domain,
                    file_type,
                    output_main_subdir
                )
                if df_cleaned is not None:
                    domain_data.append(df_cleaned)
            else:
                print(f"警告: 文件 {input_path} 不存在，跳过处理。")

        # 合并该领域的所有数据并存储
        if domain_data:
            print(f"\n--- 合并领域 {domain} 的所有数据 ---")
            combined_domain_df = pd.concat(domain_data, ignore_index=True)
            # 确保没有任务编号重复
            if '任务编号' in combined_domain_df.columns:
                initial_count = len(combined_domain_df)
                combined_domain_df = combined_domain_df.dropna(subset=['任务编号'])
                combined_domain_df = combined_domain_df.drop_duplicates(subset=['任务编号'], keep='first')
                final_count = len(combined_domain_df)
                print(
                    f"领域 {domain} 数据合并完成，去重后共 {final_count} 条记录 (减少 {initial_count - final_count} 条)")
            all_domains_data[domain] = combined_domain_df

    # 合并所有领域的数据
    if all_domains_data:
        all_df = pd.concat(all_domains_data.values(), ignore_index=True)
        print(f"所有领域数据合并完成，共 {len(all_df)} 条记录。")

        # 生成全局节点和边
        generate_nodes_and_edges(all_df, output_dir_base_main)

        # 按年度划分并输出
        if '发布时间' in all_df.columns:
            all_df['年份'] = all_df['发布时间'].dt.year
            years = sorted(all_df['年份'].unique())
            for year in years:
                year_df = all_df[all_df['年份'] == year].copy()
                year_output_dir = os.path.join(output_dir_base_main, str(year))
                if not os.path.exists(year_output_dir):
                    os.makedirs(year_output_dir)
                    print(f"创建年度输出目录: {year_output_dir}")

                # 保存年度清洗数据
                year_cleaned_path = os.path.join(year_output_dir, f"年份{year}-清洗后完整数据.csv")
                year_df.to_csv(year_cleaned_path, index=False, encoding='utf-8-sig')
                print(f"年度清洗数据已保存到: {year_cleaned_path}")

                # 生成年度节点和边
                generate_nodes_and_edges(year_df, year_output_dir, year)
        else:
            print("警告: 缺少 '发布时间' 列，无法按年度划分。")
    else:
        print("警告: 没有有效数据用于生成节点和边表。")

# --- 执行处理 ---
if __name__ == "__main__":
    if not os.path.exists(input_base_dir):
        print(f"致命错误: 输入根目录不存在 - {input_base_dir}")
        print("请确保 'input_base_dir' 配置正确。")
    else:
        process_all_files()