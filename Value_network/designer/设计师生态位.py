'''import pandas as pd
import numpy as np
import cupy as cp
from tqdm import tqdm
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# 文件路径配置
input_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\处理后\\设计师订单信息-招标雇佣.csv"
matrix_output_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\计算\\设计师重叠度矩阵3.csv"
single_overlap_output_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\计算\\设计师宽度单一重叠度-招标雇佣3.csv"
resource_matrix_output_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\计算\\设计师资源利用矩阵3.csv"


# 计算生态位宽度
def calculate_niche_width(task_counts, debug_limit=10, debug_count=[0]):
    log_func = np.log
    try:
        total = task_counts.sum()
        if total == 0:
            if debug_count[0] < debug_limit:
                print(f"警告：task_counts 全为 0: {task_counts}")
                debug_count[0] += 1
            return 0
        proportions = task_counts / total
        if not np.isfinite(proportions).all():
            if debug_count[0] < debug_limit:
                print(f"警告：proportions 包含无效值: {proportions}")
                debug_count[0] += 1
            return 0
        proportions = proportions[proportions > 0]
        if len(proportions) == 0:
            if debug_count[0] < debug_limit:
                print(f"警告：proportions 为空（可能全 0 或单一任务）: {task_counts}")
                debug_count[0] += 1
            return 0
        if len(proportions) == 1:
            if debug_count[0] < debug_limit:
                print(f"信息：单一任务类型，宽度为 0: {task_counts}")
                debug_count[0] += 1
            return 0
        result = -np.sum(proportions * log_func(proportions))
        if not np.isfinite(result):
            if debug_count[0] < debug_limit:
                print(f"警告：result 无效: {result}, proportions: {proportions}")
                debug_count[0] += 1
            return 0
        return result
    except Exception as e:
        if debug_count[0] < debug_limit:
            print(f"计算生态位宽度出错: {e}, task_counts: {task_counts}")
            debug_count[0] += 1
        return 0


# 批量 GPU 计算重叠度（优化向量化）
def calculate_niche_overlap_batch(counts_list, task_types, batch_indices):
    try:
        n_tasks = len(batch_indices)
        n_types = len(task_types)
        indices_i, indices_j = zip(*batch_indices)
        indices_i = np.array(indices_i)
        indices_j = np.array(indices_j)

        # 预计算 totals
        totals = np.array([sum(counts_list[i].values()) for i in range(len(counts_list))])
        totals_i = totals[indices_i]
        totals_j = totals[indices_j]
        valid = (totals_i > 0) & (totals_j > 0)

        # 初始化 GPU 数组
        p_matrix = cp.zeros((n_tasks * 2, n_types), dtype=cp.float32)

        # 向量化填充 p_matrix
        for k, task in enumerate(task_types):
            counts_i = np.array([counts_list[i].get(task, 0) for i in indices_i])
            counts_j = np.array([counts_list[j].get(task, 0) for j in indices_j])
            p_matrix[:n_tasks, k] = cp.array(counts_i / totals_i, dtype=cp.float32)[valid]
            p_matrix[n_tasks:, k] = cp.array(counts_j / totals_j, dtype=cp.float32)[valid]

        p1 = p_matrix[:n_tasks]
        p2 = p_matrix[n_tasks:]
        numerator = cp.sum(p1 * p2, axis=1)
        denom1 = cp.sqrt(cp.sum(p1 ** 2, axis=1))
        denom2 = cp.sqrt(cp.sum(p2 ** 2, axis=1))
        denominator = denom1 * denom2
        overlaps = cp.where(denominator != 0, numerator / denominator, 0)

        result = np.zeros(n_tasks)
        result[valid] = overlaps.get()
        return result
    except Exception as e:
        print(f"批量重叠度计算出错: {e}")
        return np.zeros(len(batch_indices))


# 并行批量处理函数
def process_batch(start, end, indices, counts_list, task_types):
    batch_indices = indices[start:end]
    return calculate_niche_overlap_batch(counts_list, task_types, batch_indices), batch_indices


# 主处理流程
def process_niche_analysis():
    # 检查 np.log 是否为函数
    if not callable(np.log):
        raise ValueError(f"np.log 不是可调用的函数，当前类型: {type(np.log)}")

    # 检查 GPU 可用性
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            print("错误：未检测到 GPU 设备")
            return
    except Exception as e:
        print(f"GPU 初始化失败: {e}. 请检查 CUDA 和 CuPy 安装.")
        return

    # 读取设计师订单信息
    start_time = time.time()
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"错误：无法找到输入文件 {input_path}")
        return
    except Exception as e:
        print(f"读取文件出错: {e}")
        return
    print(f"读取数据耗时: {time.time() - start_time:.2f} 秒")

    # 验证数据
    required_columns = ['设计师', '任务类型', '任务编号']
    if not all(col in df.columns for col in required_columns):
        print(f"错误：输入文件中缺少以下列: {set(required_columns) - set(df.columns)}")
        return

    print(f"数据预览:\n{df.head()}")
    print(f"任务类型空值数量: {df['任务类型'].isna().sum()}")
    print(f"设计师空值数量: {df['设计师'].isna().sum()}")
    print(f"订单编号空值数量: {df['任务编号'].isna().sum()}")

    # 清理数据
    df = df.dropna(subset=required_columns)
    df['任务类型'] = df['任务类型'].str.strip()
    df['设计师'] = df['设计师'].str.strip()
    df['订单编号'] = df['任务编号'].str.strip()

    # 获取所有任务类型
    task_types = df['任务类型'].unique().tolist()
    print(f"任务类型数量: {len(task_types)}")
    print(f"任务类型: {task_types}")

    # 计算每个设计师的任务类型分布、订单编号和任务类型
    try:
        designer_task_counts = df.groupby('设计师')['任务类型'].value_counts().unstack(fill_value=0)
        designer_orders = df.groupby('设计师')['任务编号'].unique().apply(lambda x: ','.join(x)).to_dict()
        designer_task_types = df.groupby('设计师')['任务类型'].unique().apply(lambda x: ','.join(x)).to_dict()
    except Exception as e:
        print(f"生成 designer_task_counts 或 orders/task_types 出错: {e}")
        return
    designers = designer_task_counts.index.tolist()
    print(f"设计师数量: {len(designers)}")

    # 统计任务分布
    print("分析任务分布...")
    task_sums = designer_task_counts.sum(axis=1)
    print(f"任务总数范围: {task_sums.min()} - {task_sums.max()}")
    print(f"无任务设计师: {(task_sums == 0).sum()}")
    print(f"单一任务设计师: {(designer_task_counts.gt(0).sum(axis=1) == 1).sum()}")

    # 预计算 counts
    print("预计算任务分布...")
    counts_list = [dict(designer_task_counts.loc[designer]) for designer in
                   tqdm(designers, desc="预计算", mininterval=0.5, ncols=100, ascii=True, file=sys.stdout)]

    # 计算生态位宽度
    niche_widths = {}
    zero_width_count = 0
    print("计算生态位宽度...")
    for designer in tqdm(designers, desc="生态位宽度计算", mininterval=0.5, ncols=100, ascii=True, file=sys.stdout):
        counts = designer_task_counts.loc[designer]
        width = calculate_niche_width(counts)
        niche_widths[designer] = width
        if width == 0:
            zero_width_count += 1

    print(f"生态位宽度为 0 的设计师数量: {zero_width_count}/{len(designers)}")

    # 初始化重叠度矩阵
    overlap_matrix = np.zeros((len(designers), len(designers)))

    # 批量 GPU 计算重叠度
    print("计算重叠度矩阵...")
    pair_count = (len(designers) * (len(designers) - 1)) // 2
    batch_size = 100000  # 增大批量
    indices = [(i, j) for i in range(len(designers)) for j in range(i, len(designers))]

    start_time = time.time()
    with tqdm(total=pair_count, desc="重叠度矩阵计算", mininterval=2.0, ncols=100, ascii=True, file=sys.stdout,
              disable=False) as pbar:
        pbar.update(0)
        batch_starts = list(range(0, len(indices), batch_size))
        batch_ends = [min(start + batch_size, len(indices)) for start in batch_starts]

        # 并行处理批量
        max_workers = 4  # 调整为 CPU 核心数
        batch_func = partial(process_batch, indices=indices, counts_list=counts_list, task_types=task_types)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda args: batch_func(*args),
                                        [(start, end) for start, end in zip(batch_starts, batch_ends)]))

        # 合并结果
        for overlaps, batch_indices in results:
            for (i, j), overlap in zip(batch_indices, overlaps):
                overlap_matrix[i, j] = overlap
                if i != j:
                    overlap_matrix[j, i] = overlap
            pbar.update(len(batch_indices))
            pbar.refresh()
            if pbar.n % 100000 == 0 and pbar.n > 0:
                print(f"已处理 {pbar.n}/{pair_count} 对，耗时 {time.time() - start_time:.2f} 秒")

    print(f"重叠度矩阵计算完成，耗时 {time.time() - start_time:.2f} 秒")

    # 释放 GPU 内存
    cp.get_default_memory_pool().free_all_blocks()

    # 创建重叠度矩阵DataFrame
    overlap_matrix_df = pd.DataFrame(overlap_matrix, index=designers, columns=designers)

    # 计算单一重叠度和生态位宽度
    single_overlaps = []
    print("计算单一重叠度...")
    for i, designer in tqdm(enumerate(designers), desc="单一重叠度计算", mininterval=0.5, ncols=100, ascii=True,
                            file=sys.stdout):
        other_overlaps = [overlap_matrix[i, j] for j in range(len(designers)) if i != j]
        mean_overlap = np.mean(other_overlaps) if other_overlaps else 0
        niche_width = niche_widths.get(designer, 0)
        single_overlaps.append({
            '设计师': designer,
            '单一重叠度': mean_overlap,
            '生态位宽度': niche_width,
            '任务编号': designer_orders.get(designer, ''),
            '任务类型': designer_task_types.get(designer, '')
        })

    # 创建单一重叠度和生态位宽度DataFrame
    single_overlap_df = pd.DataFrame(single_overlaps)

    # 分析生态位宽度分布
    print("生态位宽度分布：")
    print(f"最小值: {single_overlap_df['生态位宽度'].min():.4f}")
    print(f"最大值: {single_overlap_df['生态位宽度'].max():.4f}")
    print(f"平均值: {single_overlap_df['生态位宽度'].mean():.4f}")
    print(f"零值数量: {(single_overlap_df['生态位宽度'] == 0).sum()}/{len(designers)}")

    # 分析任务编号和类型
    print("任务编号和类型统计：")
    print(f"平均任务数量: {task_sums.mean():.2f}")
    print(f"平均任务类型数量: {(designer_task_counts.gt(0).sum(axis=1)).mean():.2f}")

    # 保存资源利用矩阵
    try:
        designer_task_counts.to_csv(resource_matrix_output_path, encoding='utf-8-sig')
        print(f"资源利用矩阵已保存到 {resource_matrix_output_path}")
    except Exception as e:
        print(f"保存资源利用矩阵出错: {e}")
        return

    # 保存其他结果
    try:
        overlap_matrix_df.to_csv(matrix_output_path, encoding='utf-8-sig')
        print(f"重叠度矩阵已保存到 {matrix_output_path}")
        single_overlap_df.to_csv(single_overlap_output_path, encoding='utf-8-sig')
        print(f"单一重叠度和生态位宽度已保存到 {single_overlap_output_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return

    # 打印统计信息
    print(f"共处理 {len(designers)} 个设计师")
    print(f"单一重叠度范围: {single_overlap_df['单一重叠度'].min():.4f} - {single_overlap_df['单一重叠度'].max():.4f}")
    print(f"生态位宽度范围: {single_overlap_df['生态位宽度'].min():.4f} - {single_overlap_df['生态位宽度'].max():.4f}")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")


# 执行处理
if __name__ == "__main__":
    process_niche_analysis()'''
import pandas as pd
import numpy as np
import cupy as cp
from tqdm import tqdm
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# 文件路径配置
input_path = "D:/Documents/Desktop/一品威客数据采集4月/527test-设计师处理终极版/清洗预处理20251111/按年份合并结果/设计师订单信息-2014.csv"
#input_files = [
    #"D:\\Documents\\Desktop\\一品威客数据采集4月\\处理后\\设计师订单信息-招标&雇佣.csv",
    # 在这里添加其他CSV文件的完整路径，例如:
    # "D:\\path\\to\\your\\second_file.csv",
    # "D:\\path\\to\\your\\third_file.csv",
#]
matrix_output_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\test63\\设计师重叠度矩阵-2014.csv"
single_overlap_output_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\test63\\设计师宽度单一重叠度-2014.csv"
resource_matrix_output_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\test63\\设计师资源利用矩阵-2014.csv"

# 计算生态位宽度
def calculate_niche_width(task_counts, debug_limit=10, debug_count=[0]):
    log_func = np.log
    try:
        total = task_counts.sum()
        if total == 0:
            if debug_count[0] < debug_limit:
                print(f"警告：task_counts 全为 0: {task_counts}")
                debug_count[0] += 1
            return 0
        proportions = task_counts / total
        if not np.isfinite(proportions).all():
            if debug_count[0] < debug_limit:
                print(f"警告：proportions 包含无效值: {proportions}")
                debug_count[0] += 1
            return 0
        proportions = proportions[proportions > 0]
        if len(proportions) == 0:
            if debug_count[0] < debug_limit:
                print(f"警告：proportions 为空（可能全 0 或单一任务）: {task_counts}")
                debug_count[0] += 1
            return 0
        if len(proportions) == 1:
            if debug_count[0] < debug_limit:
                print(f"信息：单一任务类型，宽度为 0: {task_counts}")
                debug_count[0] += 1
            return 0
        result = -np.sum(proportions * log_func(proportions))
        if not np.isfinite(result):
            if debug_count[0] < debug_limit:
                print(f"警告：result 无效: {result}, proportions: {proportions}")
                debug_count[0] += 1
            return 0
        return result
    except Exception as e:
        if debug_count[0] < debug_limit:
            print(f"计算生态位宽度出错: {e}, task_counts: {task_counts}")
            debug_count[0] += 1
        return 0

# 批量 GPU 计算重叠度（优化向量化）
def calculate_niche_overlap_batch(counts_list, task_types, batch_indices):
    try:
        n_tasks = len(batch_indices)
        n_types = len(task_types)
        indices_i, indices_j = zip(*batch_indices)
        indices_i = np.array(indices_i)
        indices_j = np.array(indices_j)

        # 预计算 totals
        totals = np.array([sum(counts_list[i].values()) for i in range(len(counts_list))])
        totals_i = totals[indices_i]
        totals_j = totals[indices_j]
        valid = (totals_i > 0) & (totals_j > 0)

        # 初始化 GPU 数组
        p_matrix = cp.zeros((n_tasks * 2, n_types), dtype=cp.float32)

        # 向量化填充 p_matrix
        for k, task in enumerate(task_types):
            counts_i = np.array([counts_list[i].get(task, 0) for i in indices_i])
            counts_j = np.array([counts_list[j].get(task, 0) for j in indices_j])
            p_matrix[:n_tasks, k] = cp.array(counts_i / totals_i, dtype=cp.float32)[valid]
            p_matrix[n_tasks:, k] = cp.array(counts_j / totals_j, dtype=cp.float32)[valid]

        p1 = p_matrix[:n_tasks]
        p2 = p_matrix[n_tasks:]
        numerator = cp.sum(p1 * p2, axis=1)
        denom1 = cp.sqrt(cp.sum(p1 ** 2, axis=1))
        denom2 = cp.sqrt(cp.sum(p2 ** 2, axis=1))
        denominator = denom1 * denom2
        overlaps = cp.where(denominator != 0, numerator / denominator, 0)

        result = np.zeros(n_tasks)
        result[valid] = overlaps.get()
        return result
    except Exception as e:
        print(f"批量重叠度计算出错: {e}")
        return np.zeros(len(batch_indices))

# 并行批量处理函数
def process_batch(start, end, indices, counts_list, task_types):
    batch_indices = indices[start:end]
    return calculate_niche_overlap_batch(counts_list, task_types, batch_indices), batch_indices

# 主处理流程
def process_niche_analysis():
    # 检查 np.log 是否为函数
    if not callable(np.log):
        raise ValueError(f"np.log 不是可调用的函数，当前类型: {type(np.log)}")

    # 检查 GPU 可用性
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            print("错误：未检测到 GPU 设备")
            return
    except Exception as e:
        print(f"GPU 初始化失败: {e}. 请检查 CUDA 和 CuPy 安装.")
        return

    # 读取设计师订单信息
    start_time = time.time()
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"错误：无法找到输入文件 {input_path}")
        return
    except Exception as e:
        print(f"读取文件出错: {e}")
        return
    print(f"读取数据耗时: {time.time() - start_time:.2f} 秒")

    # 验证数据
    required_columns = ['设计师', '任务类型', '任务编号', '交易价格']
    if not all(col in df.columns for col in required_columns):
        print(f"错误：输入文件中缺少以下列: {set(required_columns) - set(df.columns)}")
        return

    print(f"数据预览:\n{df.head()}")
    print(f"任务类型空值数量: {df['任务类型'].isna().sum()}")
    print(f"设计师空值数量: {df['设计师'].isna().sum()}")
    print(f"任务编号空值数量: {df['任务编号'].isna().sum()}")
    print(f"交易价格空值数量: {df['交易价格'].isna().sum()}")

    # 清理数据
    df = df.dropna(subset=required_columns)
    df['任务类型'] = df['任务类型'].str.strip()
    df['设计师'] = df['设计师'].str.strip()
    df['任务编号'] = df['任务编号'].str.strip()



    # 调试：检查无法解析的价格
    invalid_prices = df[df['交易价格'].isna() | (df['交易价格'] == 0)]
    if not invalid_prices.empty:
        print(f"警告：发现 {len(invalid_prices)} 个无法解析的交易价格：")
        print(invalid_prices[['设计师', '任务编号', '交易价格_raw']].head(10))
    else:
        print("所有交易价格解析成功")

    # 将无效价格标记为 NaN，不填 0
    df['交易价格'] = df['交易价格'].replace(0, np.nan)

    # 获取所有任务类型
    task_types = df['任务类型'].unique().tolist()
    print(f"任务类型数量: {len(task_types)}")
    print(f"任务类型: {task_types}")

    # 计算每个设计师的任务类型分布、订单编号、任务类型和交易价格
    try:
        designer_task_counts = df.groupby('设计师')['任务类型'].value_counts().unstack(fill_value=0)
        designer_orders = df.groupby('设计师')['任务编号'].unique().apply(lambda x: ','.join(x)).to_dict()
        designer_task_types = df.groupby('设计师')['任务类型'].unique().apply(lambda x: ','.join(x)).to_dict()
        designer_prices = df.groupby('设计师')['交易价格'].apply(lambda x: ','.join(x.dropna().astype(str))).to_dict()
        designer_price_means = df.groupby('设计师')['交易价格'].mean().to_dict()
    except Exception as e:
        print(f"生成 designer_task_counts 或 orders/task_types/prices 出错: {e}")
        return
    designers = designer_task_counts.index.tolist()
    print(f"设计师数量: {len(designers)}")

    # 调试：检查交易价格为空或全为 0 的设计师
    print("检查交易价格数据...")
    for designer in designers[:10]:  # 仅打印前 10 个以减少输出
        prices = designer_prices.get(designer, '')
        mean_price = designer_price_means.get(designer, np.nan)
        print(f"设计师 {designer}: 交易价格={prices}, 价值位势={mean_price}")

    # 统计任务分布
    print("分析任务分布...")
    task_sums = designer_task_counts.sum(axis=1)
    print(f"任务总数范围: {task_sums.min()} - {task_sums.max()}")
    print(f"无任务设计师: {(task_sums == 0).sum()}")
    print(f"单一任务设计师: {(designer_task_counts.gt(0).sum(axis=1) == 1).sum()}")

    # 统计交易价格分布
    print("分析交易价格分布...")
    price_means = pd.Series(designer_price_means)
    print(f"交易价格平均值范围: {price_means.min():.2f} - {price_means.max():.2f}")
    print(f"平均交易价格均值: {price_means.mean():.2f}")
    print(f"交易价格缺失的设计师数量: {price_means.isna().sum()}")

    # 预计算 counts
    print("预计算任务分布...")
    counts_list = [dict(designer_task_counts.loc[designer]) for designer in
                   tqdm(designers, desc="预计算", mininterval=0.5, ncols=100, ascii=True, file=sys.stdout)]

    # 计算生态位宽度
    niche_widths = {}
    zero_width_count = 0
    print("计算生态位宽度...")
    for designer in tqdm(designers, desc="生态位宽度计算", mininterval=0.5, ncols=100, ascii=True, file=sys.stdout):
        counts = designer_task_counts.loc[designer]
        width = calculate_niche_width(counts)
        niche_widths[designer] = width
        if width == 0:
            zero_width_count += 1

    print(f"生态位宽度为 0 的设计师数量: {zero_width_count}/{len(designers)}")

    # 初始化重叠度矩阵
    overlap_matrix = np.zeros((len(designers), len(designers)))

    # 批量 GPU 计算重叠度
    print("计算重叠度矩阵...")
    pair_count = (len(designers) * (len(designers) - 1)) // 2
    batch_size = 100000
    indices = [(i, j) for i in range(len(designers)) for j in range(i, len(designers))]

    start_time = time.time()
    with tqdm(total=pair_count, desc="重叠度矩阵计算", mininterval=2.0, ncols=100, ascii=True, file=sys.stdout,
              disable=False) as pbar:
        pbar.update(0)
        batch_starts = list(range(0, len(indices), batch_size))
        batch_ends = [min(start + batch_size, len(indices)) for start in batch_starts]

        # 并行处理批量
        max_workers = 4
        batch_func = partial(process_batch, indices=indices, counts_list=counts_list, task_types=task_types)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda args: batch_func(*args),
                                        [(start, end) for start, end in zip(batch_starts, batch_ends)]))

        # 合并结果
        for overlaps, batch_indices in results:
            for (i, j), overlap in zip(batch_indices, overlaps):
                overlap_matrix[i, j] = overlap
                if i != j:
                    overlap_matrix[j, i] = overlap
            pbar.update(len(batch_indices))
            pbar.refresh()
            if pbar.n % 100000 == 0 and pbar.n > 0:
                print(f"已处理 {pbar.n}/{pair_count} 对，耗时 {time.time() - start_time:.2f} 秒")

    print(f"重叠度矩阵计算完成，耗时 {time.time() - start_time:.2f} 秒")

    # 释放 GPU 内存
    cp.get_default_memory_pool().free_all_blocks()

    # 创建重叠度矩阵DataFrame
    overlap_matrix_df = pd.DataFrame(overlap_matrix, index=designers, columns=designers)

    # 计算单一重叠度和生态位宽度
    single_overlaps = []
    print("计算单一重叠度...")
    for i, designer in tqdm(enumerate(designers), desc="单一重叠度计算", mininterval=0.5, ncols=100, ascii=True,
                            file=sys.stdout):
        other_overlaps = [overlap_matrix[i, j] for j in range(len(designers)) if i != j]
        mean_overlap = np.mean(other_overlaps) if other_overlaps else 0
        niche_width = niche_widths.get(designer, 0)
        prices = designer_prices.get(designer, '')
        value_potential = designer_price_means.get(designer, np.nan)  # 使用 NaN 避免 0
        single_overlaps.append({
            '设计师': designer,
            '单一重叠度': mean_overlap,
            '生态位宽度': niche_width,
            '任务编号': designer_orders.get(designer, ''),
            '任务类型': designer_task_types.get(designer, ''),
            '交易价格': prices,
            '价值位势': value_potential
        })

    # 创建单一重叠度和生态位宽度DataFrame
    single_overlap_df = pd.DataFrame(single_overlaps)

    # 分析生态位宽度分布
    print("生态位宽度分布：")
    print(f"最小值: {single_overlap_df['生态位宽度'].min():.4f}")
    print(f"最大值: {single_overlap_df['生态位宽度'].max():.4f}")
    print(f"平均值: {single_overlap_df['生态位宽度'].mean():.4f}")
    print(f"零值数量: {(single_overlap_df['生态位宽度'] == 0).sum()}/{len(designers)}")

    # 分析任务编号和类型
    print("任务编号和类型统计：")
    print(f"平均任务数量: {task_sums.mean():.2f}")
    print(f"平均任务类型数量: {(designer_task_counts.gt(0).sum(axis=1)).mean():.2f}")

    # 保存资源利用矩阵
    try:
        designer_task_counts.to_csv(resource_matrix_output_path, encoding='utf-8-sig')
        print(f"资源利用矩阵已保存到 {resource_matrix_output_path}")
    except Exception as e:
        print(f"保存资源利用矩阵出错: {e}")
        return

    # 保存其他结果
    try:
        overlap_matrix_df.to_csv(matrix_output_path, encoding='utf-8-sig')
        print(f"重叠度矩阵已保存到 {matrix_output_path}")
        single_overlap_df.to_csv(single_overlap_output_path, encoding='utf-8-sig')
        print(f"单一重叠度和生态位宽度已保存到 {single_overlap_output_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return

    # 打印统计信息
    print(f"共处理 {len(designers)} 个设计师")
    print(f"单一重叠度范围: {single_overlap_df['单一重叠度'].min():.4f} - {single_overlap_df['单一重叠度'].max():.4f}")
    print(f"生态位宽度范围: {single_overlap_df['生态位宽度'].min():.4f} - {single_overlap_df['生态位宽度'].max():.4f}")
    print(f"价值位势范围: {single_overlap_df['价值位势'].min():.2f} - {single_overlap_df['价值位势'].max():.2f}")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

# 执行处理
if __name__ == "__main__":
    process_niche_analysis()