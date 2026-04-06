import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os

# 文件路径配置
input_path = "D:/Documents/Desktop/一品威客数据采集4月/521计算/合并去重并清洗交易价格后数据.csv"
matrix_output_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\521计算\\设计师重叠度矩阵.csv"
single_overlap_output_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\521计算\\设计师宽度单一重叠度.csv"
resource_matrix_output_path = "D:\\Documents\\Desktop\\一品威客数据采集4月\\521计算\\设计师资源利用矩阵.csv"

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

# 批量 CPU 计算重叠度（优化向量化，移除CuPy）
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

        if not valid.any():
            print(f"警告：批次 {batch_indices[0]} - {batch_indices[-1]} 全无效")
            return np.zeros(n_tasks)

        # 一次性构建 counts 数组 - 使用float64提高精度
        counts_i = np.array([[counts_list[i].get(task, 0) for task in task_types] for i in indices_i], dtype=np.float64)
        counts_j = np.array([[counts_list[j].get(task, 0) for task in task_types] for j in indices_j], dtype=np.float64)

        # 计算比例
        p1 = np.zeros_like(counts_i)
        p2 = np.zeros_like(counts_j)

        # 只对有效的行计算比例
        valid_mask = valid.reshape(-1, 1)
        p1[valid] = counts_i[valid] / totals_i[valid].reshape(-1, 1)
        p2[valid] = counts_j[valid] / totals_j[valid].reshape(-1, 1)

        # 计算重叠度
        numerator = np.sum(p1 * p2, axis=1)
        denom1 = np.sqrt(np.sum(p1 ** 2, axis=1))
        denom2 = np.sqrt(np.sum(p2 ** 2, axis=1))
        denominator = denom1 * denom2

        # 处理对角线元素（自己与自己的重叠度应该为1）
        diagonal_mask = (indices_i == indices_j)
        overlaps = np.zeros(n_tasks)

        # 对角线元素：如果有任务则设为1，否则为0
        overlaps[diagonal_mask & valid] = 1.0

        # 非对角线元素：正常计算重叠度
        non_diagonal_mask = ~diagonal_mask & valid
        non_diagonal_valid = denominator[non_diagonal_mask] > 0
        if non_diagonal_valid.any():
            overlaps[non_diagonal_mask] = np.where(
                non_diagonal_valid,
                numerator[non_diagonal_mask][non_diagonal_valid] / denominator[non_diagonal_mask][non_diagonal_valid],
                0
            )

        return overlaps

    except Exception as e:
        print(f"批量重叠度计算出错: {e}")
        return np.zeros(len(batch_indices))


# 并行批量处理函数
def process_batch(start, end, indices, counts_list, task_types):
    batch_indices = indices[start:end]
    print(f"开始处理批次: {start} - {end}, 批次大小: {len(batch_indices)}")
    result = calculate_niche_overlap_batch(counts_list, task_types, batch_indices)
    print(f"完成批次: {start} - {end}")
    return result, batch_indices

# 主处理流程
def process_niche_analysis():
    # 检查 np.log 是否为函数
    if not callable(np.log):
        raise ValueError(f"np.log 不是可调用的函数，当前类型: {type(np.log)}")

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

    # 清理异常交易价格
    invalid_prices = df[df['交易价格'] > 1e10]
    if not invalid_prices.empty:
        print(f"警告：发现 {len(invalid_prices)} 个异常交易价格：")
        print(invalid_prices[['设计师', '任务编号', '交易价格']].head())
        df.loc[df['交易价格'] > 1e10, '交易价格'] = np.nan

    # 将无效价格标记为 NaN
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

    # 过滤单一任务类型设计师
    indices = [(i, j) for i in range(len(designers)) for j in range(i, len(designers))]

    print(f"总设计师对数: {len(indices)}")

    # 初始化重叠度矩阵
    overlap_matrix = np.zeros((len(designers), len(designers)), dtype=np.float64)  # 使用float64

    # 批量计算重叠度
    print("计算重叠度矩阵...")
    batch_size = 10000
    batch_starts = list(range(0, len(indices), batch_size))
    batch_ends = [min(start + batch_size, len(indices)) for start in batch_starts]

    start_time = time.time()
    with tqdm(total=len(indices), desc="重叠度矩阵计算", mininterval=2.0, ncols=100, ascii=True,
              file=sys.stdout) as pbar:
        max_workers = min(os.cpu_count() or 4, 16)
        batch_func = partial(process_batch, indices=indices, counts_list=counts_list, task_types=task_types)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda args: batch_func(*args),
                                        [(start, end) for start, end in zip(batch_starts, batch_ends)]))

        # 合并结果
        for overlaps, batch_indices in results:
            for (i, j), overlap in zip(batch_indices, overlaps):
                overlap_matrix[i, j] = overlap
                if i != j:  # 只有非对角线元素需要对称填充
                    overlap_matrix[j, i] = overlap
            pbar.update(len(batch_indices))

    # 验证对角线元素
    diagonal_elements = np.diag(overlap_matrix)
    zero_diagonal_count = np.sum(diagonal_elements == 0)
    print(f"对角线为0的元素数量: {zero_diagonal_count}/{len(designers)}")
    print(f"对角线元素范围: {diagonal_elements.min():.4f} - {diagonal_elements.max():.4f}")

    # 如果仍有对角线为0的情况，手动修复
    if zero_diagonal_count > 0:
        print("手动修复对角线元素...")
        for i in range(len(designers)):
            if diagonal_elements[i] == 0:
                # 检查该设计师是否有任务
                total_tasks = sum(counts_list[i].values())
                if total_tasks > 0:
                    overlap_matrix[i, i] = 1.0
                    print(f"修复设计师 {designers[i]} 的对角线元素: {total_tasks} 个任务")
    '''valid_designers = [d for d in designers if niche_widths.get(d, 0) > 0]
    valid_indices = [(i, j) for i in range(len(designers)) for j in range(i, len(designers))
                     if niche_widths[designers[i]] > 0 and niche_widths[designers[j]] > 0]
    pair_count = len(valid_indices)
    print(f"有效设计师对: {pair_count}/{len(designers) * (len(designers) - 1) // 2}")
    indices = valid_indices if valid_indices else [(i, j) for i in range(len(designers)) for j in range(i, len(designers))]     修复了重叠度矩阵对角线有0的情况'''

    # 初始化重叠度矩阵


    '''overlap_matrix = np.zeros((len(designers), len(designers)))

    # 批量 CPU 计算重叠度
    print("计算重叠度矩阵...")
    batch_size = 10000  # CPU下建议较大批次
    batch_starts = list(range(0, len(indices), batch_size))
    batch_ends = [min(start + batch_size, len(indices)) for start in batch_starts]

    start_time = time.time()
    with tqdm(total=pair_count, desc="重叠度矩阵计算", mininterval=2.0, ncols=100, ascii=True, file=sys.stdout,
              disable=False) as pbar:
        pbar.update(0)
        max_workers = min(os.cpu_count() or 4, 16)  # 动态调整线程数
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
            if pbar.n % 1000000 == 0 and pbar.n > 0:
                overlap_matrix_df = pd.DataFrame(overlap_matrix, index=designers, columns=designers)
                overlap_matrix_df.to_csv(f"{matrix_output_path}.checkpoint", encoding='utf-8-sig')
                print(f"保存检查点: {matrix_output_path}.checkpoint")'''

    print(f"重叠度矩阵计算完成，耗时 {time.time() - start_time:.2f} 秒")

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
        value_potential = designer_price_means.get(designer, np.nan)
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