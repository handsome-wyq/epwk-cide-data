import pandas as pd
import os
import re

# =====================
# 路径配置
# =====================

# 1）主平台任务（单人悬赏/招标/雇佣/计件/多人悬赏）
main_base_dir = r"D:/Documents/Desktop/Crowd_intelligence_wyq/Knowledge_network/知识系统"
main_input_dir = os.path.join(main_base_dir, "数据")  # 原始任务数据根目录（按领域划分）

# 2）直接雇佣
direct_base_dir = r"D:/Documents/Desktop/Crowd_intelligence_wyq/Knowledge_network"
direct_orders_dir = os.path.join(direct_base_dir, "825补充数据-直接雇佣")

# 3）输出目录：能力向量 & 金额
output_base_dir = r"D:/Documents/Desktop/Crowd_intelligence_wyq/Knowledge_network/知识网络分析/能力向量与金额"
os.makedirs(output_base_dir, exist_ok=True)

# 类型列表
file_types_main = ["单人悬赏", "招标", "雇佣", "计件", "多人悬赏"]
years = list(range(2014, 2026))

# =====================
# 工具函数
# =====================

def extract_bracket_content(title: str):
    """
    从标题中提取【】内的能力标签
    """
    if pd.isna(title):
        return None
    m = re.search(r'【(.*?)】', str(title))
    return m.group(1).strip() if m else None


def clean_price(price):
    """
    通用价格清洗函数（主平台 + 直接雇佣统一使用）

    规则：
    1）先去掉英文/中文逗号（当千位符或辅助分隔）；
    2）统一货币符号，把 ¥ 换成 ￥；
    3）如果包含“万”或“千”，不处理，返回 None；
    4）按 '￥' 分段（有多个价格就取第一个非空段）：
         例如 "￥200,￥20" -> 去逗号 -> "￥200￥20" -> ["", "200", "20"] -> 取 "200"
    5）如果价格段中存在明显的区间形式（如 2000-3000），也不处理，返回 None；
    6）在选中的那一段里：
         - 去掉 '￥'、'元' 等文字
         - 提取第一个数字（兼容 5000/套 等）
    """
    if pd.isna(price):
        return None

    s = str(price)
    # 去掉不间断空格等奇怪空格
    s = s.replace('\u00a0', ' ').strip()
    if not s:
        return None

    # 如果价格中包含 "万" 或 "千"，按你的要求：不处理，直接跳过
    if '万' in s or '千' in s:
        return None

    # 先去掉英文/中文逗号（当千位符或辅助分隔）
    s = s.replace(',', '').replace('，', '')

    # 统一货币符号
    s = s.replace('¥', '￥')

    # 按 '￥' 分段，如果没有 '￥'，就整个串作为一段
    if '￥' in s:
        parts = [p.strip() for p in s.split('￥') if p.strip()]
    else:
        parts = [s.strip()] if s.strip() else []

    if not parts:
        return None

    # 取第一个价格段
    seg = parts[0]

    # 如果这段里面有明显的区间形式（2000-3000），也跳过
    if re.search(r'\d+\s*-\s*\d+', seg):
        return None

    # 去掉单位&货币文字
    seg = seg.replace('元', '').replace('￥', '').strip()
    if not seg:
        return None

    # 提取第一个数字（兼容 5000/套 之类）
    m = re.search(r'(\d+\.?\d*)', seg)
    if not m:
        return None

    try:
        return float(m.group(1))
    except ValueError:
        return None


# =====================
# ① 主平台：生成 designer-year-ability-price 明细
# =====================

def load_main_orders_for_vectors():
    """
    返回 DataFrame，列包含：
      designer  : 设计师（名称或虚拟ID）
      year      : 年份
      ability   : 能力标签（标题【】）
      price     : 金额（float，可为 NaN）
    """
    all_rows = []

    if not os.path.exists(main_input_dir):
        print(f"[主任务] 输入目录不存在: {main_input_dir}")
        return pd.DataFrame(columns=['designer', 'year', 'ability', 'price'])

    for domain in os.listdir(main_input_dir):
        domain_path = os.path.join(main_input_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        for ft in file_types_main:
            input_filename = f"{domain}-{ft}.csv"
            input_path = os.path.join(domain_path, input_filename)
            if not os.path.exists(input_path):
                print(f"[主任务] 跳过不存在的文件: {input_path}")
                continue

            try:
                df = pd.read_csv(input_path, encoding='utf-8-sig')
                print(f"[主任务] 读取 {input_path}, {len(df)} 条")
            except Exception as e:
                print(f"[主任务] 读取 {input_path} 出错: {e}")
                continue

            df_cleaned = df.copy()

            # 基础字段检查
            if '标题' not in df_cleaned.columns or '任务编号' not in df_cleaned.columns:
                print(f"[主任务] {input_path} 缺少 '标题' 或 '任务编号'，跳过。")
                continue

            df_cleaned.dropna(subset=['标题', '任务编号'], inplace=True)

            # 发布时间 -> 年份
            if '发布时间' in df_cleaned.columns:
                df_cleaned['发布时间'] = pd.to_datetime(
                    df_cleaned['发布时间'],
                    format='%Y.%m.%d',
                    errors='coerce'
                )
                df_cleaned.dropna(subset=['发布时间'], inplace=True)
                df_cleaned = df_cleaned[df_cleaned['发布时间'].dt.year >= 2014].copy()
                df_cleaned['year'] = df_cleaned['发布时间'].dt.year.astype(int)
            else:
                df_cleaned['year'] = None

            # 价格（交易价格优先，其次需求价格）
            if '交易价格' in df_cleaned.columns:
                if '需求价格' in df_cleaned.columns:
                    df_cleaned['交易价格'] = df_cleaned['交易价格'].where(
                        df_cleaned['交易价格'].notna(),
                        df_cleaned['需求价格']
                    )
                df_cleaned['price'] = df_cleaned['交易价格'].apply(clean_price)
            else:
                df_cleaned['price'] = None

            # 能力标签
            df_cleaned['ability'] = df_cleaned['标题'].apply(extract_bracket_content)

            # 设计师字段处理
            if ft == "多人悬赏":
                # 多人悬赏：设计师1 / 设计师2
                if '设计师1' in df_cleaned.columns and '设计师2' in df_cleaned.columns:
                    def get_designers(row):
                        ds = []
                        if pd.notna(row['设计师1']):
                            ds.append(str(row['设计师1']).strip())
                        if pd.notna(row['设计师2']):
                            ds.append(str(row['设计师2']).strip())
                        return ds

                    df_cleaned['设计师_list'] = df_cleaned.apply(get_designers, axis=1)
                    df_cleaned = df_cleaned.explode('设计师_list')
                    df_cleaned['designer'] = df_cleaned['设计师_list']
                    df_cleaned.drop(columns=['设计师_list'], inplace=True)
                else:
                    if '设计师' in df_cleaned.columns:
                        df_cleaned['designer'] = df_cleaned['设计师'].astype(str).str.strip()
                    else:
                        df_cleaned['designer'] = None
            else:
                # 非多人悬赏
                if '设计师' in df_cleaned.columns:
                    def fill_designer(row):
                        if pd.notna(row['设计师']) and str(row['设计师']).strip() != "":
                            return str(row['设计师']).strip()
                        # 原始设计师为空时，用虚拟ID：B+任务编号
                        return "B" + str(row['任务编号']).strip()
                    df_cleaned['designer'] = df_cleaned.apply(fill_designer, axis=1)
                else:
                    df_cleaned['designer'] = df_cleaned['任务编号'].apply(
                        lambda x: "B" + str(x).strip() if pd.notna(x) else None
                    )

            # 保留需要的列
            sub = df_cleaned[['designer', 'year', 'ability', 'price']].copy()
            sub.dropna(subset=['designer', 'year', 'ability'], inplace=True)
            all_rows.append(sub)

    if not all_rows:
        print("[主任务] 没有有效数据。")
        return pd.DataFrame(columns=['designer', 'year', 'ability', 'price'])

    all_df_main = pd.concat(all_rows, ignore_index=True)
    print(f"[主任务] 合并后共有 {len(all_df_main)} 条 designer-year-ability 记录")
    return all_df_main


# =====================
# ② 直接雇佣：生成 designer-year-ability-price 明细
# =====================

def load_direct_orders_for_vectors():
    """
    返回 DataFrame，列包含：
      designer : B+订单编号后缀（虚拟设计师）
      year     : 年份（文件名中的年份）
      ability  : 能力标签（标题【】）
      price    : 金额（float，可为 NaN）
    """
    all_rows = []

    for year in years:
        orders_path = os.path.join(direct_orders_dir, f"A{year}.csv")
        if not os.path.exists(orders_path):
            print(f"[直接雇佣] {orders_path} 不存在，跳过该年。")
            continue

        try:
            df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
            print(f"[直接雇佣] 读取 {orders_path}, {len(df_orders)} 条")
        except Exception as e:
            print(f"[直接雇佣] 读取 {orders_path} 出错：{e}")
            continue

        if '字段3' not in df_orders.columns:
            print(f"[直接雇佣] {orders_path} 缺少 '字段3'（订单编号），跳过该年。")
            continue

        df_orders['字段3'] = df_orders['字段3'].astype(str).str.strip()
        df_orders = df_orders[df_orders['字段3'] != ""].copy()
        if df_orders.empty:
            print(f"[直接雇佣] {orders_path} 无有效订单编号。")
            continue

        # 能力标签
        if '标题' in df_orders.columns:
            df_orders['ability'] = df_orders['标题'].apply(extract_bracket_content)
        else:
            df_orders['ability'] = None

        # 价格：列名为“价格”
        if '价格' in df_orders.columns:
            df_orders['price'] = df_orders['价格'].apply(clean_price)
        else:
            df_orders['price'] = None

        # 设计师虚拟ID：B+订单编号里的数字后缀
        def get_suffix(order_raw):
            if pd.isna(order_raw):
                return None
            s = str(order_raw).strip()
            if not s:
                return None
            m = re.search(r'(\d+)', s)
            return m.group(1) if m else s

        df_orders['订单编号后缀'] = df_orders['字段3'].apply(get_suffix)
        df_orders['designer'] = df_orders['订单编号后缀'].apply(
            lambda x: f"B{x}" if pd.notna(x) else None
        )

        df_orders['year'] = year

        sub = df_orders[['designer', 'year', 'ability', 'price']].copy()
        sub.dropna(subset=['designer', 'year', 'ability'], inplace=True)
        all_rows.append(sub)

    if not all_rows:
        print("[直接雇佣] 没有有效数据。")
        return pd.DataFrame(columns=['designer', 'year', 'ability', 'price'])

    all_df_direct = pd.concat(all_rows, ignore_index=True)
    print(f"[直接雇佣] 合并后共有 {len(all_df_direct)} 条 designer-year-ability 记录")
    return all_df_direct


# =====================
# ③ 构建能力向量 + numeric index + Heatmap 透视表
# =====================

def build_designer_ability_vectors():
    # 1. 主平台数据
    df_main = load_main_orders_for_vectors()

    # 2. 直接雇佣数据
    df_direct = load_direct_orders_for_vectors()

    # 3. 合并
    df_all = pd.concat([df_main, df_direct], ignore_index=True)
    if df_all.empty:
        print("没有可用的 designer-year-ability 数据，无法构建能力向量。")
        return

    # 确保 year 为 int
    df_all['year'] = df_all['year'].astype(int)

    # 4. 计算 designer-year-ability 的订单数 & 总金额
    grouped = (
        df_all
        .groupby(['designer', 'year', 'ability'], as_index=False)
        .agg(
            order_count=('designer', 'size'),
            total_amount=('price', 'sum')
        )
    )
    grouped['total_amount'] = grouped['total_amount'].fillna(0.0)

    # 5. 输出总表（不含 index，作为整体统计用）
    all_path = os.path.join(output_base_dir, "designer_ability_vectors_all_years.csv")
    grouped.to_csv(all_path, index=False, encoding='utf-8-sig')
    print(f"[输出] 全部年份的 designer-year-ability 向量已保存到: {all_path}")

    # 6. 按年度输出：加 numeric index + Heatmap 透视表
    for year in sorted(grouped['year'].unique()):
        year_df = grouped[grouped['year'] == year].copy()

        # ---- 6.1 生成 numeric index ----
        designers = sorted(year_df['designer'].unique())
        abilities = sorted(year_df['ability'].unique())

        designer_to_idx = {d: i + 1 for i, d in enumerate(designers)}  # 从 1 开始
        ability_to_idx = {a: j + 1 for j, a in enumerate(abilities)}

        year_df['designer_idx'] = year_df['designer'].map(designer_to_idx)
        year_df['ability_idx'] = year_df['ability'].map(ability_to_idx)

        # ---- 6.2 保存“向量+index”的年度表 ----
        year_dir = os.path.join(output_base_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        year_csv_path = os.path.join(year_dir, f"designer_ability_vectors_{year}.csv")
        year_df.to_csv(year_csv_path, index=False, encoding='utf-8-sig')
        print(f"[输出] 年份 {year} 的 designer-ability 向量（含 index）已保存到: {year_csv_path}")

        # ---- 6.3 生成 2D Heatmap 透视表：designer × ability，值为 total_amount ----
        pivot = year_df.pivot_table(
            index='designer',
            columns='ability',
            values='total_amount',
            aggfunc='sum',
            fill_value=0.0
        )

        heatmap_csv_path = os.path.join(year_dir, f"heatmap_amount_designer_by_ability_{year}.csv")
        pivot.to_csv(heatmap_csv_path, encoding='utf-8-sig')
        print(f"[输出] 年份 {year} 的 Heatmap 透视表已保存到: {heatmap_csv_path}")

    print("\n=== 设计师年度能力向量（含 numeric index + Heatmap 透视表）全部生成完成 ===")
    print(f"总记录数: {len(grouped)}")
    print(f"设计师数: {grouped['designer'].nunique()}")
    print(f"能力标签数: {grouped['ability'].nunique()}")
    print(f"年份数: {grouped['year'].nunique()}")


# =====================
# 主入口
# =====================

if __name__ == "__main__":
    build_designer_ability_vectors()

