import pandas as pd
import os
import re
import json

# ========= 知识字符串拆分与清洗 =========

def split_knowledge_items(text):
    """
    输入一条“知识字符串”，可能包含多个知识点：
    - 用  ， 、 , 空格 等分隔
    - 用 + 连接的，例如: php+mysql, 名称+域名+名字来源+商标是否被注册
      但要保留 C++ 这类（不在 ++ 上拆）

    同时：
    - 剔除网址（含 http://, https://, www.）
    - 剔除纯数字或数字区间（如 2000-3000元、100、3.14）
    返回：清洗后且去重的知识列表
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    s = str(text).strip()
    if not s:
        return []

    # 统一括号
    s = s.replace('（', '(').replace('）', ')')

    # 第一步：按“单个 +（不是 ++）”切一刀
    parts_plus = re.split(r'(?<!\+)\+(?!\+)', s)

    tokens = []
    for part in parts_plus:
        # 再按中文逗号、英文逗号、顿号、分号、斜杠、空白拆分
        sub_parts = re.split(r'[，,、;/\s]+', part)
        for t in sub_parts:
            t = t.strip("()（）[]【】\"' ")
            if t:
                tokens.append(t)

    # 过滤：网址、纯数字/价格
    cleaned = []
    for tok in tokens:
        # 过滤网址
        if re.search(r'https?://|www\.', tok):
            continue
        # 去掉单位只看数字和 - . 判断是否数值或区间
        num_like = re.sub(r'[^\d\.\-]', '', tok)
        if num_like and re.fullmatch(r'[\d\.\-]+', num_like):
            # 2000, 2000-3000, 3.14 这一类当作数值丢弃
            continue
        cleaned.append(tok)

    # 去重（保持原有顺序）
    seen = set()
    result = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            result.append(t)

    return result


# ========= 路径配置 =========

# 1）主平台任务（单人悬赏/招标/雇佣/计件/多人悬赏）
main_base_dir   = r"D:/Documents/Desktop/Crowd_intelligence_wyq/Knowledge_network/知识系统"
main_input_dir  = os.path.join(main_base_dir, "数据")  # 原始任务数据根目录（按领域划分）

# 2）主平台任务的知识 JSON（按年）
json_knowledge_dir = r"D:/Documents/Desktop/Crowd_intelligence_wyq/Knowledge_network/抽取输出/2-添加时间后/SortedByYearNEW"

# 3）直接雇佣的订单与知识
direct_base_dir      = r"D:/Documents/Desktop/Crowd_intelligence_wyq/Knowledge_network"
direct_orders_dir    = os.path.join(direct_base_dir, "825补充数据-直接雇佣")
direct_knowledge_dir = os.path.join(direct_base_dir, "直接雇佣知识抽取")

# 4）总输出目录
global_output_dir = r"D:/Documents/Desktop/Crowd_intelligence_wyq/F1_网络构建/网络数据1"
os.makedirs(global_output_dir, exist_ok=True)

file_types_main = ["单人悬赏", "招标", "雇佣", "计件", "多人悬赏"]
years = list(range(2014, 2026))


# ========= 通用小工具函数 =========

def extract_bracket_content(title):
    """从标题中提取【】内的能力标签"""
    if pd.isna(title):
        return None
    m = re.search(r'【(.*?)】', str(title))
    return m.group(1).strip() if m else None


def parse_price_to_float(price):
    """
    通用价格解析函数，处理以下形式并转为单一数值（float）：
    - '¥1,000' / '￥1,000' / '1000元'
    - '1000-3000元'
    - '2万-5万元'
    - '1000元以内' / '1000元以下' / '1000元以上'
    - '17.62万' / '20万-30万元'
    - '￥200,￥20' 取第一个价格

    规则：
    - 若是区间：取区间中点 (low + high) / 2
    - 若出现“万”：按 1 万 = 10000 元处理
    - “以内/以下”：取上限值；“以上”：也取该数字作为代表值
    """
    if pd.isna(price):
        return None

    s = str(price).strip()
    if not s:
        return None

    # 去掉千分位逗号
    s = s.replace(',', '').replace('，', '').replace(' ', '')

    # 如果出现多个“￥/¥”，只取第一个价格
    parts = re.split(r'[￥¥]', s)
    parts = [p for p in parts if p]  # 去掉空串
    if parts:
        s = parts[0]

    # 统一把“元”去掉，方便正则
    s = s.replace('元', '').replace('块', '')

    # ---------- 1. 处理“万”区间：如 '2万-5万'、'20万-30万' ----------
    m = re.search(r'(\d+(\.\d+)?)万\s*[-~－—]\s*(\d+(\.\d+)?)万?', s)
    if m:
        low = float(m.group(1)) * 10000
        high = float(m.group(3)) * 10000
        return (low + high) / 2.0

    # ---------- 2. 处理普通区间：'1000-3000' ----------
    m = re.search(r'(\d+(\.\d+)?)\s*[-~－—]\s*(\d+(\.\d+)?)', s)
    if m:
        low = float(m.group(1))
        high = float(m.group(3))
        return (low + high) / 2.0

    # ---------- 3. 单值带“万”：'17.62万' ----------
    m = re.search(r'(\d+(\.\d+)?)万', s)
    if m:
        return float(m.group(1)) * 10000

    # ---------- 4. 单值（含“以内/以下/以上”等） ----------
    m = re.search(r'(\d+(\.\d+)?)', s)
    if m:
        return float(m.group(1))

    return None


# 为兼容旧名字，如果其它地方还在调用这两个函数，可以直接转调
def clean_transaction_price(price):
    return parse_price_to_float(price)

def clean_demand_price(price):
    return parse_price_to_float(price)



def parse_date(dt_str):
    """解析日期字符串为 Timestamp"""
    return pd.to_datetime(dt_str, errors='coerce')


def parse_date_to_str(dt_str):
    """把任何日期字符串转成 YYYY-MM-DD 字符串"""
    dt = parse_date(dt_str)
    if pd.isna(dt):
        return None
    return dt.strftime('%Y-%m-%d')


# ========= ① 读取&清洗 主平台任务 数据 =========

def load_main_orders():
    """
    返回 df_main_all：
    至少包含列：
    ['任务编号', '发布时间', '需求方', '设计师', '能力标签', '需求文本', '标题',
     '交易价格'(数值), '需求价格数值'(数值) ...]
    """
    all_dfs = []

    if not os.path.exists(main_input_dir):
        print(f"主平台任务输入目录不存在: {main_input_dir}")
        return pd.DataFrame()

    for domain in os.listdir(main_input_dir):
        domain_path = os.path.join(main_input_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        for ft in file_types_main:
            input_filename = f"{domain}-{ft}.csv"
            input_path = os.path.join(domain_path, input_filename)
            if not os.path.exists(input_path):
                print(f"  [主任务] 跳过不存在的文件: {input_path}")
                continue

            try:
                df = pd.read_csv(input_path, encoding='utf-8-sig')
                print(f"  [主任务] 读取 {input_path}, {len(df)} 条")
            except Exception as e:
                print(f"  [主任务] 读取 {input_path} 出错: {e}")
                continue

            df_cleaned = df.copy()

            # a) 过滤关键列空值
            df_cleaned.dropna(subset=['标题', '需求文本', '任务编号'], inplace=True)

            # b) 价格清洗：同时处理“交易价格”和“需求价格”，并双向补全
            # 保留原始文本列不动，只新建数值列
            if '交易价格' in df_cleaned.columns:
                df_cleaned['交易价格数值'] = df_cleaned['交易价格'].apply(parse_price_to_float)
            else:
                df_cleaned['交易价格数值'] = None

            if '需求价格' in df_cleaned.columns:
                df_cleaned['需求价格数值'] = df_cleaned['需求价格'].apply(parse_price_to_float)
            else:
                df_cleaned['需求价格数值'] = None

            df_cleaned['交易价格数值'] = df_cleaned['交易价格数值'].where(
                df_cleaned['交易价格数值'].notna(),
                df_cleaned['需求价格数值']
            )

            # e) 发布时间 >= 2014
            if '发布时间' in df_cleaned.columns:
                df_cleaned['发布时间'] = pd.to_datetime(
                    df_cleaned['发布时间'],
                    format='%Y.%m.%d',
                    errors='coerce'
                )
                df_cleaned.dropna(subset=['发布时间'], inplace=True)
                df_cleaned = df_cleaned[df_cleaned['发布时间'].dt.year >= 2014].copy()
            else:
                df_cleaned['发布时间'] = pd.NaT

            # f) 去重任务编号（保持每个任务一条“基础记录”）
            df_cleaned = df_cleaned.sort_values(by='发布时间')
            df_cleaned = df_cleaned.drop_duplicates(subset=['任务编号'], keep='first')

            # g) 补充主体（需求方/设计师）
            if '需求方' in df_cleaned.columns:
                df_cleaned['需求方'] = df_cleaned.apply(
                    lambda row: "A" + str(row['任务编号'])
                    if pd.isna(row['需求方']) and pd.notna(row['任务编号'])
                    else row['需求方'],
                    axis=1
                )
            else:
                df_cleaned['需求方'] = df_cleaned['任务编号'].apply(
                    lambda x: "A" + str(x) if pd.notna(x) else None
                )

            if ft == "多人悬赏":
                # 多人悬赏：设计师1 / 设计师2 列
                if '设计师1' in df_cleaned.columns and '设计师2' in df_cleaned.columns:
                    def get_designers(row):
                        ds = []
                        if pd.notna(row['设计师1']):
                            ds.append(row['设计师1'])
                        if pd.notna(row['设计师2']):
                            ds.append(row['设计师2'])
                        if not ds and pd.notna(row['任务编号']):
                            ds.append("B" + str(row['任务编号']))
                        return ds

                    df_cleaned['设计师_list'] = df_cleaned.apply(get_designers, axis=1)
                    df_cleaned = df_cleaned.explode('设计师_list')
                    df_cleaned['设计师'] = df_cleaned['设计师_list']
                    df_cleaned.drop(columns=['设计师_list', '设计师1', '设计师2'],
                                    inplace=True, errors='ignore')
                else:
                    df_cleaned['设计师'] = df_cleaned['任务编号'].apply(
                        lambda x: "B" + str(x) if pd.notna(x) else None
                    )
            else:
                if '设计师' in df_cleaned.columns:
                    df_cleaned['设计师'] = df_cleaned.apply(
                        lambda row: "B" + str(row['任务编号'])
                        if pd.isna(row['设计师']) and pd.notna(row['任务编号'])
                        else row['设计师'],
                        axis=1
                    )
                else:
                    df_cleaned['设计师'] = df_cleaned['任务编号'].apply(
                        lambda x: "B" + str(x) if pd.notna(x) else None
                    )

            # h) 能力标签
            df_cleaned['能力标签'] = df_cleaned['标题'].apply(extract_bracket_content)

            all_dfs.append(df_cleaned)

    if not all_dfs:
        print("主平台任务没有有效数据。")
        return pd.DataFrame()

    all_df_main = pd.concat(all_dfs, ignore_index=True)
    print(f"[主任务] 合并后共有 {len(all_df_main)} 条记录")
    return all_df_main


# ========= ①-1 主体 ID 映射（解决“既当需求方又当设计师”的同名问题） =========

def build_subject_id_mappers(df_main_all):
    """
    对于既是需求方又是设计师的主体：
      - 需求方 ID = 'A' + Name
      - 设计师 ID = 'B' + Name
    否则 ID = Name
    返回两个函数：map_demander_id, map_designer_id
    """
    demanders = set(df_main_all['需求方'].dropna().unique())
    designers = set(df_main_all['设计师'].dropna().unique())
    duplicates = demanders.intersection(designers)

    def map_demander_id(name):
        if name in duplicates:
            return 'A' + str(name)
        return str(name)

    def map_designer_id(name):
        if name in duplicates:
            return 'B' + str(name)
        return str(name)

    return map_demander_id, map_designer_id


# ========= ①-2 主平台：构建“全局”主体/能力/订单 + 边 =========
#          新路径：需求方→订单→知识，设计师→能力→订单→知识

def build_main_nodes_and_edges_global(df_main_all, map_demander_id, map_designer_id):
    """
    返回：
      df_subject, df_ability, df_order,
      df_edges_sa (设计师->能力),
      df_edges_ao (能力->订单),
      df_edges_so (需求方->订单)
    """
    subject_nodes = []
    ability_nodes = []
    order_nodes   = []
    edges_sa = []  # 设计师 -> 能力（完成）
    edges_ao = []  # 能力 -> 订单
    edges_so = []  # 需求方 -> 订单（发起）

    # 主体节点
    demanders = set(df_main_all['需求方'].dropna().unique())
    designers = set(df_main_all['设计师'].dropna().unique())

    for d in demanders:
        sid = map_demander_id(d)
        subject_nodes.append([sid, d, '需求方', 1])

    for d in designers:
        sid = map_designer_id(d)
        subject_nodes.append([sid, d, '设计师', 1])

    # 能力节点
    if '能力标签' in df_main_all.columns:
        for ab in df_main_all['能力标签'].dropna().unique():
            ability_nodes.append([ab, ab, '能力', 1])

    # 订单节点（带需求文本、需求/交易价格、标题、时间）
    if '任务编号' in df_main_all.columns:
        cols_exist = [c for c in ['任务编号', '发布时间', '需求文本', '标题', '需求价格数值', '交易价格数值'] if c in df_main_all.columns]
        orders_df = df_main_all[cols_exist].dropna(subset=['任务编号'])
        orders_df = orders_df.sort_values(by='发布时间').drop_duplicates(subset=['任务编号'], keep='first')

        for _, row in orders_df.iterrows():
            oid = row['任务编号']
            pub = row['发布时间']
            pub_str = pub.strftime('%Y-%m-%d') if pd.notna(pub) else None
            demand_text = row.get('需求文本', None)
            title       = row.get('标题', None)
            demand_price = row.get('需求价格数值', None)
            demand_price = row.get('需求价格数值', None)
            trans_price = row.get('交易价格数值', None)

            order_nodes.append([
                oid,
                title,
                demand_text,
                demand_price,
                trans_price,
                pub_str,
                '订单',
                1
            ])

    # 边：设计师 -> 能力（只保留设计师路径）
    tmp_sa = df_main_all.dropna(subset=['能力标签', '设计师'])
    for _, row in tmp_sa.iterrows():
        ab = row['能力标签']
        des_name = row['设计师']
        if pd.notna(des_name):
            src_id = map_designer_id(des_name)
            edges_sa.append([src_id, ab, '完成'])

    # 边：能力 -> 订单
    tmp_ao = df_main_all.dropna(subset=['能力标签', '任务编号'])
    for _, row in tmp_ao.iterrows():
        edges_ao.append([row['能力标签'], row['任务编号'], '属于'])

    # 边：需求方 -> 订单（需求方路径）
    tmp_so = df_main_all.dropna(subset=['需求方', '任务编号'])
    tmp_so = tmp_so[['需求方', '任务编号']].drop_duplicates()
    for _, row in tmp_so.iterrows():
        dem_name = row['需求方']
        oid      = row['任务编号']
        sid = map_demander_id(dem_name)
        edges_so.append([sid, oid, '发起'])

    # 转 DataFrame 去重
    df_subject = pd.DataFrame(subject_nodes, columns=['ID', 'Name', 'type', '[z]']).drop_duplicates()
    # 主体层：需求方在前，设计师在后
    df_subject = pd.concat(
        [
            df_subject[df_subject['type'] == '需求方'],
            df_subject[df_subject['type'] == '设计师']
        ],
        ignore_index=True
    )

    df_ability = pd.DataFrame(ability_nodes, columns=['ID', 'Name', 'type', '[z]']).drop_duplicates()
    df_order   = pd.DataFrame(order_nodes,   columns=['ID', 'Title', 'Demand_Text',
                                                      'Demand_Price', 'Transaction_Price',
                                                      'Publication_Time', 'type', '[z]']).drop_duplicates()
    df_edges_sa = pd.DataFrame(edges_sa, columns=['Source', 'Target', 'Type']).drop_duplicates()
    df_edges_ao = pd.DataFrame(edges_ao, columns=['Source', 'Target', 'Type']).drop_duplicates()
    df_edges_so = pd.DataFrame(edges_so, columns=['Source', 'Target', 'Type']).drop_duplicates()

    return df_subject, df_ability, df_order, df_edges_sa, df_edges_ao, df_edges_so


# ========= ①-3 主平台：构建“某一年”的主体/能力/订单 + 边 =========

def build_main_nodes_and_edges_for_year(year, df_main_all, map_demander_id, map_designer_id):
    """
    严格按年份：只用当年 df_main_all 中的记录来构建这年的主体/能力/订单及边
    路径：
      需求方 → 订单
      设计师 → 能力 → 订单
    """
    df_y = df_main_all[df_main_all['发布时间'].dt.year == year].copy()
    if df_y.empty:
        # 返回空表
        empty_sub = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ab  = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ord = pd.DataFrame(columns=['ID', 'Title', 'Demand_Text',
                                          'Demand_Price', 'Transaction_Price',
                                          'Publication_Time', 'type', '[z]'])
        empty_es  = pd.DataFrame(columns=['Source', 'Target', 'Type'])
        empty_ea  = pd.DataFrame(columns=['Source', 'Target', 'Type'])
        empty_so  = pd.DataFrame(columns=['Source', 'Target', 'Type'])
        return empty_sub, empty_ab, empty_ord, empty_es, empty_ea, empty_so

    subject_nodes = []
    ability_nodes = []
    order_nodes   = []
    edges_sa = []  # 设计师 -> 能力
    edges_ao = []  # 能力 -> 订单
    edges_so = []  # 需求方 -> 订单

    # 订单节点（按当年记录去重）
    cols_exist = [c for c in ['任务编号', '发布时间', '需求文本', '标题', '需求价格数值', '交易价格数值'] if c in df_y.columns]
    orders_df = df_y[cols_exist].dropna(subset=['任务编号'])
    orders_df = orders_df.sort_values(by='发布时间').drop_duplicates(subset=['任务编号'], keep='first')

    for _, row in orders_df.iterrows():
        oid = row['任务编号']
        pub = row['发布时间']
        pub_str = pub.strftime('%Y-%m-%d') if pd.notna(pub) else None
        demand_text = row.get('需求文本', None)
        title       = row.get('标题', None)
        demand_price = row.get('需求价格数值', None)
        trans_price = row.get('交易价格数值', None)

        order_nodes.append([
            oid,
            title,
            demand_text,
            demand_price,
            trans_price,
            pub_str,
            '订单',
            1
        ])

    # 能力节点 & 边
    for _, row in df_y.iterrows():
        oid  = row['任务编号']
        ab   = row['能力标签']
        dem  = row['需求方']
        des  = row['设计师']

        if pd.isna(oid):
            continue

        # 能力节点 & 能力 -> 订单
        if pd.notna(ab) and ab != "":
            ability_nodes.append([ab, ab, '能力', 1])
            edges_ao.append([ab, oid, '属于'])

        # 主体节点 & 边
        if pd.notna(dem):
            sid = map_demander_id(dem)
            subject_nodes.append([sid, dem, '需求方', 1])
            # 需求方 -> 订单
            edges_so.append([sid, oid, '发起'])

        if pd.notna(des):
            sid = map_designer_id(des)
            subject_nodes.append([sid, des, '设计师', 1])
            if pd.notna(ab) and ab != "":
                # 设计师 -> 能力
                edges_sa.append([sid, ab, '完成'])

    # 去重
    if subject_nodes:
        df_subject = pd.DataFrame(subject_nodes, columns=['ID', 'Name', 'type', '[z]']).drop_duplicates()
        df_subject = pd.concat(
            [
                df_subject[df_subject['type'] == '需求方'],
                df_subject[df_subject['type'] == '设计师']
            ],
            ignore_index=True
        )
    else:
        df_subject = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])

    if ability_nodes:
        df_ability = pd.DataFrame(ability_nodes, columns=['ID', 'Name', 'type', '[z]']).drop_duplicates()
    else:
        df_ability = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])

    if order_nodes:
        df_order = pd.DataFrame(order_nodes, columns=['ID', 'Title', 'Demand_Text',
                                                      'Demand_Price', 'Transaction_Price',
                                                      'Publication_Time', 'type', '[z]']).drop_duplicates()
    else:
        df_order = pd.DataFrame(columns=['ID', 'Title', 'Demand_Text',
                                         'Demand_Price', 'Transaction_Price',
                                         'Publication_Time', 'type', '[z]'])

    if edges_sa:
        df_edges_sa = pd.DataFrame(edges_sa, columns=['Source', 'Target', 'Type']).drop_duplicates()
    else:
        df_edges_sa = pd.DataFrame(columns=['Source', 'Target', 'Type'])

    if edges_ao:
        df_edges_ao = pd.DataFrame(edges_ao, columns=['Source', 'Target', 'Type']).drop_duplicates()
    else:
        df_edges_ao = pd.DataFrame(columns=['Source', 'Target', 'Type'])

    if edges_so:
        df_edges_so = pd.DataFrame(edges_so, columns=['Source', 'Target', 'Type']).drop_duplicates()
    else:
        df_edges_so = pd.DataFrame(columns=['Source', 'Target', 'Type'])

    return df_subject, df_ability, df_order, df_edges_sa, df_edges_ao, df_edges_so


# ========= ② 主平台 NER JSON 知识（全局 + 按年） =========

def load_main_knowledge_for_year(year):
    """
    从 {year}.json 中提取：
      - 知识节点
      - 订单->知识 边
      - 知识->知识 关系边（relations）

    注意：JSON 只作为“知识抽取结果”，
    不再在这里构建“订单节点”，订单只来自主平台/直接雇佣原始数据。
    """
    json_file_path = os.path.join(json_knowledge_dir, f"{year}.json")
    if not os.path.exists(json_file_path):
        print(f"[JSON] {year}.json 不存在，跳过该年。")
        empty_k  = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ok = pd.DataFrame(columns=['Source', 'Target', 'Type'])
        empty_kk = pd.DataFrame(columns=['Source', 'Target', 'Type'])
        # 为兼容返回结构，订单表为空表
        empty_ord = pd.DataFrame(columns=['ID', 'Publication_Time', 'type', '[z]'])
        return empty_k, empty_ok, empty_kk, empty_ord

    knowledge_nodes = []
    edges_ok = []
    edges_kk = []

    print(f"[JSON] 处理 {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception as e:
            print(f"[JSON] 解析某行出错，跳过：{e}")
            continue

        task_id = record.get('任务编号')
        if not task_id:
            continue

        sd = record.get('提取实体', {}).get('structured_data', {})

        # entities -> 知识节点 & 订单-知识边
        entities = sd.get('entities', []) or []

        for ent in entities:
            name = ent.get('id')
            if not name:
                continue
            kws = split_knowledge_items(name)
            for kw in kws:
                kid = f"knowledge:{kw}"
                knowledge_nodes.append([kid, kw, '知识', 0])
                edges_ok.append([task_id, kid, '关联'])

        # relations -> 知识-知识边
        relations = sd.get('relations', []) or []
        for rel in relations:
            src_raw = rel.get('source')
            tgt_raw = rel.get('target')
            rtype   = rel.get('relation')
            if not src_raw or not tgt_raw or not rtype:
                continue

            src_kws = split_knowledge_items(src_raw)
            tgt_kws = split_knowledge_items(tgt_raw)
            if not src_kws or not tgt_kws:
                continue

            for sk in src_kws:
                sid = f"knowledge:{sk}"
                for tk in tgt_kws:
                    tid = f"knowledge:{tk}"
                    edges_kk.append([sid, tid, rtype])

    df_k  = pd.DataFrame(knowledge_nodes, columns=['ID', 'Name', 'type', '[z]']).drop_duplicates()
    df_ok = pd.DataFrame(edges_ok,       columns=['Source', 'Target', 'Type']).drop_duplicates()
    df_kk = pd.DataFrame(edges_kk,       columns=['Source', 'Target', 'Type']).drop_duplicates()

    # 订单表：不再从 JSON 生成，这里返回空表即可
    df_ord = pd.DataFrame(columns=['ID', 'Publication_Time', 'type', '[z]'])

    return df_k, df_ok, df_kk, df_ord


def load_main_knowledge_global():
    """汇总 2014-2025 所有 JSON 的知识层（全局）"""
    all_k = []
    all_ok = []
    all_kk = []

    for year in years:
        df_k, df_ok, df_kk, _ = load_main_knowledge_for_year(year)
        if not df_k.empty:
            all_k.append(df_k)
        if not df_ok.empty:
            all_ok.append(df_ok)
        if not df_kk.empty:
            all_kk.append(df_kk)

    if all_k:
        df_k_all = pd.concat(all_k, ignore_index=True).drop_duplicates()
    else:
        df_k_all = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])

    if all_ok:
        df_ok_all = pd.concat(all_ok, ignore_index=True).drop_duplicates()
    else:
        df_ok_all = pd.DataFrame(columns=['Source', 'Target', 'Type'])

    if all_kk:
        df_kk_all = pd.concat(all_kk, ignore_index=True).drop_duplicates()
    else:
        df_kk_all = pd.DataFrame(columns=['Source', 'Target', 'Type'])

    # 订单层不再由 JSON 提供
    df_ord_all = pd.DataFrame(columns=['ID', 'Publication_Time', 'type', '[z]'])

    return df_k_all, df_ok_all, df_kk_all, df_ord_all


# ========= ③ 直接雇佣：单年度处理 =========

def process_direct_employment_for_year(year):
    """
    处理某一年的直接雇佣数据（A{year}.csv + A{year}_with_knowledge.csv），
    返回：
      df_sub_dir, df_ab_dir, df_ord_dir, df_k_dir,
      df_edges_sa_dir, df_edges_ao_dir,
      df_edges_ok_dir, df_edges_kk_dir,
      df_edges_so_dir   # 新增：需求方->订单 边
    """
    orders_path = os.path.join(direct_orders_dir, f"A{year}.csv")
    if not os.path.exists(orders_path):
        print(f"[直接雇佣] {orders_path} 不存在，跳过该年。")
        empty_sub = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ab  = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ord = pd.DataFrame(columns=['ID', 'Title', 'Demand_Text',
                                          'Demand_Price', 'Transaction_Price',
                                          'Publication_Time', 'type', '[z]'])
        empty_k   = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_e   = pd.DataFrame(columns=['Source', 'Target', 'Type'])
        return empty_sub, empty_ab, empty_ord, empty_k, empty_e, empty_e.copy(), empty_e.copy(), empty_e.copy(), empty_e.copy()

    try:
        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        print(f"[直接雇佣] 读取 {orders_path}, {len(df_orders)} 条")
    except Exception as e:
        print(f"[直接雇佣] 读取 {orders_path} 出错：{e}")
        empty_sub = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ab  = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ord = pd.DataFrame(columns=['ID', 'Title', 'Demand_Text',
                                          'Demand_Price', 'Transaction_Price',
                                          'Publication_Time', 'type', '[z]'])
        empty_k   = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_e   = pd.DataFrame(columns=['Source', 'Target', 'Type'])
        return empty_sub, empty_ab, empty_ord, empty_k, empty_e, empty_e.copy(), empty_e.copy(), empty_e.copy(), empty_e.copy()

    if '字段3' not in df_orders.columns:
        print(f"[直接雇佣] {orders_path} 缺少 '字段3'（订单编号），跳过该年。")
        empty_sub = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ab  = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ord = pd.DataFrame(columns=['ID', 'Title', 'Demand_Text',
                                          'Demand_Price', 'Transaction_Price',
                                          'Publication_Time', 'type', '[z]'])
        empty_k   = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_e   = pd.DataFrame(columns=['Source', 'Target', 'Type'])
        return empty_sub, empty_ab, empty_ord, empty_k, empty_e, empty_e.copy(), empty_e.copy(), empty_e.copy(), empty_e.copy()

    df_orders['字段3'] = df_orders['字段3'].astype(str).str.strip()
    df_orders = df_orders[df_orders['字段3'] != ""].copy()
    if df_orders.empty:
        print(f"[直接雇佣] {orders_path} 无有效订单编号。")
        empty_sub = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ab  = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_ord = pd.DataFrame(columns=['ID', 'Title', 'Demand_Text',
                                          'Demand_Price', 'Transaction_Price',
                                          'Publication_Time', 'type', '[z]'])
        empty_k   = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])
        empty_e   = pd.DataFrame(columns=['Source', 'Target', 'Type'])
        return empty_sub, empty_ab, empty_ord, empty_k, empty_e, empty_e.copy(), empty_e.copy(), empty_e.copy(), empty_e.copy()

    # 能力标签
    df_orders['能力标签'] = df_orders['标题'].apply(extract_bracket_content) if '标题' in df_orders.columns else None

    # 价格：直接雇佣中只有一列“价格”，视为“需求价格=交易价格”
    if '价格' in df_orders.columns:
        df_orders['价格数值'] = df_orders['价格'].apply(parse_price_to_float)
        df_orders['交易价格数值'] = df_orders['价格数值']
        df_orders['需求价格数值'] = df_orders['价格数值']
    else:
        df_orders['价格数值'] = None
        df_orders['交易价格数值'] = None
        df_orders['需求价格数值'] = None

    # 订单编号后缀（用于构造 A/B 虚拟主体）
    def get_suffix(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        if not s:
            return None
        m = re.search(r'(\d+)', s)
        return m.group(1) if m else s

    df_orders['订单编号后缀'] = df_orders['字段3'].apply(get_suffix)
    df_orders['需求方'] = df_orders['订单编号后缀'].apply(lambda x: f"A{x}" if pd.notna(x) else None)
    df_orders['设计师'] = df_orders['订单编号后缀'].apply(lambda x: f"B{x}" if pd.notna(x) else None)

    # 知识列
    knowledge_path = os.path.join(direct_knowledge_dir, f"A{year}_with_knowledge.csv")
    if os.path.exists(knowledge_path):
        try:
            df_know = pd.read_csv(knowledge_path, encoding='utf-8-sig')
            if '字段3' in df_know.columns and '知识' in df_know.columns:
                df_know['字段3'] = df_know['字段3'].astype(str).str.strip()
                df_orders = df_orders.merge(df_know[['字段3', '知识']], on='字段3', how='left')
            else:
                df_orders['知识'] = None
        except Exception as e:
            print(f"[直接雇佣] 读取 {knowledge_path} 出错：{e}")
            df_orders['知识'] = None
    else:
        df_orders['知识'] = None

    subject_nodes = []
    ability_nodes = []
    order_nodes   = []
    knowledge_nodes = []
    edges_sa = []   # 设计师 -> 能力
    edges_ao = []   # 能力 -> 订单
    edges_ok = []   # 订单 -> 知识
    edges_kk = []   # 知识 -> 知识（共现）
    edges_so = []   # 需求方 -> 订单

    # 主体 / 能力 / 订单
    for _, row in df_orders.iterrows():
        oid = str(row['字段3']).strip()
        if not oid:
            continue
        ability  = row['能力标签']
        demander = row['需求方']
        designer = row['设计师']

        if pd.notna(demander):
            subject_nodes.append([demander, demander, '需求方', 1])
        if pd.notna(designer):
            subject_nodes.append([designer, designer, '设计师', 1])

        # 设计师 -> 能力
        if pd.notna(ability) and ability != "":
            ability_nodes.append([ability, ability, '能力', 1])
            if pd.notna(designer):
                edges_sa.append([designer, ability, '完成'])
            edges_ao.append([ability, oid, '属于'])

        # 需求方 -> 订单
        if pd.notna(demander):
            edges_so.append([demander, oid, '发起'])

        # 订单节点（无具体日期，用年份-01-01 作为近似时间）
        title        = row.get('标题', None)
        demand_text  = row.get('详情标题', None)
        demand_price = row.get('需求价格数值', None)
        trans_price  = row.get('交易价格数值', None)
        pub_str      = f"{year}-01-01"
        order_nodes.append([
            oid,
            title,
            demand_text,
            demand_price,
            trans_price,
            pub_str,
            '订单',
            1
        ])

    # 知识 / 订单-知识 / 共现
    for _, row in df_orders.iterrows():
        oid = str(row['字段3']).strip()
        if not oid:
            continue
        know_str = row.get('知识', None)
        if pd.isna(know_str) or know_str is None:
            continue

        kws = split_knowledge_items(know_str)
        if not kws:
            continue

        for kw in kws:
            kid = f"knowledge:{kw}"
            knowledge_nodes.append([kid, kw, '知识', 0])
            edges_ok.append([oid, kid, '关联'])

        # 共现边（无向去重）
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                id_i = f"knowledge:{kws[i]}"
                id_j = f"knowledge:{kws[j]}"
                if id_i <= id_j:
                    s, t = id_i, id_j
                else:
                    s, t = id_j, id_i
                edges_kk.append([s, t, '共现'])

    # 去重
    if subject_nodes:
        df_sub = pd.DataFrame(subject_nodes, columns=['ID', 'Name', 'type', '[z]']).drop_duplicates()
        df_sub = pd.concat(
            [
                df_sub[df_sub['type'] == '需求方'],
                df_sub[df_sub['type'] == '设计师']
            ],
            ignore_index=True
        )
    else:
        df_sub = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])

    if ability_nodes:
        df_ab = pd.DataFrame(ability_nodes, columns=['ID', 'Name', 'type', '[z]']).drop_duplicates()
    else:
        df_ab = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])

    if order_nodes:
        df_ord = pd.DataFrame(order_nodes, columns=['ID', 'Title', 'Demand_Text',
                                                    'Demand_Price', 'Transaction_Price',
                                                    'Publication_Time', 'type', '[z]']).drop_duplicates()
    else:
        df_ord = pd.DataFrame(columns=['ID', 'Title', 'Demand_Text',
                                       'Demand_Price', 'Transaction_Price',
                                       'Publication_Time', 'type', '[z]'])

    if knowledge_nodes:
        df_k = pd.DataFrame(knowledge_nodes, columns=['ID', 'Name', 'type', '[z]']).drop_duplicates()
    else:
        df_k = pd.DataFrame(columns=['ID', 'Name', 'type', '[z]'])

    if edges_sa:
        df_es = pd.DataFrame(edges_sa, columns=['Source', 'Target', 'Type']).drop_duplicates()
    else:
        df_es = pd.DataFrame(columns=['Source', 'Target', 'Type'])

    if edges_ao:
        df_ea = pd.DataFrame(edges_ao, columns=['Source', 'Target', 'Type']).drop_duplicates()
    else:
        df_ea = pd.DataFrame(columns=['Source', 'Target', 'Type'])

    if edges_ok:
        df_eo = pd.DataFrame(edges_ok, columns=['Source', 'Target', 'Type']).drop_duplicates()
    else:
        df_eo = pd.DataFrame(columns=['Source', 'Target', 'Type'])

    if edges_kk:
        df_ek = pd.DataFrame(edges_kk, columns=['Source', 'Target', 'Type']).drop_duplicates()
    else:
        df_ek = pd.DataFrame(columns=['Source', 'Target', 'Type'])

    if edges_so:
        df_so = pd.DataFrame(edges_so, columns=['Source', 'Target', 'Type']).drop_duplicates()
    else:
        df_so = pd.DataFrame(columns=['Source', 'Target', 'Type'])

    return df_sub, df_ab, df_ord, df_k, df_es, df_ea, df_eo, df_ek, df_so


# ========= ③-1 直接雇佣：全局汇总 =========

def load_direct_employment_global():
    all_sub = []
    all_ab  = []
    all_ord = []
    all_k   = []
    all_es  = []
    all_ea  = []
    all_eo  = []
    all_ek  = []
    all_so  = []

    for year in years:
        (df_sub_y, df_ab_y, df_ord_y, df_k_y,
         df_es_y, df_ea_y, df_eo_y, df_ek_y, df_so_y) = process_direct_employment_for_year(year)

        if not df_sub_y.empty:
            all_sub.append(df_sub_y)
        if not df_ab_y.empty:
            all_ab.append(df_ab_y)
        if not df_ord_y.empty:
            all_ord.append(df_ord_y)
        if not df_k_y.empty:
            all_k.append(df_k_y)
        if not df_es_y.empty:
            all_es.append(df_es_y)
        if not df_ea_y.empty:
            all_ea.append(df_ea_y)
        if not df_eo_y.empty:
            all_eo.append(df_eo_y)
        if not df_ek_y.empty:
            all_ek.append(df_ek_y)
        if not df_so_y.empty:
            all_so.append(df_so_y)

    def cat_or_empty(lst, cols):
        if lst:
            return pd.concat(lst, ignore_index=True).drop_duplicates()
        return pd.DataFrame(columns=cols)

    df_sub_all = cat_or_empty(all_sub, ['ID', 'Name', 'type', '[z]'])
    df_sub_all = pd.concat(
        [
            df_sub_all[df_sub_all['type'] == '需求方'],
            df_sub_all[df_sub_all['type'] == '设计师']
        ],
        ignore_index=True
    )
    df_ab_all  = cat_or_empty(all_ab,  ['ID', 'Name', 'type', '[z]'])
    df_ord_all = cat_or_empty(all_ord, ['ID', 'Title', 'Demand_Text',
                                        'Demand_Price', 'Transaction_Price',
                                        'Publication_Time', 'type', '[z]'])
    df_k_all   = cat_or_empty(all_k,   ['ID', 'Name', 'type', '[z]'])
    df_es_all  = cat_or_empty(all_es,  ['Source', 'Target', 'Type'])
    df_ea_all  = cat_or_empty(all_ea,  ['Source', 'Target', 'Type'])
    df_eo_all  = cat_or_empty(all_eo,  ['Source', 'Target', 'Type'])
    df_ek_all  = cat_or_empty(all_ek,  ['Source', 'Target', 'Type'])
    df_so_all  = cat_or_empty(all_so,  ['Source', 'Target', 'Type'])

    return df_sub_all, df_ab_all, df_ord_all, df_k_all, df_es_all, df_ea_all, df_eo_all, df_ek_all, df_so_all


# ========= ④ 构建全局网络 + 按年度子网络 =========

def build_global_and_yearly_network():
    # ① 主平台全量数据
    df_main_all = load_main_orders()
    if df_main_all.empty:
        print("警告：主平台任务数据为空。")

    # 主体ID映射（解决同名冲突）
    map_demander_id, map_designer_id = build_subject_id_mappers(df_main_all)

    # 主平台：全局 主体/能力/订单 + 边
    (df_sub_main,
     df_ab_main,
     df_ord_main,
     df_es_main,
     df_ea_main,
     df_so_main) = build_main_nodes_and_edges_global(df_main_all, map_demander_id, map_designer_id)

    # ② 主平台JSON知识：全局（只提供知识及边）
    df_k_json, df_ok_json, df_kk_json, _ = load_main_knowledge_global()

    # ③ 直接雇佣：全局
    (df_sub_dir, df_ab_dir, df_ord_dir, df_k_dir,
     df_es_dir, df_ea_dir, df_eo_dir, df_ek_dir, df_so_dir) = load_direct_employment_global()

    # ---------- 给全局边加上统一的 EdgeType（边的大类） ----------
    # 设计师 -> 能力
    if not df_es_main.empty:
        df_es_main['EdgeType'] = '设计师-能力'
    if not df_es_dir.empty:
        df_es_dir['EdgeType'] = '设计师-能力'

    # 能力 -> 订单
    if not df_ea_main.empty:
        df_ea_main['EdgeType'] = '能力-订单'
    if not df_ea_dir.empty:
        df_ea_dir['EdgeType'] = '能力-订单'

    # 需求方 -> 订单
    if not df_so_main.empty:
        df_so_main['EdgeType'] = '需求方-订单'
    if not df_so_dir.empty:
        df_so_dir['EdgeType'] = '需求方-订单'

    # 订单 -> 知识
    if not df_ok_json.empty:
        df_ok_json['EdgeType'] = '订单-知识'
    if not df_eo_dir.empty:
        df_eo_dir['EdgeType'] = '订单-知识'

    # 知识 -> 知识
    if not df_kk_json.empty:
        df_kk_json['EdgeType'] = '知识-知识'
    if not df_ek_dir.empty:
        df_ek_dir['EdgeType'] = '知识-知识'

    # ---------- 全局节点合并 ----------
    # 主体层
    df_subject_all = pd.concat([df_sub_main, df_sub_dir], ignore_index=True).drop_duplicates()
    df_subject_all = pd.concat(
        [
            df_subject_all[df_subject_all['type'] == '需求方'],
            df_subject_all[df_subject_all['type'] == '设计师']
        ],
        ignore_index=True
    )
    # 分开成两张表
    df_demanders_all = df_subject_all[df_subject_all['type'] == '需求方'].copy()
    df_designers_all = df_subject_all[df_subject_all['type'] == '设计师'].copy()
    # 能力层
    df_ability_all = pd.concat([df_ab_main, df_ab_dir], ignore_index=True).drop_duplicates()

    # 订单层（主平台 + 直接雇佣）
    df_orders_all = pd.concat([df_ord_main, df_ord_dir], ignore_index=True) \
                     .drop_duplicates(subset=['ID', 'type'])

    # 知识层（JSON + 直接雇佣）
    df_knowledge_all = pd.concat([df_k_json, df_k_dir], ignore_index=True).drop_duplicates()

    # ---------- 全局边合并 ----------
    df_edges_sa_all = pd.concat([df_es_main, df_es_dir], ignore_index=True).drop_duplicates()  # 设计师-能力
    df_edges_ao_all = pd.concat([df_ea_main, df_ea_dir], ignore_index=True).drop_duplicates()  # 能力-订单
    df_edges_so_all = pd.concat([df_so_main, df_so_dir], ignore_index=True).drop_duplicates()  # 需求方-订单
    df_edges_ok_all = pd.concat([df_ok_json, df_eo_dir], ignore_index=True).drop_duplicates()  # 订单-知识
    df_edges_kk_all = pd.concat([df_kk_json, df_ek_dir], ignore_index=True).drop_duplicates()  # 知识-知识

    # ---------- 输出全局网络 ----------
    df_demanders_all.to_csv(os.path.join(global_output_dir, "nodes_demanders_all.csv"),
                            index=False, encoding='utf-8-sig')
    df_designers_all.to_csv(os.path.join(global_output_dir, "nodes_designers_all.csv"),
                            index=False, encoding='utf-8-sig')
    #df_subject_all.to_csv(os.path.join(global_output_dir, "nodes_subjects_all.csv"),
                          #index=False, encoding='utf-8-sig')
    df_ability_all.to_csv(os.path.join(global_output_dir, "nodes_abilities_all.csv"),
                          index=False, encoding='utf-8-sig')
    df_orders_all.to_csv(os.path.join(global_output_dir, "nodes_orders_all.csv"),
                         index=False, encoding='utf-8-sig')
    df_knowledge_all.to_csv(os.path.join(global_output_dir, "nodes_knowledge_all.csv"),
                            index=False, encoding='utf-8-sig')

    df_edges_sa_all.to_csv(os.path.join(global_output_dir, "edges_designer_ability_all.csv"),
                           index=False, encoding='utf-8-sig')
    df_edges_ao_all.to_csv(os.path.join(global_output_dir, "edges_ability_order_all.csv"),
                           index=False, encoding='utf-8-sig')
    df_edges_so_all.to_csv(os.path.join(global_output_dir, "edges_demander_order_all.csv"),
                           index=False, encoding='utf-8-sig')
    df_edges_ok_all.to_csv(os.path.join(global_output_dir, "edges_order_knowledge_all.csv"),
                           index=False, encoding='utf-8-sig')
    df_edges_kk_all.to_csv(os.path.join(global_output_dir, "edges_knowledge_knowledge_all.csv"),
                           index=False, encoding='utf-8-sig')

    print("\n=== 全局节点和边表已输出完成 ===")
    print(f"主体节点: {len(df_subject_all)}")
    print(f"能力节点: {len(df_ability_all)}")
    print(f"订单节点: {len(df_orders_all)}")
    print(f"知识节点: {len(df_knowledge_all)}")
    print(f"设计师-能力边: {len(df_edges_sa_all)}")
    print(f"能力-订单边: {len(df_edges_ao_all)}")
    print(f"需求方-订单边: {len(df_edges_so_all)}")
    print(f"订单-知识边: {len(df_edges_ok_all)}")
    print(f"知识-知识边: {len(df_edges_kk_all)}")

    # ---------- 按年度输出子网络 ----------
    print("\n=== 开始按年度输出子网络（主平台 + 直接雇佣 + JSON 知识） ===")

    for year in years:
        print(f"\n--- 处理年度 {year} ---")

        year_dir = os.path.join(global_output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        # 1）主平台：当年的子网
        (df_sub_main_y,
         df_ab_main_y,
         df_ord_main_y,
         df_es_main_y,
         df_ea_main_y,
         df_so_main_y) = build_main_nodes_and_edges_for_year(year, df_main_all,
                                                             map_demander_id, map_designer_id)

        # 2）JSON 知识：当年
        df_k_json_y, df_ok_json_y, df_kk_json_y, _ = load_main_knowledge_for_year(year)

        # 3）直接雇佣：当年
        (df_sub_dir_y, df_ab_dir_y, df_ord_dir_y, df_k_dir_y,
         df_es_dir_y, df_ea_dir_y, df_eo_dir_y, df_ek_dir_y, df_so_dir_y) = process_direct_employment_for_year(year)

        # ----- 合并当年各部分 -----
        # 主体
        df_subject_y = pd.concat([df_sub_main_y, df_sub_dir_y], ignore_index=True).drop_duplicates()
        df_subject_y = pd.concat(
            [
                df_subject_y[df_subject_y['type'] == '需求方'],
                df_subject_y[df_subject_y['type'] == '设计师']
            ],
            ignore_index=True
        )
        # 按类型拆分
        df_demanders_y = df_subject_y[df_subject_y['type'] == '需求方'].copy()
        df_designers_y = df_subject_y[df_subject_y['type'] == '设计师'].copy()
        # 能力
        df_ability_y = pd.concat([df_ab_main_y, df_ab_dir_y], ignore_index=True).drop_duplicates()

        # 订单
        df_orders_y = pd.concat([df_ord_main_y, df_ord_dir_y],
                                ignore_index=True).drop_duplicates(subset=['ID', 'type'])

        # 知识
        df_knowledge_y = pd.concat([df_k_json_y, df_k_dir_y], ignore_index=True).drop_duplicates()

        # 边：设计师-能力、能力-订单、需求方-订单、订单-知识
        df_edges_sa_y = pd.concat([df_es_main_y, df_es_dir_y], ignore_index=True).drop_duplicates()
        df_edges_ao_y = pd.concat([df_ea_main_y, df_ea_dir_y], ignore_index=True).drop_duplicates()
        df_edges_so_y = pd.concat([df_so_main_y, df_so_dir_y], ignore_index=True).drop_duplicates()
        df_edges_ok_y = pd.concat([df_ok_json_y, df_eo_dir_y], ignore_index=True).drop_duplicates()

        # 知识-知识：只保留端点都在当年知识集合里的边
        knowledge_ids_y = set(df_knowledge_y['ID'].unique())
        df_edges_kk_y = pd.concat([df_kk_json_y, df_ek_dir_y], ignore_index=True)
        df_edges_kk_y = df_edges_kk_y[
            df_edges_kk_y['Source'].isin(knowledge_ids_y)
            & df_edges_kk_y['Target'].isin(knowledge_ids_y)
        ].drop_duplicates()

        # ----- 给年度边也加 EdgeType（边的大类） -----
        if not df_edges_sa_y.empty:
            df_edges_sa_y['EdgeType'] = '设计师-能力'
        if not df_edges_ao_y.empty:
            df_edges_ao_y['EdgeType'] = '能力-订单'
        if not df_edges_so_y.empty:
            df_edges_so_y['EdgeType'] = '需求方-订单'
        if not df_edges_ok_y.empty:
            df_edges_ok_y['EdgeType'] = '订单-知识'
        if not df_edges_kk_y.empty:
            df_edges_kk_y['EdgeType'] = '知识-知识'

        # ----- 写出当年子网 -----
        df_demanders_y.to_csv(os.path.join(year_dir, f"nodes_demanders_{year}.csv"),
                              index=False, encoding='utf-8-sig')
        df_designers_y.to_csv(os.path.join(year_dir, f"nodes_designers_{year}.csv"),
                              index=False, encoding='utf-8-sig')
        #df_subject_y.to_csv(os.path.join(year_dir, f"nodes_subjects_{year}.csv"),
                            #index=False, encoding='utf-8-sig')
        df_ability_y.to_csv(os.path.join(year_dir, f"nodes_abilities_{year}.csv"),
                            index=False, encoding='utf-8-sig')
        df_orders_y.to_csv(os.path.join(year_dir, f"nodes_orders_{year}.csv"),
                           index=False, encoding='utf-8-sig')
        df_knowledge_y.to_csv(os.path.join(year_dir, f"nodes_knowledge_{year}.csv"),
                              index=False, encoding='utf-8-sig')

        df_edges_sa_y.to_csv(os.path.join(year_dir, f"edges_designer_ability_{year}.csv"),
                             index=False, encoding='utf-8-sig')
        df_edges_ao_y.to_csv(os.path.join(year_dir, f"edges_ability_order_{year}.csv"),
                             index=False, encoding='utf-8-sig')
        df_edges_so_y.to_csv(os.path.join(year_dir, f"edges_demander_order_{year}.csv"),
                             index=False, encoding='utf-8-sig')
        df_edges_ok_y.to_csv(os.path.join(year_dir, f"edges_order_knowledge_{year}.csv"),
                             index=False, encoding='utf-8-sig')
        df_edges_kk_y.to_csv(os.path.join(year_dir, f"edges_knowledge_knowledge_{year}.csv"),
                             index=False, encoding='utf-8-sig')

        print(f"  年度 {year}: 主体 {len(df_subject_y)}，能力 {len(df_ability_y)}，"
              f"订单 {len(df_orders_y)}，知识 {len(df_knowledge_y)}")

    print("\n=== 全局 + 各年度子网络 已全部输出完成 ===")


if __name__ == "__main__":
    build_global_and_yearly_network()
