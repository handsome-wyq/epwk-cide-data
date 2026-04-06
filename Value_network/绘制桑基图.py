import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ================= 配置区域 =================

# 1. 文件路径 (请确保指向上一歩生成的包含全生命周期数据的CSV)
root_dir = r"D:\Documents\Desktop\Crowd_intelligence_wyq\Knowledge_network\知识网络分析\1125计算结果-对齐版"
input_path = os.path.join(root_dir, "价值网络聚类分析结果", "Sankey_Data_Full_Lifecycle_15-24.csv")
output_html_path = os.path.join(root_dir, "价值网络聚类分析结果", "Beautiful_Sankey_Chart.html")

# 2. 图表标题配置
CHART_TITLE = "群智设计生态演化路径桑基图 (2015-2024)"
FONT_FAMILY = "Microsoft YaHei"  # 微软雅黑，显示中文更好看

# 3. 美学色彩配置 (核心部分)
# 定义基础色板 (Hex格式)
COLOR_PALETTE = {
    "Type 1": "#FF6B6B",  # 柔和红 (例如: 卷王/头部)
    "Type 2": "#4ECDC4",  # 青绿色 (例如: 专家)
    "Type 3": "#FFE66D",  # 明黄色 (例如: 中坚力量)
    "Type 4": "#1A535C",  # 深青色 (例如: 底层/新手)
    "Entry": "#FDCB58",  # 金色/亮橙色 (新增，代表注入活力)
    "Exit": "#2F3542",  # 深灰色/黑色 (流失，代表沉淀/离开)
    "Default": "#A4B0BE"  # 默认灰
}

# 链接透明度 (0.0 - 1.0)，0.5 左右效果较好，有丝滑感
LINK_OPACITY = 0.5


# ================= 核心逻辑 =================

# 辅助函数：将Hex颜色转换为RGBA字符串，用于设置透明度
def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'


# 辅助函数：根据节点名称决定节点颜色
def assign_node_color(label):
    if "[新增]" in label: return COLOR_PALETTE["Entry"]
    if "[流失]" in label: return COLOR_PALETTE["Exit"]
    if "Type 1" in label: return COLOR_PALETTE["Type 1"]
    if "Type 2" in label: return COLOR_PALETTE["Type 2"]
    if "Type 3" in label: return COLOR_PALETTE["Type 3"]
    if "Type 4" in label: return COLOR_PALETTE["Type 4"]
    return COLOR_PALETTE["Default"]


def draw_beautiful_sankey():
    print("=== 开始绘制美观桑基图 (Plotly) ===")

    if not os.path.exists(input_path):
        print(f"[错误] 数据文件不存在: {input_path}\n请先运行上一步生成 CSV 数据。")
        return

    # 1. 读取数据
    df = pd.read_csv(input_path)
    print(f"成功加载数据，共 {len(df)} 条流动记录。")

    # 2. 数据预处理：构建节点索引映射
    # Plotly 需要将 Source 和 Target 的字符串名称转换为数字索引 (0, 1, 2...)

    # 获取所有唯一的节点名称集合
    all_nodes = list(pd.concat([df['Source'], df['Target']]).unique())

    # 创建映射字典 {节点名: 索引ID}
    node_map = {name: i for i, name in enumerate(all_nodes)}

    # 将 DataFrame 中的字符串映射为索引
    df['source_idx'] = df['Source'].map(node_map)
    df['target_idx'] = df['Target'].map(node_map)

    print(f"识别到 {len(all_nodes)} 个唯一节点。正在计算颜色...")

    # 3. 颜色配置 (美观的关键)

    # A. 生成节点颜色列表
    node_colors = [assign_node_color(label) for label in all_nodes]

    # B. 生成链接颜色列表 (跟随起点的颜色，并增加透明度)
    # 这样线条看起来是从起点“流”出来的，视觉连贯性极佳
    link_colors = [hex_to_rgba(assign_node_color(row['Source']), LINK_OPACITY) for _, row in df.iterrows()]

    # 4. 构建 Plotly Sankey 图表对象
    print("正在构建图表对象...")
    fig = go.Figure(data=[go.Sankey(
        # 配置节点
        node=dict(
            pad=20,  # 节点纵向间距
            thickness=25,  # 节点宽度
            line=dict(color="white", width=0.5),  # 节点边框
            label=all_nodes,  # 节点名称
            color=node_colors,  # 节点颜色
            hovertemplate='%{label}<br>总流量: %{value}<extra></extra>'  # 鼠标悬停提示
        ),
        # 配置连线
        link=dict(
            source=df['source_idx'],  # 起点索引列表
            target=df['target_idx'],  # 终点索引列表
            value=df['Value'],  # 流量值列表
            color=link_colors,  # 连线颜色列表
            hovertemplate='%{source.label} -> %{target.label}<br>流量: %{value}<extra></extra>'
        )
    )])

    # 5. 配置全局布局 (标题、字体、背景)
    fig.update_layout(
        title_text=CHART_TITLE,
        title_x=0.5,  # 标题居中
        font=dict(size=12, family=FONT_FAMILY),
        plot_bgcolor='white',  # 背景色
        paper_bgcolor='white',
        height=800,  # 图表高度，太矮了密集恐惧症
        margin=dict(t=80, b=20, l=20, r=20)  # 边距
    )

    # 6. 显示与保存
    print("正在渲染图表...")
    # 在浏览器中弹出互动窗口 (如果运行环境支持)
    fig.show()

    # 保存为 HTML 文件 (方便发给别人或嵌入网页)
    pio.write_html(fig, file=output_html_path, auto_open=False)
    print(f"\n[成功] 美观桑基图已保存为 HTML 文件:\n{output_html_path}")
    print("请在浏览器中打开该 HTML 文件查看交互效果。")


if __name__ == "__main__":
    draw_beautiful_sankey()