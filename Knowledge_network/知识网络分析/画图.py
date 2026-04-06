import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


# ============ 配置区域：改这里就行 ============

# 能力向量与金额输出目录（和前面脚本一致）
base_dir = r"D:/Documents/Desktop/Crowd_intelligence_wyq/Knowledge_network/知识网络分析/能力向量与金额"

# 想画图的年份，例如 2014、2020、2025……
year = 2014


# ============ 工具函数 ============

def load_year_vector_data(year: int):
    """
    读取某一年的 designer_ability_vectors_YYYY.csv
    返回 DataFrame
    """
    year_dir = os.path.join(base_dir, str(year))
    csv_path = os.path.join(year_dir, f"designer_ability_vectors_{year}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    return df


def load_year_heatmap_data(year: int):
    """
    读取某一年的 Heatmap 透视表（如果你之前已经保存了）
    文件名: heatmap_amount_designer_by_ability_YYYY.csv

    如果这个文件不存在，就在这里现场 pivot 一份出来。
    """
    year_dir = os.path.join(base_dir, str(year))
    heatmap_path = os.path.join(year_dir, f"heatmap_amount_designer_by_ability_{year}.csv")

    if os.path.exists(heatmap_path):
        pivot = pd.read_csv(heatmap_path, encoding='utf-8-sig', index_col=0)
    else:
        print(f"警告：{heatmap_path} 不存在，尝试从 designer_ability_vectors_{year}.csv 现场生成透视表。")
        df = load_year_vector_data(year)
        pivot = df.pivot_table(
            index='designer',
            columns='ability',
            values='total_amount',
            aggfunc='sum',
            fill_value=0.0
        )
    return pivot


# ============ 画 3D 图：designer_idx – ability_idx – total_amount ============

def plot_3d_designer_ability_amount(year: int, save_fig: bool = True):
    """
    读取某年的向量文件，画 3D 散点图：
      X: designer_idx
      Y: ability_idx
      Z: total_amount
    """
    df = load_year_vector_data(year)

    # 只保留有 index 和金额的行
    df = df.dropna(subset=['designer_idx', 'ability_idx', 'total_amount'])

    x = df['designer_idx'].values
    y = df['ability_idx'].values
    z = df['total_amount'].values

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 3D 散点
    ax.scatter(x, y, z)

    ax.set_xlabel('designer_idx')
    ax.set_ylabel('ability_idx')
    ax.set_zlabel('total_amount')

    ax.set_title(f'3D: Designer–Ability–Amount ({year})')

    plt.tight_layout()

    # 保存图片
    if save_fig:
        out_path = os.path.join(base_dir, str(year), f"3D_designer_ability_amount_{year}.png")
        plt.savefig(out_path, dpi=300)
        print(f"[3D] 图已保存到: {out_path}")

    plt.show()


# ============ 画 2D Heatmap：designer × ability ============

def plot_heatmap_designer_ability(year: int, save_fig: bool = True):
    """
    使用透视表（designer × ability，值为 total_amount）画 2D 热力图。
    """
    pivot = load_year_heatmap_data(year)

    # 行：designer，列：ability，值：total_amount
    data = pivot.values
    designers = list(pivot.index)
    abilities = list(pivot.columns)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 用 imshow 画热力图，不指定 cmap，使用 matplotlib 默认色带
    im = ax.imshow(data, aspect='auto')

    # 坐标轴刻度：如果太多可以适当采样或旋转
    ax.set_xticks(range(len(abilities)))
    ax.set_yticks(range(len(designers)))

    # 可以根据需要选择是否全显示（很多时可能太挤）
    ax.set_xticklabels(abilities, rotation=90)
    ax.set_yticklabels(designers)

    ax.set_xlabel('Ability')
    ax.set_ylabel('Designer')
    ax.set_title(f'Heatmap: Designer × Ability – Amount ({year})')

    # 加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('total_amount')

    plt.tight_layout()

    if save_fig:
        out_path = os.path.join(base_dir, str(year), f"Heatmap_designer_ability_amount_{year}.png")
        plt.savefig(out_path, dpi=300)
        print(f"[Heatmap] 图已保存到: {out_path}")

    plt.show()


# ============ 主入口 ============

if __name__ == "__main__":
    print(f"开始绘制年份 {year} 的图")

    # 3D 图
    plot_3d_designer_ability_amount(year)

    # 2D Heatmap
    plot_heatmap_designer_ability(year)

    print("绘图结束。")
