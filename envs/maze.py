import matplotlib.pyplot as plt
import numpy as np

# 1. 按照清单定义 12x12 地图：0=通路，1=墙
# 起点为 (1, 1)，终点为 (10, 10)，中央巡逻区在第 5-6 行附近
MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 中央巡逻区
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

START = (1, 1)
GOAL = (10, 10)


def draw(agent_pos=START, monster_pos=(5, 5), goal_pos=GOAL, save_path=None):
    """
    用 Matplotlib 可视化迷宫
    :param agent_pos: 智能体坐标 (row, col)
    :param monster_pos: 怪物坐标 (row, col)，若没有怪可传 None
    :param goal_pos: 终点坐标 (row, col)
    :param save_path: 图片保存路径（可选）
    """
    maze_np = np.array(MAZE, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))

    # 显示墙面（深灰色）和通路（白色）
    # cmap='binary' 中 1 是黑/深灰，0 是白
    ax.imshow(maze_np, cmap="binary", vmin=0, vmax=1)

    # 绘制网格辅助线（在每格边缘对齐）
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)

    # 绘制关键角色（注意：Matplotlib 中 x 轴对应 col 列，y 轴对应 row 行）
    # 终点：绿色五角星
    ax.plot(
        goal_pos[1],
        goal_pos[0],
        marker="*",
        color="limegreen",
        markersize=18,
        label="Goal ",
    )

    # 智能体：蓝色圆形
    ax.plot(
        agent_pos[1],
        agent_pos[0],
        marker="o",
        color="dodgerblue",
        markersize=14,
        label="Agent ",
    )

    # 怪物：红色方块（若有）
    if monster_pos is not None:
        ax.plot(
            monster_pos[1],
            monster_pos[0],
            marker="s",
            color="crimson",
            markersize=14,
            label="Monster ",
        )

    # 图表标题与标注
    ax.set_title("12x12 Maze Environment (Day 4)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))

    # 保持坐标轴整洁
    ax.set_xticks(range(0, 12, 2))
    ax.set_yticks(range(0, 12, 2))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"迷宫图像已保存至: {save_path}")

    plt.show()


if __name__ == "__main__":
    # 运行此文件进行测试
    print("正在渲染迷宫地图...")
    draw(agent_pos=START, monster_pos=(5, 5), goal_pos=GOAL)