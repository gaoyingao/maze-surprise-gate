import random
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体，防止中文方块乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 12x12 地图：0=通路，1=墙
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

# 方向字典：上、下、左、右对应的坐标位移 (d_row, d_col)
ACTION_MAP = {
    0: (-1, 0),  # 上
    1: (1, 0),   # 下
    2: (0, -1),  # 左
    3: (0, 1),   # 右
}

class MazeGame:
    def __init__(self, max_steps=200):
        self.max_steps = max_steps
        self.grid = np.array(MAZE, dtype=np.int32)
        self.height, self.width = self.grid.shape
        
        # 怪物固定巡逻矩形轨迹（四个在通路上的拐点）
        self.patrol_path = [
            (5, 4), (5, 8), (7, 8), (7, 4)
        ]
        self.reset()

    def reset(self):
        self.agent_pos = list(START)
        self.monster_pos = list(self.patrol_path[0])
        self.patrol_idx = 0
        self.steps = 0
        
        # 追击状态管理
        self.chase_countdown = 0
        return self._get_status()

    def is_valid(self, pos):
        """判断坐标是否越界或撞墙"""
        r, c = pos
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid[r, c] == 0
        return False

    def _move_agent(self, action):
        dr, dc = ACTION_MAP.get(action, (0, 0))
        next_pos = [self.agent_pos[0] + dr, self.agent_pos[1] + dc]
        # 撞墙停在原地
        if self.is_valid(next_pos):
            self.agent_pos = next_pos

    def _move_monster(self):
        """怪物巡逻与追击逻辑"""
        dist_to_agent = abs(self.monster_pos[0] - self.agent_pos[0]) + abs(
            self.monster_pos[1] - self.agent_pos[1]
        )

        # 追击触发判定：曼哈顿距离 <= 3，或 5% 概率随机触发
        if dist_to_agent <= 3 or random.random() < 0.05:
            if self.chase_countdown == 0:
                self.chase_countdown = 5  # 激活追击 5 步

        if self.chase_countdown > 0:
            # === 追击模式：贪心逼近玩家 ===
            self.chase_countdown -= 1
            candidates = []
            # 候选方向包含 上下左右 以及 原地不动
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                nr, nc = self.monster_pos[0] + dr, self.monster_pos[1] + dc
                if (nr, nc) == (self.monster_pos[0], self.monster_pos[1]) or self.is_valid((nr, nc)):
                    d = abs(nr - self.agent_pos[0]) + abs(nc - self.agent_pos[1])
                    candidates.append((d, [nr, nc]))
            if candidates:
                # 距离优先，距离为0即扑中抓获
                candidates.sort(key=lambda x: x[0])
                self.monster_pos = candidates[0][1]
        else:
            # === 巡逻模式：沿矩形路径巡逻 ===
            target = self.patrol_path[self.patrol_idx]
            if self.monster_pos == list(target):
                self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_path)
                target = self.patrol_path[self.patrol_idx]

            dr = np.sign(target[0] - self.monster_pos[0])
            dc = np.sign(target[1] - self.monster_pos[1])
            next_pos = [
                self.monster_pos[0] + dr,
                self.monster_pos[1] + (0 if dr != 0 else dc),
            ]
            if self.is_valid(next_pos):
                self.monster_pos = next_pos
            else:
                self.monster_pos = [self.monster_pos[0], self.monster_pos[1] + dc]

    def step(self, action):
        """执行一步移动"""
        self.steps += 1
        prev_agent_pos = list(self.agent_pos)
        prev_monster_pos = list(self.monster_pos)

        # 1. 玩家移动
        self._move_agent(action)

        # 判定 A：玩家主动迎面撞入怪物位置
        if self.agent_pos == self.monster_pos:
            return self.agent_pos, self.monster_pos, True, False, {"status": "die"}

        # 2. 怪物移动
        self._move_monster()

        # 判定 B：怪物移动后抓到玩家，或者两人擦肩互换位置
        if (self.agent_pos == self.monster_pos) or (
            self.agent_pos == prev_monster_pos and self.monster_pos == prev_agent_pos
        ):
            return self.agent_pos, self.monster_pos, True, False, {"status": "die"}

        # 判定 C：到终点或超时
        terminated = False
        truncated = False
        info = {}

        if tuple(self.agent_pos) == GOAL:
            terminated = True
            info["status"] = "win"
        elif self.steps >= self.max_steps:
            truncated = True
            info["status"] = "timeout"

        return self.agent_pos, self.monster_pos, terminated, truncated, info

    def _get_status(self):
        return self.agent_pos, self.monster_pos

    def render_cli(self):
        """在终端输出 ASCII 字符地图，方便手动键盘试玩"""
        char_map = []
        for r in range(self.height):
            line = []
            for c in range(self.width):
                if [r, c] == self.agent_pos:
                    line.append(" A ")  # Agent
                elif [r, c] == self.monster_pos:
                    line.append(" M ")  # Monster
                elif (r, c) == GOAL:
                    line.append(" G ")  # Goal
                elif self.grid[r, c] == 1:
                    line.append("###")  # 墙
                else:
                    line.append(" . ")  # 路
            char_map.append("".join(line))
        
        mode = "【追击模式!!】" if self.chase_countdown > 0 else "【巡逻模式】"
        print(f"\n--- 步数: {self.steps}/{self.max_steps} | 怪物状态: {mode} ---")
        print("\n".join(char_map))


# ================= 手动游玩循环 (Day 5 验证) =================
def play_manual():
    game = MazeGame()
    key_mapping = {'w': 0, 's': 1, 'a': 2, 'd': 3}
    
    print("=" * 40)
    print("迷宫手动测试：输入 w(上) / s(下) / a(左) / d(右) 并按回车；输入 q 退出")
    print("=" * 40)
    
    game.render_cli()
    
    while True:
        cmd = input("请输入方向 (w/s/a/d): ").strip().lower()
        if cmd == 'q':
            break
        if cmd not in key_mapping:
            print("输入无效，只能输入 w、s、a、d")
            continue
            
        action = key_mapping[cmd]
        _, _, terminated, truncated, info = game.step(action)
        game.render_cli()

        if terminated:
            if info.get("status") == "win":
                print("\n🎉 成功到达终点！胜利！")
            elif info.get("status") == "die":
                print("\n💀 你被怪物抓到了！游戏结束！")
            break
        elif truncated:
            print("\n⏰ 超过最大步数上限，挑战失败！")
            break

if __name__ == "__main__":
    play_manual()