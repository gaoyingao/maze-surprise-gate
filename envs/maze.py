import random
import gymnasium as gym
from gymnasium import spaces
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

ACTION_MAP = {
    0: (-1, 0),  # 上
    1: (1, 0),   # 下
    2: (0, -1),  # 左
    3: (0, 1),   # 右
}

class MazeGame:
    def __init__(self, with_monster=True, max_steps=200):
        self.with_monster = with_monster
        self.max_steps = max_steps
        self.grid = np.array(MAZE, dtype=np.int32)
        self.height, self.width = self.grid.shape
        self.patrol_path = [(5, 4), (5, 8), (7, 8), (7, 4)]
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.agent_pos = list(START)
        self.monster_pos = list(self.patrol_path[0])
        self.patrol_idx = 0
        self.steps = 0
        self.chase_countdown = 0
        return self.agent_pos, self.monster_pos

    def is_valid(self, pos):
        r, c = pos
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid[r, c] == 0
        return False

    def _move_agent(self, action):
        dr, dc = ACTION_MAP.get(action, (0, 0))
        next_pos = [self.agent_pos[0] + dr, self.agent_pos[1] + dc]
        if self.is_valid(next_pos):
            self.agent_pos = next_pos

    def _move_monster(self):
        if not self.with_monster:
            return
        dist_to_agent = abs(self.monster_pos[0] - self.agent_pos[0]) + abs(
            self.monster_pos[1] - self.agent_pos[1]
        )
        if dist_to_agent <= 3 or random.random() < 0.05:
            if self.chase_countdown == 0:
                self.chase_countdown = 5

        if self.chase_countdown > 0:
            self.chase_countdown -= 1
            candidates = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                nr, nc = self.monster_pos[0] + dr, self.monster_pos[1] + dc
                if (nr, nc) == (self.monster_pos[0], self.monster_pos[1]) or self.is_valid((nr, nc)):
                    d = abs(nr - self.agent_pos[0]) + abs(nc - self.agent_pos[1])
                    candidates.append((d, [nr, nc]))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                self.monster_pos = candidates[0][1]
        else:
            target = self.patrol_path[self.patrol_idx]
            if self.monster_pos == list(target):
                self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_path)
                target = self.patrol_path[self.patrol_idx]
            dr = np.sign(target[0] - self.monster_pos[0])
            dc = np.sign(target[1] - self.monster_pos[1])
            next_pos = [self.monster_pos[0] + dr, self.monster_pos[1] + (0 if dr != 0 else dc)]
            if self.is_valid(next_pos):
                self.monster_pos = next_pos
            else:
                self.monster_pos = [self.monster_pos[0], self.monster_pos[1] + dc]

    def step(self, action):
        self.steps += 1
        prev_agent_pos = list(self.agent_pos)
        prev_monster_pos = list(self.monster_pos)

        self._move_agent(action)

        if self.with_monster and self.agent_pos == self.monster_pos:
            return self.agent_pos, self.monster_pos, True, False, {"status": "die"}

        self._move_monster()

        if self.with_monster and (
            (self.agent_pos == self.monster_pos)
            or (self.agent_pos == prev_monster_pos and self.monster_pos == prev_agent_pos)
        ):
            return self.agent_pos, self.monster_pos, True, False, {"status": "die"}

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


# ================= Day 6 Gymnasium 封装 =================
class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, with_monster=True, max_steps=200):
        super().__init__()
        self.game = MazeGame(with_monster=with_monster, max_steps=max_steps)
        
        # 动作空间：0=上, 1=下, 2=左, 3=右
        self.action_space = spaces.Discrete(4)
        
        # 观测空间：79 维连续向量，数值在 [-1.0, 1.0] 范围内
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(79,),
            dtype=np.float32
        )

    def _get_obs(self):
        """构造 79 维观测向量"""
        ar, ac = self.game.agent_pos
        mr, mc = self.game.monster_pos
        gr, gc = GOAL

        # 1. 5x5 局部视野 x 3 通道 (75 维)
        view_wall = np.zeros((5, 5), dtype=np.float32)
        view_monster = np.zeros((5, 5), dtype=np.float32)
        view_agent = np.zeros((5, 5), dtype=np.float32)

        for i, dr in enumerate(range(-2, 3)):
            for j, dc in enumerate(range(-2, 3)):
                r, c = ar + dr, ac + dc
                # 越界视为墙壁
                if 0 <= r < self.game.height and 0 <= c < self.game.width:
                    view_wall[i, j] = 1.0 if self.game.grid[r, c] == 1 else 0.0
                else:
                    view_wall[i, j] = 1.0

                # 怪物视野（若存在怪物）
                if self.game.with_monster and [r, c] == [mr, mc]:
                    view_monster[i, j] = 1.0

                # 自己位于中心 (dr=0, dc=0)
                if dr == 0 and dc == 0:
                    view_agent[i, j] = 1.0

        local_view = np.concatenate([
            view_wall.flatten(),
            view_monster.flatten(),
            view_agent.flatten()
        ])  # 25 + 25 + 25 = 75

        # 2. 怪物相对位置（归一化到 [-1, 1] 范围，地图宽/高最大距离为 11）
        if self.game.with_monster:
            rel_monster = np.array([(mr - ar) / 11.0, (mc - ac) / 11.0], dtype=np.float32)
        else:
            rel_monster = np.array([0.0, 0.0], dtype=np.float32)

        # 3. 终点相对位置（归一化到 [-1, 1]）
        rel_goal = np.array([(gr - ar) / 11.0, (gc - ac) / 11.0], dtype=np.float32)

        # 组合成 79 维向量
        obs = np.concatenate([local_view, rel_monster, rel_goal]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        _, _, terminated, truncated, info = self.game.step(action)
        obs = self._get_obs()

        # Day 6 基础奖励机制（v0 版本预热：赢+1，超时-0.5，死-1）
        reward = 0.0
        status = info.get("status")
        if status == "win":
            reward = 1.0
        elif status == "die":
            reward = -1.0
        elif status == "timeout":
            reward = -0.5

        return obs, reward, terminated, truncated, info


# ================= Day 6 校验入口 =================
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("创建 MazeEnv 实例...")
    env = MazeEnv(with_monster=True)

    print("正在执行 Gymnasium 官方环境合规性检查 (check_env)...")
    check_env(env)
    print("✅ check_env 校验通过！无警告无报错。")

    # 验证单步输出维度
    obs, info = env.reset(seed=42)
    print(f"reset() 返回观测 shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape == (79,), "观测维度不等于 79！"

    next_obs, reward, terminated, truncated, info = env.step(1)
    print(f"step() 执行成功，next_obs shape: {next_obs.shape}, reward: {reward}")
    print("🎉 Day 6 全部目标达成！")