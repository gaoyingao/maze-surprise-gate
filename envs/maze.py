import random
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
        act_idx = int(action)
        dr, dc = ACTION_MAP.get(act_idx, (0, 0))
        next_pos = [self.agent_pos[0] + dr, self.agent_pos[1] + dc]
        hit_wall = False
        if self.is_valid(next_pos):
            self.agent_pos = next_pos
        else:
            hit_wall = True
        return hit_wall

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

        hit_wall = self._move_agent(action)

        if self.with_monster and self.agent_pos == self.monster_pos:
            return self.agent_pos, self.monster_pos, True, False, {"status": "die", "hit_wall": hit_wall}

        self._move_monster()

        if self.with_monster and (
            (self.agent_pos == self.monster_pos)
            or (self.agent_pos == prev_monster_pos and self.monster_pos == prev_agent_pos)
        ):
            return self.agent_pos, self.monster_pos, True, False, {"status": "die", "hit_wall": hit_wall}

        terminated = False
        truncated = False
        info = {"hit_wall": hit_wall}
        if tuple(self.agent_pos) == GOAL:
            terminated = True
            info["status"] = "win"
        elif self.steps >= self.max_steps:
            truncated = True
            info["status"] = "timeout"

        return self.agent_pos, self.monster_pos, terminated, truncated, info


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, with_monster=True, max_steps=200):
        super().__init__()
        self.game = MazeGame(with_monster=with_monster, max_steps=max_steps)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(79,),
            dtype=np.float32
        )

    def _get_obs(self):
        ar, ac = self.game.agent_pos
        mr, mc = self.game.monster_pos
        gr, gc = GOAL

        view_wall = np.zeros((5, 5), dtype=np.float32)
        view_monster = np.zeros((5, 5), dtype=np.float32)
        view_agent = np.zeros((5, 5), dtype=np.float32)

        for i, dr in enumerate(range(-2, 3)):
            for j, dc in enumerate(range(-2, 3)):
                r, c = ar + dr, ac + dc
                if 0 <= r < self.game.height and 0 <= c < self.game.width:
                    view_wall[i, j] = 1.0 if self.game.grid[r, c] == 1 else 0.0
                else:
                    view_wall[i, j] = 1.0

                if self.game.with_monster and [r, c] == [mr, mc]:
                    view_monster[i, j] = 1.0

                if dr == 0 and dc == 0:
                    view_agent[i, j] = 1.0

        local_view = np.concatenate([
            view_wall.flatten(),
            view_monster.flatten(),
            view_agent.flatten()
        ])

        if self.game.with_monster:
            rel_monster = np.array([(mr - ar) / 11.0, (mc - ac) / 11.0], dtype=np.float32)
        else:
            rel_monster = np.array([0.0, 0.0], dtype=np.float32)

        rel_goal = np.array([(gr - ar) / 11.0, (gc - ac) / 11.0], dtype=np.float32)

        obs = np.concatenate([local_view, rel_monster, rel_goal]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        prev_ar, prev_ac = self.game.agent_pos
        prev_dist = abs(prev_ar - GOAL[0]) + abs(prev_ac - GOAL[1])

        _, _, terminated, truncated, info = self.game.step(action)
        obs = self._get_obs()

        curr_ar, curr_ac = self.game.agent_pos
        curr_dist = abs(curr_ar - GOAL[0]) + abs(curr_ac - GOAL[1])

        # 核心奖励设定：
        reward = -0.01  # 每走一步轻微时间消耗
        if info.get("hit_wall", False):
            reward -= 0.05  # 撞墙额外惩罚，不要卡在死角
        else:
            progress = prev_dist - curr_dist
            reward += progress * 0.1  # 靠近终点给 0.1 奖励

        status = info.get("status")
        if status == "win":
            reward += 10.0  # 终点给予强力主奖励，吸引力拉满
        elif status == "die":
            reward -= 5.0
        elif status == "timeout":
            reward -= 1.0

        return obs, reward, terminated, truncated, info