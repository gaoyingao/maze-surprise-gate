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
    """
    迷宫游戏规则。

    Day 20 起支持可配置的怪物行为，便于做威胁度对照实验：

        extra_walls    : 追加的墙 [(r,c), ...]（用于收窄地图、制造咽喉）
        patrol_path    : 巡逻航点序列
        chase_radius   : 进入多少格内触发追击（默认 3）
        chase_prob     : 每步随机触发追击的概率（默认 0.05）
        chase_len      : 一次追击持续多少步（默认 5）
        monster_speed  : 怪物每步移动几格（默认 1）
    """

    def __init__(self, with_monster=True, max_steps=200,
                 extra_walls=None, patrol_path=None,
                 chase_radius=3, chase_prob=0.05, chase_len=5,
                 monster_speed=1):
        self.with_monster = with_monster
        self.max_steps = max_steps
        self.grid = np.array(MAZE, dtype=np.int32)
        for (r, c) in (extra_walls or []):
            if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
                self.grid[r, c] = 1
        self.height, self.width = self.grid.shape
        self.patrol_path = list(patrol_path) if patrol_path else \
            [(5, 4), (5, 8), (7, 8), (7, 4)]
        self.chase_radius = chase_radius
        self.chase_prob = chase_prob
        self.chase_len = chase_len
        self.monster_speed = monster_speed
        # 起点/终点在任何配置下都必须是通路
        assert self.grid[START] == 0, "起点被墙堵住"
        assert self.grid[GOAL] == 0, "终点被墙堵住"
        for p in self.patrol_path:
            assert self.grid[p] == 0, f"巡逻点 {p} 是墙"
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.agent_pos = list(START)
        # 无怪模式下把怪物放到地图外，确保它既不在观测里、也不可能碰撞。
        # 旧实现把它冻在 patrol_path[0]（如 (5,4)），虽然观测里被置零，
        # 但会把「有怪但不动」变成策略可分辨的分布外状态，污染对照实验。
        self.monster_pos = list(self.patrol_path[0]) if self.with_monster \
            else [-1, -1]
        self.patrol_idx = 0
        self.steps = 0
        self.chase_countdown = 0
        self.monster_chasing = False
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
            self.monster_chasing = False
            return
        self.monster_chasing = False
        for _ in range(max(1, int(self.monster_speed))):
            self._move_monster_once()

    def _move_monster_once(self):
        """怪物走一格：追击态朝智能体曼哈顿下降；否则沿巡逻航点前进"""
        dist_to_agent = abs(self.monster_pos[0] - self.agent_pos[0]) + abs(
            self.monster_pos[1] - self.agent_pos[1]
        )
        if dist_to_agent <= self.chase_radius or random.random() < self.chase_prob:
            if self.chase_countdown == 0:
                self.chase_countdown = self.chase_len

        # 记录本步是否处于追击态 —— 供 predict/gate 做模式切分。
        # 必须在倒计时递减【之前】判定，否则"追击的第 1 步"会被漏掉。
        if self.chase_countdown > 0:
            self.monster_chasing = True

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
                # 兜底：只走列方向，但仍要校验合法性，避免走进墙里
                alt = [self.monster_pos[0], self.monster_pos[1] + dc]
                if self.is_valid(alt):
                    self.monster_pos = alt

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

    def __init__(self, with_monster=True, max_steps=200, **game_kwargs):
        """
        game_kwargs 透传给 MazeGame，用于配置怪物行为 / 追加墙体：
            extra_walls, patrol_path, chase_radius, chase_prob,
            chase_len, monster_speed
        """
        super().__init__()
        self.game = MazeGame(with_monster=with_monster, max_steps=max_steps,
                             **game_kwargs)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(79,),
            dtype=np.float32
        )

    def _get_obs(self):
        ar, ac = self.game.agent_pos

        # 无怪模式下怪物不在图中，用哨兵值，保证任何通道都不会出现它
        if self.game.with_monster:
            mr, mc = self.game.monster_pos
        else:
            mr, mc = -1, -1

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
        prev_dist_goal = abs(prev_ar - GOAL[0]) + abs(prev_ac - GOAL[1])

        _, _, terminated, truncated, info = self.game.step(action)
        obs = self._get_obs()

        curr_ar, curr_ac = self.game.agent_pos
        curr_dist_goal = abs(curr_ar - GOAL[0]) + abs(curr_ac - GOAL[1])

        # 1. 步数时间消耗（给前进压力）
        reward = -0.01

        # 2. 撞墙惩罚
        if info.get("hit_wall", False):
            reward -= 0.05
        else:
            # 靠近终点奖励
            progress = prev_dist_goal - curr_dist_goal
            reward += progress * 0.1

        # 3. 怪物危险感知惩罚（靠近怪物 <= 2 格扣分，鼓励绕道避险）
        if self.game.with_monster:
            dist_to_monster = abs(curr_ar - self.game.monster_pos[0]) + abs(curr_ac - self.game.monster_pos[1])
            if dist_to_monster <= 2:
                reward -= 0.05

        # 4. 关键事件结算（根治挂机：超时惩罚严厉于被怪抓）
        status = info.get("status")
        if status == "win":
            reward += 10.0   # 成功主奖励
        elif status == "die":
            reward -= 4.0    # 撞怪被击杀
        elif status == "timeout":
            reward -= 10.0   # 超时摆烂直接判定重罚，逼它必须向前冲

        return obs, reward, terminated, truncated, info