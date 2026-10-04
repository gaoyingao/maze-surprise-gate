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
        # 起点/终点（子类可覆盖，用于泛化实验的生成地图）
        self.start = tuple(START)
        self.goal = tuple(GOAL)
        # 起点/终点在任何配置下都必须是通路
        assert self.grid[self.start] == 0, "起点被墙堵住"
        assert self.grid[self.goal] == 0, "终点被墙堵住"
        for p in self.patrol_path:
            assert self.grid[p] == 0, f"巡逻点 {p} 是墙"
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.agent_pos = list(self.start)
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
        if tuple(self.agent_pos) == self.goal:
            terminated = True
            info["status"] = "win"
        elif self.steps >= self.max_steps:
            truncated = True
            info["status"] = "timeout"

        return self.agent_pos, self.monster_pos, terminated, truncated, info


# ======================================================================
# 泛化实验：任意生成地图上的同规则游戏与环境
# ======================================================================
class MapMazeGame(MazeGame):
    """
    与 MazeGame 规则完全相同，但地图 / 起点 / 终点 / 巡逻路径
    来自 envs.map_gen.generate_map 的 map_spec。

    存在的意义：MazeGame 把 grid 写死为模块级 MAZE 常量，无法用于
    泛化实验的多张随机地图。
    """

    def __init__(self, map_spec, with_monster=True, max_steps=200, **game_kwargs):
        self.with_monster = with_monster
        self.max_steps = max_steps
        self.grid = np.array(map_spec["grid"], dtype=np.int32)
        self.height, self.width = self.grid.shape
        self.start = tuple(map_spec["start"])
        self.goal = tuple(map_spec["goal"])
        self.patrol_path = [tuple(p) for p in map_spec["patrol"]]
        self.chase_radius = game_kwargs.get("chase_radius", 3)
        self.chase_prob = game_kwargs.get("chase_prob", 0.05)
        self.chase_len = game_kwargs.get("chase_len", 5)
        self.monster_speed = game_kwargs.get("monster_speed", 1)
        assert self.grid[self.start] == 0, "起点被墙堵住"
        assert self.grid[self.goal] == 0, "终点被墙堵住"
        for p in self.patrol_path:
            assert self.grid[p] == 0, f"巡逻点 {p} 是墙"
        self.reset()


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    # 局部视野半径：观测量为 (2R+1)^2 的窗口。
    # 注意：局部视野大小【不随地图尺寸变化】—— 这样同一个策略架构可以
    # 在不同尺寸的地图上训练与评估，也是泛化实验能对比的前提。
    VIEW_R = 2

    def __init__(self, with_monster=True, max_steps=200, **game_kwargs):
        """
        game_kwargs 透传给 MazeGame，用于配置怪物行为 / 追加墙体：
            extra_walls, patrol_path, chase_radius, chase_prob,
            chase_len, monster_speed

        额外支持 map_spec（泛化实验用）：传入 envs.map_gen.generate_map 的
        返回值，即可在任意尺寸的生成地图上构造环境。start/goal 随之改变。
        """
        super().__init__()
        self.map_spec = game_kwargs.pop("map_spec", None)
        if self.map_spec is not None:
            self.game = MapMazeGame(map_spec=self.map_spec,
                                    with_monster=with_monster,
                                    max_steps=max_steps, **game_kwargs)
        else:
            self.game = MazeGame(with_monster=with_monster, max_steps=max_steps,
                                 **game_kwargs)

        k = 2 * self.VIEW_R + 1
        self._view_shape = (k, k)
        # 观测 = 墙视野 k*k + 怪物视野 k*k + 自身 k*k + 怪物相对 2
        #        + 终点相对 2 + 访问计数 1
        self.obs_dim = 3 * k * k + 5
        # 归一化分母：用实际地图边长，保证不同尺寸下尺度一致
        self._norm = float(max(self.game.height, self.game.width) - 1)
        # 访问计数表（破退化震荡用）
        self._visits = {}

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

    def _get_obs(self):
        ar, ac = self.game.agent_pos

        # a) 访问计数标量：告诉策略"这个格子你来过几次"。
        #    没有它时，观测是马尔可夫的 —— 原地来回震荡与探索在观测上
        #    完全等价，策略会收敛到"在两个格子间反复横跳"的退化解
        #    （实测 gen2 地图：200 步只访问 2 个不同格，全程超时）。
        visit = float(self._visits.get((ar, ac), 0))

        # 无怪模式下怪物不在图中，用哨兵值，保证任何通道都不会出现它
        if self.game.with_monster:
            mr, mc = self.game.monster_pos
        else:
            mr, mc = -1, -1

        gr, gc = self.game.goal

        k = 2 * self.VIEW_R + 1
        view_wall = np.zeros((k, k), dtype=np.float32)
        view_monster = np.zeros((k, k), dtype=np.float32)
        view_agent = np.zeros((k, k), dtype=np.float32)

        for i, dr in enumerate(range(-self.VIEW_R, self.VIEW_R + 1)):
            for j, dc in enumerate(range(-self.VIEW_R, self.VIEW_R + 1)):
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

        n = self._norm
        if self.game.with_monster:
            rel_monster = np.array([(mr - ar) / n, (mc - ac) / n],
                                   dtype=np.float32)
        else:
            rel_monster = np.array([0.0, 0.0], dtype=np.float32)

        rel_goal = np.array([(gr - ar) / n, (gc - ac) / n], dtype=np.float32)
        # 归一化访问计数：>=5 次视为饱和
        visit_feat = np.array([min(visit, 5.0) / 5.0], dtype=np.float32)

        obs = np.concatenate([local_view, rel_monster, rel_goal,
                              visit_feat]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self._visits = {}
        self._visits[tuple(self.game.agent_pos)] = 1
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        prev_ar, prev_ac = self.game.agent_pos
        prev_dist_goal = abs(prev_ar - self.game.goal[0]) + abs(prev_ac - self.game.goal[1])

        _, _, terminated, truncated, info = self.game.step(action)

        curr_ar, curr_ac = self.game.agent_pos
        curr_dist_goal = abs(curr_ar - self.game.goal[0]) + abs(curr_ac - self.game.goal[1])

        # 累加访问计数（在 _get_obs 之前，使观测含"本次到达"）
        key = (curr_ar, curr_ac)
        self._visits[key] = self._visits.get(key, 0) + 1
        obs = self._get_obs()

        # 1. 步数时间消耗（给前进压力）
        reward = -0.01

        # 2. 撞墙惩罚
        if info.get("hit_wall", False):
            reward -= 0.05
        else:
            # 靠近终点奖励
            progress = prev_dist_goal - curr_dist_goal
            reward += progress * 0.1

        # 2b. 重复访问惩罚 —— 破除"在两格之间反复横跳"的退化解。
        #     纯马尔可夫观测下，原地震荡与探索在状态上无法区分，
        #     策略会卡在 -0.01*n 的局部最优里（实测 gen2 图全程超时）。
        #     每次回到同一格额外扣分，使震荡的代价高于一路向前。
        reward -= 0.05 * min(self._visits[key] - 1, 4)

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


# ======================================================================
# MapMazeEnv 的真实实现（必须定义在 MazeEnv 之后）
# ======================================================================
class MapMazeEnv(MazeEnv):
    """
    在生成地图上运行的 MazeEnv（泛化实验用）。

    观测结构、奖励、终止条件全部继承 MazeEnv，只有地图来源不同 ——
    这样 A/B/C 三组对照的环境语义在新地图上完全一致，
    结果差异才能归因到触发机制，而不是环境实现差异。
    """

    def __init__(self, map_spec, with_monster=True, max_steps=200, **game_kwargs):
        super().__init__(with_monster=with_monster, max_steps=max_steps,
                         map_spec=map_spec, **game_kwargs)
        # 基类已按 map_spec 构造 MapMazeGame，这里显式记录 spec 便于追溯
        self.map_spec = map_spec
