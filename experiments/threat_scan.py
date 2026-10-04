# -*- coding: utf-8 -*-
"""
experiments/threat_scan.py —— 威胁度快速评估（重训前的筛选器）

为什么要它：每训练一个 PPO 要 ~10 分钟，不能盲目扫参数。
本脚本用三种【不需要训练】的求解器估计每个配置的威胁度：

  ① BFS 最优 agent  vs  巡逻怪物      —— 上界（会躲的智能体）
  ② 贪心 agent      vs  巡逻怪物      —— 中界（只朝终点走，不躲）
  ③ 随机 agent      vs  巡逻怪物      —— 下界

取这三个成功率作为威胁度的近似。真正合格与否仍需重训后复测，
但能有效筛掉"怪物根本挡不住路"的配置。

用法：
    python experiments/threat_scan.py
    python experiments/threat_scan.py --episodes 60
"""

import argparse
import os
import sys
from collections import deque

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.maze import MAZE, START, GOAL, ACTION_MAP

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ======================================================================
# 配置定义
# ======================================================================
MAPS = {
    "M0 原图": [],
    "M1 底行留缺口": [(10, 4), (10, 8)],
    "M2 底行两点": [(10, 4), (10, 7)],
    "M3 底行三点": [(10, 4), (10, 7), (10, 9)],
}

PATROLS = {
    "P0 原矩形": [(5, 4), (5, 8), (7, 8), (7, 4)],
    "P1 中央横巡": [(5, 4), (5, 8), (5, 5), (5, 7)],
    "P2 咽喉竖巡": [(3, 5), (8, 5), (7, 6), (4, 6)],
}

BEHAVIORS = {
    "B0 现状(3格/5%/5步)": dict(radius=3, prob=0.05, clen=5, speed=1),
    "B1 加强(4格/10%/6步)": dict(radius=4, prob=0.10, clen=6, speed=1),
    "B2 凶猛(5格/15%/8步)": dict(radius=5, prob=0.15, clen=8, speed=2),
}


class Sim:
    """轻量环境模拟器（与 MazeGame 规则一致，但不依赖 gym）"""

    def __init__(self, walls, patrol, behav, with_monster=True, max_steps=200):
        self.grid = np.array(MAZE, dtype=np.int32)
        for (r, c) in walls:
            self.grid[r, c] = 1
        self.patrol = list(patrol)
        self.b = behav
        self.with_monster = with_monster
        self.max_steps = max_steps

    def is_valid(self, pos):
        r, c = pos
        return (0 <= r < 12 and 0 <= c < 12 and self.grid[r, c] == 0)

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.agent = list(START)
        self.monster = list(self.patrol[0])
        self.pidx = 0
        self.steps = 0
        self.cd = 0

    def move_agent(self, a):
        dr, dc = ACTION_MAP[a]
        n = [self.agent[0] + dr, self.agent[1] + dc]
        if self.is_valid(n):
            self.agent = n
            return False
        return True

    def move_monster(self):
        if not self.with_monster:
            return
        for _ in range(self.b["speed"]):
            self._step_once()

    def _step_once(self):
        d = abs(self.monster[0] - self.agent[0]) + abs(self.monster[1] - self.agent[1])
        if d <= self.b["radius"] or self.rng.random() < self.b["prob"]:
            if self.cd == 0:
                self.cd = self.b["clen"]
        if self.cd > 0:
            self.cd -= 1
            cand = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                n = [self.monster[0] + dr, self.monster[1] + dc]
                if n == self.monster or self.is_valid(n):
                    dd = abs(n[0] - self.agent[0]) + abs(n[1] - self.agent[1])
                    cand.append((dd, n))
            if cand:
                cand.sort(key=lambda x: x[0])
                self.monster = cand[0][1]
        else:
            t = self.patrol[self.pidx]
            if self.monster == list(t):
                self.pidx = (self.pidx + 1) % len(self.patrol)
                t = self.patrol[self.pidx]
            dr = int(np.sign(t[0] - self.monster[0]))
            dc = int(np.sign(t[1] - self.monster[1]))
            n = [self.monster[0] + dr, self.monster[1] + (0 if dr != 0 else dc)]
            if self.is_valid(n):
                self.monster = n
            else:
                alt = [self.monster[0], self.monster[1] + dc]
                if self.is_valid(alt):
                    self.monster = alt

    def step(self, a):
        """返回 status: win / die / timeout / None"""
        self.steps += 1
        pa = list(self.agent)
        pm = list(self.monster)
        self.move_agent(a)
        if self.with_monster and self.agent == self.monster:
            return "die"
        self.move_monster()
        if self.with_monster:
            if self.agent == self.monster:
                return "die"
            # 对穿
            if self.agent == pm and self.monster == pa:
                return "die"
        if tuple(self.agent) == GOAL:
            return "win"
        if self.steps >= self.max_steps:
            return "timeout"
        return None


# ======================================================================
# 三种求解器
# ======================================================================
def bfs_dist_map(grid, src):
    dist = {tuple(src): 0}
    q = deque([tuple(src)])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            n = (r + dr, c + dc)
            if (0 <= n[0] < 12 and 0 <= n[1] < 12 and grid[n] == 0
                    and n not in dist):
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


def policy_bfs(sim, dist):
    """朝终点最优走（全知，但不躲怪）"""
    r, c = sim.agent
    best, ba = None, 0
    dcur = dist.get((r, c), 999)
    for a in range(4):
        dr, dc = ACTION_MAP[a]
        n = (r + dr, c + dc)
        if sim.is_valid(n) and dist.get(n, 999) < dcur:
            if best is None or dist[n] < best:
                best, ba = dist[n], a
    return ba


def policy_random(sim):
    return int(sim.rng.integers(0, 4))


def policy_greedy(sim, dist):
    """只朝终点走，但撞墙就换方向"""
    a = policy_bfs(sim, dist)
    if not sim.is_valid([sim.agent[0] + ACTION_MAP[a][0],
                         sim.agent[1] + ACTION_MAP[a][1]]):
        a = policy_random(sim)
    return a


def eval_config(walls, patrol, behav, episodes, max_steps=200):
    res = {}
    for pname, pol in [("bfs", policy_bfs), ("greedy", policy_greedy),
                       ("random", policy_random)]:
        for wm in [False, True]:
            wins = 0
            for ep in range(episodes):
                sim = Sim(walls, patrol, behav, with_monster=wm, max_steps=max_steps)
                sim.reset(seed=1000 + ep)
                dist = bfs_dist_map(sim.grid, GOAL)
                done = None
                while done is None:
                    a = pol(sim, dist) if pol is not policy_random else pol(sim)
                    done = sim.step(a)
                if done == "win":
                    wins += 1
            res[(pname, wm)] = 100 * wins / episodes
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    print("=" * 100)
    print(f"威胁度扫描（无需训练）| 每种配置 {args.episodes} 回合 × 3 种 agent")
    print("=" * 100)

    rows = []
    # 先扫地图 × 行为（巡逻用原矩形）
    for mname, walls in MAPS.items():
        for bname, behav in BEHAVIORS.items():
            r = eval_config(walls, PATROLS["P0 原矩形"], behav,
                            args.episodes, args.max_steps)
            margin_bfs = r[("bfs", False)] - r[("bfs", True)]
            margin_gr = r[("greedy", False)] - r[("greedy", True)]
            rows.append((f"{mname} + {bname}", r, margin_bfs, margin_gr))

    print(f"{'配置':<42}{'BFS无怪':>8}{'BFS有怪':>8}{'Δbfs':>7}"
          f"{'贪心无怪':>9}{'贪心有怪':>9}{'Δ贪心':>7}")
    print("-" * 100)
    for name, r, mb, mg in rows:
        print(f"{name:<42}{r[('bfs',False)]:>7.0f}%{r[('bfs',True)]:>7.0f}%"
              f"{mb:>7.0f}{r[('greedy',False)]:>8.0f}%{r[('greedy',True)]:>8.0f}%"
              f"{mg:>7.0f}")
    print("-" * 100)
    print("Δ = 无怪成功率 - 有怪成功率（越大威胁越强）")
    print()
    print("解读：BFS 是'全知上界'。若连 BFS 都被打下 20pp 以上，")
    print("      说明怪物能拦住最优路，威胁充分。")
    print()
    best = max(rows, key=lambda x: x[3])
    print(f"贪心口径下威胁最大的配置: {best[0]}  Δ={best[3]:.0f}pp")
    best2 = max(rows, key=lambda x: x[2])
    print(f"BFS   口径下威胁最大的配置: {best2[0]}  Δ={best2[2]:.0f}pp")


if __name__ == "__main__":
    main()
