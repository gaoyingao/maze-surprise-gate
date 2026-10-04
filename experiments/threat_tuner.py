# -*- coding: utf-8 -*-
"""
experiments/threat_tuner.py —— 怪物威胁度快速迭代框架

目的：把「有怪 vs 无怪成功率差 ≥ 20pp」这件事从手工试错变成可扫的网格。

三个要一起看清楚的数：
  1. 无怪天花板 ceiling  —— 无怪时的纯 PPO 成功率
  2. 有怪存留   survival —— 用同一个策略在带怪环境里跑
  3. 理论可达上限          用 BFS 判断：怪物若完全封死必经路，agent 还能不能过

注意：本脚本是【快速筛选】工具。因为 PPO 是在旧环境上训练的，
      它对「新怪物行为」的适应度不是最优的 —— 筛选出候选配置后
      必须重新训练 PPO 才能得到可信的 threat margin。

用法：
    python experiments/threat_tuner.py                 # 扫预设配置
    python experiments/threat_tuner.py --episodes 60
"""

import argparse
import os
import sys
from collections import deque

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from envs.maze import MazeEnv, MAZE, START, GOAL
from envs.a_config import A_ENV_KWARGS, A_MODEL_PATH

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ----------------------------------------------------------------------
# BFS 工具：判断在给定"禁区"下起点到终点是否仍连通
# ----------------------------------------------------------------------
def reachable(grid, start, goal, blocked=()):
    if grid[start] != 0 or grid[goal] != 0:
        return False
    blocked = set(blocked)
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in DIRS4:
            n = (r + dr, c + dc)
            if (0 <= n[0] < 12 and 0 <= n[1] < 12
                    and grid[n] == 0 and n not in seen and n not in blocked):
                seen.add(n)
                q.append(n)
    return False


def bfs_dist(grid, src):
    dist = {src: 0}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            n = (r + dr, c + dc)
            if (0 <= n[0] < 12 and 0 <= n[1] < 12
                    and grid[n] == 0 and n not in dist):
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


# ----------------------------------------------------------------------
# 配置：每种 = 名称 + 对环境的改写
# ----------------------------------------------------------------------
def make_cfg_default():
    return {}


def make_cfg_corridor():
    """把巡逻路径压到西侧走廊 (1..10, 1) —— 那是必经路的咽喉"""
    return {"patrol_path": [(1, 1), (10, 1), (10, 2), (1, 2)]}


def make_cfg_block_wide():
    """在迷宫中央加两道墙，压缩可绕行空间（改地图）"""
    return {"extra_walls": [(4, 2), (4, 3), (6, 6), (6, 7), (8, 5), (8, 6)]}


def make_cfg_aggressive():
    """追击更凶：3 格 -> 5 格触发，5% -> 15% 随机，持续 5 -> 8 步"""
    return {"chase_radius": 5, "chase_prob": 0.15, "chase_len": 8}


def make_cfg_fast_monster():
    """怪物每步移动 2 格（更快）"""
    return {"monster_speed": 2}


def make_cfg_combined():
    """组合：走廊巡逻 + 中央加墙 + 更凶追击"""
    c = {}
    c.update(make_cfg_corridor())
    c.update(make_cfg_block_wide())
    c.update(make_cfg_aggressive())
    return c


CONFIGS = {
    "baseline(现状)": make_cfg_default,
    "A 走廊巡逻": make_cfg_corridor,
    "B 中央加墙": make_cfg_block_wide,
    "C 更凶追击": make_cfg_aggressive,
    "D 怪物加速": make_cfg_fast_monster,
    "E A+B+C 组合": make_cfg_combined,
}


# ----------------------------------------------------------------------
# 打补丁：把配置应用到 MazeGame
# ----------------------------------------------------------------------
def patch_env(cfg):
    """返回一个打了补丁的 MazeEnv 类（每次调用独立，避免互相污染）"""
    import envs.maze as M

    grid = np.array(MAZE, dtype=np.int32)
    for (r, c) in cfg.get("extra_walls", []):
        grid[r, c] = 1
        # 加墙后若起点到终点不再连通，说明该配置破坏性地图 -> 报错
    if not reachable(grid, START, GOAL):
        raise ValueError("加墙后起点到终点不连通，配置无效")

    patrol = cfg.get("patrol_path")
    radius = cfg.get("chase_radius", 3)
    prob = cfg.get("chase_prob", 0.05)
    clen = cfg.get("chase_len", 5)
    speed = cfg.get("monster_speed", 1)

    class PatchedGame(M.MazeGame):
        def __init__(self, with_monster=True, max_steps=200):
            super().__init__(with_monster=with_monster, max_steps=max_steps)
            if cfg.get("extra_walls"):
                self.grid = grid.copy()
            if patrol:
                self.patrol_path = list(patrol)
                self.monster_pos = list(self.patrol_path[0])

        def _move_monster(self):
            if not self.with_monster:
                return
            for _ in range(speed):
                self._move_monster_once(radius, prob, clen)

        def _move_monster_once(self, radius, prob, clen):
            dist = abs(self.monster_pos[0] - self.agent_pos[0]) + \
                   abs(self.monster_pos[1] - self.agent_pos[1])
            if dist <= radius or np.random.random() < prob:
                if self.chase_countdown == 0:
                    self.chase_countdown = clen

            if self.chase_countdown > 0:
                self.chase_countdown -= 1
                cand = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                    nr, nc = self.monster_pos[0] + dr, self.monster_pos[1] + dc
                    if (nr, nc) == tuple(self.monster_pos) or self.is_valid((nr, nc)):
                        d = abs(nr - self.agent_pos[0]) + abs(nc - self.agent_pos[1])
                        cand.append((d, [nr, nc]))
                if cand:
                    cand.sort(key=lambda x: x[0])
                    self.monster_pos = cand[0][1]
            else:
                target = self.patrol_path[self.patrol_idx]
                if self.monster_pos == list(target):
                    self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_path)
                    target = self.patrol_path[self.patrol_idx]
                dr = int(np.sign(target[0] - self.monster_pos[0]))
                dc = int(np.sign(target[1] - self.monster_pos[1]))
                nxt = [self.monster_pos[0] + dr, self.monster_pos[1] + (0 if dr != 0 else dc)]
                if self.is_valid(nxt):
                    self.monster_pos = nxt
                else:
                    alt = [self.monster_pos[0], self.monster_pos[1] + dc]
                    if self.is_valid(alt):
                        self.monster_pos = alt

    class PatchedEnv(M.MazeEnv):
        def __init__(self, with_monster=True, max_steps=200):
            super().__init__(with_monster=with_monster, max_steps=max_steps)
            self.game = PatchedGame(with_monster=with_monster, max_steps=max_steps)

    return PatchedEnv, grid


def run(model, EnvCls, with_monster, episodes, base_seed, max_steps):
    env = EnvCls(with_monster=with_monster, max_steps=max_steps)
    wins = deaths = timeouts = 0
    steps_win = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep * 100)
        done = False
        steps = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(a))
            steps += 1
            done = term or trunc
            if term:
                if info.get("status") == "win":
                    wins += 1
                    steps_win.append(steps)
                else:
                    deaths += 1
                break
            if trunc:
                timeouts += 1
                break
    n = episodes
    return (100 * wins / n, 100 * deaths / n, 100 * timeouts / n,
            float(np.mean(steps_win)) if steps_win else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--configs", default="", help="逗号分隔，默认全部")
    args = ap.parse_args()

    ckpt = A_MODEL_PATH
    if not os.path.exists(ckpt):
        print(f"找不到 {ckpt}")
        return
    model = PPO.load(ckpt, env=MazeEnv(with_monster=True,
                                       max_steps=args.max_steps, **A_ENV_KWARGS))

    names = ([s.strip() for s in args.configs.split(",") if s.strip()]
             if args.configs else list(CONFIGS))

    print("=" * 88)
    print(f"怪物威胁度扫描 | 回合数 {args.episodes} | 用旧策略 {ckpt} 近似评估")
    print("=" * 88)
    print(f"{'配置':<18}{'无怪成功':>9}{'有怪成功':>9}{'落差pp':>9}"
          f"{'击杀':>7}{'超时':>7}{'步数':>7}   判定")
    print("-" * 88)

    results = []
    for name in names:
        cfg = CONFIGS[name]()
        try:
            EnvCls, grid = patch_env(cfg)
        except Exception as e:
            print(f"{name:<18} 配置无效: {e}")
            continue

        no_succ, _, _, _ = run(model, EnvCls, False, args.episodes,
                               args.base_seed, args.max_steps)
        yes_succ, die, to, st = run(model, EnvCls, True, args.episodes,
                                    args.base_seed, args.max_steps)
        margin = no_succ - yes_succ
        verdict = "✅ 达标" if margin >= 20 else ("⚠️ 接近" if margin >= 12 else "❌ 不足")
        results.append((name, no_succ, yes_succ, margin, die, to, st, verdict))
        print(f"{name:<18}{no_succ:>8.1f}%{yes_succ:>8.1f}%{margin:>9.1f}"
              f"{die:>6.1f}%{to:>6.1f}%{st:>7.1f}   {verdict}")

    print("-" * 88)
    print("门槛: 落差 >= 20pp")
    print()
    print("⚠️ 提醒：PPO 是在旧环境上训练的，对新怪物行为不适应。")
    print("   本表只用于筛掉明显无效的配置；选中的配置必须重训 PPO 后复测。")

    best = max(results, key=lambda x: x[3]) if results else None
    if best:
        print(f"\n当前最佳候选: {best[0]}  落差 {best[3]:.1f}pp")


if __name__ == "__main__":
    main()
