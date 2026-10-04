# -*- coding: utf-8 -*-
"""
envs/map_gen.py —— 泛化实验用的可变地图生成器

为什么需要它
    目前所有结论都基于同一张固定 12x12 迷宫（79 个可通行格），存在地图过拟合
    风险：PPO 完全可能把这张图记下来，从而掩盖"需要真会走"的差距。
    **把地图放大**是最直接的检验手段 —— 格子一多就记不住，必须真会走。

设计原则
    1. 完全由 seed 决定，可复现；地图以 seed 为唯一身份。
    2. 尺寸可配（size x size），起点/终点/巡逻路径随之自适应。
    3. **连通性保证**：生成后做 BFS 校验。
    4. **无绝对咽喉**：任一巡逻点被单独封堵后仍可达。否则怪物站上去就能
       永久堵死任务（怪物不离开巡逻路径），智能体无法绕过，训练必然失败。
    5. 威胁度适中：单格封堵导致的最大绕路比落在 [min_threat, max_threat]。
       太低说明怪物挡不住路；太高说明一步走错就完蛋。
    6. 怪物初始位置距出生点足够远，避免开局即死。

地图表示
    {"seed", "gen_seed", "size", "grid", "start", "goal", "patrol",
     "base_dist", "worst_detour_dist", "threat"}

用法
    python envs/map_gen.py --generate 3 --size 16
    python envs/map_gen.py --generate 2 --size 20
"""

import argparse
import os
import sys
from collections import deque

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ----------------------------------------------------------------------
def make_grid(seed, size, wall_prob=0.30, border=True):
    """随机生成一张 size x size 地图：外圈全墙，内部按概率设墙"""
    rng = np.random.default_rng(seed)
    g = np.zeros((size, size), dtype=np.int32)
    if border:
        g[0, :] = g[size-1, :] = 1
        g[:, 0] = g[:, size-1] = 1
    inner = rng.random((size - 2, size - 2)) < wall_prob
    g[1:size-1, 1:size-1] = inner.astype(np.int32)
    return g


def bfs(grid, start, goal, blocked=frozenset()):
    H, W = grid.shape
    if tuple(start) in blocked or tuple(goal) in blocked:
        return None
    if grid[tuple(start)] != 0 or grid[tuple(goal)] != 0:
        return None
    q = deque([tuple(start)])
    dist = {tuple(start): 0}
    while q:
        cur = q.popleft()
        if cur == tuple(goal):
            return dist[cur]
        for dr, dc in DIRS4:
            n = (cur[0] + dr, cur[1] + dc)
            if (0 <= n[0] < H and 0 <= n[1] < W and grid[n] == 0
                    and n not in dist and n not in blocked):
                dist[n] = dist[cur] + 1
                q.append(n)
    return None


def free_cells(grid):
    H, W = grid.shape
    return [(r, c) for r in range(H) for c in range(W) if grid[r, c] == 0]


def shortest_path(grid, start, goal):
    """返回一条最短路径（格子列表），不可达返回 None"""
    H, W = grid.shape
    q = deque([tuple(start)])
    prev = {tuple(start): None}
    while q:
        cur = q.popleft()
        if cur == tuple(goal):
            break
        for dr, dc in DIRS4:
            n = (cur[0] + dr, cur[1] + dc)
            if (0 <= n[0] < H and 0 <= n[1] < W and grid[n] == 0
                    and n not in prev):
                prev[n] = cur
                q.append(n)
    if tuple(goal) not in prev:
        return None
    path, cur = [], tuple(goal)
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def pick_patrol(grid, start, goal, n_way=4, band_lo=2, band_hi=4):
    """
    在"必经之路的侧翼带"上选巡逻航点。

    为什么要侧翼带而不是直接取最短路上的点：
        直接取最短路会让怪物初始位置就蹲在智能体的必经之路上。
        实测 gen2/gen3 两张 16x16 地图上，**即使最温和的怪物参数**
        （半径 3 / 5% / 5 步）也会在 4~6 步内击杀 —— 因为智能体沿
        最短路前进时与怪物迎面相撞，无论怎么调参数都救不回来，
        威胁度失去区分度。

        放在距离最短路 band_lo~band_hi 格的自由格上，怪物仍能威胁
        必经之路（追击半径通常 >= 3），但智能体有绕行余地。

    做法：
        1. 求一条最短路径
        2. 收集所有"距最短路在 [band_lo, band_hi] 格内"的自由格
        3. 按到起点的距离排序后均匀采样 n_way 个作为航点
    """
    path = shortest_path(grid, start, goal)
    if path is None:
        return None
    L = len(path)
    if L < n_way + 4:
        return None
    H, W = grid.shape

    # 到最短路的曼哈顿距离（只算一定范围内的，避免全图 O(N*L)）
    def dist_to_path(cell):
        r, c = cell
        best = 10 ** 9
        for (pr, pc) in path:
            d = abs(r - pr) + abs(c - pc)
            if d < best:
                best = d
                if best <= band_lo:
                    return best
        return best

    band = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0:
                continue
            d = dist_to_path((r, c))
            if band_lo <= d <= band_hi:
                band.append((r, c))
    if len(band) < n_way:
        return None

    # 按路径进度排序：用"到路径上最近点的下标"近似
    def progress(cell):
        r, c = cell
        return min(range(len(path)),
                   key=lambda i: abs(r - path[i][0]) + abs(c - path[i][1]))

    band.sort(key=progress)
    # 掐头去尾，避开紧贴起点/终点的位置
    lo = max(0, int(len(band) * 0.12))
    hi = max(lo + 1, int(len(band) * 0.88))
    body = band[lo:hi]
    if len(body) < n_way:
        body = band
    idx = np.linspace(0, len(body) - 1, n_way).round().astype(int)
    way = [body[i] for i in idx]
    uniq = []
    for w in way:
        if not uniq or uniq[-1] != w:
            uniq.append(w)
    if len(uniq) < 3:
        # 去重后太少，放宽带宽再试一次
        return None
    return uniq


def threat_of(grid, patrol, start, goal):
    """
    威胁度 = 怪物堵住【单个】巡逻点时，最短路径相对基线的最大增幅。

    不用"全巡逻区封堵"：那几乎必然不可达，比值恒为上限，没有区分度。
    真实场景里怪物一次只占一格，效果是"逼迫绕路"。

    ⚠️ ratio=None 表示存在"单格封死"的咽喉点 —— 该图不可用。
    """
    base = bfs(grid, start, goal)
    if base is None:
        return None, None, 0.0
    worst = base
    for p in patrol:
        d = bfs(grid, start, goal, frozenset([p]))
        if d is None:
            return base, None, None
        worst = max(worst, d)
    return base, worst, worst / base


# ----------------------------------------------------------------------
def generate_map(seed, size=12, wall_prob=0.30, min_dist=None, max_dist=None,
                 min_threat=1.20, max_threat=2.2, max_tries=800, verbose=False):
    """
    生成一张合格地图。尺寸自适应：默认最短路径要求约为 size*1.4 上下。
    """
    H = W = size
    if min_dist is None:
        min_dist = int(size * 1.3)
    if max_dist is None:
        max_dist = int(size * 2.6)

    for k in range(max_tries):
        s = seed * 1000 + k
        g = make_grid(s, size, wall_prob=wall_prob)
        free = free_cells(g)
        if len(free) < size * size * 0.30:
            continue
        rng = np.random.default_rng(s + 7)
        a = free[int(rng.integers(0, len(free)))]
        # 从 a 做 BFS，取最远的几格之一作为终点
        q = deque([a])
        dist = {a: 0}
        while q:
            cur = q.popleft()
            for dr, dc in DIRS4:
                n = (cur[0]+dr, cur[1]+dc)
                if (0 <= n[0] < H and 0 <= n[1] < W and g[n] == 0
                        and n not in dist):
                    dist[n] = dist[cur] + 1
                    q.append(n)
        far = sorted(dist.items(), key=lambda kv: -kv[1])[:8]
        if not far or far[0][1] < min_dist:
            continue
        b = far[int(rng.integers(0, min(len(far), 4)))][0]
        d = bfs(g, a, b)
        if d is None or not (min_dist <= d <= max_dist):
            continue

        patrol = pick_patrol(g, a, b)
        if patrol is None:
            continue
        # 怪物初始位置必须离出生点足够远
        if abs(patrol[0][0] - a[0]) + abs(patrol[0][1] - a[1]) < max(5, size // 3):
            continue

        base, blk, ratio = threat_of(g, patrol, a, b)
        if ratio is None:            # 有绝对咽喉，弃用
            continue
        if not (min_threat <= ratio <= max_threat):
            continue

        m = dict(seed=seed, gen_seed=s, size=size, grid=g.tolist(),
                 start=list(a), goal=list(b),
                 patrol=[list(p) for p in patrol],
                 base_dist=base, worst_detour_dist=blk,
                 threat=round(ratio, 3))
        if verbose:
            print(f"  候选 {k}: start={tuple(a)} goal={tuple(b)} dist={d} "
                  f"threat={ratio:.2f}")
        return m
    return None


# ----------------------------------------------------------------------
def render(m, show_patrol=True):
    g = np.array(m["grid"])
    H, W = g.shape
    start, goal = tuple(m["start"]), tuple(m["goal"])
    patrol = {tuple(p) for p in m["patrol"]} if show_patrol else set()
    lines = []
    hdr = "   " + "".join(f"{c%10}" for c in range(W))
    lines.append(hdr)
    for r in range(H):
        row = ""
        for c in range(W):
            if (r, c) == start:
                row += "S"
            elif (r, c) == goal:
                row += "G"
            elif (r, c) in patrol:
                row += "M"
            elif g[r, c] == 0:
                row += "."
            else:
                row += "#"
        lines.append(f"{r:>2} " + row)
    return "\n".join(lines)


def show(m):
    print(f"  seed={m['seed']} size={m['size']} (gen={m['gen_seed']}) "
          f"start={tuple(m['start'])} goal={tuple(m['goal'])}")
    print(f"  最短路径 {m['base_dist']} 步 | 单格封堵后最长 "
          f"{m['worst_detour_dist']} | 威胁度 {m['threat']}")
    print(f"  巡逻航点 {[tuple(p) for p in m['patrol']]}")
    print(render(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--wall-prob", type=float, default=0.30)
    ap.add_argument("--seed-start", type=int, default=1)
    args = ap.parse_args()

    if args.generate:
        print("=" * 72)
        print(f"生成 {args.generate} 张 {args.size}x{args.size} 地图"
              f"（wall_prob={args.wall_prob}）")
        print("=" * 72)
        ok = 0
        for seed in range(args.seed_start, args.seed_start + args.generate * 10):
            if ok >= args.generate:
                break
            m = generate_map(seed, size=args.size, wall_prob=args.wall_prob)
            if m is None:
                continue
            ok += 1
            print()
            show(m)
        print(f"\n共生成 {ok}/{args.generate} 张")
    else:
        print("用 --generate N --size S 生成地图")


if __name__ == "__main__":
    main()
