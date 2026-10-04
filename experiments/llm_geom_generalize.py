# -*- coding: utf-8 -*-
"""
experiments/llm_geom_generalize.py —— LLM 几何能力的跨地图检验

要回答的问题
    在 12x12 原图上，qwen3.5:4b 的首选合法率是 92.4%，死角层（1 个出口）
    4/4 全对，标签置换 16/20 —— 当时判定为"真几何判断"。

    **但那只有 79 个可通行格。** 模型完全可能是把这张图背下来了。
    换一张它从没见过的地图（尤其更大、格子更多的），如果合法率暴跌，
    就说明之前的高分是记忆而非几何推理。

方法
    在指定地图上做分层抽样（层 = 该格可走方向数 1/2/3/4），
    每个格子配多个怪物位置各问一次，统计首选方向的合法率。

    判决标准（沿用 stratified_probe 的口径）：
        死角层(1 个出口) 合法率 >= 85%  -> 真几何判断
        死角层 < 60% 而十字层 > 90%     -> 模式补全
    随机基线：层号/4（死角 25%，十字 100%）

用法
    python experiments/llm_geom_generalize.py --map orig --per-layer 8
    python experiments/llm_geom_generalize.py --map gen --size 16 --per-layer 8
    python experiments/llm_geom_generalize.py --map gen --size 20 --per-layer 6
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OLLAMA_URL = "http://localhost:11434/api/generate"
DIRS = ["上", "下", "左", "右"]
OFF = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}


# ----------------------------------------------------------------------
def load_map(kind, size, seed):
    """返回 (grid: np.ndarray, goal: (r,c), name)"""
    if kind == "orig":
        import envs.maze as M
        g = np.array(M.MAZE)
        return g, tuple(M.GOAL), "orig12"
    from envs.map_gen import generate_map
    m = generate_map(seed, size=size, max_tries=1500)
    if m is None:
        raise SystemExit(f"生成地图失败: size={size} seed={seed}")
    return np.array(m["grid"]), tuple(m["goal"]), f"gen{size}_{seed}"


def neighbors(grid, r, c):
    H, W = grid.shape
    return [(d, r + OFF[d][0], c + OFF[d][1]) for d in DIRS
            if 0 <= r + OFF[d][0] < H and 0 <= c + OFF[d][1] < W]


def legal_dirs(grid, r, c):
    return [d for d, rr, cc in neighbors(grid, r, c) if grid[rr, cc] == 0]


def is_free(grid, r, c):
    H, W = grid.shape
    return 0 <= r < H and 0 <= c < W and grid[r, c] == 0


def build_prompt(grid, r, c, goal, mon, size):
    H, W = grid.shape
    return (f"你在一个 {H}x{W} 的迷宫里，出口在 (x={goal[1]}, y={goal[0]})。\n"
            f"怪物在 (x={mon[1]}, y={mon[0]})，它正在追击你。\n"
            f"你现在在 (x={c}, y={r})。\n\n"
            f"你四周四个方向的格子情况如下：\n"
            f"上 方那一格是：{'路' if is_free(grid, r-1, c) else '墙'}\n"
            f"下 方那一格是：{'路' if is_free(grid, r+1, c) else '墙'}\n"
            f"左 方那一格是：{'路' if is_free(grid, r, c-1) else '墙'}\n"
            f"右 方那一格是：{'路' if is_free(grid, r, c+1) else '墙'}\n\n"
            "请把你希望走的方向按优先顺序排序。\n"
            "严格按格式输出 4 行：\n上=？\n下=？\n左=？\n右=？\n"
            "每行填 1/2/3/4，各用一次。\n不要解释。")


def ask(model, prompt, npred=400):
    r = requests.post(OLLAMA_URL, json={
        "model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
        "think": False,
        "options": {"temperature": 0.0, "seed": 42, "num_predict": npred}},
        timeout=1800).json()
    resp = (r.get("response") or "").strip()
    if not resp and (r.get("thinking") or "").strip():
        resp = r["thinking"].strip()[-300:]
    return resp


def parse_ranking(raw):
    got = {}
    for line in raw.splitlines():
        m = re.match(r"\s*([上下左右])\s*[=＝:：]\s*([1-4])", line)
        if m:
            got[m.group(1)] = int(m.group(2))
    seq = [d for d, _ in sorted(got.items(), key=lambda kv: kv[1])]
    return (seq[0] if seq else None), seq


# ----------------------------------------------------------------------
def run_audit(model, grid, goal, per_layer=8, n_monsters=2, seed=7,
              verbose=True):
    H, W = grid.shape
    cells = [(r, c) for r in range(H) for c in range(W) if grid[r, c] == 0]
    rng = np.random.default_rng(seed)

    # 分层：按可走方向数
    layers = {1: [], 2: [], 3: [], 4: []}
    for cell in cells:
        k = len(legal_dirs(grid, *cell))
        if k >= 1:
            layers[k].append(cell)

    # 怪物位置：在地图上随机取，避开被测格子
    monsters = []
    while len(monsters) < n_monsters:
        p = cells[int(rng.integers(0, len(cells)))]
        if all(abs(p[0]-m[0]) + abs(p[1]-m[1]) > 2 for m in monsters):
            monsters.append(p)

    sample = []
    for k in [1, 2, 3, 4]:
        grp = layers[k][:per_layer]
        sample += [(k, cell) for cell in grp]
    total = len(sample) * len(monsters)

    if verbose:
        print("=" * 78)
        print(f"LLM 几何分层检验 —— {model}")
        print(f"地图 {H}x{W} | 可通行 {len(cells)} 格 | "
              f"分层规模 1出口={len(layers[1])} 2={len(layers[2])} "
              f"3={len(layers[3])} 4={len(layers[4])}")
        print(f"每层抽 {per_layer} 格 × {len(monsters)} 怪物位 = {total} 次调用")
        print("层 = 该格可走方向数；随机猜中率 = 层号/4")
        print("=" * 78)

    t0 = time.time()
    rows = []
    for k in [1, 2, 3, 4]:
        grp = layers[k][:per_layer]
        if not grp:
            if verbose:
                print(f"  {k} 个出口: 该层无格子，跳过")
            continue
        ok = tot = 0
        raw_set = set()
        for (r, c) in grp:
            lg = legal_dirs(grid, r, c)
            for mon in monsters:
                raw = ask(model, build_prompt(grid, r, c, goal, mon, H))
                raw_set.add(raw.strip())
                first, _ = parse_ranking(raw)
                tot += 1
                if first in lg:
                    ok += 1
        rows.append({"layer": k, "cells": len(grp), "calls": tot,
                     "legal_hits": ok,
                     "legal_pct": round(100 * ok / tot, 1),
                     "random_pct": round(100 * k / 4, 1),
                     "unique_outputs": len(raw_set)})
        if verbose:
            print(f"  {k} 个出口: 合法 {ok}/{tot} = {100*ok/tot:5.1f}%   "
                  f"(随机 {100*k/4:.0f}%)   不同输出 {len(raw_set)}")

    elapsed = time.time() - t0
    if verbose:
        print("-" * 78)
        print(f"耗时 {elapsed:.0f}s  ({elapsed/max(total,1):.2f}s/次)")
    return rows, elapsed, total


def verdict(rows):
    l1 = next((r for r in rows if r["layer"] == 1), None)
    l4 = next((r for r in rows if r["layer"] == 4), None)
    if not l1:
        return "无死角层样本，无法判决"
    l1p = l1["legal_pct"]
    l4p = l4["legal_pct"] if l4 else float("nan")
    if l1p >= 85:
        return f"真几何判断（死角层 {l1p}% >= 85%）"
    if l1p < 60 and (l4 is None or l4p > 90):
        return f"模式补全（死角层 {l1p}% < 60%，十字层 {l4p}%）"
    return f"部分几何能力，紧约束下不可靠（死角层 {l1p}%）"


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--map", dest="kind", default="orig",
                    choices=["orig", "gen"])
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-layer", type=int, default=8)
    ap.add_argument("--monsters", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grid, goal, name = load_map(args.kind, args.size, args.seed)
    print(f"地图: {name}  ({grid.shape[0]}x{grid.shape[1]})  出口={goal}")
    rows, elapsed, total = run_audit(args.model, grid, goal,
                                     per_layer=args.per_layer,
                                     n_monsters=args.monsters)
    v = verdict(rows)
    print()
    print("=" * 78)
    print(f"判决: {v}")
    print("=" * 78)

    out = {"model": args.model, "map": name,
           "size": [int(grid.shape[0]), int(grid.shape[1])],
           "free_cells": int((grid == 0).sum()),
           "goal": list(goal), "per_layer": args.per_layer,
           "n_monsters": args.monsters, "layers": rows,
           "verdict": v, "elapsed_s": round(elapsed, 1), "calls": total}
    os.makedirs("results/generalize", exist_ok=True)
    path = args.out or f"results/generalize/llm_geom_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已保存: {path}")


if __name__ == "__main__":
    main()
