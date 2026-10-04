# -*- coding: utf-8 -*-
"""
experiments/threat_sweep_map.py —— 在指定生成地图上快速扫描怪物威胁档位

为什么需要
    generalization 实验里 gen2/gen3 两张 16x16 地图上，怪物参数（4/15%/8）
    过强，导致无怪策略在有怪环境下 40/40 全被击杀，威胁度无法测量
    （有怪成功率恒为 0，与参数无关，没有区分度）。

做法
    用训练好的「无怪策略」当标尺 —— 它在无怪环境是 100%，所以它在
    有怪环境下的存活率**直接就是该档位的威胁强度**，无需重新训练。

用法
    python experiments/threat_sweep_map.py --seeds 2 3
    python experiments/threat_sweep_map.py --seeds 2 --episodes 60
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from envs.maze import MapMazeEnv

# 待扫描的威胁档位：从温和到凶猛
GRID = [
    dict(chase_radius=3, chase_prob=0.05, chase_len=5),
    dict(chase_radius=3, chase_prob=0.08, chase_len=5),
    dict(chase_radius=4, chase_prob=0.05, chase_len=5),
    dict(chase_radius=4, chase_prob=0.08, chase_len=6),
    dict(chase_radius=4, chase_prob=0.10, chase_len=6),
    dict(chase_radius=5, chase_prob=0.08, chase_len=6),
    dict(chase_radius=4, chase_prob=0.15, chase_len=8),
    dict(chase_radius=5, chase_prob=0.15, chase_len=8),
]


def evaluate(model_path, map_spec, kw, episodes, base_seed=42,
             with_monster=True, max_steps=None):
    if max_steps is None:
        max_steps = max(200, (map_spec.get("base_dist") or 18) * 4)
    env = MapMazeEnv(map_spec, with_monster=with_monster,
                     max_steps=max_steps, **kw)
    model = PPO.load(model_path, env=env)
    wins = die = to = 0
    steps_win, steps_die = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep * 100)
        done, steps = False, 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, t, tr, info = env.step(int(a))
            steps += 1
            done = t or tr
            if t:
                if info.get("status") == "win":
                    wins += 1
                    steps_win.append(steps)
                else:
                    die += 1
                    steps_die.append(steps)
                break
            if tr:
                to += 1
                break
    n = episodes
    return dict(win=100 * wins / n, die=100 * die / n, timeout=100 * to / n,
                win_steps=float(np.mean(steps_win)) if steps_win else 0.0,
                die_steps=float(np.mean(steps_die)) if steps_die else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--out", default="results/generalize/threat_sweep.json")
    args = ap.parse_args()

    all_rows = []
    for seed in args.seeds:
        mp = f"results/generalize/maps/map_{args.size}_gen{seed}.json"
        if not os.path.exists(mp):
            print(f"跳过 seed={seed}: 找不到 {mp}")
            continue
        m = json.load(open(mp, encoding="utf-8"))
        model_path = f"results/generalize/ppo_gen{seed}_{args.size}_no_monster.zip"
        if not os.path.exists(model_path):
            print(f"跳过 seed={seed}: 找不到 {model_path}")
            continue

        print("=" * 84)
        print(f"地图 gen{seed}_{args.size}  最短 {m['base_dist']} 步 | "
              f"静态威胁度 {m['threat']} | 标尺 = 无怪策略（无怪环境 100%）")
        print("=" * 84)
        # 先确认标尺在无怪环境下确实是 100%
        base = evaluate(model_path, m, dict(), args.episodes,
                        args.base_seed, with_monster=False)
        print(f"  标尺校验: 无怪 {base['win']:.1f}%  "
              f"{'✅' if base['win'] >= 95 else '⚠️ 标尺不达标，结果仅供参考'}")
        print()
        print(f"  {'档位 (半径/概率/时长)':<26}{'存活率':>9}{'击杀率':>9}"
              f"{'超时':>8}{'存活步数':>10}{'阵亡步数':>10}{'威胁pp':>9}")
        print("  " + "-" * 82)
        for kw in GRID:
            r = evaluate(model_path, m, kw, args.episodes,
                         args.base_seed, with_monster=True)
            threat = base["win"] - r["win"]
            tag = f"{kw['chase_radius']}/{kw['chase_prob']:.2f}/{kw['chase_len']}"
            print(f"  {tag:<26}{r['win']:>8.1f}%{r['die']:>8.1f}%"
                  f"{r['timeout']:>7.1f}%{r['win_steps']:>10.1f}"
                  f"{r['die_steps']:>10.1f}{threat:>9.1f}")
            all_rows.append(dict(seed=seed, kw=kw, base_win=base["win"],
                                 win=r["win"], die=r["die"],
                                 timeout=r["timeout"], threat_pp=threat,
                                 win_steps=r["win_steps"],
                                 die_steps=r["die_steps"]))
        print()

    if not all_rows:
        print("没有可用数据")
        return

    # 汇总：哪个档位在各图上最接近 20~50pp 的合理区间
    print("=" * 84)
    print("汇总：各档位在三张图上的威胁度（目标区间 20~55pp，且不能全灭）")
    print("=" * 84)
    keys = {(r["kw"]["chase_radius"], r["kw"]["chase_prob"],
             r["kw"]["chase_len"]) for r in all_rows}
    print(f"  {'档位':<20}{'gen1':>10}{'gen2':>10}{'gen3':>10}{'均值':>9}{'可用':>7}")
    print("  " + "-" * 68)
    best = None
    for k in sorted(keys):
        vals = [r["threat_pp"] for r in all_rows
                if (r["kw"]["chase_radius"], r["kw"]["chase_prob"],
                    r["kw"]["chase_len"]) == k]
        avg = float(np.mean(vals))
        usable = all(15 <= v <= 65 for v in vals)
        tag = f"{k[0]}/{k[1]:.2f}/{k[2]}"
        cells = "".join(f"{v:>10.1f}" for v in vals)
        print(f"  {tag:<20}{cells}{avg:>9.1f}{'✅' if usable else '':>7}")
        if usable and (best is None or abs(avg - 35) < abs(best[1] - 35)):
            best = (tag, avg)
    print("  " + "-" * 68)
    if best:
        print(f"  推荐档位: {best[0]}  (均值威胁 {best[1]:.1f}pp)")
    else:
        print("  没有档位同时满足所有地图；请扩大扫描范围或调整地图难度")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
