# -*- coding: utf-8 -*-
"""
experiments/threat_report.py —— 怪物威胁度最终报告

按【同一策略跨环境】的公平口径测量，这是唯一站得住脚的对照方式：
    用同一个策略，分别在无怪 / 有怪环境各跑 N 回合，比较成功率。

用法：
    python experiments/threat_report.py
"""

import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from envs.maze import MazeEnv

CONFIGS = {
    "baseline": dict(),
    "medium": dict(chase_radius=4, chase_prob=0.10, chase_len=6),
    "strong": dict(chase_radius=5, chase_prob=0.15, chase_len=8, monster_speed=2),
}
SEEDS = [42, 777, 2024]
EPISODES = 100


def evaluate(model, kw, with_monster, episodes, base_seed, max_steps=200):
    env = MazeEnv(with_monster=with_monster, max_steps=max_steps, **kw)
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
    return dict(win=100 * wins / n, die=100 * deaths / n,
                timeout=100 * timeouts / n,
                steps=float(np.mean(steps_win)) if steps_win else 0.0)


def main():
    report = {}
    print("=" * 90)
    print("怪物威胁度最终报告 —— 同一策略跨环境对照（每格 3 个种子 × 100 回合）")
    print("=" * 90)

    for cfg_name, kw in CONFIGS.items():
        mpath = f"models/threat_{cfg_name}_with_monster.zip"
        if not os.path.exists(mpath):
            print(f"\n[{cfg_name}] 缺少 {mpath}，跳过")
            continue
        model = PPO.load(mpath, env=MazeEnv(with_monster=True, max_steps=200))

        print(f"\n【{cfg_name}】{kw if kw else '（现状行为）'}")
        print(f"  策略: {mpath}")
        print(f"  {'种子':<8}{'无怪成功':>10}{'有怪成功':>10}{'落差pp':>9}"
              f"{'有怪击杀':>10}{'无怪步数':>10}{'有怪步数':>10}")
        print("  " + "-" * 76)

        margins = []
        wins_no, wins_yes = [], []
        for sd in SEEDS:
            no = evaluate(model, kw, False, EPISODES, sd)
            yes = evaluate(model, kw, True, EPISODES, sd)
            margin = no["win"] - yes["win"]
            margins.append(margin)
            wins_no.append(no["win"])
            wins_yes.append(yes["win"])
            print(f"  {sd:<8}{no['win']:>9.1f}%{yes['win']:>9.1f}%{margin:>9.1f}"
                  f"{yes['die']:>9.1f}%{no['steps']:>10.1f}{yes['steps']:>10.1f}")

        # 合并统计（300 回合）
        all_no = evaluate(model, kw, False, EPISODES * len(SEEDS), SEEDS[0])
        all_yes = evaluate(model, kw, True, EPISODES * len(SEEDS), SEEDS[0])
        # 注意：这里用第一个种子的长序列，作为"更大样本"的稳健估计
        avg_no = float(np.mean(wins_no))
        avg_yes = float(np.mean(wins_yes))
        margin_avg = avg_no - avg_yes

        print("  " + "-" * 76)
        print(f"  三种子均值: 无怪 {avg_no:.1f}% | 有怪 {avg_yes:.1f}% | "
              f"落差 {margin_avg:.1f}pp")
        print(f"  逐种子落差: {[round(m,1) for m in margins]}  "
              f"(最小 {min(margins):.1f}pp)")
        passed = min(margins) >= 20
        print(f"  判定（每个种子都要 ≥20pp）: "
              f"{'✅ 通过' if passed else '⚠️ 部分种子不足'}")

        report[cfg_name] = dict(
            config=kw, seeds=SEEDS, episodes=EPISODES,
            per_seed=[dict(seed=s, no=w1, yes=w2, margin=m)
                      for s, w1, w2, m in zip(SEEDS, wins_no, wins_yes, margins)],
            avg_no=avg_no, avg_yes=avg_yes, avg_margin=margin_avg,
            min_margin=min(margins), all_seeds_pass=bool(passed))

    os.makedirs("results", exist_ok=True)
    out = "results/threat_margin_final.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*90}")
    print(f"报告已保存: {out}")

    # 推荐
    ok = {k: v for k, v in report.items() if v["all_seeds_pass"]}
    if ok:
        best = min(ok.values(), key=lambda v: v["avg_yes"])  # 有怪成功率最低=威胁最大但达标
        name = [k for k, v in report.items() if v is best][0]
        print(f"推荐配置: 【{name}】 三种子均值落差 {best['avg_margin']:.1f}pp，"
              f"最小 {best['min_margin']:.1f}pp")
    else:
        cand = max(report.values(), key=lambda v: v["min_margin"])
        name = [k for k, v in report.items() if v is cand][0]
        print(f"无配置在所有种子达标；最接近的是【{name}】"
              f"（最小落差 {cand['min_margin']:.1f}pp）")


if __name__ == "__main__":
    main()
