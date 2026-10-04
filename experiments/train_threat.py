# -*- coding: utf-8 -*-
"""
experiments/train_threat.py —— 重训 PPO 并测量怪物威胁度

背景（诊断结论）
    旧模型有两个问题，导致威胁度无法可信测量：
      1. models/ppo_maze_no_monster.zip 已退化：无怪 100%，有怪 0%（完全不躲）
      2. models/ppo_maze_A.zip 退化方向相反：无怪 0%（超时卡死），有怪 93%
         —— 它在用怪物位置当导航信号
    因此必须在【新环境】上重训，才能得到可信的威胁度。

本脚本训练两个策略并测三种口径的落差：
    口径① 同一策略跨环境：no_monster 策略在无怪 / 有怪下的成功率
    口径② 同一策略跨环境：with_monster 策略在无怪 / 有怪下的成功率
    口径③ 训练目标口径：无怪天花板 与 有怪策略成功率 的差

用法：
    python experiments/train_threat.py --config baseline
    python experiments/train_threat.py --config strong
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from envs.maze import MazeEnv

# ----------------------------------------------------------------------
# 环境配置（威胁度逐级加强）
# ----------------------------------------------------------------------
CONFIGS = {
    "baseline": dict(),                     # 现状行为
    "medium": dict(chase_radius=4, chase_prob=0.10, chase_len=6),   # 追击加强
    "strong": dict(chase_radius=5, chase_prob=0.15, chase_len=8,
                   monster_speed=2),        # 又凶又快
}


def train_one(env_kwargs, timesteps, seed, tag, max_steps=200,
              with_monster=True, curriculum=False, ent_coef=0.01):
    """
    curriculum=True 时先在不含怪物的副本里预训练，再切到目标环境继续训练。
    这样能保证策略先学会「走到终点」，而不是靠怪物位置当导航信号。

    背景：Day 20 实测发现，直接在带怪环境训练会得到退化的策略 ——
    无怪时 0%（在 (10,2) 原地撞墙 188 次），有怪时 90%。
    根因是旧版 with_monster=False 只是把怪物位置置零、怪物还冻在 (5,4)，
    导致「无怪」成为一种分布外状态。已修复为怪物移出地图外。
    """
    if curriculum:
        pre = MazeEnv(with_monster=False, max_steps=max_steps, **env_kwargs)
        model = PPO("MlpPolicy", pre, learning_rate=3e-4, n_steps=2048,
                    batch_size=64, gamma=0.99, ent_coef=ent_coef,
                    seed=seed, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=timesteps)
        print(f"    [{tag}] 课程①：无怪预训练 {timesteps} 步，"
              f"用时 {time.time()-t0:.0f}s")
        env = MazeEnv(with_monster=with_monster, max_steps=max_steps, **env_kwargs)
        model.set_env(env)
        t0 = time.time()
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        print(f"    [{tag}] 课程②：目标环境续训 {timesteps} 步，"
              f"用时 {time.time()-t0:.0f}s")
    else:
        env = MazeEnv(with_monster=with_monster, max_steps=max_steps, **env_kwargs)
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                    batch_size=64, gamma=0.99, ent_coef=ent_coef,
                    seed=seed, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=timesteps)
        print(f"    [{tag}] 训练 {timesteps} 步完成，用时 {time.time()-t0:.0f}s")

    path = f"models/threat_{tag}"
    os.makedirs("models", exist_ok=True)
    model.save(path)
    print(f"    [{tag}] 已保存 -> {path}.zip")
    return model


def evaluate(model, env_kwargs, with_monster, episodes, base_seed=42, max_steps=200):
    env = MazeEnv(with_monster=with_monster, max_steps=max_steps, **env_kwargs)
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
    return dict(win=100 * wins / n, die=100 * deaths / n, timeout=100 * timeouts / n,
                steps=float(np.mean(steps_win)) if steps_win else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline",
                    choices=list(CONFIGS.keys()) + ["all"])
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--curriculum", action="store_true",
                    help="有怪策略先做无怪预训练，避免退化")
    args = ap.parse_args()

    names = list(CONFIGS) if args.config == "all" else [args.config]
    summary = {}

    for name in names:
        kw = CONFIGS[name]
        print("=" * 84)
        print(f"配置 [{name}]  {kw if kw else '（现状行为）'}")
        print("=" * 84)

        # ---- 训练两个策略：无怪 / 有怪 ----
        print("  训练中...")
        m_no = train_one({**kw}, args.timesteps, args.seed, f"{name}_no_monster",
                         args.max_steps, with_monster=False, ent_coef=args.ent_coef)
        m_yes = train_one({**kw}, args.timesteps, args.seed, f"{name}_with_monster",
                          args.max_steps, with_monster=True,
                          curriculum=args.curriculum, ent_coef=args.ent_coef)

        # ---- 评估 ----
        print("  评估中...")
        no_nomon = evaluate(m_no, kw, False, args.episodes, max_steps=args.max_steps)
        no_yesmon = evaluate(m_no, kw, True, args.episodes, max_steps=args.max_steps)
        yes_nomon = evaluate(m_yes, kw, False, args.episodes, max_steps=args.max_steps)
        yes_yesmon = evaluate(m_yes, kw, True, args.episodes, max_steps=args.max_steps)

        margin1 = no_nomon["win"] - no_yesmon["win"]
        margin2 = yes_nomon["win"] - yes_yesmon["win"]
        margin3 = no_nomon["win"] - yes_yesmon["win"]

        print()
        print(f"  {'策略':<18}{'无怪成功':>10}{'有怪成功':>10}{'落差pp':>9}"
              f"{'有怪击杀':>10}{'有怪超时':>10}")
        print("  " + "-" * 74)
        print(f"  {'no_monster':<18}{no_nomon['win']:>9.1f}%{no_yesmon['win']:>9.1f}%"
              f"{margin1:>9.1f}{no_yesmon['die']:>9.1f}%{no_yesmon['timeout']:>9.1f}%")
        print(f"  {'with_monster':<18}{yes_nomon['win']:>9.1f}%{yes_yesmon['win']:>9.1f}%"
              f"{margin2:>9.1f}{yes_yesmon['die']:>9.1f}%{yes_yesmon['timeout']:>9.1f}%")
        print("  " + "-" * 74)
        print(f"  口径① 同一 no_monster 策略跨环境落差 : {margin1:.1f} pp")
        print(f"  口径② 同一 with_monster 策略跨环境落差: {margin2:.1f} pp")
        print(f"  口径③ 无怪天花板 - 有怪策略成功率    : {margin3:.1f} pp")
        ok = margin1 >= 20 and margin3 >= 20
        print(f"  判定（要求 ≥20pp）: {'✅ 通过' if ok else '❌ 未通过'}")
        print()

        summary[name] = dict(config=kw, margin_same_policy_no=margin1,
                             margin_same_policy_yes=margin2, margin_target=margin3,
                             no_monster=no_nomon, with_monster=yes_yesmon,
                             passed=bool(ok))

    out = "results/threat_margin.json"
    os.makedirs("results", exist_ok=True)
    if os.path.exists(out):
        old = json.load(open(out, encoding="utf-8"))
        old.update(summary)
        summary = old
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
