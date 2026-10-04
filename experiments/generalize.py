# -*- coding: utf-8 -*-
"""
experiments/generalize.py —— 泛化实验：多张地图上重做 A/B/C 三组对照

要回答的问题
    当前结论（A 71% > C 66% > B 54%；固定频率显著有害、门控中性）
    全部来自同一张固定 12x12 迷宫。这是不是地图过拟合？
    换地图之后，三个机制的相对排序还成立吗？

实验设计
    对每张地图（原图 + N 张生成图）执行完整流水线：
      0. 生成/载入地图（保证连通、无绝对咽喉、威胁度适中）
      1. 采集该图的怪物轨迹并训练该图专属的 GRU 预测器
         （巡逻路径每张图不同，用原图预测器会让惊讶度恒高，门控失效）
      2. 训练该图专属的 PPO 策略：无怪版 + 有怪版（课程学习）
      3. 测该图的威胁度（同一策略跨环境）
      4. 跑 A/B/C 三组（B 组间隔按该图实测 T̄ 与 C 组 n̄ 标定）
    最后汇总成跨地图的对比表，检验结论是否稳健。

用法
    python experiments/generalize.py --maps 2 --timesteps 400000 --episodes 30
    python experiments/generalize.py --maps 3 --skip-train     # 只重跑评估
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from envs.maze import MazeEnv, MapMazeEnv
from envs.a_config import A_ENV_KWARGS, A_MODEL_PATH, A_MAX_STEPS, A_BASE_SEED
from envs.map_gen import generate_map, render
from prediction.collect_data import collect, MODE_PATROL, MODE_CHASE
from prediction.train_predictor import MonsterPredictor
import torch


OUT_DIR = "results/generalize"
MAPS_DIR = "results/generalize/maps"

# 泛化实验使用的怪物行为（可被 --chase 覆盖）。
# 注意：这里可以独立于 envs/a_config.py 的 A_ENV_KWARGS 设置 ——
# 每张生成地图会重新标定自己的威胁度，目的是让每张图的落差都达标。
CHASE = dict(chase_radius=4, chase_prob=0.15, chase_len=8)


def set_chase(kw):
    global CHASE
    CHASE = dict(kw)


# ----------------------------------------------------------------------
def env_factory_for(map_spec, max_steps=None):
    """返回一个 (with_monster) -> env 的工厂，供 run.evaluate 使用。

    max_steps 随地图尺寸放宽：大地图最短路径更长，沿用 200 会导致
    大量本可完成的回合被判超时，污染成功率。
    """
    if max_steps is None:
        base = map_spec.get("base_dist") or 18
        max_steps = int(max(A_MAX_STEPS, base * 4))
    def f(with_monster):
        return MapMazeEnv(map_spec, with_monster=with_monster,
                          max_steps=max_steps, **CHASE)
    return f


def train_ppo(env, timesteps, seed, save_path, curriculum=False):
    """课程学习：先无怪预训练，再切到目标环境续训"""
    if curriculum:
        pre = MazeEnv(with_monster=False, max_steps=A_MAX_STEPS, **CHASE)
        model = PPO("MlpPolicy", pre, learning_rate=3e-4, n_steps=2048,
                    batch_size=64, gamma=0.99, ent_coef=0.01, seed=seed,
                    verbose=0)
        model.learn(total_timesteps=timesteps)
        model.set_env(env)
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                    batch_size=64, gamma=0.99, ent_coef=0.01, seed=seed,
                    verbose=0)
        model.learn(total_timesteps=timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return model


def eval_policy(model, env_factory, with_monster, episodes, base_seed):
    env = env_factory(with_monster)
    wins = deaths = timeouts = 0
    steps_win = []
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
                    deaths += 1
                break
            if tr:
                timeouts += 1
                break
    n = episodes
    return dict(win=100 * wins / n, die=100 * deaths / n,
                timeout=100 * timeouts / n,
                steps=float(np.mean(steps_win)) if steps_win else 0.0)


def calibrate_b(run_eval, ef, mpath, target_calls, T0, args,
                max_rounds=4, tol=0.15, seed=A_BASE_SEED):
    """
    迭代标定 B 组间隔，使其每回合调用次数逼近 C 组的 n̄。

    为什么不直接用 N = T̄/n̄：
        B 组自己的回合长度会随 N 变化（LLM 介入时机影响存活/结束步数），
        所以这是一条反馈回路。用 A 组的 T̄ 一次算出会系统性偏大 ——
        实测 16x16 上算出 N=9，结果 B 调用 2.64 次 vs C 1.88 次，
        差距 40%，超出计划书要求的 20%。

    做法：以 N = T̄/n̄ 为初值，跑一轮 B，按实测调用数近似反比外推修正 N，
    直到 |B-C|/C <= tol 或达到轮数上限。
    """
    N = max(1, int(round(T0 / max(target_calls, 1e-9))))
    best = None
    for it in range(max_rounds):
        b = run_eval("B", args.episodes, seed, interval=N, verbose=False,
                     env_factory=ef, model_path=mpath)
        got = b["avg_llm_calls_per_episode"]
        diff = abs(got - target_calls) / max(target_calls, 1e-9) * 100
        print(f"        标定轮 {it+1}: N={N:>3} -> B 调用 {got:.2f} 次/回合 "
              f"(目标 {target_calls:.2f}, 差 {diff:.1f}%)")
        if best is None or diff < best[2]:
            best = (N, b, diff)
        if diff <= tol * 100:
            break
        N = max(1, int(round(N * got / max(target_calls, 1e-9))))
        if N == best[0]:
            break
    return best


# ----------------------------------------------------------------------
def run_one_map(map_spec, name, args, log):
    """在单张地图上跑完整流水线"""
    print("\n" + "=" * 78)
    print(f"地图【{name}】 {map_spec.get('desc', '')}")
    print("=" * 78)
    if map_spec.get("grid") is not None:
        print(f"  最短路径 {map_spec.get('base_dist')} 步 | "
              f"威胁度 {map_spec.get('threat')}")
        print(render(map_spec))

    ef = env_factory_for(map_spec)
    res = {"name": name, "map": {k: v for k, v in map_spec.items()
                                 if k != "grid"}}

    # ---- 1. 该图专属预测器 ----
    pred_path = f"{OUT_DIR}/predictor_{name}.pth"
    data_path = f"{OUT_DIR}/data_{name}.npz"
    if not args.skip_train or not os.path.exists(pred_path):
        print("\n  [1/4] 采集该图怪物轨迹 ...")
        X, y, m = collect(episodes=args.data_episodes, policy="random",
                          env_kwargs=CHASE, save_path=data_path,
                          verbose=True, env_factory=ef,
                          seed_base=10000 + (map_spec.get("seed") or 0) * 37,
                          max_steps=ef(True).game.max_steps)
        print("  [2/4] 训练该图预测器 ...")
        hp = MonsterPredictor(2, 128, 5)
        d = np.load(data_path, allow_pickle=True)
        Xa, ya = d["X"], d["y"]
        idx = np.arange(len(Xa))
        np.random.seed(42)
        np.random.shuffle(idx)
        spl = int(len(Xa) * 0.85)
        tr, va = idx[:spl], idx[spl:]
        Xt = torch.tensor(Xa[tr]); yt = torch.tensor(ya[tr])
        Xv = torch.tensor(Xa[va]); yv = torch.tensor(ya[va])
        opt = torch.optim.Adam(hp.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        for ep in range(args.predictor_epochs):
            hp.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 128):
                b = perm[i:i+128]
                opt.zero_grad()
                loss = crit(hp(Xt[b]), yt[b])
                loss.backward()
                opt.step()
        hp.eval()
        with torch.no_grad():
            vacc = (torch.argmax(hp(Xv), 1) == yv).float().mean().item()
        torch.save(hp.state_dict(), pred_path)
        print(f"        预测器验证准确率 {100*vacc:.1f}% -> {pred_path}")
        res["predictor_val_acc"] = round(100 * vacc, 1)

    # ---- 2. 该图专属 PPO ----
    mpath_yes = f"{OUT_DIR}/ppo_{name}_with_monster"
    mpath_no = f"{OUT_DIR}/ppo_{name}_no_monster"
    if not args.skip_train:
        print("\n  [3/4] 训练该图 PPO（无怪 / 有怪+课程学习）...")
        t0 = time.time()
        train_ppo(ef(True), args.timesteps, 42, mpath_yes, curriculum=True)
        print(f"        有怪策略完成 {time.time()-t0:.0f}s")
        t0 = time.time()
        train_ppo(ef(False), args.timesteps, 42, mpath_no, curriculum=False)
        print(f"        无怪策略完成 {time.time()-t0:.0f}s")

    model_yes = PPO.load(mpath_yes, env=ef(True))

    # ---- 3. 威胁度 ----
    no = eval_policy(model_yes, ef, False, args.episodes, A_BASE_SEED)
    yes = eval_policy(model_yes, ef, True, args.episodes, A_BASE_SEED)
    margin = no["win"] - yes["win"]
    res["threat"] = dict(no_win=no["win"], yes_win=yes["win"],
                         margin_pp=margin, passed=bool(margin >= 20))
    print(f"\n  威胁度: 无怪 {no['win']:.1f}% | 有怪 {yes['win']:.1f}% | "
          f"落差 {margin:.1f}pp -> {'✅' if margin >= 20 else '❌'}")

    # ---- 4. A/B/C ----
    print("\n  [4/4] 跑 A/B/C 三组 ...")
    from experiments.run import evaluate as run_eval

    a = run_eval("A", args.episodes, A_BASE_SEED, verbose=False,
                 env_factory=ef, model_path=mpath_yes)
    c = run_eval("C", args.episodes, A_BASE_SEED, verbose=False,
                 env_factory=ef, model_path=mpath_yes,
                 predictor_path=pred_path, tau=args.tau)
    T = a["avg_steps"]
    nbar = c["avg_llm_calls_per_episode"]
    if nbar <= 0:
        print("        ⚠️ C 组从未触发，无法匹配调用率；B 组用间隔 1")
        N, b, diff = 1, run_eval("B", args.episodes, A_BASE_SEED, interval=1,
                                 verbose=False, env_factory=ef,
                                 model_path=mpath_yes), 100.0
    else:
        print(f"        初值: T̄={T:.2f} 步, C 组 n̄={nbar} 次/回合")
        N, b, diff = calibrate_b(run_eval, ef, mpath_yes, nbar, T, args)

    res["groups"] = {"A": a, "B": b, "C": c}
    res["interval"] = N
    used = {"B_vs_C_calls_pct": round(diff, 1),
            "matched": bool(diff < 20.0)}
    res["call_match"] = used

    print(f"\n  ── 地图【{name}】结果 ──")
    print(f"  {'组':<4}{'成功率':>9}{'击杀率':>9}{'调用/回合':>11}")
    for k in ("A", "B", "C"):
        g = res["groups"][k]
        print(f"  {k:<4}{g['success_rate']:>8.1f}%{g['death_rate']:>8.1f}%"
              f"{g['avg_llm_calls_per_episode']:>11.2f}")
    if used:
        print(f"  调用率匹配 B vs C 差距: {used['B_vs_C_calls_pct']}%")
    return res


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=2, help="额外生成的随机地图数")
    ap.add_argument("--timesteps", type=int, default=400_000)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--data-episodes", type=int, default=250)
    ap.add_argument("--predictor-epochs", type=int, default=40)
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--wall-prob", type=float, default=0.30)
    ap.add_argument("--size", type=int, default=16,
                    help="生成地图边长（原图固定 12；放大可检验过拟合）")
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--chase-radius", type=int, default=4)
    ap.add_argument("--chase-prob", type=float, default=0.15)
    ap.add_argument("--chase-len", type=int, default=8)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    set_chase(dict(chase_radius=args.chase_radius,
                   chase_prob=args.chase_prob,
                   chase_len=args.chase_len))
    print(f"怪物行为: {CHASE}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)

    t_start = time.time()
    results = []

    # --- 地图 0：原图（作为基线对照） ---
    import envs.maze as M
    orig = dict(seed=0, grid=[list(r) for r in M.MAZE], start=list(M.START),
                goal=list(M.GOAL), base_dist=18, threat=None,
                desc="原始固定地图（对照组）")
    orig["patrol"] = [[5, 4], [5, 8], [7, 8], [7, 4]]
    # 原图用已有的 A 组基线结果，不重跑
    orig_spec = dict(orig)
    orig_spec["desc"] = "原始固定地图"
    with open(f"{MAPS_DIR}/map_orig.json", "w", encoding="utf-8") as f:
        json.dump(orig_spec, f, ensure_ascii=False)

    # --- 生成 N 张随机地图 ---
    gen_maps = []
    for i in range(args.maps):
        m = generate_map(args.seed_start + i, size=args.size,
                         wall_prob=args.wall_prob, max_tries=1500)
        if m is None:
            print(f"⚠️ seed={args.seed_start+i} 生成失败，跳过")
            continue
        m["desc"] = f"{args.size}x{args.size} 生成地图 seed={args.seed_start+i}"
        gen_maps.append(m)
        with open(f"{MAPS_DIR}/map_{args.size}_gen{args.seed_start+i}.json", "w",
                  encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        print(f"  生成 {args.size}x{args.size} 地图 seed={m['seed']}: "
              f"最短 {m['base_dist']} 步, 威胁度 {m['threat']}, "
              f"巡逻 {m['patrol']}")

    for m in gen_maps:
        r = run_one_map(m, f"gen{m['seed']}_{args.size}", args, None)
        results.append(r)
        with open(f"{OUT_DIR}/summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # --- 汇总 ---
    print("\n" + "=" * 78)
    print(f"泛化实验汇总（{len(results)} 张地图，每图 {args.episodes} 回合）")
    print("=" * 78)
    print(f"{'地图':<10}{'威胁度pp':>10}{'A':>9}{'B':>9}{'C':>9}"
          f"{'B调用':>9}{'C调用':>9}{'匹配%':>8}")
    print("-" * 78)
    for r in results:
        g = r["groups"]
        mc = r["call_match"].get("B_vs_C_calls_pct", float("nan"))
        print(f"{r['name']:<10}{r['threat']['margin_pp']:>10.1f}"
              f"{g['A']['success_rate']:>8.1f}%{g['B']['success_rate']:>8.1f}%"
              f"{g['C']['success_rate']:>8.1f}%"
              f"{g['B']['avg_llm_calls_per_episode']:>9.2f}"
              f"{g['C']['avg_llm_calls_per_episode']:>9.2f}{mc:>8.1f}")

    print(f"\n总耗时 {(time.time()-t_start)/60:.1f} 分钟")
    print(f"结果已保存: {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
