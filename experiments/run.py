# -*- coding: utf-8 -*-
"""
experiments/run.py —— Day 17：三组对照统一入口

    MODE = A | B | C
      A = 纯 RL（不调用 LLM）
      B = 固定频率调用 LLM
      C = 惊讶度门控调用 LLM

三组共用同一环境、同一底层策略、同一 LLM、同一"偏好排序 + 环境掩码"接口，
唯一的差别是【何时唤醒 LLM】。这样结果差异才能归因到触发机制本身。

──────────────────────────────────────────────────────────────────────
调用率匹配（计划书 Day 17 的关键步骤）
──────────────────────────────────────────────────────────────────────
计划书给的公式是  N = 200 / n̄  （200 = max_steps）。

⚠️ 本环境的实测口径修正
    本迷宫最短路径 18 步，A 组策略平均 18 步就到达终点，回合长度远小于
    max_steps=200。若照搬 200/n̄，当 n̄=2.12 时得到 N=94 —— 而回合只有
    ~18 步，B 组将【一次都不调用】，退化成 A 组，对照失效。

    因此改用【实测平均回合长度】做基数：
        N = T̄ / n̄            T̄ = 实测平均回合步数
    语义仍是"每次调用的代价相同"，只是基数换成真实回合长度。
    脚本会自动先测 T̄（跑一遍 A 组），再算 N，并实测 B 组调用率做校验。

用法
    python experiments/run.py --mode C --episodes 50     # 先测 C 的 n̄
    python experiments/run.py --mode B --episodes 50 --target-calls 2.12
    python experiments/run.py --mode A --episodes 100
    python experiments/run.py --calibrate                # 只测 T̄ 与建议 N
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
from envs.a_config import A_ENV_KWARGS, A_MODEL_PATH, A_MAX_STEPS, A_BASE_SEED
from llm.advisor import ask_plan, DIR_NAMES, DIRECTION_MAP


# ----------------------------------------------------------------------
def _llm_pick(gate_or_none, env, agent_pos, monster_pos, goal_pos, log):
    """
    统一的 LLM 接口：取偏好排序 -> 按顺序取第一个合法方向（环境掩码）。
    返回 (动作编号, 是否被掩码顺延, 首选是否可行, 原始偏好)
    """
    actions, ranking, raw, lat = ask_plan(
        agent_pos=agent_pos, monster_pos=monster_pos, goal_pos=goal_pos,
        env=env, plan_len=3)
    if not ranking:
        log["parse_failures"] += 1
        return None, False, False, []
    chosen, first_legal = None, False
    for i, d in enumerate(ranking):
        dr, dc = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}[d]
        if env.game.is_valid((agent_pos[0] + dr, agent_pos[1] + dc)):
            chosen = d
            first_legal = (i == 0)
            break
    if chosen is None:
        log["no_legal_choice"] += 1
        return None, False, False, ranking
    if not first_legal:
        log["masked_fallbacks"] += 1
    if not first_legal:
        log["first_choice_illegal"] += 1
    log["latency"].append(lat)
    return DIRECTION_MAP[chosen], (not first_legal), first_legal, ranking


# ----------------------------------------------------------------------
def run_episode(mode, model, env, seed, interval=1, gate=None, log=None):
    obs, _ = env.reset(seed=seed)
    steps = 0
    total_reward = 0.0
    done = False
    info = {}
    calls = 0

    while not done:
        steps += 1
        agent_p = [int(x) for x in env.game.agent_pos]
        monster_p = [int(x) for x in env.game.monster_pos]

        # ---- 1. 底层策略动作 ----
        ppo_act, _ = model.predict(obs, deterministic=True)
        action = int(ppo_act)
        source = "PPO"

        # ---- 2. 决定是否唤醒 LLM ----
        if mode == "C" and gate is not None:
            if len(gate.monster_history) >= 2:
                prev_m = [int(round(p * 11.0)) for p in gate.monster_history[-2]]
                surprisal = gate.compute_surprisal(prev_m, monster_p)
            else:
                surprisal = 0.0
            a, masked, first_ok, ranking = gate.decide(
                llm_pick_fn=_llm_pick,
                ppo_action=action,
                agent_pos=agent_p,
                monster_pos=monster_p,
                goal_pos=(10, 10),
                env=env, surprisal=surprisal, log=log)
            if a is not None:
                action = a
                calls += 1
                source = "LLM_C"
        elif mode == "B":
            # 固定频率：从第 1 步起，每 interval 步调用一次
            if interval > 0 and ((steps - 1) % interval == 0):
                a, masked, first_ok, ranking = _llm_pick(
                    None, env, agent_p, monster_p, (10, 10), log)
                if a is not None:
                    action = a
                    calls += 1
                    source = "LLM_B"

        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        done = term or trunc
        if gate is not None:
            gate.update_history(env.game.monster_pos)

    return dict(status=info.get("status"), steps=steps,
                reward=total_reward, calls=calls)


# ----------------------------------------------------------------------
def evaluate(mode, episodes, base_seed, interval=1, tau=1.5, cooldown=5,
             verbose=True):
    from dual_system.gate_controller import SurprisalGateController

    env = MazeEnv(with_monster=True, max_steps=A_MAX_STEPS, **A_ENV_KWARGS)
    model = PPO.load(A_MODEL_PATH, env=env)

    gate = None
    if mode == "C":
        gate = SurprisalGateController(
            predictor_path="prediction/monster_predictor.pth", env=env,
            tau=tau, max_ranking=3, cooldown_steps=cooldown, history_len=4)
        gate.reset()

    log = dict(parse_failures=0, no_legal_choice=0, masked_fallbacks=0,
               first_choice_illegal=0, latency=[])
    wins = deaths = timeouts = 0
    steps_all, steps_win, rewards, calls_all = [], [], [], []
    t0 = time.time()

    for ep in range(episodes):
        if gate is not None:
            gate.ranking = []
            gate.cooldown_counter = 0
            gate.monster_history.clear()
        r = run_episode(mode, model, env, base_seed + ep * 100,
                        interval=interval, gate=gate, log=log)
        steps_all.append(r["steps"])
        rewards.append(r["reward"])
        calls_all.append(r["calls"])
        if r["status"] == "win":
            wins += 1
            steps_win.append(r["steps"])
        elif r["status"] == "die":
            deaths += 1
        else:
            timeouts += 1

    n = episodes
    out = dict(
        group=mode, episodes=n,
        success_rate=round(100 * wins / n, 1),
        death_rate=round(100 * deaths / n, 1),
        timeout_rate=round(100 * timeouts / n, 1),
        avg_steps=round(float(np.mean(steps_all)), 2),
        avg_win_steps=round(float(np.mean(steps_win)), 2) if steps_win else 0.0,
        avg_reward=round(float(np.mean(rewards)), 3),
        total_llm_calls=int(sum(calls_all)),
        avg_llm_calls_per_episode=round(float(np.mean(calls_all)), 3),
        interval=interval, tau=tau if mode == "C" else None,
        elapsed_s=round(time.time() - t0, 1),
    )
    if mode in ("B", "C"):
        c = max(out["total_llm_calls"], 1)
        out["first_choice_illegal_calls"] = log["first_choice_illegal"]
        out["first_choice_illegal_pct"] = round(
            100 * log["first_choice_illegal"] / c, 1)
        out["masked_fallbacks"] = log["masked_fallbacks"]
        out["parse_failures"] = log["parse_failures"]
        out["no_legal_choice"] = log["no_legal_choice"]
        out["avg_llm_latency_s"] = round(float(np.mean(log["latency"])), 2) \
            if log["latency"] else 0.0

    if verbose:
        print(f"\n【{mode} 组】{n} 回合 | {out['elapsed_s']:.0f}s")
        print(f"  成功率 {out['success_rate']}% | 击杀 {out['death_rate']}% | "
              f"超时 {out['timeout_rate']}%")
        print(f"  平均步数 {out['avg_steps']} | 成功步数 {out['avg_win_steps']} | "
              f"平均回报 {out['avg_reward']}")
        print(f"  LLM 调用 {out['total_llm_calls']} 次 "
              f"({out['avg_llm_calls_per_episode']} 次/回合)")
        if mode in ("B", "C"):
            print(f"  首选不可行 {out['first_choice_illegal_pct']}% | "
                  f"掩码顺延 {out['masked_fallbacks']} | "
                  f"解析失败 {out['parse_failures']} | "
                  f"平均延迟 {out['avg_llm_latency_s']}s")
    return out


# ----------------------------------------------------------------------
def calibrate(episodes):
    """测 T̄（平均回合步数）与 C 组 n̄，给出建议的 B 组间隔"""
    print("=" * 74)
    print("调用率匹配标定")
    print("=" * 74)
    print("\n[1/2] 测 A 组（纯 PPO）的平均回合长度 T̄ ...")
    a = evaluate("A", episodes, A_BASE_SEED, verbose=False)
    T = a["avg_steps"]
    print(f"      T̄ = {T:.2f} 步（max_steps={A_MAX_STEPS}）")
    plan_N = A_MAX_STEPS / max(T, 1e-9)
    print(f"      计划书口径 N = max_steps/T̄ = {plan_N:.1f} "
          f"（仅作参考：它假设回合会跑满 max_steps）")

    print("\n[2/2] 测 C 组（惊讶度门控）的平均调用次数 n̄ ...")
    c = evaluate("C", episodes, A_BASE_SEED, verbose=False)
    nbar = c["avg_llm_calls_per_episode"]
    print(f"      n̄ = {nbar} 次/回合（{c['total_llm_calls']} 次 / "
          f"{episodes} 回合）")

    if nbar <= 0:
        print("\n⚠️ C 组从不触发，无法做调用率匹配；请先检查 τ 与预测器。")
        return None, None, None

    N = T / nbar
    print(f"\n      实际回合长度 {T:.1f} 步 / n̄ {nbar} = 建议间隔 N ≈ {N:.2f}")
    print(f"      取整后 N = {max(1, int(round(N)))} 步调用一次")
    print(f"\n      ⚠️ 计划书的 N = 200/n̄ = {A_MAX_STEPS/nbar:.0f} 步在此不可用：")
    print(f"         回合平均只有 {T:.1f} 步，间隔 {A_MAX_STEPS/nbar:.0f} 步 "
          f"会导致 B 组一次都不调用，退化成 A 组。")
    return T, nbar, max(1, int(round(N)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="A", choices=["A", "B", "C"])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--base-seed", type=int, default=A_BASE_SEED)
    ap.add_argument("--interval", type=int, default=0,
                    help="B 组间隔；0=自动按 C 组 n̄ 标定")
    ap.add_argument("--target-calls", type=float, default=0.0,
                    help="已知 C 组 n̄ 时直接指定，跳过标定")
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--cooldown", type=int, default=5)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.episodes)
        return

    interval = args.interval
    if args.mode == "B" and interval <= 0:
        if args.target_calls > 0:
            a = evaluate("A", args.episodes, args.base_seed, verbose=False)
            T = a["avg_steps"]
            interval = max(1, int(round(T / args.target_calls)))
            print(f"按已知 n̄={args.target_calls} 与 T̄={T:.2f} 标定："
                  f"间隔 N = {interval} 步")
        else:
            T, nbar, interval = calibrate(args.episodes)
            if interval is None:
                return

    out = evaluate(args.mode, args.episodes, args.base_seed,
                   interval=interval, tau=args.tau, cooldown=args.cooldown)

    os.makedirs("results", exist_ok=True)
    path = args.out or f"results/eval_{args.mode}_run.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {path}")


if __name__ == "__main__":
    main()
