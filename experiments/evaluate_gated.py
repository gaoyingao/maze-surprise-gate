"""
experiments/evaluate_gated.py —— 通用多回合评估（A 组纯 PPO / C 组惊讶度门控）

存在的理由：单回合结果没有任何统计意义。seed 8011 在 temperature=0.1 时
曾"逃脱"，改成 temperature=0 后同一 seed 变成"被抓"——前者是采样运气，
后者才是这个 seed 下确定的结果。任何结论都必须来自多回合统计。

用法：
    python experiments/evaluate_gated.py                 # A 组基线
    python experiments/evaluate_gated.py --gated         # C 组（惊讶度门控 + LLM 偏好掩码）
    python experiments/evaluate_gated.py --gated --episodes 50 --tau 1.5

注意：max_steps 必须与 evaluate.py 一致（200），否则成功率不可比。
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from envs.maze import MazeEnv
from envs.a_config import A_ENV_KWARGS, A_MODEL_PATH

DEFAULT_MODEL = A_MODEL_PATH            # models/threat_medium_with_monster.zip
DEFAULT_PREDICTOR = "prediction/monster_predictor.pth"
MAX_STEPS = 200          # 必须与 evaluate.py 一致
BASE_SEED = 42           # 必须与 evaluate.py 一致


def run_one(model, env, gate, seed, max_steps=MAX_STEPS):
    """跑一个回合，返回统计字典"""
    from llm.advisor import DIR_NAMES  # noqa: F401  (确保模块可导入)

    obs, _ = env.reset(seed=seed)
    if gate is not None:
        # 注意：不能在这里调 gate.reset()，否则每个回合都会清零累计指标。
        # 累计指标在 main() 里只清零一次。
        gate.ranking = []
        gate.cooldown_counter = 0
        gate.monster_history.clear()
        gate.update_history(env.game.monster_pos)

    steps = 0
    total_reward = 0.0
    done = False
    info = {}
    calls = 0
    illegal = 0
    masked = 0

    while not done:
        steps += 1
        ppo_act, _ = model.predict(obs, deterministic=True)

        if gate is not None:
            agent_p = [int(x) for x in env.game.agent_pos]
            monster_p = [int(x) for x in env.game.monster_pos]
            if len(gate.monster_history) >= 2:
                prev_m = [int(round(p * 11.0)) for p in gate.monster_history[-2]]
                surprisal = gate.compute_surprisal(prev_m, monster_p)
            else:
                surprisal = 0.0
            before_calls = gate.total_llm_calls
            before_ill = gate.llm_first_choice_illegal
            before_mask = gate.llm_masked_fallbacks
            action, _src = gate.decide_action(
                ppo_action=int(ppo_act), agent_pos=agent_p,
                monster_pos=monster_p, goal_pos=(10, 10), surprisal=surprisal)
            calls += gate.total_llm_calls - before_calls
            illegal += gate.llm_first_choice_illegal - before_ill
            masked += gate.llm_masked_fallbacks - before_mask
        else:
            action = int(ppo_act)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if gate is not None:
            gate.update_history(env.game.monster_pos)

    return {
        "status": info.get("status"),
        "steps": steps,
        "reward": total_reward,
        "llm_calls": calls,
        "first_choice_illegal": illegal,
        "masked_fallbacks": masked,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", action="store_true", help="启用惊讶度门控 + LLM 偏好掩码")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--cooldown", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--predictor", default=DEFAULT_PREDICTOR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    group = "C（惊讶度门控 + LLM 偏好掩码）" if args.gated else "A（纯 PPO 基线）"
    print("=" * 74)
    print(f"多回合评估 —— {group}")
    print(f"回合数: {args.episodes} | max_steps: {MAX_STEPS} | base_seed: {BASE_SEED}")
    if args.gated:
        print(f"门控参数: tau={args.tau} cooldown={args.cooldown}")
    print("=" * 74)

    if not os.path.exists(args.model):
        print(f"❌ 找不到模型: {args.model}")
        return

    env = MazeEnv(with_monster=True, max_steps=MAX_STEPS, **A_ENV_KWARGS)
    model = PPO.load(args.model, env=env)

    gate = None
    if args.gated:
        from dual_system.gate_controller import SurprisalGateController
        gate = SurprisalGateController(
            predictor_path=args.predictor, env=env,
            tau=args.tau, max_ranking=3,
            cooldown_steps=args.cooldown, history_len=4)
        gate.reset()   # 累计指标只在整轮评估开始时清零一次

    wins = deaths = timeouts = 0
    win_steps = []
    rewards = []
    calls = []
    ill_total = 0
    mask_total = 0
    t0 = time.time()

    for ep in range(args.episodes):
        seed = BASE_SEED + ep * 100
        r = run_one(model, env, gate, seed)
        rewards.append(r["reward"])
        calls.append(r["llm_calls"])
        ill_total += r["first_choice_illegal"]
        mask_total += r["masked_fallbacks"]
        if r["status"] == "win":
            wins += 1
            win_steps.append(r["steps"])
        elif r["status"] == "die":
            deaths += 1
        else:
            timeouts += 1
        if (ep + 1) % 10 == 0:
            print(f"  ... 已完成 {ep+1}/{args.episodes} 回合 "
                  f"(胜 {wins} 负 {deaths} 超时 {timeouts})")

    n = args.episodes
    dur = time.time() - t0
    total_calls = int(sum(calls))
    out = {
        "group": "C" if args.gated else "A",
        "episodes": n,
        "success_rate": round(100 * wins / n, 1),
        "death_rate": round(100 * deaths / n, 1),
        "timeout_rate": round(100 * timeouts / n, 1),
        "avg_win_steps": round(float(np.mean(win_steps)), 2) if win_steps else 0.0,
        "avg_reward": round(float(np.mean(rewards)), 3),
        "total_llm_calls": total_calls,
        "avg_llm_calls_per_episode": round(float(np.mean(calls)), 3),
    }

    if gate is not None:
        c = max(total_calls, 1)
        out["first_choice_illegal_calls"] = ill_total
        out["first_choice_illegal_pct"] = round(100 * ill_total / c, 1)
        out["masked_fallbacks"] = mask_total
        out["parse_failures"] = gate.llm_parse_failures
        out["no_legal_choice"] = gate.llm_no_legal_choice

    print("-" * 74)
    print(f"🏆 成功率      : {out['success_rate']}%  ({wins}/{n})")
    print(f"💀 被怪击杀率  : {out['death_rate']}%  ({deaths}/{n})")
    print(f"⏰ 超时率      : {out['timeout_rate']}%  ({timeouts}/{n})")
    print(f"👣 平均成功步数: {out['avg_win_steps']}")
    print(f"💰 平均回报    : {out['avg_reward']}")
    if args.gated:
        print(f"🔔 LLM 总调用  : {out['total_llm_calls']} 次 "
              f"(平均 {out['avg_llm_calls_per_episode']} 次/回合)")
        print(f"🚫 首选不可行率: {out['first_choice_illegal_pct']}%  "
              f"({out['first_choice_illegal_calls']}/{total_calls})")
        print(f"↪️  掩码顺延    : {out['masked_fallbacks']} 次")
        print(f"❓ 解析失败    : {out['parse_failures']} 次")
    print(f"⏱️  耗时        : {dur:.1f}s")
    print("=" * 74)

    if args.out:
        import json
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {args.out}")

    return out


if __name__ == "__main__":
    main()
