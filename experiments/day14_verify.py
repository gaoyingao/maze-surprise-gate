# -*- coding: utf-8 -*-
"""
experiments/day14_verify.py —— Day 14 三项 Go/No-Go 检查（大样本一次性验证）

计划书三项：
  ① 怪物真构成威胁   : 有怪 vs 无怪成功率差 > 20pp
  ② 惊讶度真会惊讶   : 巡逻 <0.5 nats，切换瞬间 >2.0 nats
  ③ LLM 可用         : 方向词解析成功率 > 90%

本脚本把①②做大样本统计（③需要 Ollama，单独跑 test_llm_availability）。
输出机器可读的 JSON 到 results/。

用法：
    python experiments/day14_verify.py --episodes 100
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from envs.maze import MazeEnv
from envs.a_config import A_ENV_KWARGS, A_MODEL_PATH, A_MAX_STEPS, A_BASE_SEED
from prediction.train_predictor import MonsterPredictor
from prediction.collect_data import OFFSET_TO_ACTION, MODE_PATROL, MODE_CHASE

SEEDS = [42, 777, 2024]


# ----------------------------------------------------------------------
def check_threat(episodes, seeds):
    """① 同一策略跨环境的成功率落差"""
    model = PPO.load(A_MODEL_PATH,
                     env=MazeEnv(with_monster=True, max_steps=A_MAX_STEPS,
                                 **A_ENV_KWARGS))
    per_seed, margins = [], []
    for sd in seeds:
        res = {}
        for wm in (False, True):
            env = MazeEnv(with_monster=wm, max_steps=A_MAX_STEPS, **A_ENV_KWARGS)
            wins = deaths = 0
            for ep in range(episodes):
                obs, _ = env.reset(seed=sd + ep * 100)
                done = False
                while not done:
                    a, _ = model.predict(obs, deterministic=True)
                    obs, _, t, tr, info = env.step(int(a))
                    done = t or tr
                    if t:
                        wins += info.get("status") == "win"
                        deaths += info.get("status") == "die"
                        break
            res["no" if not wm else "yes"] = 100 * wins / episodes
            res["die" if wm else "die_no"] = 100 * deaths / episodes
        m = res["no"] - res["yes"]
        margins.append(m)
        per_seed.append(dict(seed=sd, no=res["no"], yes=res["yes"], margin=m))
    return dict(per_seed=per_seed, avg_margin=float(np.mean(margins)),
                min_margin=float(min(margins)),
                passed=bool(min(margins) >= 20.0))


# ----------------------------------------------------------------------
def check_surprisal(episodes, history_len=4, seed_base=8000):
    """② 巡逻 vs 追击的惊讶度分布（大样本聚合）"""
    dev = torch.device("cpu")
    pred = MonsterPredictor(2, 128, 5).to(dev)
    pred.load_state_dict(torch.load("prediction/monster_predictor.pth",
                                    map_location=dev))
    pred.eval()

    env = MazeEnv(with_monster=True, max_steps=A_MAX_STEPS, **A_ENV_KWARGS)
    ppo = PPO.load(A_MODEL_PATH, env=env)

    pat, cha = [], []
    acc = {MODE_PATROL: [0, 0], MODE_CHASE: [0, 0]}
    n_switch = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        hist = []
        prev_chase = False
        while not done:
            mr, mc = env.game.monster_pos
            hist.append([mr / 11.0, mc / 11.0])
            prev_m = [int(mr), int(mc)]
            probs = None
            if len(hist) >= history_len:
                seq = torch.tensor([hist[-history_len:]], dtype=torch.float32)
                with torch.no_grad():
                    probs = F.softmax(pred(seq), dim=1).squeeze(0).numpy()
            a, _ = ppo.predict(obs, deterministic=True)
            obs, _, t, tr, _ = env.step(int(a))
            done = t or tr
            mode = MODE_CHASE if env.game.monster_chasing else MODE_PATROL

            if mode == MODE_CHASE and not prev_chase:
                n_switch += 1
            prev_chase = (mode == MODE_CHASE)

            cur = [int(x) for x in env.game.monster_pos]
            ta = OFFSET_TO_ACTION.get((cur[0]-prev_m[0], cur[1]-prev_m[1]), 4)
            if probs is not None:
                s = float(-np.log(max(float(probs[ta]), 1e-4)))
                (cha if mode == MODE_CHASE else pat).append(s)
                acc[mode][0] += int(int(np.argmax(probs)) == ta)
                acc[mode][1] += 1

    pat, cha = np.array(pat), np.array(cha)
    res = dict(
        patrol_n=len(pat), chase_n=len(cha),
        patrol_mean=float(pat.mean()) if len(pat) else 0.0,
        patrol_p95=float(np.percentile(pat, 95)) if len(pat) else 0.0,
        patrol_max=float(pat.max()) if len(pat) else 0.0,
        chase_mean=float(cha.mean()) if len(cha) else 0.0,
        chase_p95=float(np.percentile(cha, 95)) if len(cha) else 0.0,
        chase_max=float(cha.max()) if len(cha) else 0.0,
        patrol_acc=100 * acc[MODE_PATROL][0] / max(acc[MODE_PATROL][1], 1),
        chase_acc=100 * acc[MODE_CHASE][0] / max(acc[MODE_CHASE][1], 1),
        n_switches=n_switch,
    )
    res["acc_gap"] = res["patrol_acc"] - res["chase_acc"]
    res["patrol_ok"] = bool(res["patrol_mean"] < 0.5)
    res["peak_ok"] = bool(res["chase_max"] > 2.0)
    res["acc_ok"] = bool(res["patrol_acc"] > 90)
    res["passed"] = bool(res["patrol_ok"] and res["peak_ok"] and res["acc_ok"])
    return res


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()

    print("=" * 78)
    print("Day 14 Go/No-Go 验收（大样本）")
    print("=" * 78)

    print("\n【检查项 ①】怪物威胁度（同一策略跨环境）")
    t = check_threat(args.episodes, SEEDS)
    for r in t["per_seed"]:
        print(f"   seed {r['seed']:<5} 无怪 {r['no']:5.1f}% | 有怪 {r['yes']:5.1f}% "
              f"| 落差 {r['margin']:5.1f}pp")
    print(f"   均值 {t['avg_margin']:.1f}pp | 最小 {t['min_margin']:.1f}pp "
          f"| 门槛 ≥20pp -> {'✅ 通过' if t['passed'] else '❌ 未通过'}")

    print("\n【检查项 ②】惊讶度动力学（模式由环境权威切分）")
    s = check_surprisal(args.episodes)
    print(f"   巡逻段: n={s['patrol_n']:<5} 均值 {s['patrol_mean']:.2f} nats  "
          f"p95 {s['patrol_p95']:.2f}  最大 {s['patrol_max']:.2f}  "
          f"-> <0.5 {'✅' if s['patrol_ok'] else '❌'}")
    print(f"   追击段: n={s['chase_n']:<5} 均值 {s['chase_mean']:.2f} nats  "
          f"p95 {s['chase_p95']:.2f}  最大 {s['chase_max']:.2f}  "
          f"-> >2.0 {'✅' if s['peak_ok'] else '❌'}")
    print(f"   预测准确率: 巡逻 {s['patrol_acc']:.1f}% | 追击 {s['chase_acc']:.1f}% "
          f"| 落差 {s['acc_gap']:.1f}pp  -> >90% {'✅' if s['acc_ok'] else '❌'}")
    print(f"   模式切换次数: {s['n_switches']}")

    out = dict(episodes=args.episodes,
               env_kwargs=A_ENV_KWARGS, model=A_MODEL_PATH,
               check1_threat=t, check2_surprisal=s)
    os.makedirs("results", exist_ok=True)
    with open("results/day14_gonogo.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: results/day14_gonogo.json")
    print(f"检查项①② 总体: "
          f"{'✅ 全部通过' if (t['passed'] and s['passed']) else '❌ 有未通过项'}")
    print("（检查项 ③ LLM 解析率需另跑 experiments/day14_check.test_llm_availability）")


if __name__ == "__main__":
    main()
