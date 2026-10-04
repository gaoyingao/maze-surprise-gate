# -*- coding: utf-8 -*-
"""
prediction/collect_data.py —— Day 12：采集 GRU 预测器训练数据

计划书要求
    用随机策略跑若干回合，每步记录 (最近 T 帧怪物位置, 怪物本步动作)。
    完成标志：predictor_data.npz 存好，能打印样本核对格式。

Day 21 修订要点（对照 A 组新基线）
    1. 环境配置从 envs/a_config.py 取，与 A 组评估环境完全一致。
       旧版用默认 chase 参数（3/5%/5），而 A 组是 4/10%/6 —— 动力学不匹配，
       预测器学到的追击行为不是它实际会遇到的那一种。
    2. 同时记录 `mode`（patrol / chase），用于按模式评估准确率。
       旧版靠 env.game.chase_countdown > 0 事后切分，读的是**移动之后**的
       倒计时，与产生该动作时的模式差一步，会把边界步分错桶。
       这里改为记录**产生该动作时的模式**。
    3. 位置用绝对坐标归一化 (r/11, c/11)，与 gate_controller 和
       train_predictor 的评估保持一致。

用法
    python prediction/collect_data.py
    python prediction/collect_data.py --episodes 600 --policy random
    python prediction/collect_data.py --policy ppo      # 用 A 组策略采（分布更贴近部署）
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.maze import MazeEnv
from envs.a_config import A_ENV_KWARGS, A_MODEL_PATH, A_MAX_STEPS

# 怪物动作编号：0上 1下 2左 3右 4不动
OFFSET_TO_ACTION = {
    (-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3, (0, 0): 4,
}
MODE_PATROL, MODE_CHASE = 0, 1


def collect(episodes=600, history_len=4, max_steps=A_MAX_STEPS,
            policy="random", seed_base=10000, env_kwargs=None,
            save_path="prediction/predictor_data.npz", verbose=True):
    env_kwargs = dict(A_ENV_KWARGS if env_kwargs is None else env_kwargs)
    env = MazeEnv(with_monster=True, max_steps=max_steps, **env_kwargs)

    ppo = None
    if policy == "ppo":
        from stable_baselines3 import PPO
        ppo = PPO.load(A_MODEL_PATH, env=env)

    if verbose:
        print("=" * 66)
        print("【Day 12】采集 GRU 预测器训练数据")
        print("=" * 66)
        print(f"  环境配置 : {env_kwargs}")
        print(f"  采集策略 : {policy}")
        print(f"  回合数   : {episodes} | 历史长度 T={history_len} | "
              f"max_steps={max_steps}")

    seqs, labels, modes = [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        hist = []

        while not done:
            mr, mc = env.game.monster_pos
            hist.append([mr / 11.0, mc / 11.0])

            prev_m = [int(mr), int(mc)]

            if policy == "ppo" and ppo is not None:
                action, _ = ppo.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = int(env.action_space.sample())

            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

            cur_m = [int(x) for x in env.game.monster_pos]
            dr, dc = cur_m[0] - prev_m[0], cur_m[1] - prev_m[1]
            m_act = OFFSET_TO_ACTION.get((dr, dc), 4)

            # 模式由环境权威给出（env.game.monster_chasing 在本步移动时置位），
            # 不再事后从 chase_countdown 推测 —— 后者会因为倒计时在移动中
            # 递减而把"追击的第 1 步"错分进巡逻桶。
            mode = MODE_CHASE if env.game.monster_chasing else MODE_PATROL

            if len(hist) >= history_len:
                seqs.append(hist[-history_len:])
                labels.append(m_act)
                modes.append(mode)

    X = np.array(seqs, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    m = np.array(modes, dtype=np.int64)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X, y=y, mode=m,
                        env_kwargs=np.array([str(env_kwargs)], dtype=object),
                        policy=np.array([policy], dtype=object))

    if verbose:
        n_p = int((m == MODE_PATROL).sum())
        n_c = int((m == MODE_CHASE).sum())
        print(f"\n  样本数     : {len(X)} 条  (X.shape={X.shape})")
        print(f"  巡逻段样本 : {n_p} ({100*n_p/len(m):.1f}%)")
        print(f"  追击段样本 : {n_c} ({100*n_c/len(m):.1f}%)")
        print(f"  动作分布   : {np.bincount(y, minlength=5).tolist()} "
              f"(0上 1下 2左 3右 4不动)")
        print(f"\n  样例行（核对格式）：")
        for i in range(min(3, len(X))):
            print(f"    X[{i}] = {np.round(X[i], 3).tolist()}  "
                  f"y={y[i]}  mode={'chase' if m[i] else 'patrol'}")
        print(f"\n  已保存: {save_path}")

    return X, y, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--history-len", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=A_MAX_STEPS)
    ap.add_argument("--policy", default="random", choices=["random", "ppo"])
    ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--save", default="prediction/predictor_data.npz")
    args = ap.parse_args()
    collect(episodes=args.episodes, history_len=args.history_len,
            max_steps=args.max_steps, policy=args.policy,
            seed_base=args.seed_base, save_path=args.save)


if __name__ == "__main__":
    main()
