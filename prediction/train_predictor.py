# -*- coding: utf-8 -*-
"""
prediction/train_predictor.py —— Day 13：训练 GRU 怪物动作预测器

计划书要求
    输入 (B, T=4, 2) 怪物位置序列 -> GRU + Linear -> 5 类（上下左右不动）
    交叉熵损失、训练若干轮
    **巡逻段准确率应 > 90%，追击段应明显下降**（这个落差就是惊讶度的来源）

为什么追击段必然下降
    怪物在巡逻时沿固定航点走，只看它自己的历史位置就能预测；
    但追击时它朝智能体做曼哈顿下降，而智能体的位置**不在预测器输入里**，
    所以追击动作在给定输入下本质上不可预测 —— 准确率下降是设计使然，
    不是模型没训好。这一点必须在报告里写清楚。

Day 21 修订
    1. 训练/评估环境均取自 envs/a_config.py，与 A 组基线一致
    2. 按数据里记录的 mode 字段切分巡逻/追击（不再事后用 chase_countdown 猜）
    3. 评估在【固定 A 组策略】下进行，贴近部署分布

用法
    python prediction/train_predictor.py
    python prediction/train_predictor.py --epochs 60 --history-len 4
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.maze import MazeEnv
from envs.a_config import A_ENV_KWARGS, A_MODEL_PATH, A_MAX_STEPS
from prediction.collect_data import OFFSET_TO_ACTION, MODE_PATROL, MODE_CHASE


class MonsterPredictor(nn.Module):
    """
    输入怪物自身过去 T 步的绝对坐标（归一化），预测它的下一步动作。
    GRU(hidden) + Linear(hidden, 5)
    """

    def __init__(self, input_dim=2, hidden_dim=128, num_classes=5, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ----------------------------------------------------------------------
def train(data_path, save_path, epochs, batch_size, lr, hidden_dim, seed):
    print("=" * 68)
    print("【Day 13】训练 GRU 怪物动作预测器")
    print("=" * 68)

    if not os.path.exists(data_path):
        print(f"❌ 未找到数据集: {data_path}")
        print("   请先运行: python prediction/collect_data.py")
        return None

    d = np.load(data_path, allow_pickle=True)
    X, y = d["X"], d["y"]
    mode = d["mode"] if "mode" in d else np.zeros(len(y), dtype=np.int64)
    history_len = X.shape[1]
    print(f"  数据: {data_path}")
    print(f"  样本 {len(X)} 条 | T={history_len} | 特征 {X.shape[2]} 维")
    print(f"  巡逻样本 {int((mode==MODE_PATROL).sum())} | "
          f"追击样本 {int((mode==MODE_CHASE).sum())}")
    if "env_kwargs" in d:
        print(f"  采集时环境: {d['env_kwargs'][0]}")
        print(f"  采集时策略: {d['policy'][0]}")
    print(f"  当前 A 组环境: {A_ENV_KWARGS}")

    # ---- 划分 ----
    idx = np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(idx)
    split = int(len(X) * 0.85)
    tr, va = idx[:split], idx[split:]

    def ds(ix):
        return TensorDataset(torch.tensor(X[ix]), torch.tensor(y[ix]))

    tr_loader = DataLoader(ds(tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(ds(va), batch_size=256, shuffle=False)

    device = torch.device("cpu")
    model = MonsterPredictor(2, hidden_dim, 5).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n  超参: hidden={hidden_dim} epochs={epochs} lr={lr} "
          f"batch={batch_size} 划分=85/15")
    print("-" * 68)

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tot += loss.item() * len(by)
        if ep % 10 == 0 or ep == epochs:
            model.eval()
            c = t = 0
            with torch.no_grad():
                for bx, by in va_loader:
                    p = torch.argmax(model(bx.to(device)), dim=1).cpu()
                    c += (p == by).sum().item()
                    t += len(by)
            print(f"  Epoch {ep:>3}/{epochs} | loss {tot/len(tr):.4f} | "
                  f"验证准确率 {100*c/t:.1f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n  💾 已保存: {save_path}")
    return model


# ----------------------------------------------------------------------
def evaluate_segments(model, history_len, episodes=50, seed_base=20000,
                      policy="ppo", device="cpu"):
    """
    在【A 组策略】下跑回合，按模式分别统计预测准确率。
    计划书标准：巡逻段 > 90%，追击段明显下降。
    """
    print("\n" + "=" * 68)
    print("环境实测：巡逻段 vs 追击段 预测准确率")
    print("=" * 68)

    env = MazeEnv(with_monster=True, max_steps=A_MAX_STEPS, **A_ENV_KWARGS)
    ppo = None
    if policy == "ppo":
        from stable_baselines3 import PPO
        ppo = PPO.load(A_MODEL_PATH, env=env)
    print(f"  评估策略: {policy} | 环境: {A_ENV_KWARGS} | {episodes} 回合")

    model.eval()
    stat = {MODE_PATROL: [0, 0], MODE_CHASE: [0, 0]}   # mode -> [correct, total]
    probs_by_mode = {MODE_PATROL: [], MODE_CHASE: []}
    chase_count = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        hist = []
        while not done:
            mr, mc = env.game.monster_pos
            hist.append([mr / 11.0, mc / 11.0])
            prev_m = [int(mr), int(mc)]

            probs = None
            if len(hist) >= history_len:
                seq = torch.tensor([hist[-history_len:]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(seq), dim=1).squeeze(0).cpu().numpy()

            if ppo is not None:
                a, _ = ppo.predict(obs, deterministic=True)
                a = int(a)
            else:
                a = int(env.action_space.sample())
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc

            # 模式由环境权威给出，与采集脚本口径完全一致
            mode = MODE_CHASE if env.game.monster_chasing else MODE_PATROL
            if mode == MODE_CHASE:
                chase_count += 1

            cur_m = [int(x) for x in env.game.monster_pos]
            true_act = OFFSET_TO_ACTION.get((cur_m[0]-prev_m[0], cur_m[1]-prev_m[1]), 4)

            if probs is not None:
                pred = int(np.argmax(probs))
                stat[mode][0] += int(pred == true_act)
                stat[mode][1] += 1
                probs_by_mode[mode].append(float(probs[true_act]))

    names = {MODE_PATROL: "巡逻段", MODE_CHASE: "追击段"}
    result = {}
    print("-" * 68)
    for m in (MODE_PATROL, MODE_CHASE):
        c, t = stat[m]
        acc = 100 * c / t if t else 0.0
        ps = probs_by_mode[m]
        avg_s = float(-np.log(max(np.mean(ps), 1e-9))) if ps else 0.0
        result[m] = dict(acc=acc, correct=c, total=t, avg_surprisal=avg_s)
        print(f"  {names[m]}: 准确率 {acc:5.1f}%  ({c}/{t})   "
              f"平均惊讶度 {avg_s:.2f} nats")

    pa = result[MODE_PATROL]["acc"]
    ca = result[MODE_CHASE]["acc"]
    gap = pa - ca
    print("-" * 68)
    print(f"  准确率落差 (巡逻 - 追击): {gap:.1f} 个百分点")
    print(f"  计划书标准: 巡逻段 > 90% 且 追击段明显下降")
    c1 = pa > 90
    c2 = gap > 15
    print(f"    巡逻段 >90%      : {'✅' if c1 else '❌'} ({pa:.1f}%)")
    print(f"    追击段明显下降    : {'✅' if c2 else '❌'} (落差 {gap:.1f}pp)")
    return result


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prediction/predictor_data.npz")
    ap.add_argument("--save", default="prediction/monster_predictor.pth")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--eval-policy", default="ppo", choices=["ppo", "random"])
    args = ap.parse_args()

    model = train(args.data, args.save, args.epochs, args.batch_size,
                  args.lr, args.hidden_dim, args.seed)
    if model is None:
        return
    d = np.load(args.data, allow_pickle=True)
    evaluate_segments(model, d["X"].shape[1], episodes=args.episodes,
                      policy=args.eval_policy)


if __name__ == "__main__":
    main()
