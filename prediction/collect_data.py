import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.maze import MazeEnv

OFFSET_TO_ACTION = {
    (-1, 0): 0,  # 上
    (1, 0): 1,   # 下
    (0, -1): 2,  # 左
    (0, 1): 3,   # 右
    (0, 0): 4,   # 不动
}

def collect_predictor_data(episodes=600, history_len=4, save_path="prediction/predictor_data.npz"):
    print("=" * 60)
    print("【修正版】采集怪物绝对轨迹数据（彻底解耦智能体走位干扰）")
    print("=" * 60)

    env = MazeEnv(with_monster=True, max_steps=200)
    all_sequences = []
    all_labels = []

    for ep in range(episodes):
        env.reset(seed=10000 + ep)
        done = False
        monster_history = []

        while not done:
            # 记录怪物自身当前的绝对坐标 (归一化到 [0, 1])
            mr, mc = env.game.monster_pos
            monster_history.append([mr / 11.0, mc / 11.0])

            prev_m_pos = list(env.game.monster_pos)
            # 智能体动作
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated

            # 怪物真实动作
            curr_m_pos = list(env.game.monster_pos)
            dr = curr_m_pos[0] - prev_m_pos[0]
            dc = curr_m_pos[1] - prev_m_pos[1]
            m_act = OFFSET_TO_ACTION.get((dr, dc), 4)

            # 当历史够 4 步即可精准预测巡逻下一步
            if len(monster_history) >= history_len:
                all_sequences.append(monster_history[-history_len:])
                all_labels.append(m_act)

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X, y=y)
    print(f"数据采集完成: 样本数 = {len(X)} 条 | 特征维度 = {X.shape}")

if __name__ == "__main__":
    collect_predictor_data(episodes=600, history_len=4)