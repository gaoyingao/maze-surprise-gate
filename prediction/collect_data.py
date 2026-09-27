import os
import sys
import numpy as np

# 将项目根目录加入模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.maze import MazeEnv

# 怪物移动向量到离散动作标签的映射
OFFSET_TO_ACTION = {
    (-1, 0): 0,  # 上
    (1, 0): 1,   # 下
    (0, -1): 2,  # 左
    (0, 1): 3,   # 右
    (0, 0): 4,   # 不动
}

def collect_predictor_data(episodes=300, history_len=4, save_path="prediction/predictor_data.npz"):
    """
    Day 12 数据采集：
    使用随机策略驱动智能体，记录环境运行中怪物的相对位置历史与实际位移动作
    """
    print("=" * 60)
    print("【Day 12】开始采集怪物动作预测器训练数据")
    print(f"目标轮数: {episodes} 回合 | 历史窗口长度 T: {history_len}")
    print("=" * 60)

    env = MazeEnv(with_monster=True, max_steps=200)

    all_sequences = []  # 形状为 (N, T=4, 2)
    all_labels = []     # 形状为 (N,)

    total_steps = 0
    patrol_count = 0
    chase_count = 0

    for ep in range(episodes):
        env.reset(seed=10000 + ep)
        done = False
        
        # 记录单个回合内的怪物相对位置序列
        monster_rel_history = []

        while not done:
            # 记录移动前怪物位置
            prev_m_pos = list(env.game.monster_pos)
            
            # 当前相对位置 (怪物坐标 - 智能体坐标)，归一化到 [-1, 1]
            ar, ac = env.game.agent_pos
            mr, mc = prev_m_pos
            rel_pos = [(mr - ar) / 11.0, (mc - ac) / 11.0]
            monster_rel_history.append(rel_pos)

            # 随机策略选择智能体动作
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(random_action)
            done = terminated or truncated
            total_steps += 1

            # 统计巡逻与追击状态
            if env.game.chase_countdown > 0:
                chase_count += 1
            else:
                patrol_count += 1

            # 结算移动后怪物的实际位移动作
            curr_m_pos = list(env.game.monster_pos)
            dr = curr_m_pos[0] - prev_m_pos[0]
            dc = curr_m_pos[1] - prev_m_pos[1]
            m_action = OFFSET_TO_ACTION.get((dr, dc), 4)

            # 当历史帧数凑满 T=4 帧时，保存为一个有效训练样本
            if len(monster_rel_history) >= history_len:
                seq = monster_rel_history[-history_len:]
                all_sequences.append(seq)
                all_labels.append(m_action)

    # 转换为 NumPy 格式数组
    X = np.array(all_sequences, dtype=np.float32)  # (N, 4, 2)
    y = np.array(all_labels, dtype=np.int64)        # (N,)

    # 确保存储目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X, y=y)

    print("\n" + "-" * 50)
    print(" 数据采集完成并打包！")
    print(f"  - 总采集步数: {total_steps} 步")
    print(f"  - 生成训练样本数: {len(X)} 条")
    print(f"  - 特征维度 X.shape: {X.shape} (样本数, 历史窗口, 相对坐标)")
    print(f"  - 标签维度 y.shape: {y.shape} (0=上, 1=下, 2=左, 3=右, 4=不动)")
    print(f"  - 样本分布: 巡逻段步数 = {patrol_count} | 追击段步数 = {chase_count}")
    print(f"  - 存储路径: {save_path}")
    print("-" * 50)

    # 打印前 3 条样本进行格式核对（清单验证要求）
    print(" 核对前 3 条样本数据格式：")
    for i in range(min(3, len(X))):
        print(f"  [样本 {i+1}] 历史输入序列:\n{X[i]}")
        print(f"            怪物实际动作标签: {y[i]} ({list(OFFSET_TO_ACTION.keys())[list(OFFSET_TO_ACTION.values()).index(y[i])]})")
    print("=" * 60)


if __name__ == "__main__":
    # 随机策略速度极快，跑 300 回合只需大约 3~5 秒
    collect_predictor_data(episodes=1000, history_len=4)