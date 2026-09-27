import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from envs.maze import MazeEnv


def evaluate_policy(model_path="models/ppo_maze_A.zip", episodes=50, base_seed=42):
    """
    Day 11 通用评估函数：
    在统一固定的种子序列下，评估策略在带怪迷宫中的基线表现
    """
    print("=" * 60)
    print("【Day 11】A 组基线标准评估（纯 PPO 策略）")
    print(f"评估模型: {model_path}")
    print(f"测试轮数: {episodes} 回合 | 基础随机种子: {base_seed}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重文件: {model_path}")
        return None

    # 初始化带怪迷宫评估环境
    env = MazeEnv(with_monster=True, max_steps=200)
    model = PPO.load(model_path, env=env)

    wins = 0
    deaths = 0
    timeouts = 0
    step_records = []
    reward_records = []

    for ep in range(episodes):
        # 严格固定每个回合的种子，确保后续三组对照的迷宫怪物轨迹可完全复现
        eval_seed = base_seed + ep * 100
        obs, _ = env.reset(seed=eval_seed)
        done = False
        steps = 0
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if terminated:
                status = info.get("status")
                if status == "win":
                    wins += 1
                    step_records.append(steps)
                elif status == "die":
                    deaths += 1
                break
            elif truncated:
                timeouts += 1
                break

        reward_records.append(total_reward)

    success_rate = (wins / episodes) * 100.0
    death_rate = (deaths / episodes) * 100.0
    timeout_rate = (timeouts / episodes) * 100.0
    avg_steps = np.mean(step_records) if step_records else 0.0
    avg_reward = np.mean(reward_records)

    print("📊 A 组基线定型结果：")
    print(f"  🏆 成功率 (Success Rate):     {success_rate:.1f}% ({wins}/{episodes})")
    print(f"  💀 击杀率 (Monster Kill Rate): {death_rate:.1f}% ({deaths}/{episodes})")
    print(f"  ⏰ 超时率 (Timeout Rate):      {timeout_rate:.1f}% ({timeouts}/{episodes})")
    print(f"  👣 平均成功步数 (Avg Steps):    {avg_steps:.2f} 步")
    print(f"  💰 平均回合回报 (Avg Reward):   {avg_reward:.2f}")
    print("=" * 60)

    # 保存评估基线记录至文本文件，供第三周撰写报告直接查阅
    os.makedirs("results", exist_ok=True)
    report_file = "results/day11_baseline_A.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("A组基线评估报告 (Day 11 定型)\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"成功率: {success_rate:.1f}%\n")
        f.write(f"击杀率: {death_rate:.1f}%\n")
        f.write(f"超时率: {timeout_rate:.1f}%\n")
        f.write(f"平均成功步数: {avg_steps:.2f}\n")
        f.write(f"平均回报: {avg_reward:.2f}\n")
    print(f"📝 基线数据已持久化保存至: {report_file}")

    return {
        "success_rate": success_rate,
        "death_rate": death_rate,
        "timeout_rate": timeout_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward
    }


if __name__ == "__main__":
    evaluate_policy(episodes=100, base_seed=42)