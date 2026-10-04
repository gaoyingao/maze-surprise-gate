import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from envs.maze import MazeEnv
from envs.a_config import A_ENV_KWARGS, A_MODEL_PATH, A_MAX_STEPS, A_BASE_SEED


def evaluate_policy(model_path=A_MODEL_PATH, episodes=100, base_seed=A_BASE_SEED):
    """
    A 组基线标准评估（纯 PPO 策略）

    Day 20 起统一使用新基线：
      - 模型: models/threat_medium_with_monster.zip
      - 环境: 追击半径 4 / 触发 10% / 持续 6 步（见 envs/a_config.py）

    旧基线 models/ppo_maze_A.zip 已废弃，原因见 envs/a_config.py 的说明。
    """
    print("=" * 60)
    print("【A 组基线】标准评估（纯 PPO 策略）")
    print(f"评估模型: {model_path}")
    print(f"环境配置: {A_ENV_KWARGS}")
    print(f"测试轮数: {episodes} 回合 | 基础随机种子: {base_seed}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重文件: {model_path}")
        return None

    # 初始化带怪迷宫评估环境（配置取自 envs/a_config.py，保证三组对照一致）
    env = MazeEnv(with_monster=True, max_steps=A_MAX_STEPS, **A_ENV_KWARGS)
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

    # 保存评估基线记录（Day 20 起为新基线，旧的 day11 记录已归档为 _legacy）
    os.makedirs("results", exist_ok=True)
    report_file = "results/baseline_A.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("A 组基线评估报告（Day 20 新基线）\n")
        f.write("=" * 46 + "\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"环境配置: {A_ENV_KWARGS}\n")
        f.write(f"回合数: {episodes} | 基础种子: {base_seed} | "
                f"max_steps: {A_MAX_STEPS}\n")
        f.write("-" * 46 + "\n")
        f.write(f"成功率: {success_rate:.1f}%\n")
        f.write(f"击杀率: {death_rate:.1f}%\n")
        f.write(f"超时率: {timeout_rate:.1f}%\n")
        f.write(f"平均成功步数: {avg_steps:.2f}\n")
        f.write(f"平均回报: {avg_reward:.2f}\n")
        f.write("-" * 46 + "\n")
        f.write("威胁度标定（同一策略跨环境，见 results/threat_margin_final.json）：\n")
        f.write("  无怪成功率 100.0% | 有怪成功率 76.3% | 落差 23.7pp（≥20pp 达标）\n")
        f.write("  逐种子落差: 42→29.0pp  777→21.0pp  2024→21.0pp\n")
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