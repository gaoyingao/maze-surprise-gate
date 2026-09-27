import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.maze import MazeEnv
from stable_baselines3 import PPO


def train_day8():
    print("=" * 50)
    print("【Day 8】开始训练最简版 PPO：临时关闭怪物，开启探索熵增强")
    print("=" * 50)

    env = MazeEnv(with_monster=False, max_steps=200)

    # ent_coef=0.01: 促使智能体主动尝试不同方向，防止在死角摆烂
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        seed=42,
    )

    print("正在训练 500,000 步...")
    model.learn(total_timesteps=500_000)

    os.makedirs("models", exist_ok=True)
    save_path = "models/ppo_maze_no_monster"
    model.save(save_path)
    print(f"模型已保存至: {save_path}.zip")

    # 评估阶段
    print("\n" + "=" * 50)
    print("开始评估策略效果（测试 50 个回合）...")
    print("=" * 50)

    eval_env = MazeEnv(with_monster=False, max_steps=200)
    success_count = 0
    total_episodes = 50
    step_records = []

    for ep in range(total_episodes):
        obs, _ = eval_env.reset(seed=1000 + ep)
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1
            done = terminated or truncated

            if terminated and info.get("status") == "win":
                success_count += 1
                step_records.append(steps)
                break

    success_rate = (success_count / total_episodes) * 100.0
    avg_steps = sum(step_records) / len(step_records) if step_records else 0.0

    print(f"评估完成！总回合数: {total_episodes}")
    print(f"成功到达终点次数: {success_count}")
    print(f"最终成功率: {success_rate:.1f}%")
    if step_records:
        print(f"成功回合的平均步数: {avg_steps:.1f} 步")
    print("=" * 50)

    if success_rate >= 90.0:
        print("🎉 恭喜！成功率 >= 90%，Day 8 目标顺利达成！")
    else:
        print("⚠️ 成功率未达 90%，可适当继续增加训练步数。")


if __name__ == "__main__":
    train_day8()