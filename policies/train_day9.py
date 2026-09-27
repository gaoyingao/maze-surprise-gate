import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# 将项目根目录加入模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat
from envs.maze import MazeEnv


class FullMetricsCollector(KVWriter):
    """
    全量指标拦截器：
    直接作为 Logger 的后端输出，当控制台表格生成时，截获所有真实展示的参数（含 fps, ep_rew_mean 等）。
    """
    def __init__(self):
        self.metrics = defaultdict(list)
        self.steps = []

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        # 当有训练数据输出时提取总步数
        if "time/total_timesteps" in key_values:
            curr_step = key_values["time/total_timesteps"]
        else:
            curr_step = step

        # 只记录包含有效指标的轮次
        if len(key_values) > 0:
            self.steps.append(curr_step)
            for k, v in key_values.items():
                self.metrics[k].append(v)

    def close(self) -> None:
        pass


def plot_all_metrics(collector: FullMetricsCollector, save_path="figures/day9_full_metrics.png"):
    """将收集到的核心指标绘制成 3x3 的全景监控图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 关注的核心维度
    target_plots = [
        ("rollout/ep_rew_mean", "Average Reward (ep_rew_mean)", "royalblue"),
        ("rollout/ep_len_mean", "Episode Length (ep_len_mean)", "crimson"),
        ("train/value_loss", "Value Loss (Critic)", "darkorange"),
        ("train/explained_variance", "Explained Variance (Critic)", "forestgreen"),
        ("train/policy_gradient_loss", "Policy Gradient Loss", "purple"),
        ("train/entropy_loss", "Policy Entropy Loss (Exploration)", "teal"),
        ("train/approx_kl", "Approx KL Divergence", "brown"),
        ("train/clip_fraction", "PPO Clip Fraction", "gray"),
        ("time/fps", "Training FPS (Throughput)", "olive"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (key, title, color) in enumerate(target_plots):
        ax = axes[idx]
        if key in collector.metrics and len(collector.metrics[key]) > 0:
            vals = collector.metrics[key]
            steps = collector.steps[-len(vals):]
            ax.plot(steps, vals, color=color, linewidth=1.8)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Timesteps", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.6)
            
            # 限制解释方差纵坐标，防止初期负值拉变形图表
            if key == "train/explained_variance":
                ax.set_ylim(-1.5, 1.1)
        else:
            ax.text(0.5, 0.5, "Collecting / No Data", ha="center", va="center")
            ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\n📊 全变量监控曲线已生成并保存至: {save_path}")
    plt.show()


def train_day9():
    print("=" * 60)
    print("【Day 9 & 10】全量参数监控训练模式（全图修复版）")
    print("环境设定: with_monster=True | 训练步数: 800,000 步")
    print("=" * 60)

    # 1. 实例化环境
    env = MazeEnv(with_monster=True, max_steps=200)

    # 2. 初始化 PPO
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

    # 3. 挂载全量数据拦截器，安全替换 logger
    collector = FullMetricsCollector()
    custom_logger = Logger(
        folder=None,
        output_formats=[
            HumanOutputFormat(sys.stdout),  # 维持终端表格打印
            collector                       # 截获表格全部变量
        ]
    )
    model.set_logger(custom_logger)

    # 4. 开始训练 80 万步
    print("开始训练 800,000 步...")
    model.learn(total_timesteps=800_000)

    # 5. 训练结束，绘制并展示 9 宫格完整监控大图
    plot_all_metrics(collector, save_path="figures/day9_full_metrics.png")

    # 6. 保存模型检查点
    os.makedirs("models", exist_ok=True)
    save_path = "models/ppo_maze_day9"
    model.save(save_path)
    print(f"模型权重已保存至: {save_path}.zip")

    # 7. 跑 50 回合实测评估
    print("\n" + "=" * 60)
    print("开始进行 50 回合综合评估...")
    print("=" * 60)

    eval_env = MazeEnv(with_monster=True, max_steps=200)
    wins, deaths, timeouts = 0, 0, 0
    step_records = []

    for ep in range(50):
        obs, _ = eval_env.reset(seed=2000 + ep)
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
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

    success_rate = (wins / 50) * 100.0
    death_rate = (deaths / 50) * 100.0
    timeout_rate = (timeouts / 50) * 100.0
    avg_steps = sum(step_records) / len(step_records) if step_records else 0.0

    print(f"评估完成（总计 50 回合）：")
    print(f"  🏆 成功到达终点: {wins} 回合 ({success_rate:.1f}%)")
    print(f"  💀 被怪物击杀:   {deaths} 回合 ({death_rate:.1f}%)")
    print(f"  ⏰ 步数耗尽超时: {timeouts} 回合 ({timeout_rate:.1f}%)")
    if step_records:
        print(f"  👣 成功回合平均步数: {avg_steps:.1f} 步")
    print("=" * 60)


if __name__ == "__main__":
    train_day9()