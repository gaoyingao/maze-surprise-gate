import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import requests
import matplotlib.pyplot as plt

# 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from envs.maze import MazeEnv
from prediction.train_predictor import MonsterPredictor
from prediction.collect_data import OFFSET_TO_ACTION


def test_llm_availability(test_times=20):
    """
    检查项 ③：测试本地 Ollama 方向词解析稳定性
    """
    print("\n" + "=" * 60)
    print("【检查项 ③】正在测试 Ollama (qwen2.5:0.5b) 方向词解析稳定性...")
    print("=" * 60)
    
    url = "http://localhost:11434/api/generate"
    prompt_template = (
        "你是一个迷宫向导。智能体在(3,4)，怪物在(4,4)，出口在(10,10)。\n"
        "请给出一个避开怪物的建议方向。只回答一个汉字方向词：上、下、左、右。绝对不要解释。"
    )
    
    valid_dirs = ["上", "下", "左", "右"]
    success_count = 0
    latencies = []

    for i in range(test_times):
        payload = {
            "model": "qwen2.5:0.5b",
            "prompt": prompt_template,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 5}
        }
        try:
            import time
            t0 = time.time()
            res = requests.post(url, json=payload, timeout=10)
            res.raise_for_status()
            lat = time.time() - t0
            latencies.append(lat)
            
            raw_text = res.json()["response"].strip()
            for ch in raw_text:
                if ch in valid_dirs:
                    success_count += 1
                    break
        except Exception as e:
            print(f"  [调用失败 {i+1}]: {e}")

    llm_acc = (success_count / test_times) * 100.0
    avg_lat = np.mean(latencies) if latencies else 0.0
    print(f"  - 测试次数: {test_times} 次")
    print(f"  - 成功解析次数: {success_count} 次")
    print(f"  - 解析成功率: {llm_acc:.1f}%")
    print(f"  - 平均单次延迟: {avg_lat:.2f} 秒")
    return llm_acc


def run_surprisal_check(episodes=50, history_len=4):
    """
    检查项 ① & ②：因果时序校准 + 怪物自身绝对轨迹惊讶度计算
    """
    print("\n" + "=" * 60)
    print("【检查项 ① & ②】运行 PPO 策略，计算惊讶度并生成多模态时序图...")
    print("=" * 60)

    rl_model_path = "models/ppo_maze_A.zip"
    predictor_path = "prediction/monster_predictor.pth"

    if not os.path.exists(rl_model_path) or not os.path.exists(predictor_path):
        print("❌ 未找到模型权重，请确认 Day 11/13 文件存在！")
        return

    device = torch.device("cpu")
    predictor = MonsterPredictor(input_dim=2, hidden_dim=128, num_classes=5).to(device)
    predictor.load_state_dict(torch.load(predictor_path, map_location=device))
    predictor.eval()

    env = MazeEnv(with_monster=True, max_steps=200)
    ppo_model = PPO.load(rl_model_path, env=env)

    best_record = None

    for ep in range(episodes):
        obs, _ = env.reset(seed=8000 + ep)
        done = False

        monster_history = []
        surprisals = []
        is_chase_list = []
        switch_moments = []
        step = 0
        prev_chase = False

        while not done:
            # 1. 记录移动前怪物自身坐标
            mr, mc = env.game.monster_pos
            monster_history.append([mr / 11.0, mc / 11.0])

            # 2. 预测怪物下一动作分布 P(a | history)
            probs = None
            if len(monster_history) >= history_len:
                seq = torch.tensor([monster_history[-history_len:]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = predictor(seq)
                    probs = F.softmax(logits, dim=1).squeeze(0).numpy()

            # 3. 推进环境（智能体与怪物移动）
            prev_m_pos = list(env.game.monster_pos)
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            # 4. 结算怪物实际位移动作
            curr_m_pos = list(env.game.monster_pos)
            dr = curr_m_pos[0] - prev_m_pos[0]
            dc = curr_m_pos[1] - prev_m_pos[1]
            true_act = OFFSET_TO_ACTION.get((dr, dc), 4)

            # 5. 追击模式判定
            curr_is_chase = (env.game.chase_countdown > 0)
            if curr_is_chase and not prev_chase:
                switch_moments.append(step)
            prev_chase = curr_is_chase

            # 6. 计算香农惊讶度
            if probs is not None:
                p_actual = max(float(probs[true_act]), 1e-4)
                s = -np.log(p_actual)
                surprisals.append(s)
                is_chase_list.append(curr_is_chase)
                step += 1

        # 选取包含追击事件、步数在 18~35 步之间的典型回合
        if len(switch_moments) >= 1 and 18 <= step <= 35:
            best_record = {
                "surprisals": surprisals,
                "switch_moments": switch_moments,
                "is_chase": is_chase_list
            }
            break

    if not best_record:
        print("未抓取到典型长度回合，请调大 episodes。")
        return

    surps = np.array(best_record["surprisals"])
    chases = np.array(best_record["is_chase"])

    # 稳态巡逻均值：剔除追击及追击后 3 步折返期的影响
    pure_patrol_surps = []
    cooldown = 0
    for s_val, is_c in zip(surps, chases):
        if is_c:
            cooldown = 3
        else:
            if cooldown > 0:
                cooldown -= 1
            else:
                pure_patrol_surps.append(s_val)

    mean_patrol = np.mean(pure_patrol_surps) if pure_patrol_surps else np.mean(surps[~chases])
    max_chase = np.max(surps[chases]) if np.any(chases) else 0.0

    print("📊 修正后的惊讶度统计：")
    print(f"  - 稳态巡逻平均惊讶度: {mean_patrol:.2f} nats (标准要求: 处于低位)")
    print(f"  - 追击段最高爆发峰值: {max_chase:.2f} nats (标准要求: 产生 >2.0 尖峰)")

    # 绘图
    os.makedirs("figures", exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x_steps = np.arange(len(surps))

    # 1. 惊讶度曲线
    ax.plot(x_steps, surps, color="#0b2e59", linewidth=2.2, label="惊讶度 $s = -\\log P$", zorder=3)
    ax.axhline(1.5, color="gray", linestyle="--", linewidth=1.5, label="门控建议阈值 $\\tau = 1.5$", zorder=2)

    # 2. 模式切换时刻红线
    first_switch = True
    for sm in best_record["switch_moments"]:
        ax.axvline(sm, color="crimson", linestyle="-.", linewidth=2.0,
                   label="模式突变起点 (巡逻 $\\to$ 追击)" if first_switch else None, zorder=4)
        first_switch = False

    # 3. 追击持续阶段区间阴影
    in_chase = False
    start_idx = 0
    for idx, is_c in enumerate(chases):
        if is_c and not in_chase:
            in_chase = True
            start_idx = idx
        elif not is_c and in_chase:
            in_chase = False
            ax.axvspan(start_idx, idx - 1, color="crimson", alpha=0.18,
                       label="怪物追击阶段 (持续寻路)" if start_idx == best_record["switch_moments"][0] else None)
    if in_chase:
        ax.axvspan(start_idx, len(chases) - 1, color="crimson", alpha=0.18)

    ax.set_title("Day 14: Surprisal Dynamics Across Monster Mode Switches", fontsize=12)
    ax.set_xlabel("Time Step (t)", fontsize=10)
    ax.set_ylabel("Surprisal [nats]", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()

    save_path = "figures/day14_surprisal_sequence.png"
    plt.savefig(save_path, dpi=200)
    print(f"\n📈 最终学术级时序图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    test_llm_availability(5)
    run_surprisal_check(episodes=50, history_len=4)