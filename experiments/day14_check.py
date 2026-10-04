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
from envs.a_config import A_ENV_KWARGS, A_MODEL_PATH, A_MAX_STEPS, A_BASE_SEED
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


def _run_episodes(model, env_kwargs, with_monster, episodes, base_seed, max_steps):
    """跑一批回合，返回 (成功率%, 击杀率%, 超时率%, 平均步数)"""
    env = MazeEnv(with_monster=with_monster, max_steps=max_steps, **env_kwargs)
    wins = deaths = timeouts = 0
    step_records = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep * 100)
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            steps += 1
            done = terminated or truncated
            if terminated:
                if info.get("status") == "win":
                    wins += 1
                    step_records.append(steps)
                else:
                    deaths += 1
                break
            if truncated:
                timeouts += 1
                break

    avg_steps = float(np.mean(step_records)) if step_records else 0.0
    return (100.0 * wins / episodes, 100.0 * deaths / episodes,
            100.0 * timeouts / episodes, avg_steps)


def check_threat_margin(episodes=100, base_seed=A_BASE_SEED, max_steps=A_MAX_STEPS,
                        threshold_pp=20.0, seeds=None):
    """
    检查项 ①（生死线）：怪物是否构成实质威胁。

    Day 20 起改用【同一策略跨环境】口径 —— 这是唯一站得住脚的对照方式：
        用同一个策略，分别在无怪 / 有怪环境各跑 N 回合，比较成功率。

    旧实现用两个不同模型（ppo_maze_A vs ppo_maze_no_monster）互相比，
    两者都退化（一个无怪 0% 超时、一个有怪 0% 被吃），结论不可信。

    返回 (平均落差pp, 是否通过)
    """
    print("\n" + "=" * 66)
    print("【检查项 ①】怪物威胁度对照（同一策略跨环境）")
    print("=" * 66)

    model_path = A_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重: {model_path}")
        return None, False

    model = PPO.load(model_path,
                     env=MazeEnv(with_monster=True, max_steps=max_steps,
                                 **A_ENV_KWARGS))
    if seeds is None:
        seeds = [42, 777, 2024]

    print(f"策略: {model_path}")
    print(f"环境配置: {A_ENV_KWARGS}")
    print(f"每个种子 {episodes} 回合 | max_steps: {max_steps}")
    print("-" * 66)
    print(f"{'种子':<8}{'无怪成功':>10}{'有怪成功':>10}{'落差pp':>9}"
          f"{'有怪击杀':>10}")
    print("-" * 66)

    margins = []
    for sd in seeds:
        no_succ, no_die, no_to, _ = _run_episodes(
            model, A_ENV_KWARGS, False, episodes, sd, max_steps)
        yes_succ, yes_die, yes_to, yes_steps = _run_episodes(
            model, A_ENV_KWARGS, True, episodes, sd, max_steps)
        m = no_succ - yes_succ
        margins.append(m)
        print(f"{sd:<8}{no_succ:>9.1f}%{yes_succ:>9.1f}%{m:>9.1f}{yes_die:>9.1f}%")

    avg_margin = float(np.mean(margins))
    print("-" * 66)
    print(f"平均落差: {avg_margin:.1f} pp   最小落差: {min(margins):.1f} pp")
    print(f"计划书门槛: ≥ {threshold_pp:.1f} pp（每个种子都要达标）")

    passed = min(margins) >= threshold_pp
    print(f"判定: {'✅ 通过' if passed else '❌ 未通过'}")
    print("=" * 66)

    return avg_margin, passed


def run_surprisal_check(episodes=50, history_len=4):
    """
    检查项 ① & ②：因果时序校准 + 怪物自身绝对轨迹惊讶度计算
    """
    print("\n" + "=" * 60)
    print("【检查项 ① & ②】运行 PPO 策略，计算惊讶度并生成多模态时序图...")
    print("=" * 60)

    rl_model_path = A_MODEL_PATH
    predictor_path = "prediction/monster_predictor.pth"

    if not os.path.exists(rl_model_path) or not os.path.exists(predictor_path):
        print(f"❌ 未找到模型权重: {rl_model_path} 或 {predictor_path}")
        return

    device = torch.device("cpu")
    predictor = MonsterPredictor(input_dim=2, hidden_dim=128, num_classes=5).to(device)
    predictor.load_state_dict(torch.load(predictor_path, map_location=device))
    predictor.eval()

    env = MazeEnv(with_monster=True, max_steps=A_MAX_STEPS, **A_ENV_KWARGS)
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
    # 检查项 ①：怪物威胁度（生死线）—— 真正计算，不再硬编码通过
    check_threat_margin(episodes=100, base_seed=A_BASE_SEED,
                        max_steps=A_MAX_STEPS)
    # 检查项 ③：LLM 方向解析稳定性（计划书要求 20 次）
    test_llm_availability(20)
    # 检查项 ②：惊讶度动力学
    run_surprisal_check(episodes=50, history_len=4)