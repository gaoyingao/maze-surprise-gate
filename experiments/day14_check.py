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
    best_score = -1

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

            # 5. 追击模式判定 —— 由环境权威给出，与采集/训练脚本口径一致。
            #    旧版用 chase_countdown > 0 事后判断，会把"追击的第 1 步"
            #    错分进巡逻桶，导致巡逻均值虚高。
            curr_is_chase = bool(env.game.monster_chasing)
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

        # 选取"两种模式都有足够样本、且至少一次切换"的回合做可视化。
        # 旧条件要求 18<=step<=35，但 A 组策略平均 18 步到终点，该窗口选不中；
        # 并且必须同时含巡逻与追击样本，否则图上无法展示"低-高"对比。
        n_chase = int(sum(is_chase_list))
        n_patrol = int(len(is_chase_list) - n_chase)
        score = min(n_patrol, n_chase) * 10 + len(switch_moments)
        if (len(switch_moments) >= 1 and n_chase >= 3 and n_patrol >= 3
                and score > best_score):
            best_score = score
            best_record = {
                "surprisals": surprisals,
                "switch_moments": switch_moments,
                "is_chase": is_chase_list,
                "episode": ep,
                "n_chase": n_chase,
                "n_patrol": n_patrol,
                "total_steps": step,
            }

    if not best_record:
        print("未抓取到同时含巡逻与追击的回合，请调大 episodes。")
        return

    print(f"  代表回合: ep={best_record['episode']} | 总步数 "
          f"{best_record['total_steps']} | 巡逻 {best_record['n_patrol']} 步 "
          f"| 追击 {best_record['n_chase']} 步 "
          f"| 模式切换 {len(best_record['switch_moments'])} 次")

    surps = np.array(best_record["surprisals"])
    chases = np.array(best_record["is_chase"])

    # 模式切分已由环境权威给出，不再需要"追击后 N 步冷却"这类人工启发式
    pure_patrol_surps = surps[~chases]
    chase_surps = surps[chases]

    mean_patrol = float(np.mean(pure_patrol_surps)) if len(pure_patrol_surps) else 0.0
    max_chase = float(np.max(chase_surps)) if len(chase_surps) else 0.0
    mean_chase = float(np.mean(chase_surps)) if len(chase_surps) else 0.0

    # Go/No-Go 判定（计划书：巡逻 <0.5，切换瞬间 >2.0）
    c_patrol = mean_patrol < 0.5
    c_peak = max_chase > 2.0

    print("📊 惊讶度统计（模式由环境权威切分）：")
    print(f"  - 巡逻段平均惊讶度: {mean_patrol:.2f} nats   "
          f"(标准 <0.5) {'✅' if c_patrol else '❌'}")
    print(f"  - 追击段平均惊讶度: {mean_chase:.2f} nats")
    print(f"  - 追击段最高峰值  : {max_chase:.2f} nats   "
          f"(标准 >2.0) {'✅' if c_peak else '❌'}")
    print(f"  - 巡逻样本 {len(pure_patrol_surps)} | 追击样本 {len(chase_surps)}")

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