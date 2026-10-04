import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from envs.maze import MazeEnv
from envs.a_config import A_ENV_KWARGS, A_MODEL_PATH, A_MAX_STEPS
from dual_system.gate_controller import SurprisalGateController


def run_day16_integration_test():
    print("=" * 88)
    print("【Day 16】双系统闭环控制测试 (PPO + GRU 惊讶度门控 + LLM 偏好排序掩码接管)")
    print("=" * 88)

    rl_model_path = A_MODEL_PATH
    if not os.path.exists(rl_model_path):
        print(f"❌ 未找到底层 RL 模型: {rl_model_path}")
        return

    # 与 evaluate.py 统一为 200 步、同一套 A 组环境配置
    env = MazeEnv(with_monster=True, max_steps=A_MAX_STEPS, **A_ENV_KWARGS)
    ppo_model = PPO.load(rl_model_path, env=env)

    gate = SurprisalGateController(
        predictor_path="prediction/monster_predictor.pth",
        env=env,
        tau=1.5,
        max_ranking=3,
        cooldown_steps=5,
        history_len=4,
    )

    test_seed = 8011
    obs, _ = env.reset(seed=test_seed)
    gate.reset()
    gate.update_history(env.game.monster_pos)

    print(f"\n回合启动 | seed={test_seed} | 地图固定 | 出口 (10, 10)")
    print("-" * 88)
    print(f"{'步':^4} | {'智能体':^8} | {'怪物':^8} | {'惊讶度':>7} | {'决策来源':<30} | 动作")
    print("-" * 88)

    dir_chars = ["上", "下", "左", "右"]
    done = False
    step_count = 0
    terminated = truncated = False
    info = {}

    while not done:
        step_count += 1
        agent_p = [int(x) for x in env.game.agent_pos]
        monster_p = [int(x) for x in env.game.monster_pos]

        ppo_act, _ = ppo_model.predict(obs, deterministic=True)

        # 惊讶度：用历史里倒数第二帧（t-2）到当前（t-1）的怪物位移
        if len(gate.monster_history) >= 2:
            prev_m = [int(round(p * 11.0)) for p in gate.monster_history[-2]]
            surprisal = gate.compute_surprisal(prev_m, monster_p)
        else:
            surprisal = 0.0

        final_act, source = gate.decide_action(
            ppo_action=int(ppo_act),
            agent_pos=agent_p,
            monster_pos=monster_p,
            goal_pos=(10, 10),
            surprisal=surprisal,
        )

        print(f"{step_count:4d} | {str(agent_p):^8} | {str(monster_p):^8} | "
              f"{surprisal:7.2f} | {source:<30} | {dir_chars[final_act]}")

        obs, reward, terminated, truncated, info = env.step(final_act)
        done = terminated or truncated

        # 掩码方案下不会出现非法步；这里只做断言式的健全性检查
        if info.get("hit_wall", False):
            print(f"     {'':>4} | {'':>8} | {'':>8} | {'':>7} | "
                  f"{'⚠️ 意外撞墙！掩码逻辑可能有误':<30} |")

        gate.update_history(env.game.monster_pos)

    print("-" * 88)
    status = info.get("status", "unknown")
    if status == "win":
        result = "🎉 成功逃脱达到出口！"
    elif status == "die":
        result = "💀 被怪物捕获！"
    elif status == "timeout":
        result = "⏰ 步数耗尽超时"
    else:
        result = f"未知({status})"
    print(f"回合结束结果: {result} | 总步数: {step_count} 步")
    print()
    calls = max(gate.total_llm_calls, 1)
    print("实验指标：")
    print(f"  LLM 唤醒次数              : {gate.total_llm_calls} 次")
    print(f"  LLM 解析失败次数          : {gate.llm_parse_failures} 次")
    print(f"  首选方向不可行（撞墙倾向）: {gate.llm_first_choice_illegal} 次 "
          f"({100*gate.llm_first_choice_illegal/calls:.0f}% of calls)")
    print(f"  被掩码顺延到次选          : {gate.llm_masked_fallbacks} 次")
    print(f"  偏好中完全无合法方向      : {gate.llm_no_legal_choice} 次")
    print("=" * 88)


if __name__ == "__main__":
    run_day16_integration_test()
