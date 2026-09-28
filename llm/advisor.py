import os
import sys
import re
import time
import requests

# Ollama 服务配置
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"

# 汉字方向到环境离散动作的严格映射 (envs/maze.py 标准动作)
DIRECTION_MAP = {
    "上": 0,
    "下": 1,
    "左": 2,
    "右": 3,
}

# 清单标准 Prompt 模板
PROMPT_TEMPLATE = """你是一个迷宫中的向导。迷宫是 12×12 的网格。
你在第 {ax} 行第 {ay} 列。怪物在第 {mx} 行第 {my} 列。出口在第 {gx} 行第 {gy} 列。
你不能穿过墙。请帮助智能体避开怪物并到达出口。
只回答一个方向词：上、下、左、右。不要解释。"""


def ask_direction(agent_pos, monster_pos, goal_pos, timeout=10):
    """
    Day 15 核心接口：
    传入坐标元组或列表 (row, col)，构造 Prompt 并请求本地 Ollama，
    使用正则表达式提取回答中的第一个方向词并返回动作编号。
    
    返回: (action_idx, direction_str, latency)
          解析失败时 action_idx 为 None
    """
    prompt = PROMPT_TEMPLATE.format(
        ax=agent_pos[0], ay=agent_pos[1],
        mx=monster_pos[0], my=monster_pos[1],
        gx=goal_pos[0], gy=goal_pos[1]
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.1,    # 低温确保确定性输出
            "num_predict": 5       # 严格截断生成长度，最大化推理速度
        }
    }

    t0 = time.time()
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        res.raise_for_status()
        latency = time.time() - t0
        raw_text = res.json().get("response", "").strip()

        # 使用正则提取文本中出现的第一个有效方向词
        match = re.search(r"[上下左右]", raw_text)
        if match:
            direction_char = match.group(0)
            return DIRECTION_MAP[direction_char], direction_char, latency
        else:
            return None, raw_text, latency

    except Exception as e:
        latency = time.time() - t0
        return None, f"Request_Error: {e}", latency


def run_day15_benchmark(trials=20):
    """
    Day 15 验收基准测试：
    在不同相对位置随机扰动下连续调用 20 次，验证解析成功率 >= 90%
    """
    print("=" * 60)
    print("【Day 15】LLM 接入与方向解析鲁棒性基准测试")
    print(f"测试模型: {MODEL_NAME} | 目标调用次数: {trials}")
    print("=" * 60)

    import random
    success_count = 0
    latencies = []
    goal = (10, 10)

    for i in range(1, trials + 1):
        # 模拟不同场景的坐标态势
        ax, ay = random.randint(1, 10), random.randint(1, 10)
        mx, my = random.randint(1, 10), random.randint(1, 10)

        act_idx, text_or_err, lat = ask_direction((ax, ay), (mx, my), goal)
        latencies.append(lat)

        if act_idx is not None:
            success_count += 1
            status_tag = f"✅ 解析成功 -> 动作: {act_idx} ('{text_or_err}')"
        else:
            status_tag = f"❌ 解析失败 -> 原文: '{text_or_err}'"

        print(f"[{i:02d}/{trials}] 智能体: ({ax:2d},{ay:2d}) | 怪物: ({mx:2d},{my:2d}) | 耗时: {lat:.2f}s | {status_tag}")

    success_rate = (success_count / trials) * 100.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n" + "-" * 50)
    print("📊 Day 15 验收指标统计：")
    print(f"  - 总测试次数: {trials} 次")
    print(f"  - 解析成功次数: {success_count} 次")
    print(f"  - 整体解析成功率: {success_rate:.1f}% (标准要求: >= 90%)")
    print(f"  - 单次推理平均耗时: {avg_lat:.2f} 秒")
    print("-" * 50)

    if success_rate >= 90.0:
        print("🎉 验收达标！方向解析稳定可靠，可直接接入 Day 16 门控接管。")
    else:
        print("⚠️ 成功率未达 90%，请检查 Ollama 后台状态或 Prompt 设置。")


if __name__ == "__main__":
    run_day15_benchmark(trials=20)