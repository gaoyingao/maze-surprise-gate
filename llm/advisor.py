import os
import re
import sys
import time

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OLLAMA_URL = "http://localhost:11434/api/generate"

# 模型选型（Day 19-20 定稿）
#
# 全图 79 格审计（158 次调用/模型）的首选合法率：
#   qwen2.5:0.5b = 0%   3b = 83.5%   7b = 80.4%   14b = 100%
#   qwen3:4b     = 0%（思维链吃光预算，跑不完，不可用）
#   gemma3n:e4b  = 86.7%（恒定偏好）  gemma4:e4b = 88.6%（恒定偏好）
#   phi4-mini    = 49.4%（低于盲猜）
#   ★ qwen3.5:4b = 92.4%  ← 最终选用
#
# 选 qwen3.5:4b 的理由：
#   1. 92.4% 合法率，接近 14B 的 100%，但只有 1/4 体量
#   2. 延迟仅 2.70s（14B 是 5.6s）
#   3. 分层检验通过：最紧约束层（1 个出口）4/4 全对（随机仅 25%）
#      —— 而 gemma4 在同一层只有 2/4，属模式补全
#   4. 标签置换检验 16/20：答案跟着标签走，证明它在读输入内容
#   5. 30/79 个格子会随怪物位置改变输出（14B 只有 7/79）
#
# ⚠️ 必须 think=False：开启思考时思维链会吃光 num_predict，
#    response 为空（done_reason=length），这正是 qwen3:4b 失败的原因。
MODEL_NAME = "qwen3.5:4b"
LLM_THINK = False   # 关思考；Qwen3.5 支持该参数

# temperature 必须为 0（可复现性）。
# 实测：temperature=0.1 时同一个局面连问 6 次会给出 4 种不同答案，其中 2 次非法；
#       temperature=0 时输出完全确定，两次完整回合逐字节一致。
TEMPERATURE = 0.0
LLM_SEED = 42

# 方向 -> 环境动作编号，必须与 envs/maze.py 的 ACTION_MAP 严格一致
DIRECTION_MAP = {"上": 0, "下": 1, "左": 2, "右": 3}
ACTION_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ["上", "下", "左", "右"]

PLAN_LEN = 3   # 从偏好序列里最多取几步交给环境执行

# ---------------------------------------------------------------------------
# 为什么用「偏好排序」而不是「判断哪个方向是墙」
#
# 实测（本项目 12x12 迷宫，36 次平行试验 × 2 个模型）：
#   1. 让模型判断"哪个方向是墙"时，0.5B/3B 在位置、四邻格值、怪物位置、
#      随机种子全部不同的 36 次提问里，都只输出了同一个字符串
#      （0.5B 甚至把模板 '上=？' 原样吐回）—— 输出与输入完全无关，是模式补全。
#   2. 把视野从 3x3 扩到 7x7，输出一个字都没变，证明"视野不够"不是原因。
#   3. 换成让模型输出"走的方向的优先顺序"后，输出开始随局面变化
#      （36 次里 2~3 种串，3/4 的局面会随怪物位置改变），首选合法率 75%。
#
# 因此职责划分改为：
#   - 几何判断（哪格是墙）交给环境 env.game.is_valid，确定且零错误；
#   - LLM 只提供战术偏好（更想往哪走），排序任务不需要几何能力。
#   最终动作 = 偏好序列里第一个合法的方向（掩码选择，见 gate_controller）。
#
# ⚠️ 关键：prompt 必须把「四邻格值」显式告诉模型，否则它无从判断方向是否可走。
#    实测教训——生产版曾经漏掉这一段，14B 的首选不可行率立刻从 0% 涨到 53.5%，
#    回落到接近随机盲猜（67%）的水平。补上后恢复 100%。
# ---------------------------------------------------------------------------
PLAN_PROMPT_TEMPLATE = """你在一个 12x12 的迷宫里，出口在 (x={gx}, y={gy})。
怪物在 (x={mx}, y={my})，它正在追击你。
你现在在 (x={ax}, y={ay})。

你四周四个方向的格子情况如下（「墙」表示走不通）：
上 方那一格是：{c_up}
下 方那一格是：{c_down}
左 方那一格是：{c_left}
右 方那一格是：{c_right}

请把你接下来希望走的方向按 优先顺序 排序，最想走的写在前面。
请综合考虑：尽量远离怪物，同时尽量朝出口 (x={gx}, y={gy}) 靠近。
注意：是「墙」的方向你根本走不过去。

严格按下面格式输出 4 行，把 ？ 换成名次数字，不要添加别的内容：
上=？
下=？
左=？
右=？

每行只能填 1、2、3、4 中的一个数字，四个数字各用一次，不能重复。
1 表示你最想走的方向，4 表示你最不想走的方向。
不要解释。"""


def get_available_directions(agent_pos, env):
    """直接调用环境底层的 is_valid 判定"""
    if env is None:
        return list(DIR_NAMES)
    r, c = int(agent_pos[0]), int(agent_pos[1])
    return [name for name, (dr, dc) in zip(DIR_NAMES, ACTION_OFFSETS)
            if env.game.is_valid((r + dr, c + dc))]


def get_adjacent_cells(agent_pos, env):
    """
    返回四邻格的可通行性文字，供 prompt 播报。

    这一步是「环境做几何」的落点：模型不必自己解析 ASCII 地图
    （实测它做不到，5x5 地图与纯文字列表的合法率持平），
    而是由环境把每一格的状态算好、直接写进 prompt。
    """
    out = {}
    if env is None:
        # 无环境时一律标"未知"，避免给出误导性的"路"
        return {d: "未知" for d in DIR_NAMES}
    r, c = int(agent_pos[0]), int(agent_pos[1])
    for name, (dr, dc) in zip(DIR_NAMES, ACTION_OFFSETS):
        out[name] = "路" if env.game.is_valid((r + dr, c + dc)) else "墙"
    return out


def parse_ranking(text):
    """
    解析偏好排序输出，返回按名次升序排列的方向列表。

    只认 方向=数字 这种结构；解析不出完整 4 个时返回已解析到的部分，
    由调用方（gate_controller）用掩码兜底。本文件不做合法性过滤，
    这样"模型首选是否可行"能被如实记成实验指标。
    """
    rank = {}
    for line in (text or "").splitlines():
        m = re.match(r"\s*([上下左右])\s*[=＝:：]\s*([1-4])", line)
        if m:
            rank[m.group(1)] = int(m.group(2))
    return [d for d, _ in sorted(rank.items(), key=lambda kv: kv[1])]


def parse_plan(text, plan_len=PLAN_LEN):
    """兼容旧接口：把偏好序列转成动作编号前缀（不做合法性过滤）"""
    return [DIRECTION_MAP[d] for d in parse_ranking(text)[:plan_len]]


def ask_plan(agent_pos, monster_pos, goal_pos, env=None, plan_len=PLAN_LEN, timeout=60):
    """
    向本地 Ollama 请求一份方向偏好排序。

    返回: (actions, dir_names, raw_text, latency)
          actions   —— 偏好序列（动作编号），长度可能为 0
          dir_names —— 同一序列的方向名，便于打日志
          raw_text  —— LLM 原始回复，务必打日志留证
    """
    cells = get_adjacent_cells(agent_pos, env)
    prompt = PLAN_PROMPT_TEMPLATE.format(
        ax=int(agent_pos[1]), ay=int(agent_pos[0]),
        mx=int(monster_pos[1]), my=int(monster_pos[0]),
        gx=int(goal_pos[1]), gy=int(goal_pos[0]),
        c_up=cells["上"], c_down=cells["下"],
        c_left=cells["左"], c_right=cells["右"],
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "think": LLM_THINK,          # Qwen3.5 必须关思考，否则答案被思维链挤掉
        "options": {
            "temperature": TEMPERATURE,
            "seed": LLM_SEED,
            "num_predict": 200,      # 关思考后 15 字符就够，留余量防长尾
        },
    }

    t0 = time.time()
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        res.raise_for_status()
        latency = time.time() - t0
        raw_text = res.json().get("response", "").strip()
        ranking = parse_ranking(raw_text)[:plan_len]
        actions = [DIRECTION_MAP[d] for d in ranking]
        return actions, ranking, raw_text, latency
    except Exception as e:
        latency = time.time() - t0
        return [], [], f"Error: {e}", latency


def ask_direction(agent_pos, monster_pos, goal_pos, env=None, timeout=60):
    """
    兼容旧接口：只取偏好里最想走的那一步，返回 (action|None, dir_name|raw, latency)。
    """
    actions, names, raw, lat = ask_plan(
        agent_pos, monster_pos, goal_pos, env=env, plan_len=1, timeout=timeout
    )
    if actions:
        return actions[0], names[0], lat
    return None, raw, lat


if __name__ == "__main__":
    # 自检：打印实际发给模型的 prompt 与一次真实调用结果
    from envs.maze import MazeEnv

    env = MazeEnv(with_monster=True, max_steps=200)
    env.reset(seed=8011)
    pos, mon, goal = [5, 4], [8, 2], (10, 10)

    print("智能体 (row, col) =", pos, "| 怪物 =", mon, "| 出口 =", goal)
    cells = get_adjacent_cells(pos, env)
    print("环境判定的四邻格值:", cells)
    print("可通行方向:", get_available_directions(pos, env))
    print("-" * 62)
    print(PLAN_PROMPT_TEMPLATE.format(
        ax=pos[1], ay=pos[0], mx=mon[1], my=mon[0], gx=goal[1], gy=goal[0],
        c_up=cells["上"], c_down=cells["下"],
        c_left=cells["左"], c_right=cells["右"]))
    print("-" * 62)
    acts, names, raw, lat = ask_plan(pos, mon, goal, env=env)
    print("模型原始输出:")
    print(raw)
    print(f"解析出的偏好顺序: {names}  ({lat:.2f}s)")
    legal = get_available_directions(pos, env)
    first_ok = names[0] in legal if names else False
    masked = next((d for d in names if d in legal), None)
    print(f"首选是否合法: {first_ok}   掩码后选择: {masked}")
