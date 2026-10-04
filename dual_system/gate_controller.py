import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction.train_predictor import MonsterPredictor
from prediction.collect_data import OFFSET_TO_ACTION
from llm.advisor import ask_plan, DIR_NAMES, ACTION_OFFSETS, DIRECTION_MAP, PLAN_LEN


class SurprisalGateController:
    """
    Day 16 双系统接管门控状态机（偏好排序 + 环境掩码版）：
    - GRU 预测怪物行为与惊讶度计算
    - 危机状态唤醒 LLM，取回一张「方向偏好排序」
    - 环境按偏好顺序取第一个合法方向执行（掩码），几何正确性由环境保证
    - 5 步冷却期保护，随后控制权交还底层 PPO

    为什么不让 LLM 自己判断合法性：
      本项目 36 次平行试验显示，在位置 / 四邻格值 / 怪物位置 / 随机种子
      全部变化的条件下，让模型判断"哪个方向是墙"时它只输出同一个字符串
      （输出与输入无关）；把视野从 3x3 扩到 7x7 输出也一字未变。
      改为只让模型输出"更想往哪走"的顺序后，输出才开始随局面变化。

    实验指标说明：
      llm_first_choice_illegal —— 模型首选方向在物理上走不通的次数。
        这是衡量「LLM 战术建议可执行性」的核心指标，比"砍掉几个非法步"更有意义。
      llm_masked_fallbacks     —— 首选非法、被掩码顺延到次选方向的次数。
    """

    def __init__(
        self,
        predictor_path="prediction/monster_predictor.pth",
        env=None,
        tau=1.5,
        max_ranking=PLAN_LEN,
        cooldown_steps=5,
        history_len=4,
        device="cpu",
    ):
        self.tau = tau
        self.max_ranking = max_ranking
        self.cooldown_duration = cooldown_steps
        self.history_len = history_len
        self.env = env
        self.device = torch.device(device)

        self.predictor = MonsterPredictor(
            input_dim=2, hidden_dim=128, num_classes=5
        ).to(self.device)
        self.predictor.load_state_dict(
            torch.load(predictor_path, map_location=self.device)
        )
        self.predictor.eval()

        self.monster_history = []
        self.ranking = []            # 本次唤醒取回的偏好序列（方向名）
        self.cooldown_counter = 0

        # ---- 实验指标（报告里要用的）----
        self.total_llm_calls = 0            # LLM 实际被唤醒次数
        self.llm_parse_failures = 0         # 没解析出任何方向
        self.llm_no_legal_choice = 0        # 偏好里没有任何合法方向（理论上不该发生）
        self.llm_first_choice_illegal = 0   # 首选方向走不通
        self.llm_masked_fallbacks = 0       # 因首选非法而顺延到次选

    # ------------------------------------------------------------------ #
    def reset(self):
        self.monster_history.clear()
        self.ranking = []
        self.cooldown_counter = 0
        self.total_llm_calls = 0
        self.llm_parse_failures = 0
        self.llm_no_legal_choice = 0
        self.llm_first_choice_illegal = 0
        self.llm_masked_fallbacks = 0

    # ------------------------------------------------------------------ #
    def compute_surprisal(self, prev_monster_pos, curr_monster_pos):
        """s = -ln P(怪物本步动作 | 怪物自身历史)，单位 nats"""
        dr = int(curr_monster_pos[0] - prev_monster_pos[0])
        dc = int(curr_monster_pos[1] - prev_monster_pos[1])
        true_act = OFFSET_TO_ACTION.get((dr, dc), 4)

        if len(self.monster_history) < self.history_len:
            return 0.0

        seq = torch.tensor(
            [self.monster_history[-self.history_len:]], dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            logits = self.predictor(seq)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        p_actual = max(float(probs[true_act]), 1e-4)
        return float(-np.log(p_actual))

    def update_history(self, monster_pos):
        self.monster_history.append(
            [float(monster_pos[0]) / 11.0, float(monster_pos[1]) / 11.0]
        )

    # ------------------------------------------------------------------ #
    def _is_legal(self, agent_pos, direction_name):
        """几何合法性完全由环境判定"""
        if self.env is None:
            return True
        dr, dc = ACTION_OFFSETS[DIR_NAMES.index(direction_name)]
        nr = int(agent_pos[0]) + dr
        nc = int(agent_pos[1]) + dc
        return self.env.game.is_valid((nr, nc))

    def mask_select(self, ranking, agent_pos):
        """
        掩码选择：按偏好顺序取第一个合法方向。
        返回 (方向名 | None, 首选是否合法)
        """
        if not ranking:
            return None, False
        first_legal = self._is_legal(agent_pos, ranking[0])
        for d in ranking:
            if self._is_legal(agent_pos, d):
                return d, first_legal
        return None, first_legal

    # ------------------------------------------------------------------ #
    def decide(self, llm_pick_fn, ppo_action, agent_pos, monster_pos, goal_pos,
               env, surprisal, log=None):
        """
        供 experiments/run.py 使用的统一接口，与 B 组的调用路径完全一致。

        返回 (动作编号 | None, 是否被掩码顺延, 首选是否可行, 偏好序列)
        None 表示本次不接管，调用方应继续用 PPO 动作。

        只做门控判定（惊讶度阈值 + 冷却），LLM 调用与掩码交由 llm_pick_fn，
        这样 C 组与 B 组走的是同一段 LLM 代码，唯一差别就是"何时调用"。
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter = max(0, self.cooldown_counter - 1)

        if surprisal > self.tau and self.cooldown_counter == 0:
            self.total_llm_calls += 1
            self.cooldown_counter = self.cooldown_duration
            act, masked, first_ok, ranking = llm_pick_fn(
                None, env, agent_pos, monster_pos, goal_pos, log)
            if act is None:
                return None, False, first_ok, ranking
            return act, masked, first_ok, ranking

        return None, False, True, []

    # ------------------------------------------------------------------ #
    def decide_action(self, ppo_action, agent_pos, monster_pos, goal_pos, surprisal):
        """
        返回 (最终动作编号, 决策来源字符串)

        注意：返回的是 int 动作编号（与环境 action_space 一致），
        不是方向名 —— 调用方会直接把它喂给 env.step()。
        """

        # 1) 推进冷却（不会变成负数）
        if self.cooldown_counter > 0:
            self.cooldown_counter = max(0, self.cooldown_counter - 1)

        # 2) 门控判定：惊讶度突变且非冷却状态，唤醒 LLM
        if surprisal > self.tau and self.cooldown_counter == 0:
            actions, ranking, raw, lat = ask_plan(
                agent_pos=agent_pos,
                monster_pos=monster_pos,
                goal_pos=goal_pos,
                env=self.env,
                plan_len=self.max_ranking,
            )
            self.total_llm_calls += 1
            self.cooldown_counter = self.cooldown_duration

            if not ranking:
                self.llm_parse_failures += 1
                print(f"    [LLM] 解析失败，原始回复={raw!r} ({lat:.2f}s)")
                return ppo_action, "PPO(LLM无效)"

            chosen, first_legal = self.mask_select(ranking, agent_pos)
            if not first_legal:
                self.llm_first_choice_illegal += 1

            if chosen is None:
                self.llm_no_legal_choice += 1
                print(f"    [LLM] 偏好 {ranking} 中无合法方向，交还 PPO")
                return ppo_action, "PPO(无合法)⛔"

            if not first_legal:
                self.llm_masked_fallbacks += 1
                print(f"    [LLM] 偏好={'、'.join(ranking)} -> 首选「{ranking[0]}」是墙，"
                      f"掩码顺延到「{chosen}」")
                return DIRECTION_MAP[chosen], f"LLM_Masked({chosen}, {lat:.2f}s)"

            return DIRECTION_MAP[chosen], f"LLM_Pick({chosen}, {lat:.2f}s)"

        return ppo_action, "PPO"
