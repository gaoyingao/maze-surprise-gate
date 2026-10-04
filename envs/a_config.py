# -*- coding: utf-8 -*-
"""
envs/a_config.py —— A 组基线环境的唯一配置源

为什么要有这个文件
    环境参数散落在各个脚本里会导致「三组对照用了不同环境」这种致命错误。
    所有评估脚本（evaluate.py / evaluate_gated.py / day14_check.py /
    day16_test_loop.py）都必须从这里取配置。

背景（Day 20 威胁度标定）
    旧 A 组策略存在退化：无怪环境 0%（在 (10,2) 原地撞墙 188 次超时），
    有怪 90% —— 它在用怪物位置当导航信号，一旦没有怪物就不会走路。

    根因有两个，都已修复：
      1. 旧版 with_monster=False 只是把怪物位置置零，怪物仍冻在 patrol_path[0]，
         使「无怪」变成一种分布外状态。已改为把怪物移出地图（[-1,-1]）。
      2. 直接在带怪环境训练会学到依赖怪物信号的策略。
         已改为课程学习：先无怪预训练 50 万步，再带怪续训 50 万步。

    修复后的标定结果（同一策略跨环境，各 3 个种子 × 100 回合）：
        配置              seed42  seed777  seed2024   均值    最小
        baseline(现状)     22pp    11pp     15pp     16.0pp   11.0pp  ❌
        medium(本配置)     29pp    21pp     21pp     23.7pp   21.0pp  ✅
    计划书门槛：≥ 20pp。选定 medium。
"""

# A 组（带怪基线）环境配置 —— 直接展开给 MazeEnv(**A_ENV_KWARGS)
A_ENV_KWARGS = dict(
    chase_radius=4,     # 进入 4 格内触发追击（原 3）
    chase_prob=0.10,    # 每步 10% 随机触发（原 5%）
    chase_len=6,        # 一次追击持续 6 步（原 5）
)

# A 组策略权重
A_MODEL_PATH = "models/threat_medium_with_monster.zip"

# 评估口径（必须与 evaluate.py 一致）
A_MAX_STEPS = 200
A_BASE_SEED = 42

# 威胁度标定结果（供报告引用）
THREAT_MARGIN = dict(
    config="medium",
    no_monster_win=100.0,
    with_monster_win=76.3,
    margin_pp=23.7,
    min_margin_pp=21.0,
    seeds=[42, 777, 2024],
    episodes_per_seed=100,
    threshold_pp=20.0,
    passed=True,
)
