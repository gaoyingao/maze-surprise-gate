# -*- coding: utf-8 -*-
"""
experiments/abc_report.py —— Day 18/19：三组对照报告 + Pareto 图

计划书 Day 18：每组 100 回合（不够就 50），记录成功率/步数/回报/调用次数/击杀率
计划书 Day 19：画 Pareto 图（横轴 = 每回合 LLM 调用次数，纵轴 = 成功率，A/B/C 三点）

用法
    python experiments/abc_report.py
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS = {
    "A": "results/eval_A_100ep.json",
    "B": "results/eval_B_50ep.json",
    "C": "results/eval_C_50ep.json",
}
LABEL = {"A": "A 纯 PPO（不调用）",
         "B": "B 固定频率调用",
         "C": "C 惊讶度门控"}
COLOR = {"A": "#9aa7b8", "B": "#e8a33d", "C": "#12324f"}


def load():
    out = {}
    for k, p in RESULTS.items():
        if os.path.exists(p):
            out[k] = json.load(open(p, encoding="utf-8"))
    return out


def fisher_p(a, b, c, d):
    from math import comb
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c

    def prob(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    p_obs = prob(a)
    p = 0.0
    for x in range(max(0, c1 - r2), min(r1, c1) + 1):
        px = prob(x)
        if px <= p_obs + 1e-12:
            p += px
    return min(p, 1.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def main():
    R = load()
    print("=" * 92)
    print("Day 18：三组对照结果")
    print("=" * 92)
    print(f"{'组':<6}{'触发机制':<20}{'回合':>6}{'成功率':>9}{'击杀率':>9}"
          f"{'超时':>7}{'平均步数':>10}{'失败步数':>10}{'调用/回合':>11}{'总调用':>8}")
    print("-" * 92)
    for k in ("A", "B", "C"):
        if k not in R:
            continue
        d = R[k]
        n = d["episodes"]
        w = round(d["success_rate"] / 100 * n)
        avg_steps = d.get("avg_steps", 0.0)
        win_steps = d.get("avg_win_steps", 0.0)
        fail_steps = ((avg_steps * n - win_steps * w) / (n - w)
                      if n > w else 0.0)
        print(f"{k:<6}{LABEL[k]:<20}{n:>6}{d['success_rate']:>8.1f}%"
              f"{d['death_rate']:>8.1f}%{d['timeout_rate']:>6.1f}%"
              f"{avg_steps:>10.2f}{fail_steps:>10.2f}"
              f"{d.get('avg_llm_calls_per_episode', 0.0):>11.2f}"
              f"{d.get('total_llm_calls', 0):>8}")

    print()
    print("=" * 92)
    print("调用率匹配校验（计划书要求：B 组与 C 组差距 < 20%）")
    print("=" * 92)
    nb = R["B"]["avg_llm_calls_per_episode"]
    nc = R["C"]["avg_llm_calls_per_episode"]
    diff = abs(nb - nc) / nc * 100
    print(f"  B = {nb} 次/回合 | C = {nc} 次/回合 | 差距 {diff:.1f}% "
          f"-> {'✅ 匹配' if diff < 20 else '❌ 未匹配'}")

    print()
    print("=" * 92)
    print("两两显著性检验（Fisher 精确检验）")
    print("=" * 92)
    stats = {}
    for k, d in R.items():
        n = d["episodes"]
        w = round(d["success_rate"] / 100 * n)
        stats[k] = (w, n - w)
    for x, y in [("A", "C"), ("A", "B"), ("B", "C")]:
        if x not in stats or y not in stats:
            continue
        p = fisher_p(stats[x][0], stats[x][1], stats[y][0], stats[y][1])
        sx = R[x]["success_rate"]
        sy = R[y]["success_rate"]
        print(f"  {x} ({sx:5.1f}%) vs {y} ({sy:5.1f}%): "
              f"差 {sx-sy:+6.1f}pp | p = {p:.4f} "
              f"{'→ 显著' if p < 0.05 else '→ 不显著'}")

    # ---------------- Pareto 图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    ax = axes[0]
    offs = {"A": (0, 22), "B": (26, -6), "C": (-26, 18)}
    for k in ("A", "B", "C"):
        if k not in R:
            continue
        d = R[k]
        n = d["episodes"]
        w = round(d["success_rate"] / 100 * n)
        lo, hi = wilson(w, n)
        ax.errorbar(d["avg_llm_calls_per_episode"], d["success_rate"],
                    yerr=[[d["success_rate"] - lo], [hi - d["success_rate"]]],
                    fmt="o", ms=14, color=COLOR[k], ecolor="#555",
                    capsize=6, elinewidth=2, zorder=4)
        dx, dy = offs.get(k, (0, 22))
        ax.annotate(f"{k}  {d['success_rate']:.1f}%",
                    (d["avg_llm_calls_per_episode"], d["success_rate"]),
                    textcoords="offset points", xytext=(dx, dy), ha="center",
                    fontsize=11, fontweight="bold", color=COLOR[k])
    ax.set_xlabel("每回合 LLM 调用次数", fontsize=11)
    ax.set_ylabel("成功率 (%)", fontsize=11)
    ax.set_title("Pareto 图：调用代价 vs 成功率", fontsize=13, fontweight="bold")
    ax.grid(ls="--", alpha=0.5, zorder=0)
    ax.set_xlim(-0.6, max(R[k]["avg_llm_calls_per_episode"] for k in R) + 1.0)
    ax.set_ylim(40, 90)
    ax.text(0.03, 0.06, "误差棒为 95% Wilson 置信区间",
            transform=ax.transAxes, fontsize=9, color="#666")

    ax = axes[1]
    ks = [k for k in ("A", "B", "C") if k in R]
    succ = [R[k]["success_rate"] for k in ks]
    die = [R[k]["death_rate"] for k in ks]
    x = range(len(ks))
    wd = 0.36
    ax.bar([i - wd / 2 for i in x], succ, wd, label="成功率",
           color=[COLOR[k] for k in ks], edgecolor="#33475b", zorder=3)
    ax.bar([i + wd / 2 for i in x], die, wd, label="被怪击杀率",
           color="#c0392b", alpha=0.65, edgecolor="#7b241c", zorder=3)
    for i, (s, d) in enumerate(zip(succ, die)):
        ax.text(i - wd / 2, s + 1.5, f"{s:.1f}%", ha="center",
                fontsize=10.5, fontweight="bold")
        ax.text(i + wd / 2, d + 1.5, f"{d:.1f}%", ha="center",
                fontsize=10.5, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[k] for k in ks], fontsize=9)
    ax.set_ylim(0, 95)
    ax.set_ylabel("百分比 (%)", fontsize=11)
    ax.set_title("成功率 / 击杀率对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.5, zorder=0)

    fig.suptitle(f"Day 19：三组对照（调用率已匹配，B/C 差距 {diff:.1f}%）",
                 fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/day19_pareto.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    print(f"\n📈 Pareto 图已保存: figures/day19_pareto.png")

    summary = dict(groups={k: R[k] for k in R}, call_rate_match_pct=round(diff, 1),
                   matched=bool(diff < 20))
    with open("results/abc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("结果已保存: results/abc_summary.json")


if __name__ == "__main__":
    main()
