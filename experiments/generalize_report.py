# -*- coding: utf-8 -*-
"""
experiments/generalize_report.py —— 泛化实验汇总与出图

读取 results/generalize/summary.json（由 experiments/generalize.py 产出）
以及 results/generalize/llm_geom_*.json（由 experiments/llm_geom_generalize.py 产出），
输出跨地图对比表、结论判定和图表。

用法
    python experiments/generalize_report.py
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

OUT = "results/generalize"


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


def load_summary():
    p = f"{OUT}/summary.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []


def load_llm():
    out = []
    if not os.path.isdir(OUT):
        return out
    for fn in os.listdir(OUT):
        if fn.startswith("llm_geom_") and fn.endswith(".json"):
            out.append(json.load(open(os.path.join(OUT, fn), encoding="utf-8")))
    return out


def main():
    R = load_summary()
    if not R:
        print(f"未找到 {OUT}/summary.json，请先运行 experiments/generalize.py")
        return

    print("=" * 96)
    print(f"泛化实验汇总 —— {len(R)} 张地图")
    print("=" * 96)
    print(f"{'地图':<14}{'规模':>7}{'威胁pp':>9}{'A':>8}{'B':>8}{'C':>8}"
          f"{'B调用':>8}{'C调用':>8}{'匹配%':>8}{'排序':>12}")
    print("-" * 96)

    for r in R:
        g = r["groups"]
        mp = r["map"]
        size = f"{mp.get('size','12')}x{mp.get('size','12')}"
        mc = r.get("call_match", {}).get("B_vs_C_calls_pct", float("nan"))
        a, b, c = (g[k]["success_rate"] for k in "ABC")
        order = " > ".join(sorted("ABC", key=lambda k: -g[k]["success_rate"]))
        print(f"{r['name']:<14}{size:>7}{r['threat']['margin_pp']:>9.1f}"
              f"{a:>7.1f}%{b:>7.1f}%{c:>7.1f}%"
              f"{g['B']['avg_llm_calls_per_episode']:>8.2f}"
              f"{g['C']['avg_llm_calls_per_episode']:>8.2f}"
              f"{mc:>8.1f}{order:>12}")

    print("-" * 96)

    # ---- 逐地图显著性 ----
    print()
    print("=" * 96)
    print("逐地图显著性检验（Fisher 精确检验）")
    print("=" * 96)
    print(f"{'地图':<14}{'A vs C':>22}{'A vs B':>22}{'B vs C':>22}")
    print("-" * 96)
    detail = []
    for r in R:
        g = r["groups"]
        n = {k: g[k]["episodes"] for k in "ABC"}
        w = {k: round(g[k]["success_rate"] / 100 * n[k]) for k in "ABC"}
        line = f"{r['name']:<14}"
        row = {}
        for x, y in [("A", "C"), ("A", "B"), ("B", "C")]:
            p = fisher_p(w[x], n[x] - w[x], w[y], n[y] - w[y])
            d = g[x]["success_rate"] - g[y]["success_rate"]
            star = "*" if p < 0.05 else " "
            line += f"{d:>+8.1f}pp p={p:.3f}{star:<8}"
            row[f"{x}{y}"] = dict(diff=d, p=p, sig=bool(p < 0.05))
        print(line)
        detail.append(dict(name=r["name"], tests=row))

    # ---- 结论判定 ----
    print()
    print("=" * 96)
    print("结论判定：原图上的三条结论是否在新地图上复现")
    print("=" * 96)
    c1 = c2 = c3 = 0
    for r in R:
        g = r["groups"]
        a, b, c = (g[k]["success_rate"] for k in "ABC")
        if a >= b:
            c1 += 1
        if c >= b:
            c2 += 1
        if r["threat"]["passed"]:
            c3 += 1
    n = len(R)
    print(f"  ① A >= B（固定频率不优于不调用）      : {c1}/{n} 张地图成立")
    print(f"  ② C >= B（门控不劣于固定频率）        : {c2}/{n} 张地图成立")
    print(f"  ③ 威胁度 >= 20pp                     : {c3}/{n} 张地图达标")

    # ---- LLM 几何 ----
    L = load_llm()
    if L:
        print()
        print("=" * 96)
        print("LLM 几何能力的跨地图检验（层 = 该格可走方向数）")
        print("=" * 96)
        print(f"{'地图':<14}{'规模':>8}{'空格':>6}{'1出口':>10}{'2出口':>10}"
              f"{'3出口':>10}{'4出口':>10}{'判决':>10}")
        print("-" * 96)
        for d in L:
            sz = f"{d['size'][0]}x{d['size'][1]}"
            byk = {r["layer"]: r for r in d["layers"]}
            def gv(k):
                r = byk.get(k)
                return f"{r['legal_pct']:.0f}%" if r else "-"
            v = "真几何" if "真几何" in d["verdict"] else (
                "模式补全" if "模式补全" in d["verdict"] else "混合")
            print(f"{d['map']:<14}{sz:>8}{d['free_cells']:>6}"
                  f"{gv(1):>10}{gv(2):>10}{gv(3):>10}{gv(4):>10}{v:>10}")
        print("-" * 96)
        print("  判决口径：死角层(1出口)>=85% 判为真几何；<60% 且十字层>90% 判为模式补全")
        print("  随机基线：1出口=25%  2出口=50%  3出口=75%  4出口=100%")

    # ---- 出图 ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    names = [r["name"] for r in R]
    x = range(len(R))
    wd = 0.26
    COLOR = {"A": "#9aa7b8", "B": "#e8a33d", "C": "#12324f"}

    ax = axes[0]
    for i, k in enumerate("ABC"):
        vals = [r["groups"][k]["success_rate"] for r in R]
        ax.bar([j + (i - 1) * wd for j in x], vals, wd,
               label={"A": "A 纯 PPO", "B": "B 固定频率",
                      "C": "C 惊讶度门控"}[k],
               color=COLOR[k], edgecolor="#33475b", zorder=3)
        for j, v in zip(x, vals):
            ax.text(j + (i - 1) * wd, v + 1.2, f"{v:.0f}", ha="center",
                    fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("成功率 (%)")
    ax.set_title("A/B/C 三组在各地图上的成功率", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.5, zorder=0)
    ax.set_ylim(0, 105)

    ax = axes[1]
    for r in R:
        g = r["groups"]
        ax.plot([g["B"]["avg_llm_calls_per_episode"],
                 g["C"]["avg_llm_calls_per_episode"]],
                [g["B"]["success_rate"], g["C"]["success_rate"]],
                "o-", color="#555", alpha=0.6, zorder=2)
    for i, r in enumerate(R):
        g = r["groups"]
        ax.scatter([g["A"]["avg_llm_calls_per_episode"]],
                   [g["A"]["success_rate"]], s=90, color=COLOR["A"], zorder=4)
        ax.scatter([g["B"]["avg_llm_calls_per_episode"]],
                   [g["B"]["success_rate"]], s=90, color=COLOR["B"], zorder=4)
        ax.scatter([g["C"]["avg_llm_calls_per_episode"]],
                   [g["C"]["success_rate"]], s=90, color=COLOR["C"], zorder=4)
        ax.annotate(r["name"], (g["C"]["avg_llm_calls_per_episode"],
                                g["C"]["success_rate"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("每回合 LLM 调用次数")
    ax.set_ylabel("成功率 (%)")
    ax.set_title("Pareto：调用代价 vs 成功率（灰线连接同图的 B-C）",
                 fontsize=13, fontweight="bold")
    ax.grid(ls="--", alpha=0.5, zorder=0)

    fig.suptitle(f"泛化实验：{len(R)} 张地图上的三组对照",
                 fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/generalize_results.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    print(f"\n📈 图已保存: figures/generalize_results.png")

    with open(f"{OUT}/report.json", "w", encoding="utf-8") as f:
        json.dump(dict(summary=R, significance=detail, llm_geometry=L,
                       conclusion=dict(A_ge_B=f"{c1}/{n}", C_ge_B=f"{c2}/{n}",
                                       threat_pass=f"{c3}/{n}")),
                  f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {OUT}/report.json")


if __name__ == "__main__":
    main()
