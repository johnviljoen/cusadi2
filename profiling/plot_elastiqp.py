"""Bar chart of the elastiqp fp64 CPU batch-1 pieces per problem -> results/elastiqp_f64_cpu.png"""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "results"
rows = [r for r in csv.DictReader(open(R / "raw_timings.csv"))
        if r["backend"] == "elastiqp" and r["dtype"] == "f64" and r["platform"] == "cpu" and r["batch"] == "1"]
pieces = [("nominal", "nominal OSC controller", "model"),
          ("dyn_terms", "2a dynamics: M, M⁻¹, bias, J", "model"),
          ("h2_only", "raw barriers h₂ (no derivatives)", "model"),
          ("barriers", "2b barriers + Lie derivatives (h, Lfh, Lgh)", "model"),
          ("qp_data", "cbf.qp_data (P, q, G, h)", "model"),
          ("solve", "QP solve (elastiqp, dual active set)", "solve"),
          ("end_to_end", "end to end: nominal + safety_filter", "total")]
col = {"model": "#2a78d6", "solve": "#eb6834", "total": "#898781"}
probs = [p for p in ("panda", "g1") if any(r["problem"] == p for r in rows)]
fig, axes = plt.subplots(1, len(probs), figsize=(6.8 * len(probs), 4.4), facecolor="#fcfcfb")
if len(probs) == 1: axes = [axes]
for ax, p in zip(axes, probs):
    t = {r["piece"]: r for r in rows if r["problem"] == p}
    labels, med, p99, colors = [], [], [], []
    for key, lab, kind in pieces:
        if key not in t: continue
        labels.append(lab); med.append(float(t[key]["median_us"])); p99.append(float(t[key]["p99_us"])); colors.append(col[kind])
    y = range(len(labels))[::-1]
    ax.set_facecolor("#fcfcfb")
    ax.barh(list(y), med, color=colors, height=0.55)
    ax.scatter(p99, list(y), marker="|", s=120, color="#0b0b0b", zorder=3, linewidths=1.2)
    for yi, m, q in zip(y, med, p99):
        ax.text(max(m, q) + 0.02 * max(p99), yi, f"{m:,.0f} µs   (p99 {q:,.0f})", va="center", fontsize=9, color="#52514e")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9, color="#0b0b0b")
    ax.set_xlim(0, max(p99) * 1.45)
    ax.set_xlabel("time per call, batch 1, CPU, fp64 (µs)", color="#52514e", fontsize=9)
    nv = {"panda": "Panda, 7 DOF, 78 QP rows", "g1": "G1 fixed root, 29 DOF, 213 QP rows"}[p]
    e2e, sol = float(t["end_to_end"]["median_us"]), float(t["solve"]["median_us"])
    ax.set_title(f"{nv}\nsolve {sol/e2e:.0%} of the tick · model side (e2e − solve) {(e2e-sol)/e2e:.0%}", fontsize=10, loc="left", color="#0b0b0b")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="x", colors="#898781", labelsize=8); ax.grid(axis="x", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=col["model"], label="frax model evaluation"), Patch(color=col["solve"], label="QP solve"),
                    Patch(color=col["total"], label="end to end"), plt.Line2D([], [], marker="|", ls="", color="#0b0b0b", label="p99")],
           loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("elastiqp backend (as published): where a safety-filter tick goes — median over 1000 calls", fontsize=11, x=0.01, ha="left", color="#0b0b0b")
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
out = R / "elastiqp_f64_cpu.png"; fig.savefig(out, dpi=160); print(out)
