"""Bar chart of the GPU fp64 (qpax) configurations with the largest model-side time
-> results/qpax_f64_gpu_model_max.png

Left panel: largest absolute model-side time (end_to_end - solve) over all GPU fp64 rows.
Right panel: largest model-side share of the tick.
"""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = Path(__file__).resolve().parent / "results"
rows = [r for r in csv.DictReader(open(R / "raw_timings.csv"))
        if r["platform"] == "gpu" and r["dtype"] == "f64"]
cfg = defaultdict(dict)
for r in rows:
    cfg[(r["problem"], int(r["batch"]))][r["piece"]] = r
model = {k: float(v["end_to_end"]["median_us"]) - float(v["solve"]["median_us"]) for k, v in cfg.items()}
share = {k: model[k] / float(v["end_to_end"]["median_us"]) for k, v in cfg.items()}
picks = [(max(model, key=model.get), "largest model-side time"), (max(share, key=share.get), "largest model-side share")]

pieces = [("nominal", "nominal OSC controller", "model"),
          ("dyn_terms", "2a dynamics: M, M⁻¹, bias, J", "model"),
          ("h2_only", "raw barriers h₂ (no derivatives)", "model"),
          ("barriers", "2b barriers + Lie derivatives (h, Lfh, Lgh)", "model"),
          ("qp_data", "cbf.qp_data (P, q, G, h)", "model"),
          ("solve", "QP solve (qpax, elastic PDIP, batch max iters)", "solve"),
          ("end_to_end", "end to end: nominal + safety_filter", "total")]
col = {"model": "#2a78d6", "solve": "#eb6834", "total": "#898781"}
name = {"panda": "Panda, 7 DOF, 78 QP rows", "g1": "G1 fixed root, 29 DOF, 213 QP rows"}

fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), facecolor="#fcfcfb")
for ax, ((prob, B), why) in zip(axes, picks):
    t = cfg[(prob, B)]
    labels, med, p99, colors = [], [], [], []
    for key, lab, kind in pieces:
        labels.append(lab); med.append(float(t[key]["median_us"]) / 1e3); p99.append(float(t[key]["p99_us"]) / 1e3); colors.append(col[kind])
    y = list(range(len(labels)))[::-1]
    ax.set_facecolor("#fcfcfb")
    ax.barh(y, med, color=colors, height=0.55)
    ax.scatter(p99, y, marker="|", s=120, color="#0b0b0b", zorder=3, linewidths=1.2)
    for yi, m, q in zip(y, med, p99):
        ax.text(max(m, q) + 0.02 * max(p99), yi, f"{m:,.1f} ms   (p99 {q:,.1f})", va="center", fontsize=9, color="#52514e")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9, color="#0b0b0b")
    ax.set_xlim(0, max(p99) * 1.45)
    n_calls = t["end_to_end"]["n_calls"]
    ax.set_xlabel(f"time per vmapped call of batch {B}, GPU, fp64 (ms) — median over {n_calls} calls", color="#52514e", fontsize=9)
    e2e, sol = float(t["end_to_end"]["median_us"]), float(t["solve"]["median_us"])
    m = e2e - sol
    ax.set_title(f"{why}: {name[prob]}, batch {B}\n"
                 f"model side (e2e − solve) {m/1e3:,.1f} ms = {m/e2e:.0%} of the tick, {m/B:,.1f} µs per instance · solve {sol/e2e:.0%}",
                 fontsize=10, loc="left", color="#0b0b0b")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="x", colors="#898781", labelsize=8); ax.grid(axis="x", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True)
fig.legend(handles=[Patch(color=col["model"], label="frax model evaluation"), Patch(color=col["solve"], label="QP solve"),
                    Patch(color=col["total"], label="end to end"), plt.Line2D([], [], marker="|", ls="", color="#0b0b0b", label="p99")],
           loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("qpax backend on the GPU, fp64: the two configurations where frax model evaluation costs most", fontsize=11, x=0.01, ha="left", color="#0b0b0b")
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
out = R / "qpax_f64_gpu_model_max.png"; fig.savefig(out, dpi=160); print(out)
for (prob, B), why in picks:
    print(why, prob, B, f"model {model[(prob,B)]/1e3:.1f} ms share {share[(prob,B)]:.1%}")
