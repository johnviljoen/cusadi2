"""Render results/cusadi2_swap.{md,png} from results/cusadi2_swap.csv (last row per (batch, piece) wins)."""
import csv, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; R = HERE / "results"
rows = list(csv.DictReader(open(R / "cusadi2_swap.csv")))
for r in rows:
    for k in ("median_us", "p99_us", "compile_s", "max_abs_du_vs_frax"): r[k] = float(r[k])
    r["batch"] = int(r["batch"])
med = {(r["batch"], r["piece"]): r["median_us"] for r in rows}
batches = sorted({r["batch"] for r in rows})
variants = sorted({r["piece"][len("model_cusadi2"):] for r in rows if r["piece"].startswith("model_cusadi2")})
pieces = ["model_frax"] + ["model_cusadi2" + v for v in variants] + ["solve", "e2e_frax"] + ["e2e_cusadi2" + v for v in variants]
def label(v):
    ml = v.split("_ml")[1] if "_ml" in v else None
    return ("serial" if "serial" in v else "level") + (f", max_live {ml}" if ml else "")
stats = {}
for v in variants:
    ml = v.split("_ml")[1] if "_ml" in v else ""
    f = HERE / "build/double" / (f"oscbf_panda_ml{ml}_stats.json" if ml else "plan_stats.json")
    if f.exists(): stats[v] = json.loads(f.read_text())
def g(B, k): return med.get((B, k), float("nan"))
hdr = "| B | model frax | " + "".join(f"model cusadi2 ({label(v)}) | " for v in variants) + "solve | e2e frax | " + "".join(f"e2e cusadi2 ({label(v)}) | " for v in variants) + "best cusadi2 model speed-up | best cusadi2 e2e speed-up | model share frax | model share cusadi2 (best) | max\\|Δu\\| |"
lines = [hdr, "|" + "---:|" * (hdr.count("|") - 1)]
for B in batches:
    mf, s, ef = g(B, "model_frax"), g(B, "solve"), g(B, "e2e_frax")
    mc = [g(B, "model_cusadi2" + v) for v in variants]; ec = [g(B, "e2e_cusadi2" + v) for v in variants]
    bm = min(x for x in mc if x == x); be = min(x for x in ec if x == x)
    du = max(r["max_abs_du_vs_frax"] for r in rows if r["batch"] == B)
    cells = [f"{B}", f"{mf:,.0f}"] + [f"{x:,.0f}" for x in mc] + [f"{s:,.0f}", f"{ef:,.0f}"] + [f"{x:,.0f}" for x in ec] + [f"{mf/bm:.2f}×", f"{ef/be:.2f}×", f"{(ef-s)/ef:.0%}", f"{(be-s)/be:.0%}", f"{du:.1e}"]
    lines.append("| " + " | ".join(cells) + " |")
md = ["# Panda OSC + CBF on GPU (qpax, fp64): frax vs cusadi2 as the model-evaluation backend", "",
      "Times in µs per batched call, median of 1000 calls (B ≤ 128) or 200 calls, jit + vmap, device-resident inputs, block_until_ready. "
      "`model` = nominal OSC torque + cbfpy's (P, q, G, h); `solve` = qpax.solve_qp_elastic_primal on precomputed data (identical for both backends); "
      "`e2e` = model + solve in one jit. Speed-up > 1 means cusadi2 is faster. `max|Δu|` is the largest difference in the safe torque between the two backends in the first batch window.", "",
      "cusadi2 side: `panda_casadi.py` (Pinocchio-casadi on frax's panda.urdf; cbfpy's barrier / Lie-derivative algebra reproduced with forward-mode directional derivatives, `ca.jtimes`, as cbfpy does with `jax.jvp`, then `ca.cse`; "
      "verified to ≤1e-9 relative against frax + cbfpy on 64 random states). It runs inside the JAX program as one XLA FFI custom call on XLA's stream (`cusadi2_ffi.cu`); "
      "the dense P and G are scattered from cusadi2's column-major nonzeros inside the same jit. The bare kernel timed with CUDA events equals the jit time at every batch, so the FFI path adds no copies. "
      "`max_live` is cusadi2's live-slot budget (shared memory per block = 8 B × peak live slots); a smaller budget windows the plan into more levels but lets more blocks share an SM.", "",
      "| cusadi2 variant | SX nodes | levels | peak live slots | shared memory / block |", "|---|---:|---:|---:|---:|"] + [
      f"| {label(v)} | {st['sx_nodes']} | {st['n_levels']} | {st['peak_live_slots']} | {st['peak_live_slots']*8/1024:.0f} KB |" for v, st in stats.items()] + [""] + lines
(R / "cusadi2_swap.md").write_text("\n".join(md) + "\n"); print("\n".join(lines))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor="#fcfcfb")
palette = ["#1a9e5c", "#8a4fd3", "#c2185b", "#00838f"]
col = {"model_frax": "#2a78d6", "solve": "#eb6834", "e2e_frax": "#2a78d6"}
for i, v in enumerate(variants): col["model_cusadi2" + v] = palette[i % 4]; col["e2e_cusadi2" + v] = palette[i % 4]
sty = {k: ("--" if k.startswith("e2e") else (":" if k == "solve" else "-")) for k in col}
nice = {k: (k if "cusadi2" not in k else k.split("_cusadi2")[0] + f" cusadi2 ({label(k[len(k.split('_cusadi2')[0]) + len('_cusadi2'):])})") for k in pieces}
for k in pieces:
    ax[0].plot(batches, [g(B, k) / 1e3 for B in batches], sty[k], marker="o", ms=4, color=col[k], label=nice[k])
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].set_xlabel("batch size"); ax[0].set_ylabel("ms per batched call (median)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both"); ax[0].set_title("time per call", loc="left", fontsize=10)
for v in variants:
    ax[1].plot(batches, [g(B, "model_frax") / g(B, "model_cusadi2" + v) for B in batches], "-o", ms=4, color=col["model_cusadi2" + v], label=f"model: frax ÷ cusadi2 ({label(v)})")
    ax[1].plot(batches, [g(B, "e2e_frax") / g(B, "e2e_cusadi2" + v) for B in batches], "--o", ms=4, color=col["e2e_cusadi2" + v], label=f"e2e: frax ÷ cusadi2 ({label(v)})")
ax[1].axhline(1, color="k", lw=0.8); ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_xlabel("batch size"); ax[1].set_ylabel("frax time ÷ cusadi2 time (>1: cusadi2 faster)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both"); ax[1].set_title("speed-up of cusadi2 over frax", loc="left", fontsize=10)
fig.suptitle("Panda OSC + CBF safety filter, GPU, qpax fp64: frax vs cusadi2 model evaluation", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(R / "cusadi2_swap.png", dpi=150); print(R / "cusadi2_swap.png")
