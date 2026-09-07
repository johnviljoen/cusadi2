"""Merge the MPC benchmark CSVs into results/mpc_latency.csv and results/mpc_latency.png.

usage: python collect.py <staged_dir> <mono_dir> [cpu_csv]
  staged_dir: holds sweep_double.csv, sweep_float.csv, probe_double.csv (mpc_cudss_bench output)
  mono_dir:   holds mono_*.csv from mpc_mono_bench (mode cudss = the 'cusadi' baseline; KEEP_LDL=1 also lists mode c2)
"""
import csv, glob, math, os, sys
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except ImportError: plt = None
def read_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try: r[k] = float(v)
            except (ValueError, TypeError): pass
    return rows

KEEP_LDL = os.environ.get("KEEP_LDL", "0") == "1"      # also list the wip's symbolic-LDL monolithic rows
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
staged_dir, mono_dir = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(os.path.join(staged_dir, "*.csv")):
    for r in read_csv(f):
        if "asm_ruiz_us" not in r: continue
        rows.append(dict(dtype=r["dtype"], system="cusadi2", launch=r["mode"], batch=int(r["batch"]),
                         assembly_us=r["asm_ruiz_us"], factor_us=r["factor_us"], iters_us=r["iters_us"], total_us=r["total_us"],
                         wall_us=r["wall_us"], max_abs_err=r["max_abs_err_ref"]))
for f in glob.glob(os.path.join(mono_dir, "mpc_solve_c2_*.log")) + glob.glob(os.path.join(mono_dir, "mono_*.csv")):
    for r in read_csv(f):
        if "kkt_us" not in r or r["mode"] not in ("cudss", "c2"): continue
        name = "cusadi" if r["mode"] == "cudss" else "cusadi (symbolic LDL)"
        if r["mode"] == "c2" and not KEEP_LDL: continue
        for launch, tot in (("eager", r["eager_total_us"]), ("graph", r["graph_total_us"])):
            rows.append(dict(dtype=r["dtype"], system=name, launch=launch, batch=int(r["batch"]),
                             assembly_us=r["kkt_us"] + r["ruiz_us"], factor_us=r["fac_us"], iters_us=r["iters_us"], total_us=tot,
                             wall_us=float("nan"), max_abs_err=r["max_abs_err"]))
if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
    d = read_csv(sys.argv[3])[0]
    rows.append(dict(dtype="f64", system="CPU (CasADi C codegen, 1 core)", launch="cpu", batch=1, assembly_us=d["kkt_us"] + d["ruiz_us"],
                     factor_us=d["fac_us"], iters_us=d["iters_us"], total_us=d["cpu_total_us"], wall_us=d["cpu_total_us"], max_abs_err=d["max_abs_err"]))
seen, uniq = set(), []
for r in sorted(rows, key=lambda r: (r["dtype"], r["system"], r["launch"], r["batch"])):
    key = (r["dtype"], r["system"], r["launch"], r["batch"])
    if key not in seen: seen.add(key); uniq.append(r)
rows = uniq
cols = ["dtype", "system", "launch", "batch", "assembly_us", "factor_us", "iters_us", "total_us", "wall_us", "max_abs_err"]
with open(os.path.join(RES, "mpc_latency.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
fmt = lambda v: (f"{v:.4g}" if isinstance(v, float) else str(v))
print(" ".join(f"{c:>12s}" if c not in ("system",) else f"{c:34s}" for c in cols))
for r in rows: print(" ".join(f"{fmt(r[c]):>12s}" if c != "system" else f"{r[c]:34s}" for c in cols))

if plt is None: sys.exit(0)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, dt in zip(axes, ("f64", "f32")):
    sub = [r for r in rows if r["dtype"] == dt and r["launch"] == "graph" and not math.isnan(r["total_us"])]
    for name in sorted(set(r["system"] for r in sub)):
        g = sorted([r for r in sub if r["system"] == name], key=lambda r: r["batch"])
        ax.plot([r["batch"] for r in g], [r["total_us"] / 1e3 for r in g], "o-", label=name)
    per = sorted([r for r in sub if r["system"] == "cusadi2"], key=lambda r: r["batch"])
    ax.plot([r["batch"] for r in per], [r["total_us"] / 1e3 / r["batch"] for r in per], "s--", color="gray", label="cusadi2, per QP")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("batch size B (QPs)"); ax.set_title(f"MIT humanoid MPC QP solve, {dt}, CUDA graph")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("latency per batch [ms]"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "mpc_latency.png"), dpi=130)
print("wrote", os.path.join(RES, "mpc_latency.csv"))
