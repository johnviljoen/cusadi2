"""Table of the fp32 timing campaign: results/bench_f32_runs.csv (5 appended runs of bench.py, every evaluator) plus
results/bench_f32_cpu_ideal_runs.csv (bench_cpu.py, evaluator "cpu_ideal") -> results/bench_f32_runs.md: per (evaluator, K, T)
the mean +- std over runs of the per-run tick medians (3 iterations per tick), the mean p99, the mean one-iteration time and
the mean single model call, then the speedup of CusADi2 at the K sweep's horizon (ratio of the run means, +- std of the per-run ratios).

    python summarize_bench.py [--T 35]
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; R = HERE / "results"
ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, default=35); a = ap.parse_args()
rows = list(csv.DictReader(open(R / "bench_f32_runs.csv")))
if (R / "bench_f32_cpu_ideal_runs.csv").exists():
    rows += list(csv.DictReader(open(R / "bench_f32_cpu_ideal_runs.csv")))
NAMES = dict(cusadi2="CusADi2 (ours)", frax="frax", mjx="MuJoCo XLA", cusadi="CusADi", cpu_ideal="CPU idealized", cpu="CPU + transfer")
ORDER = ["cusadi2", "frax", "mjx", "cusadi", "cpu_ideal", "cpu"]
D = defaultdict(lambda: defaultdict(list))
for r in rows:
    k = (r["evaluator"], int(r["K"]), int(r["T"]))
    for f in ("tick_median_ms", "tick_p99_ms", "iteration_ms", "model_ms"): D[k][f].append(float(r[f]))
    D[k]["n_iter"].append(int(r.get("n_iter", 1)))
keys = sorted(D, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1] != 256, k[2] if k[1] == 256 else k[1]))
L = ["| evaluator (fp32) | K | T | iterations | runs | tick median, mean +- std (ms) | tick p99, mean (ms) | one iteration, mean (ms) | model call, mean (ms) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for ev, K, T in keys:
    d = D[(ev, K, T)]; m = np.array(d["tick_median_ms"])
    L.append(f"| {NAMES.get(ev, ev)} | {K} | {T} | {d['n_iter'][0]} | {len(m)} | {m.mean():.2f} +- {m.std(ddof=1) if len(m) > 1 else 0:.2f} | {np.mean(d['tick_p99_ms']):.2f} | "
             f"{np.mean(d['iteration_ms']):.2f} | {np.mean(d['model_ms']):.3f} |")
Ks = sorted({K for (ev, K, T) in D if ev == "cusadi2" and T == a.T})
L += ["", f"Speedup of CusADi2 at T = {a.T} (ratio of the run means of the tick medians; +- std of the per-run ratios):"]
for ev in ORDER[1:]:
    txt = []
    for K in Ks:
        A, B = np.array(D[(ev, K, a.T)]["tick_median_ms"]), np.array(D[("cusadi2", K, a.T)]["tick_median_ms"])
        if len(A) == 0: continue
        n = min(len(A), len(B)); rt = A[:n] / B[:n]
        txt.append(f"B={K}: {A.mean() / B.mean():.2f} +- {rt.std(ddof=1) if n > 1 else 0:.2f}")
    if txt: L.append(f"- {NAMES[ev]}: " + ", ".join(txt))
Ts = sorted({T for (ev, K, T) in D if ev == "cusadi2" and K == 256})
L += ["", "Speedup of CusADi2 at K = 256 over the horizon sweep:"]
for ev in ORDER[1:]:
    txt = []
    for T in Ts:
        A, B = np.array(D[(ev, 256, T)]["tick_median_ms"]), np.array(D[("cusadi2", 256, T)]["tick_median_ms"])
        if len(A) == 0: continue
        txt.append(f"T={T}: {A.mean() / B.mean():.2f}")
    if txt: L.append(f"- {NAMES[ev]}: " + ", ".join(txt))
(R / "bench_f32_runs.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
