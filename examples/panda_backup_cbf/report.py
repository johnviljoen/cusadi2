"""Render results/report.md and results/bench.png from the CSV / JSON results of this directory.

    python report.py [--N 20]

Reads results/bench.csv (three pipelines over batch), results/frax_variants.csv (frax implementation
variants and the fp32 footnote), results/staged_times.csv (bare cusadi2 kernel times), the plan stats
in build/double and results/closed_loop_b256_N{N}.json.  Last row per key wins.
"""
import argparse, csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parent; R = HERE / "results"
ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=20); args = ap.parse_args()
N = args.N
BUDGET_US = 20000.0


def read(name):
    f = R / name
    return list(csv.DictReader(open(f))) if f.exists() else []


bench = [r for r in read("bench.csv") if int(r["N"]) == N]
med = {(int(r["batch"]), r["piece"]): float(r["median_us"]) for r in bench}
p99 = {(int(r["batch"]), r["piece"]): float(r["p99_us"]) for r in bench}
comp = {(int(r["batch"]), r["piece"]): float(r["compile_s"]) for r in bench}
du = {(int(r["batch"]), r["piece"]): float(r["max_abs_du_vs_frax_qpax"]) for r in bench if r["max_abs_du_vs_frax_qpax"] not in ("", "None")}
batches = sorted({int(r["batch"]) for r in bench})
def g(B, k, d=med): return d.get((B, k), float("nan"))
def f_us(x): return "—" if x != x else f"{x / 1e3:,.2f}"
def hz(x): return "—" if x != x else f"{1e6 / x:,.0f}"

lines = []
lines += [f"## Three pipelines over batch size (N = {N}, {bench[0]['n_rows'] if bench else '?'} rows, fp64, RTX 4080)", "",
          "Milliseconds per batched call, median (p99 in parentheses for the end-to-end columns), jit + vmap, device-resident "
          "inputs, block_until_ready, 1000 calls for B ≤ 128 else 200. The budget is 20 ms (50 Hz). "
          "`model` = nominal OSC + all QP rows; `solve` = the QP on precomputed data; `e2e` = model + solve in one jitted program.", "",
          "| B | model frax | model cusadi2 | solve qpax | solve CUDA IPM | e2e frax + qpax | e2e frax + CUDA IPM | e2e cusadi2 + CUDA IPM | ours ÷ budget | frax+qpax ÷ budget | frax+IPM ÷ budget | max\\|Δu\\| vs frax+qpax |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for B in batches:
    mf, mc, sq, sc = g(B, "model_frax"), g(B, "model_cusadi2"), g(B, "solve_qpax"), g(B, "solve_cuda")
    efq, efc, ecc = g(B, "e2e_frax_qpax"), g(B, "e2e_frax_cuda"), g(B, "e2e_cusadi2_cuda")
    dus = [v for (b, k), v in du.items() if b == B and k != "e2e_frax_qpax"]
    lines.append(f"| {B} | {f_us(mf)} | {f_us(mc)} | {f_us(sq)} | {f_us(sc)} | {f_us(efq)} ({f_us(g(B, 'e2e_frax_qpax', p99))}) | {f_us(efc)} ({f_us(g(B, 'e2e_frax_cuda', p99))}) | "
                 f"{f_us(ecc)} ({f_us(g(B, 'e2e_cusadi2_cuda', p99))}) | {ecc / BUDGET_US:.2f} | {efq / BUDGET_US:.2f} | {efc / BUDGET_US:.2f} | {max(dus) if dus else float('nan'):.1e} |")
lines += ["", "XLA compile time of the jitted end-to-end program per batch size (s), frax + CUDA IPM (the frax + qpax program was already compiled for the reference torques, so its "
          "compile time is not separately measured; it is the same order): " + "; ".join(f"B={B}: {g(B, 'e2e_frax_cuda', comp):.0f}" for B in batches) +
          ". The cusadi2 + CUDA IPM program compiles in 0.2 to 0.3 s at every batch (two FFI calls and a scatter); the cusadi2 plans are batch-independent.",
          "", "max|Δu| is the largest difference in the safe torque against the frax + qpax pipeline over the first batch window. It grows with batch because the batch is more likely to "
          "contain a QP that the solvers leave at the 1e-5 KKT tolerance after different iterates (both solvers converge on 99.9 % of the instances; the remaining ones hit the 30-iteration cap), "
          "the same effect as in profiling/results/cusadi2_swap.md.", ""]

# pass test at B = 256
if 256 in batches:
    ours_p99 = g(256, "e2e_cusadi2_cuda", p99); frax_model = g(256, "model_frax"); frax_ipm = g(256, "e2e_frax_cuda"); frax_qpax = g(256, "e2e_frax_qpax")
    lines += ["## Pass test at batch 256 (fixed before the runs)", "",
              f"- ours (cusadi2 model + CUDA IPM) p99 ≤ 16 ms: **{ours_p99 / 1e3:.2f} ms → {'PASS' if ours_p99 <= 16000 else 'FAIL'}**",
              f"- frax model side alone, fastest variant, median ≥ 30 ms: **{frax_model / 1e3:.1f} ms → {'frax FAILS real time' if frax_model >= 30000 else 'gate not met'}**",
              f"- frax + CUDA IPM (the fairness row): {frax_ipm / 1e3:.1f} ms = {frax_ipm / BUDGET_US:.2f}× the budget; frax + qpax: {frax_qpax / 1e3:.1f} ms = {frax_qpax / BUDGET_US:.2f}× the budget", ""]

# frax variants
fv = [r for r in read("frax_variants.csv") if int(r["batch"]) == 256]
if fv:
    lines += ["## frax implementation variants (model side, B = 256, median ms)", "",
              "| N | rows | scan + vmap-jvp | unrolled + vmap-jvp | scan + jacfwd | unrolled + jacfwd | fastest |", "|---:|---:|---:|---:|---:|---:|---:|"]
    byN = {}
    for r in fv:
        if r["dtype"] == "f64": byN.setdefault(int(r["N"]), {})[r["variant"]] = float(r["median_us"])
    for n in sorted(byN):
        v = byN[n]; best = min(v, key=v.get)
        lines.append(f"| {n} | {78 + 64 * n} | " + " | ".join(f_us(v.get(k, float('nan'))) for k in ("scan-jvp", "unrolled-jvp", "scan-jacfwd", "unrolled-jacfwd")) + f" | {best} |")
    f32 = {(r["dtype"], r["variant"]): float(r["median_us"]) for r in fv if r["dtype"].startswith("f32") and int(r["N"]) == N}
    if f32:
        lines += ["", "### fp32 footnote (frax model side only, same states, B = 256)", "",
                  "| precision | " + " | ".join(f"{v}" for v in ("unrolled-jvp", "scan-jvp")) + " | rows G rel. error vs fp64 | max abs Δu vs fp64 (N m) |", "|---|---:|---:|---:|---:|"]
        notes = {"f32": ("float32, XLA default (TF32 matmuls)", "5e-1", "1.0e+2"), "f32-highest": ("float32, matmul precision highest (no TF32)", "5.5e-6", "3.9e-1")}
        for dt, (label, gerr, du) in notes.items():
            if any(k[0] == dt for k in f32):
                lines.append(f"| {label} | " + " | ".join(f"{f32[(dt, v)] / 1e3:.1f} ms" if (dt, v) in f32 else "—" for v in ("unrolled-jvp", "scan-jvp")) + f" | {gerr} | {du} |")
        lines += ["", "Accuracy from `fp32_check.py` (256 states, the fp32 rows solved by the fp64 CUDA IPM and compared with the fp64 solution). "
                  "With XLA's default TF32 matmuls the fp32 rollout is unusable (50 % error in the rows after 20 steps of sensitivities, torques off by up to 100 N m); "
                  "with TF32 disabled the rows are accurate to 5e-6 and the safe torques agree to 0.4 N m worst case (median 2e-4). "
                  "qpax itself does not converge in fp32 on this QP (profiling/results.md), so the only fp32 configuration that exists is the frax model in fp32 feeding the fp64 CUDA IPM. "
                  "**Caveat to the claim:** that configuration is not measured end to end (a mixed-precision jitted program needs x64 enabled, under which frax's constants promote to fp64), "
                  "but the sum of its separately measured parts at B = 256 is about 9.0 + 6.9 = 15.9 ms, i.e. 0.8× the budget. In fp64, the precision the published stack runs at and the "
                  "only one whose default settings give correct rows, frax misses the budget by 2.1× even with our solver; in fp32 with TF32 disabled it would fit with about 20 % headroom. "
                  "Disabling TF32 costs frax nothing here (9.0 ms vs 8.9 ms), so the fp32 route is a real option for frax on a GPU with full-rate fp32 and 1/64-rate fp64. "
                  "A float32 cusadi2 build (cusadi2 has float plans) was not run."]
    lines.append("")

lines += ["## CUDA interior-point solver", "",
          "`cbf_ipm.cu`: one thread block (512 threads) per instance; each thread keeps up to 3 rows of G and their slack / dual state in registers; the 7 x 7 normal "
          "equations are block reductions (warp shuffles + one shared-memory pass) and thread 0 factors them (Cholesky). Same elastic formulation, initialisation, "
          "Mehrotra predictor-corrector, line search, 1e-5 inf-norm convergence test and 30-iteration cap as qpax's explicit backend; each block stops at its own convergence. "
          "Before the solve every CBF row with h_i > Σ_j |G_ij| u_max,j is dropped: such a row cannot be violated by any torque inside the box, so it cannot be active "
          "(exact for the box-constrained problem; the elastic relaxation makes the bounds soft at a 1e5 penalty, which never binds here). At N = 20 this removes 43 % of the rows on the dataset. "
          "Validation against qpax on 512 dataset states: identical convergence fractions, safe torques within 6e-4 N m (median 1e-6; 5e-7 with screening off), objective within 2e-5. "
          "The solver is bound by the RTX 4080's 1/64-rate FP64 pipe and by reduction latency at one block per SM: removing the H accumulation (28 FMAs per row per iteration) "
          "saves 26 %, removing the serial factorisation or the divisions saves nothing measurable.", ""]

# cusadi2 plans and bare kernel times
st = {}
for w in ("head", "step"):
    f = HERE / "build/double" / f"backup_{w}_ml0_stats.json"
    if f.exists(): st[w] = json.loads(f.read_text())
if st:
    lines += ["## cusadi2 plans", "", "| plan | SX nodes | levels | max level width | peak live slots | shared memory / block |", "|---|---:|---:|---:|---:|---:|"]
    for w, s in st.items():
        lines.append(f"| {w} | {s['sx_nodes']} | {s['n_levels']} | {s['max_level_width']} | {s['peak_live_slots']} | {s['peak_live_slots'] * 8 / 1024:.0f} KB |")
    lines.append("")
stt = read("staged_times.csv")
if stt:
    lines += ["Bare kernel time (CUDA events, no XLA): head + N step launches, µs", "", "| N | B = 1 | 64 | 256 | 1024 | 4096 | per step at 256 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    byN = {}
    for r in stt: byN.setdefault(int(r["N"]), {})[int(r["batch"])] = (float(r["total_us"]), float(r["step_us"]))
    for n in sorted(byN):
        v = byN[n]
        lines.append(f"| {n} | " + " | ".join(f"{v[b][0]:,.0f}" if b in v else "—" for b in (1, 64, 256, 1024, 4096)) + f" | {v[256][1]:.0f} |" if 256 in v else " | — |")
    lines.append("")

# closed loop
cl = R / f"closed_loop_b256_N{N}.json"
if cl.exists():
    c = json.loads(cl.read_text())
    lines += ["## Closed loop, 256 instances, 20 s at 50 Hz", "",
              f"- applied pipeline (cusadi2 + CUDA IPM): median {c['ours_us_median'] / 1e3:.2f} ms, p99 {c['ours_us_p99'] / 1e3:.2f} ms, max {c['ours_us_max'] / 1e3:.2f} ms per tick",
              (f"- frax + qpax on the same states (not applied): median {c['frax_us_median'] / 1e3:.1f} ms, p99 {c['frax_us_p99'] / 1e3:.1f} ms" if c.get("frax_us_median") else ""),
              f"- batch-minimum barrier over the run: {c['h_min_all']:+.4f} (per group: " + ", ".join(f"{k} {v:+.3f}" for k, v in c["h_min"].items()) + ")",
              f"- filter changed the torque in {100 * c['active_frac_mean']:.1f} % of instance-ticks; the predictive rows changed it (vs the one-step filter on the same state) in "
              f"{100 * c['pred_changed_frac_mean']:.2f} % (max {100 * c['pred_changed_frac_max']:.1f} % of the batch in one tick), in {100 * c['ticks_with_pred_change']:.0f} % of ticks; "
              f"a predictive row was active in {100 * c['pred_row_active_frac_mean']:.2f} % of instance-ticks",
              f"- IPM: max iterations in any batch {c['iters_max_over_run']}, lowest converged fraction in a tick {100 * c['conv_frac_min']:.1f} %",
              f"- MuJoCo (256 × 10 substeps, {12} threads): median {c['sim_us_median'] / 1e3:.1f} ms per tick", ""]
(R / "report.md").write_text("\n".join(l for l in lines if l is not None) + "\n")
print("\n".join(lines))

if batches:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor="#fcfcfb")
    col = {"e2e_frax_qpax": "#2a78d6", "e2e_frax_cuda": "#8a4fd3", "e2e_cusadi2_cuda": "#1a9e5c", "model_frax": "#2a78d6", "model_cusadi2": "#1a9e5c", "solve_qpax": "#eb6834", "solve_cuda": "#c2185b"}
    nice = {"e2e_frax_qpax": "frax + qpax (published stack)", "e2e_frax_cuda": "frax + CUDA IPM (fairness row)", "e2e_cusadi2_cuda": "cusadi2 + CUDA IPM (ours)",
            "model_frax": "model: frax", "model_cusadi2": "model: cusadi2", "solve_qpax": "solve: qpax", "solve_cuda": "solve: CUDA IPM"}
    for k in ("e2e_frax_qpax", "e2e_frax_cuda", "e2e_cusadi2_cuda"):
        ax[0].plot(batches, [g(B, k) / 1e3 for B in batches], "-o", ms=4, color=col[k], label=nice[k])
    ax[0].axhline(20, color="k", lw=0.9, ls="--", label="20 ms budget (50 Hz)")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].axvline(256, color="#898781", lw=0.8, ls=":"); ax[0].text(256 * 1.15, ax[0].get_ylim()[0] * 1.25, "B = 256", fontsize=8, color="#898781")
    ax[0].set_xlabel("batch size"); ax[0].set_ylabel("ms per tick (median)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    ax[0].set_title("end to end", loc="left", fontsize=10)
    for k in ("model_frax", "model_cusadi2", "solve_qpax", "solve_cuda"):
        ax[1].plot(batches, [g(B, k) / 1e3 for B in batches], "-o" if "model" in k else "--s", ms=4, color=col[k], label=nice[k])
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_xlabel("batch size"); ax[1].set_ylabel("ms per call (median)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    ax[1].set_title("pieces", loc="left", fontsize=10)
    fig.suptitle(f"Panda backup-trajectory CBF filter, N = {N}, GPU fp64: frax vs cusadi2 model, qpax vs CUDA interior point", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(R / "bench.png", dpi=150); print(R / "bench.png")
