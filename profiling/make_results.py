"""Build results.md from the CSV / JSON files in results/.

Tables: one per problem; columns backend, dtype, batch, t_nominal,
t_qp_data (2a dyn_terms / 2b barriers), t_solve, t_end_to_end, solve share,
iterations (mean / max per instance; batch-max under vmap). Times are
median | p99 per batch in microseconds. Hand-written notes come from
notes.md (deviations, what did not work) and are appended verbatim.
"""

import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"


def read_csv(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def latest(rows, keys):
    """Keep the last row per key tuple (re-runs append; the latest wins)."""
    out = OrderedDict()
    for r in rows:
        out[tuple(r[k] for k in keys)] = r
    return list(out.values())


def fmt_us(v):
    v = float(v)
    if v >= 1e5:
        return f"{v/1e3:,.0f} ms"
    if v >= 1e4:
        return f"{v/1e3:,.1f} ms"
    return f"{v:,.0f}"


def cell(t):
    return f"{fmt_us(t['median_us'])} \\| {fmt_us(t['p99_us'])}" if t else "–"


def pct(x):
    return f"{100*float(x):.0f} %" if x is not None else "–"


def main():
    timings = latest(read_csv(R / "raw_timings.csv"), ("problem", "backend", "platform", "dtype", "batch", "piece"))
    stats = latest(read_csv(R / "solver_stats.csv"), ("problem", "backend", "platform", "dtype", "batch"))
    tput = latest(read_csv(R / "cpu_throughput.csv"), ("problem", "backend", "dtype", "piece", "procs"))
    sysinfo = json.load(open(R / "system.json")) if (R / "system.json").exists() else {}
    trace = json.load(open(R / "trace_g1_summary.json")) if (R / "trace_g1_summary.json").exists() else None
    probs = {p: json.load(open(R / f"problem_{p}.json")) for p in ("panda", "g1") if (R / f"problem_{p}.json").exists()}

    L = []
    L.append("# Where FRAX's CBF safety filter spends its time\n")
    L.append("Measured split of one control tick between model evaluation (kinematics, dynamics, "
             "Jacobians, barriers and Lie derivatives) and the QP solve, for frax + cbfpy as shipped. "
             "Numbers only; no optimisation of either library.\n")

    # ---- setup ----
    L.append("## Setup\n")
    if sysinfo:
        v = sysinfo.get("versions", {})
        cu = sysinfo.get("jax_cuda_versions") or {}
        L.append(f"- GPU: {sysinfo.get('gpu')} (driver in the same field); CUDA runtime seen by JAX: "
                 f"{cu.get('cuda_runtime')}, cuDNN {cu.get('cudnn')}; nvcc: {sysinfo.get('nvcc')}")
        L.append(f"- CPU: {sysinfo.get('cpu')} — {sysinfo.get('physical_cores')} physical cores, "
                 f"{sysinfo.get('logical_cpus')} logical; OS {sysinfo.get('os')}; Python {sysinfo.get('python')}")
        L.append("- Versions: " + ", ".join(f"{k} {val}" for k, val in v.items() if val))
        fs = sysinfo.get("frax_source", {})
        L.append(f"- frax source: git {fs.get('commit', '')[:10]} (pyproject version {fs.get('pyproject_version')}), "
                 f"used unmodified from `profiling/third_party/frax` (the pip wheel 0.0.4 lacks the demo API)")
    L.append("- fp64 everywhere (`JAX_ENABLE_X64=True`, as the demos); qpax additionally in fp32 (x64 disabled).")
    L.append("- CPU runs use frax's demo flags: `JAX_PLATFORMS=cpu`, `--xla_cpu_multi_thread_eigen=false`, "
             "`OPENBLAS_NUM_THREADS=1`, memory-optimised XLA CPU scheduler. GPU runs: `JAX_PLATFORMS=cuda`, default flags.")
    L.append("- No MuJoCo, no viewer. States: q uniform inside the joint limits, v ~ N(0, 0.5²) rad/s per joint "
             "(measured RMS ≈ 0.5); 25 % of the states have ~15 % of their joints within 0.01 rad of a limit "
             "moving outward, 25 % are the lowest-clearance states of a 10× random pool (collision rows), 50 % "
             "plain random. Desired end-effector poses are FK of independent random q, zero desired twist.")
    L.append("- Every piece is `jax.jit(jax.vmap(f))` on GPU; on CPU batch 1 is unbatched like the demo. "
             "Warm-up 10 calls, then `block_until_ready` after each timed call; median | p99 over 1000 calls "
             "(200 for B ≥ 1024). Compile time via `.lower().compile()`.")
    L.append("- QP: cbfpy's l1-relaxed (elastic) QP, penalties 1e3 (CBF rows) / 1e5 (torque bounds), "
             "`solver_tol` 1e-5. qpax: `solve_qp_elastic_primal` (explicit PDIP, max_iter 30). "
             "elastiqp: `elastiqp.jax.solve` (dual active set, `eps_abs` 1e-5), CPU-only FFI.\n")

    # ---- headline split ----
    L.append("## Headline split: QP solve vs model evaluation, share of the end-to-end tick\n")
    L.append("`solve` = the solver alone on precomputed QP data; `model (e2e − solve)` = everything else in the tick "
             "(nominal controller, dynamics, barriers, Lie derivatives, QP assembly). Shares are of the end-to-end median. "
             "Separately timed pieces overlap and fuse differently, so `solve + nominal + qp_data` is also shown relative to e2e.\n")
    L.append("| problem | backend | dtype | batch | e2e median | solve | model (e2e − solve) | solve share | model share | (nominal + qp_data) / e2e |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in ("panda", "g1"):
        for backend, platform, dtype in [("elastiqp", "cpu", "f64"), ("qpax", "cpu", "f64"), ("qpax", "cpu", "f32"), ("qpax", "gpu", "f64"), ("qpax", "gpu", "f32")]:
            sel = [r for r in timings if r["problem"] == p and r["backend"] == backend and r["platform"] == platform and r["dtype"] == dtype]
            for B in sorted({int(r["batch"]) for r in sel}):
                t = {r["piece"]: r for r in sel if int(r["batch"]) == B}
                if not all(k in t for k in ("nominal", "qp_data", "solve", "end_to_end")):
                    continue
                e2e, sol = float(t["end_to_end"]["median_us"]), float(t["solve"]["median_us"])
                mdl = float(t["nominal"]["median_us"]) + float(t["qp_data"]["median_us"])
                L.append(f"| {p} | {backend} {platform.upper()} | {dtype} | {B} | {fmt_us(e2e)} | {fmt_us(sol)} | {fmt_us(e2e - sol)} | "
                         f"{pct(sol / e2e)} | {pct((e2e - sol) / e2e)} | {pct(mdl / e2e)} |")
    L.append("")

    # ---- dimensions ----
    L.append("## QP dimensions\n")
    L.append("| problem | nv = QP variables | inequality rows (CBF + 2·nv torque bounds) | barrier rows | self-collision pairs | collision spheres | elastic vars incl. slacks |")
    L.append("|---|---|---|---|---|---|---|")
    for p, d in probs.items():
        groups = ", ".join(f"{k} {v}" for k, v in d["barrier_groups"].items())
        L.append(f"| {p} | {d['n_vars']} | {d['n_rows']} = {d['num_cbf']} + {2*d['nv']} | {d['num_cbf']} ({groups}) | "
                 f"{d['n_self_collision_pairs']} | {d['n_collision_spheres']} | {d['elastic_qp_vars_incl_slack']} |")
    L.append("")
    L.append("All barriers are relative degree 2 on torque (`h_2`), so each barrier row of G is one row of Lgh = "
             "∂(ḣ₂ + α₂h₂)/∂z · g(z) and the RHS is Lfh + α(h); cbfpy forms them with a jvp over f and a vmapped jvp over the columns of g.\n")

    # ---- per-problem tables ----
    for p in ("panda", "g1"):
        rows = [r for r in timings if r["problem"] == p]
        if not rows:
            continue
        L.append(f"## {p.upper() if p == 'g1' else 'Panda'}: timings per batch (median | p99, µs unless marked ms)\n")
        L.append("| backend | dtype | batch | t_nominal | t_qp_data | 2a dyn_terms | 2b barriers | t_solve | t_end_to_end | sum of pieces | solve / e2e | iters mean / max (batch-max mean / max) | conv. | active CBF | compile e2e (s) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = [("elastiqp", "cpu", "f64"), ("qpax", "cpu", "f64"), ("qpax", "cpu", "f32"), ("qpax", "gpu", "f64"), ("qpax", "gpu", "f32")]
        for backend, platform, dtype in order:
            sel = [r for r in rows if r["backend"] == backend and r["platform"] == platform and r["dtype"] == dtype]
            for B in sorted({int(r["batch"]) for r in sel}):
                t = {r["piece"]: r for r in sel if int(r["batch"]) == B}
                st = next((s for s in stats if s["problem"] == p and s["backend"] == backend and s["platform"] == platform
                           and s["dtype"] == dtype and int(s["batch"]) == B), None)
                need = ("nominal", "qp_data", "solve", "end_to_end")
                if not all(k in t for k in need):
                    continue
                e2e = float(t["end_to_end"]["median_us"])
                ssum = sum(float(t[k]["median_us"]) for k in ("nominal", "qp_data", "solve"))
                share = float(t["solve"]["median_us"]) / e2e
                it = (f"{float(st['iters_mean']):.1f} / {st['iters_max']} ({float(st['batchmax_mean']):.1f} / {st['batchmax_max']})"
                      if st else "–")
                L.append(f"| {backend} {platform.upper()} | {dtype} | {B} | {cell(t.get('nominal'))} | {cell(t.get('qp_data'))} | "
                         f"{cell(t.get('dyn_terms'))} | {cell(t.get('barriers'))} | {cell(t.get('solve'))} | {cell(t.get('end_to_end'))} | "
                         f"{fmt_us(ssum)} | {pct(share)} | {it} | {pct(st['converged_frac']) if st else '–'} | "
                         f"{pct(st['active_cbf_frac']) if st else '–'} | {float(t['end_to_end']['compile_s']):.1f} |")
        L.append("")
        # extra: h2_only and compile times
        L.append(f"Extra columns for {p}: raw barrier values without derivatives (`h2_only`) and per-piece compile times.\n")
        L.append("| backend | dtype | batch | h2_only | compile nominal (s) | compile qp_data (s) | compile solve (s) | compile e2e (s) | n calls |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for backend, platform, dtype in order:
            sel = [r for r in rows if r["backend"] == backend and r["platform"] == platform and r["dtype"] == dtype]
            for B in sorted({int(r["batch"]) for r in sel}):
                t = {r["piece"]: r for r in sel if int(r["batch"]) == B}
                if "end_to_end" not in t:
                    continue
                g = lambda k: f"{float(t[k]['compile_s']):.2f}" if k in t else "–"
                L.append(f"| {backend} {platform.upper()} | {dtype} | {B} | {cell(t.get('h2_only'))} | {g('nominal')} | {g('qp_data')} | "
                         f"{g('solve')} | {g('end_to_end')} | {t['end_to_end']['n_calls']} |")
        L.append("")
        # per-kind active stats
        st_any = [s for s in stats if s["problem"] == p and s["batch"] == "1"]
        if st_any:
            L.append("Constraint activity on the 4096-state dataset (CBF row active = residual ≤ 1e-5·max(1,|h|) at the solution):\n")
            L.append("| backend / dtype | ≥1 active CBF row | mean active rows | random states | joint-limit states | collision states | filter changed u | u_des outside torque bounds | solver converged |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for s in st_any:
                L.append(f"| {s['backend']} {s['platform']} {s['dtype']} | {pct(s['active_cbf_frac'])} | {float(s['active_rows_mean']):.2f} | "
                         f"{pct(s['active_frac_random'])} | {pct(s['active_frac_jointlimit'])} | {pct(s['active_frac_collision'])} | "
                         f"{pct(s['filter_changed_frac'])} | {pct(s['u_sat_frac'])} | {pct(s['converged_frac'])} |")
            L.append("")

    # ---- CPU throughput ----
    if tput:
        L.append("## CPU throughput, one process per physical core\n")
        L.append("Each process pinned to its own core, batch 1, unbatched jit, frax's single-thread flags; "
                 "all processes start together and run back to back for 10 s.\n")
        L.append("| problem | backend | dtype | piece | processes | total calls/s | per-process calls/s (mean / min) | single-process µs equiv. | 1-proc batch-1 median µs |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in tput:
            t1 = next((t for t in timings if t["problem"] == r["problem"] and t["backend"] == r["backend"] and t["platform"] == "cpu"
                       and t["dtype"] == r["dtype"] and t["batch"] == "1" and t["piece"] == r["piece"]), None)
            L.append(f"| {r['problem']} | {r['backend']} | {r['dtype']} | {r['piece']} | {r['procs']} | {float(r['calls_per_s']):,.0f} | "
                     f"{float(r['per_proc_calls_per_s_mean']):,.0f} / {float(r['per_proc_calls_per_s_min']):,.0f} | "
                     f"{float(r['single_proc_us_equiv']):,.0f} | {fmt_us(t1['median_us']) if t1 else '–'} |")
        L.append("")

    # ---- trace ----
    if trace:
        m = trace["mean_per_call"]
        L.append(f"## GPU trace: end-to-end {trace['problem'].upper()} / qpax / {trace['dtype']} at batch {trace['batch']}\n")
        L.append(f"`jax.profiler` trace of {trace['n_calls_traced']} back-to-back calls (device stream events, means per call). "
                 "The qpax `while_loop` is bracketed by its condition evaluations (`hlo_op` `while*`, one device event per "
                 "iteration + 1); everything between the first and last of them is counted as the loop.\n")
        L.append("| quantity | per tick |")
        L.append("|---|---|")
        L.append(f"| wall time per call (ms) | {sum(trace['wall_ms_per_call'])/len(trace['wall_ms_per_call']):.1f} |")
        L.append(f"| device busy time (ms) | {m['device_busy_ms']:.1f} |")
        L.append(f"| device span first→last event (ms) | {m['device_span_ms']:.1f} |")
        L.append(f"| device events (kernels + memcpys) | {m['n_device_events']:.0f} = {m['n_kernels']:.0f} kernels + {m['n_memcpy']:.0f} memcpys ({m['n_memcpy_d2h']:.0f} D2H) |")
        L.append(f"| while-loop iterations run | {m['n_while_cond_events']-1:.0f} (qpax reports batch-max {trace['qpax_iters_batch_max']}, mean {trace['qpax_iters_mean']:.1f}, converged {100*trace['qpax_converged_frac']:.1f} %) |")
        L.append(f"| busy time before the loop: nominal + qp_data + solver init (ms) | {m['busy_pre_loop_ms']:.2f} ({m['kernels_pre_loop']:.0f} events) |")
        L.append(f"| busy time inside the while loop (ms) | {m['busy_loop_ms']:.2f} ({m['kernels_loop']:.0f} events) |")
        L.append(f"| busy time after the loop (ms) | {m['busy_post_loop_ms']:.2f} ({m['kernels_post_loop']:.0f} events) |")
        L.append(f"| **while-loop share of device busy time** | **{100*m['loop_share_of_busy']:.1f} %** (of span: {100*m['loop_share_of_span']:.1f} %) |")
        L.append("")
        L.append("Top HLO ops by device time per call:\n")
        L.append("| hlo_op | launches / call | µs / call | in loop |")
        L.append("|---|---|---|---|")
        for t in trace["top_hlo_ops_by_time"][:15]:
            L.append(f"| `{t['hlo_op']}` | {t['count_per_call']:.0f} | {t['us_per_call']:,.0f} | {pct(t['frac_in_loop'])} |")
        L.append("")

    notes = HERE / "notes.md"
    if notes.exists():
        L.append(notes.read_text())

    L.append("\n## Files\n")
    L.append("- `results/raw_timings.csv` — every timed piece (median/p99/mean/min/max µs, n, compile s).")
    L.append("- `results/solver_stats.csv` — iteration and activity statistics per configuration and batch.")
    L.append("- `results/cpu_throughput.csv` — N-process CPU throughput.")
    L.append("- `results/trace_g1_summary.json` — trace breakdown (raw trace under `results/trace/`, not committed).")
    L.append("- `results/elastiqp_f64_cpu.png` — per-piece bar chart of the published backend (`plot_elastiqp.py`).")
    L.append("- `results/panda_oscbf.mp4`, `results/panda_oscbf_gpu_b4096.mp4` (+ `.png` traces) — renders of the Panda scenario "
             "(single instance, demo backend; 4096 instances on the GPU with qpax), `render_panda_video.py`, `render_panda_batch_video.py`.")
    L.append("- `results/cusadi2_swap.md`, `.csv`, `.png` — frax vs cusadi2 as the model-evaluation backend (`bench_cusadi2_swap.py`, "
             "`panda_casadi.py`, `cusadi2_oscbf.py`, `cusadi2_ffi.cu`, `build_cusadi2_plan.py`, `report_cusadi2_swap.py`); "
             "`results/cusadi2_swap_jacobianform.*` is the first, badly formulated attempt kept for the record.")
    L.append("- `results/system.json`, `results/problem_*.json`, `results/run_all.log`.")
    L.append("- Scripts: `run_all.py` (everything), `bench.py`, `cpu_throughput.py`, `trace_g1.py`, `make_results.py`; "
             "problems in `panda_problem.py`, `g1_problem.py`; sampling in `sampling.py`; helpers in `common.py`.")
    (HERE / "results.md").write_text("\n".join(L) + "\n")
    print(f"wrote {HERE / 'results.md'} ({len(L)} lines)")


if __name__ == "__main__":
    main()
