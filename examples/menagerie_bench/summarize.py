#!/usr/bin/env python
"""Cross-robot summary outputs for the Menagerie campaign.

Reads every results/<robot>/ produced by run.py and writes, into results/:

  * summary.csv   — one row per robot x func x dtype (plus one bare row per
                    gate-skipped robot): model size (nq/nv/n_bodies), plan
                    metrics (instruction counts, per-stage/unrolled smem),
                    the best cusadi2 wall-clock latency at the saturation batch,
                    the MJX batched/rollout latency at the matching batch,
                    the speedups, and the compile times.
  * status.md     — a robot x stage markdown table of pass/fail/skip(+reason).
  * summary_speedup.png/.pdf — cusadi2-vs-MJX speedup against model size (nv),
                    one point per robot, both dtypes, one panel per function.

"Saturation batch" = the largest batch B* at which BOTH the Python-called
cusadi2 sweep (latency_py_<dtype>.csv — host-language-symmetric with MJX) has a
feasible config AND MJX batched has a non-null median; the cusadi2 number is the
best (minimum-wall) config at that B*. Falls back to the largest feasible
cusadi2 batch when MJX is unavailable.

Standalone:  python examples/menagerie_bench/run.py --summarize
"""
import csv
import json
import os

DT2FP = {"double": "fp64", "float": "fp32"}
FUNC2KIND = {"fk": "fk", "fkjac": "jac"}

STAGES = ["gate", "cusadi_double", "pylib_double", "cusadi_float", "pylib_float",
          "mjx_fp64", "mjx_fp32", "warp", "plot"]

CSV_COLS = ["robot", "func", "dtype", "gate_status", "gate_reason",
            "nq", "nv", "n_bodies", "floating",
            "n_instr", "n_instr_unr", "smem_kb", "smem_unr_kb",
            "codegen_s", "nvcc_s",
            "sat_batch", "best_config", "cusadi2_wall_us", "cusadi2_kernel_us",
            "mjx_batched_us", "mjx_rollout_us",
            "speedup_batched", "speedup_rollout", "mjx_compile_s"]


def _jload(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _read_py_csv(path):
    """latency_py CSV -> {func: {B: {config: (wall_us, kernel_us)}}} (nan dropped)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                wall, kern = float(r["wall_us"]), float(r["latency_us"])
            except ValueError:
                continue
            out.setdefault(r["func"], {}).setdefault(int(r["batch"]), {})[r["config"]] = (wall, kern)
    return out


def _mjx_series(mjx, key):
    """{B: seconds} for one series, nulls dropped."""
    if not mjx:
        return {}
    res = mjx.get("results") or {}
    return {int(b): v for b, v in (res.get(key) or {}).items() if v is not None}


def _robot_rows(robot, resdir, status, at_batch=None):
    gate = (status or {}).get("gate") or {}
    base = {"robot": robot, "gate_status": gate.get("status", ""),
            "gate_reason": gate.get("reason") or "",
            "nq": gate.get("nq", ""), "nv": gate.get("nv", ""),
            "n_bodies": gate.get("n_bodies", ""), "floating": gate.get("floating", "")}
    if gate.get("status") != "pass":
        return [dict(base, **{c: "" for c in CSV_COLS if c not in base})]

    rows = []
    for dtype, fp in DT2FP.items():
        comp = _jload(os.path.join(resdir, f"compile_cusadi2_{dtype}.json")) or {}
        pyd = _read_py_csv(os.path.join(resdir, f"latency_py_{dtype}.csv"))
        mjx = _jload(os.path.join(resdir, f"mjx_{fp}.json"))
        for func, kind in FUNC2KIND.items():
            meta = (comp.get("meta") or {}).get(func, {})
            row = dict(base, func=func, dtype=dtype,
                       n_instr=meta.get("n_instr", ""),
                       n_instr_unr=meta.get("n_instr_unr", ""),
                       smem_kb=round(meta["smem_kb"], 2) if "smem_kb" in meta else "",
                       smem_unr_kb=round(meta["smem_unr_kb"], 2) if "smem_unr_kb" in meta else "",
                       codegen_s=round((comp.get("codegen_s") or {}).get(func, 0), 2) or "",
                       nvcc_s=round(comp["nvcc_s"], 1) if "nvcc_s" in comp else "",
                       sat_batch="", best_config="", cusadi2_wall_us="",
                       cusadi2_kernel_us="", mjx_batched_us="", mjx_rollout_us="",
                       speedup_batched="", speedup_rollout="", mjx_compile_s="")
            by_b = pyd.get(func, {})
            mjx_b = _mjx_series(mjx, f"{kind}_batched")
            mjx_r = _mjx_series(mjx, f"{kind}_rollout")
            if at_batch is not None:
                # fixed operating point: only report robots measured at exactly B
                common = [at_batch] if at_batch in by_b else []
            else:
                common = sorted(set(by_b) & set(mjx_b)) or sorted(by_b)
            if common:
                Bs = common[-1]
                cfg, (wall, kern) = min(by_b[Bs].items(), key=lambda kv: kv[1][0])
                row.update(sat_batch=Bs, best_config=cfg,
                           cusadi2_wall_us=round(wall, 1), cusadi2_kernel_us=round(kern, 1))
                if Bs in mjx_b:
                    mb = mjx_b[Bs] * 1e6
                    row.update(mjx_batched_us=round(mb, 1),
                               speedup_batched=round(mb / wall, 2))
                if Bs in mjx_r:
                    mr = mjx_r[Bs] * 1e6
                    row.update(mjx_rollout_us=round(mr, 1),
                               speedup_rollout=round(mr / wall, 2))
                comp_mjx = ((mjx or {}).get("compile") or {}).get(f"{kind}_batched") or {}
                if str(Bs) in comp_mjx and comp_mjx[str(Bs)] is not None:
                    row["mjx_compile_s"] = round(comp_mjx[str(Bs)], 2)
            rows.append(row)
    return rows


def write_summary_csv(results_dir, robots, statuses, at_batch=None, tag=""):
    path = os.path.join(results_dir, f"summary{tag}.csv")
    rows = []
    for robot in robots:
        rows += _robot_rows(robot, os.path.join(results_dir, robot), statuses.get(robot),
                            at_batch=at_batch)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLS})
    print(f"wrote {path} ({len(rows)} rows)")
    return rows


def write_status_md(results_dir, robots, statuses):
    path = os.path.join(results_dir, "status.md")
    lines = ["# Menagerie campaign status", "",
             "| robot | " + " | ".join(STAGES) + " | note |",
             "|" + "---|" * (len(STAGES) + 2)]
    for robot in robots:
        st = statuses.get(robot)
        if not st:
            lines.append(f"| {robot} | " + " | ".join(["—"] * len(STAGES)) + " | not run |")
            continue
        cells = []
        for s in STAGES:
            rec = (st.get("stages") or {}).get(s)
            cells.append(rec["status"] if rec else "—")
        gate = st.get("gate") or {}
        note = (gate.get("reason") or "").replace("|", "/")[:120]
        lines.append(f"| {robot} | " + " | ".join(cells) + f" | {note} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def plot_speedup(results_dir, rows, n_stages, tag="", batch_label="sat. batch"):
    """Aggregate figure: cusadi2-vs-MJX-batched speedup at the saturation batch
    (or, with tag/batch_label, at one fixed batch) against model size (nv). One
    point per robot; fp64 and fp32 as two fixed series (color + marker); one
    panel per function; speedup > 1 favors cusadi2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    series_style = {"double": ("fp64", "C0", "o"), "float": ("fp32", "C1", "s")}
    any_pts = False
    for ax, (func, title) in zip(axes, [("fk", "forward kinematics"),
                                        ("fkjac", "frame Jacobians")]):
        for dtype, (fp, color, mk) in series_style.items():
            pts = [(r["nv"], r["speedup_batched"], r["robot"]) for r in rows
                   if r.get("func") == func and r.get("dtype") == dtype
                   and r.get("speedup_batched") not in ("", None)]
            if not pts:
                continue
            pts.sort()
            any_pts = True
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=28,
                       color=color, marker=mk, alpha=0.85, label=f"{fp}", zorder=3)
        ax.axhline(1.0, color="gray", lw=1, ls=":", zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("model size nv (dof)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].set_ylabel(f"speedup = MJX batched / cusadi2 best (wall, {batch_label})")
    if any_pts:
        axes[0].legend(fontsize=9, title="precision")
    fig.suptitle(f"cusadi2 vs MJX-batched wall-clock speedup across the Menagerie  "
                 f"(one point per robot, N={n_stages}, {batch_label})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = os.path.join(results_dir, f"summary_speedup{tag}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


def summarize(results_dir, manifest, n_stages=8, at_batch=None):
    robots = list(manifest.ROBOTS)
    statuses = {r: _jload(os.path.join(results_dir, r, "status.json")) for r in robots}
    tag = f"_B{at_batch}" if at_batch is not None else ""
    batch_label = f"B={at_batch}" if at_batch is not None else "sat. batch"
    rows = write_summary_csv(results_dir, robots, statuses, at_batch=at_batch, tag=tag)
    write_status_md(results_dir, robots, statuses)
    try:
        plot_speedup(results_dir, rows, n_stages, tag=tag, batch_label=batch_label)
    except Exception as e:
        print(f"summary figure failed (non-fatal): {e!r}")
