#!/usr/bin/env python
"""DAG shape of every manifest robot's FK, frame-Jacobian and frame-Hessian
functions: depth and per-level width of the plain longest-path level plan, plus
whether cusadi2 had to window the plan to fit this GPU's shared memory.

One subprocess per robot (`--worker`), no time limit, in ascending nv order.
Results are cached as results/dag/<robot>.json and a robot with a complete
record is skipped unless `--force`; a worker that dies (segfault, earlyoom
SIGKILL) leaves its partial record plus a `failed` block and the sweep
continues. Per function the record holds the instruction count, compute-node
count, depth, the width list, peak live slots, and per dtype whether the plan
cusadi2 returns at the device limit (`extract_level_plan(fn, max_live=
device_max_live(dtype))`, the call the benchmarks make) is the plain plan or a
windowed one, or whether that call raised.

    python dag_shape.py [--robot R] [--force]     # cusadi2 conda env

`paper_results/dag_shape.py` reads the cache to draw the menagerie on its figure.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import DTYPES, MENAGERIE, load_manifest, robot_xml, _git   # noqa: E402

STEMS = ["fk", "fkjac", "fkhess"]
OUT = os.environ.get("CUSADI2_DAG_RESULTS", os.path.join(HERE, "results", "dag"))


def compute_widths(plan):
    """Compute-node count per level, dropping levels that hold only IO nodes."""
    from cusadi2.dag import IO_OPS
    ops = np.asarray(plan.node_ops)
    is_compute = ~np.isin(ops, list(IO_OPS))
    widths = [int(is_compute[o:o + w].sum()) for o, w in zip(plan.level_offsets, plan.level_widths)]
    return [w for w in widths if w > 0]


def record_path(robot):
    return os.path.join(OUT, f"{robot}.json")


def robot_size(xml):
    """(nq, nv, nbody) from MuJoCo, for the sweep order and the record."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    return int(m.nq), int(m.nv), int(m.nbody)


# ---------------------------------------------------------------------------
def worker(manifest, robot):
    import model
    import cusadi2
    from cusadi2.dag import extract_level_plan
    xml = robot_xml(manifest, robot)
    rec = dict(robot=robot, xml=os.path.relpath(xml, MENAGERIE), status="partial",
               menagerie_head=_git(["rev-parse", "HEAD"]), functions={})
    rec["nq"], rec["nv"], rec["n_bodies"] = robot_size(xml)
    max_live = {d: cusadi2.device_max_live(d) for d in DTYPES}

    def save():
        os.makedirs(OUT, exist_ok=True)
        with open(record_path(robot) + ".tmp", "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(record_path(robot) + ".tmp", record_path(robot))

    save()
    m = model.load_mujoco(xml)
    pmodel = model.load_pinocchio(xml, m=m)            # same repaired load path as the campaign
    names = model.body_names(m)
    corr = model.root_correction(m, pmodel, model.qpos_mj_to_pin_index(m, pmodel))
    builders = {"fk": model.build_fk, "fkjac": model.build_fkjac, "fkhess": model.build_fkhess}
    for stem in STEMS:
        rec["current"] = stem                          # what was running if the worker dies
        save()
        t0 = time.perf_counter()
        fn = builders[stem](pmodel, names, name=stem, root_corr=corr)
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        plain = extract_level_plan(fn)
        plan_s = time.perf_counter() - t0
        widths = compute_widths(plain)
        r = dict(n_instr=fn.n_instructions(), nnz_out=fn.nnz_out(0), n_compute=int(sum(widths)),
                 depth=len(widths), mean_width=sum(widths) / len(widths),
                 median_width=float(np.median(widths)), max_width=max(widths), widths=widths,
                 peak_live_slots=plain.peak_live_slots, n_levels=plain.n_levels,
                 build_s=build_s, plan_s=plan_s, fit={})
        for d in DTYPES:                               # the call the benchmarks make
            try:
                fitted = extract_level_plan(fn, max_live=max_live[d])
                r["fit"][d] = dict(max_live=max_live[d], windowed=fitted.n_levels != plain.n_levels,
                                   n_levels=fitted.n_levels, peak_live_slots=fitted.peak_live_slots)
            except Exception as e:                     # noqa: BLE001
                r["fit"][d] = dict(max_live=max_live[d], error=f"{type(e).__name__}: {e}")
        rec["functions"][stem] = r
        print(f"  [{robot}] {stem}: instr={r['n_instr']} nodes={r['n_compute']} depth={r['depth']} "
              f"mean={r['mean_width']:.1f} peak_live={r['peak_live_slots']} build={build_s:.1f}s "
              f"plan={plan_s:.1f}s fit=" + ",".join(
                  f"{d}:" + ("ERR" if "error" in r["fit"][d] else ("windowed" if r["fit"][d]["windowed"] else "plain"))
                  for d in DTYPES), flush=True)
        save()
    rec.pop("current", None)
    rec["status"] = "ok"
    save()


# ---------------------------------------------------------------------------
def sweep(manifest, robots, force):
    os.makedirs(OUT, exist_ok=True)
    head = _git(["rev-parse", "HEAD"])
    if head != manifest.MENAGERIE_COMMIT:
        print(f"WARNING: menagerie HEAD {head} != manifest MENAGERIE_COMMIT {manifest.MENAGERIE_COMMIT}")
    sized = []
    for robot in robots:
        spec = manifest.ROBOTS[robot]
        if not spec.get("xml"):
            rec = dict(robot=robot, status="skip", reason=f"manifest: {spec.get('skip', 'no xml')}")
            with open(record_path(robot), "w") as f:
                json.dump(rec, f, indent=1)
            continue
        sized.append((robot_size(robot_xml(manifest, robot))[1], robot))
    for nv, robot in sorted(sized):
        path = record_path(robot)
        if not force and os.path.exists(path) and json.load(open(path)).get("status") == "ok":
            print(f"[{robot}] cached")
            continue
        if os.path.exists(path):
            os.remove(path)
        print(f"[{robot}] nv={nv} ...", flush=True)
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, __file__, "--worker", robot, "--manifest", os.environ["CUSADI2_MANIFEST"]],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        sys.stdout.write(p.stdout)
        rec = json.load(open(path)) if os.path.exists(path) else dict(robot=robot, status="partial", functions={})
        rec["wall_s"] = time.perf_counter() - t0
        if p.returncode != 0 or rec.get("status") != "ok":
            rec["status"] = "failed"
            rec["failed"] = dict(returncode=p.returncode, during=rec.pop("current", None),
                                 stderr_tail=p.stderr[-4000:])
            print(f"[{robot}] FAILED rc={p.returncode} during={rec['failed']['during']}: "
                  + (p.stderr.strip().splitlines() or ["(no stderr)"])[-1], flush=True)
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
    report(robots)


def report(robots):
    print("\n=== summary ===")
    windowed = {d: [] for d in DTYPES}
    fit_errors, failed, missing = [], [], []
    for robot in robots:
        path = record_path(robot)
        if not os.path.exists(path):
            missing.append(robot)
            continue
        rec = json.load(open(path))
        if rec["status"] == "failed":
            failed.append(f"{robot} (during {rec['failed'].get('during')}, rc={rec['failed'].get('returncode')})")
        for stem, r in rec.get("functions", {}).items():
            for d, f in r["fit"].items():
                if "error" in f:
                    fit_errors.append(f"{robot}/{stem} {d}: {f['error']}")
                elif f["windowed"]:
                    windowed[d].append(f"{robot}/{stem} ({r['n_levels']} -> {f['n_levels']} levels, "
                                       f"peak_live {r['peak_live_slots']} -> {f['peak_live_slots']})")
    for d in DTYPES:
        print(f"windowed at {d} (max_live fits): {len(windowed[d])}")
        for w in windowed[d]:
            print("   ", w)
    print(f"fit errors (library could not window under the limit): {len(fit_errors)}")
    for e in fit_errors:
        print("   ", e)
    print(f"failed robots: {len(failed)}")
    for f in failed:
        print("   ", f)
    if missing:
        print(f"no record: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", help="one robot (default: every manifest robot)")
    ap.add_argument("--force", action="store_true", help="recompute cached robots")
    ap.add_argument("--manifest")
    ap.add_argument("--worker", help="(internal) compute one robot in-process")
    ap.add_argument("--report", action="store_true", help="only print the summary of the cache")
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    if args.worker:
        return worker(manifest, args.worker)
    robots = [args.robot] if args.robot else list(manifest.ROBOTS)
    if args.report:
        return report(robots)
    sweep(manifest, robots, args.force)


if __name__ == "__main__":
    main()
