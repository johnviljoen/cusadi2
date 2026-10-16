"""The cost-shaping ladder: run closed_loop.py over a list of configurations and tabulate results/<tag>/summary.json.

    python ladder.py run  --configs rung2            (runs the named configuration set, sequentially, no video)
    python ladder.py table                           (prints every results/*/summary.json as a table)

Rung 1: the pure goal cost. Rung 2: the same cost, solver effort only (T, sigma per knot, temperature, projection
iterations; K = 256 fixed). Rung 3: + the TCP funnel (w_fun). Rung 4: + the wrist funnel (w_wrist). Rung 5: + the joint-space reference (w_q, the IK
pre-grasp configuration), since no in-plane arm posture reaches the cup (feasibility.py --inplane).
"""
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = {
    "rung2": [
        dict(tag="rung2_T150", rung=2, T=150),
        dict(tag="rung2_T200", rung=2, T=200),
        dict(tag="rung2_sig1234", rung=2, T=100, sigma="0.1 0.2 0.3 0.4"),
        dict(tag="rung2_T150_sig1234", rung=2, T=150, sigma="0.1 0.2 0.3 0.4"),
        dict(tag="rung2_temp01", rung=2, T=100, temp=0.1),
        dict(tag="rung2_temp1", rung=2, T=100, temp=1.0),
        dict(tag="rung2_proj128", rung=2, T=100, proj_iters=128),
    ],
    "rung3": [
        dict(tag="rung3_fun20", rung=3, T=100, par="w_fun=20"),
        dict(tag="rung3_fun100", rung=3, T=100, par="w_fun=100"),
    ],
    "rung5": [
        dict(tag="rung5_q1", rung=5, T=100, par="w_q=1"),
        dict(tag="rung5_q5", rung=5, T=100, par="w_q=5"),
        dict(tag="rung5_q20", rung=5, T=100, par="w_q=20"),
    ],
    "rung4": [
        dict(tag="rung4_fun20_wr20", rung=4, T=100, par="w_fun=20,w_wrist=20"),
        dict(tag="rung4_fun100_wr100", rung=4, T=100, par="w_fun=100,w_wrist=100"),
    ],
}


def run(cfgs, extra):
    for c in cfgs:
        cmd = [sys.executable, str(HERE / "closed_loop.py"), "--no-video", "--tag", c["tag"], "--rung", str(c["rung"]), "--T", str(c.get("T", 100))]
        if "sigma" in c: cmd += ["--sigma", *c["sigma"].split()]
        if "temp" in c: cmd += ["--temp", str(c["temp"])]
        if "proj_iters" in c: cmd += ["--proj-iters", str(c["proj_iters"])]
        if "par" in c: cmd += ["--par", c["par"]]
        cmd += extra
        print("==", " ".join(cmd[2:]), flush=True)
        log = HERE / "logs" / f"{c['tag']}.log"
        with open(log, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        s = HERE / "results" / c["tag"] / "summary.json"
        if s.exists():
            d = json.loads(s.read_text()); print(f"   reached {d['n_reached']}/{d['n']}  grasped {d['n_grasp']}/{d['n']}  h_min {d['h_min']:.4f}", flush=True)
        else:
            print("   FAILED (see", log, ")", flush=True)


def table():
    rows = []
    for s in sorted((HERE / "results").glob("*/summary.json")):
        d = json.loads(s.read_text())
        if "results" not in d or "rung" not in d: continue
        r = d["results"]
        rows.append((d["tag"], d["rung"], d["T"], " ".join(f"{v:.2f}" for v in d["sigma"]), d["temp"], d["proj_iters"], d["par"].get("w_fun", 0), d["par"].get("w_wrist", 0), d["par"].get("w_q", 0),
                     f"{d['n_reached']}/{d['n']}", f"{d['n_grasp']}/{d['n']}", f"{d['h_min']:.4f}",
                     " ".join(f"{x['final']['ax'] * 100:.0f}" for x in r), " ".join(f"{x['final']['rot'] * 57.3:.0f}" for x in r),
                     f"{sum(x['plant_active_frac'] for x in r) / len(r) * 100:.0f}", f"{sum(x['ours_ms_median'] for x in r) / len(r):.1f}"))
    hdr = ("tag", "rung", "T", "sigma", "temp", "proj", "w_fun", "w_wr", "w_q", "reached", "grasped", "h_min", "final ax cm", "final rot deg", "plant act %", "tick ms")
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    for line in (hdr, *rows):
        print("  ".join(str(x).ljust(wi) for x, wi in zip(line, w)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["run", "table"]); ap.add_argument("--configs", default="rung2")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    if a.what == "run":
        run(CONFIGS[a.configs], a.extra)
    else:
        table()
