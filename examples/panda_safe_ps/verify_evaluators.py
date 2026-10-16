"""Gate 1b: every evaluator computes the same controller (fp32). The driver is run with the same seed, plant state, plan
and waypoint for each evaluator, so the sampled plans are bit-identical, and the applied torque, the recorded rollout
trajectories (T + 1 states of the first 32 samples, a T-step chain of model + projection + integrate), the final costs
and the selected sample index are compared with the cusadi2 run. Gate 1c, for every evaluator: the applied torque
(the gathered step-0 torque of the selected rollout) equals the selected plan's torque recomputed at the plant state.

    python verify_evaluators.py [--T 100] [--K 256] [--evaluators frax mjx cusadi cpu]

Tolerances are fp32: 1e-5 relative on torques and trajectories, the same selected index on every tick, and the gathered
torque equal to the recompute to 1e-6 (the same kernels on the same inputs; the PJRT rows run the batch-B program on a
replicated state). Writes results/verify_evaluators.json.
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=100); ap.add_argument("--K", type=int, default=256)
ap.add_argument("--evaluators", nargs="+", default=["frax", "mjx", "cusadi", "cpu"]); ap.add_argument("--ticks", type=int, default=3)
a = ap.parse_args()
from driver import SafePS
import common
common.ensure_frax()
from frax import load_panda
q0 = np.asarray(load_panda().default_configuration)
WP = np.array([[0.05, 0.55, 0.55]])


def run(ev):
    d = SafePS(ev, K=a.K, T=a.T, graph=False)
    x = np.zeros((1, 16)); x[0, :7] = q0; d.set_plan(np.tile(q0, (1, 4)))
    out = []
    for _ in range(a.ticks):                        # external plant: the tick's plant state stays in the driver for the recompute
        u, stat, gm, wm = d.tick(x, WP)
        Y, xT = d.samples(); rc = d.recompute()
        out.append(dict(u=u.copy(), traj=d.traj(), cost=xT[:, 15].copy(), ks=int(stat[0, 3]), Y=Y, recompute=rc, plant=x.copy()))
        x = np.zeros((1, 16)); x[0, :14] = xT[stat[0, 3].astype(int), :14]      # follow the selected rollout one step
    info = d.info; d.close()
    return out, info


def rel(p, q):
    return float(np.abs(p - q).max() / max(1.0, np.abs(q).max()))


ref, ref_info = run("cusadi2")
print("cusadi2:", ref_info, flush=True)
res = {}
gate_rc = {"cusadi2": max(rel(o["u"], o["recompute"]) for o in ref)}
print(f"cusadi2  gathered torque vs recompute {gate_rc['cusadi2']:.1e}  selected {[o['ks'] for o in ref]}", flush=True)
for ev in a.evaluators:
    try:
        out, info = run(ev)
    except Exception as e:
        print(f"{ev}: {type(e).__name__}: {str(e)[:200]}", flush=True); res[ev] = dict(error=str(e)[:200]); continue
    r = dict(info=info, Y=max(rel(o["Y"], p["Y"]) for o, p in zip(out, ref)), u=max(rel(o["u"], p["u"]) for o, p in zip(out, ref)),
             traj=max(rel(o["traj"], p["traj"]) for o, p in zip(out, ref)), cost=max(rel(o["cost"], p["cost"]) for o, p in zip(out, ref)),
             selected=[o["ks"] for o in out], same_selected=all(o["ks"] == p["ks"] for o, p in zip(out, ref)),
             recompute=max(rel(o["u"], o["recompute"]) for o in out))
    res[ev] = r
    print(f"{ev:8s} vs cusadi2 over {a.ticks} ticks: applied torque {r['u']:.1e}  rollout trajectories {r['traj']:.1e}  "
          f"costs {r['cost']:.1e}  selected {r['selected']} same {r['same_selected']}  gathered vs recompute {r['recompute']:.1e}   [{info[:90]}]", flush=True)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "verify_evaluators.json").write_text(json.dumps(dict(T=a.T, K=a.K, ticks=a.ticks, cusadi2=ref_info, cusadi2_selected=[o["ks"] for o in ref],
                                                                   cusadi2_recompute=gate_rc["cusadi2"], evaluators=res), indent=2))
ok = all("error" not in r and r["traj"] < 1e-5 and r["u"] < 1e-5 and r["same_selected"] for r in res.values())
ok_rc = gate_rc["cusadi2"] < 1e-6 and all("error" not in r and r["recompute"] < 1e-6 for r in res.values())
print("GATE 1b (evaluators agree)", "PASS" if ok else "FAIL", "  GATE 1c (gathered torque = recompute)", "PASS" if ok_rc else "FAIL")
