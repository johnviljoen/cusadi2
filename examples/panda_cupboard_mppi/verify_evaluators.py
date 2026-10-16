"""Gate 1b: every evaluator computes the same controller. The driver is run with the same seed, plant state, plan and
waypoint for each evaluator, so the sampled plans are bit-identical, and the applied torque, the recorded rollout
trajectories (T + 1 states of the first 32 samples, a T-step chain of model + projection + integrate) and the MPPI
weights are compared with the cusadi2 run.

    python verify_evaluators.py [--T 100] [--K 256] [--evaluators frax mjx cusadi cpu]

Writes results/verify_evaluators.json.
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
from driver import SafeMppi
import common
common.ensure_frax()
from frax import load_panda
q0 = np.asarray(load_panda().default_configuration)
import scene
WP = scene.P_PREGRASP[None]


def run(ev):
    d = SafeMppi(ev, K=a.K, T=a.T, graph=False)
    x = np.zeros((1, 16)); x[0, :7] = q0; d.set_plant(x); d.set_plan(np.tile(q0, (1, 4)))
    out = []
    for _ in range(a.ticks):
        u, stat, gm, wm = d.tick(None, WP)
        out.append(dict(u=u.copy(), traj=d.traj(), w=d.weights(), Y=d.samples()[0], plant=d.plant()))
    info = d.info; d.close()
    return out, info


def rel(p, q):
    return float(np.abs(p - q).max() / max(1.0, np.abs(q).max()))


ref, ref_info = run("cusadi2")
print("cusadi2:", ref_info, flush=True)
res = {}
for ev in a.evaluators:
    try:
        out, info = run(ev)
    except Exception as e:
        print(f"{ev}: {type(e).__name__}: {str(e)[:200]}", flush=True); res[ev] = dict(error=str(e)[:200]); continue
    r = dict(info=info, Y=max(rel(o["Y"], p["Y"]) for o, p in zip(out, ref)), u=max(rel(o["u"], p["u"]) for o, p in zip(out, ref)),
             traj=max(rel(o["traj"], p["traj"]) for o, p in zip(out, ref)), w=max(rel(o["w"], p["w"]) for o, p in zip(out, ref)),
             plant=max(rel(o["plant"], p["plant"]) for o, p in zip(out, ref)))
    res[ev] = r
    # the sample buffer is overwritten by the plant step of the non-capturable evaluators (replicated mean plan), so `Y` is
    # only comparable for the capturable ones; the trajectories are the test of identical inputs and outputs
    print(f"{ev:8s} vs cusadi2 over {a.ticks} ticks: applied torque {r['u']:.1e}  rollout trajectories {r['traj']:.1e}  "
          f"weights {r['w']:.1e}  plant {r['plant']:.1e}   [{info[:90]}]", flush=True)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "verify_evaluators.json").write_text(json.dumps(dict(T=a.T, K=a.K, ticks=a.ticks, cusadi2=ref_info, evaluators=res), indent=2))
ok = all("error" not in r and r["traj"] < 1e-8 and r["u"] < 1e-8 for r in res.values())
print("GATE 1b", "PASS" if ok else "FAIL")
