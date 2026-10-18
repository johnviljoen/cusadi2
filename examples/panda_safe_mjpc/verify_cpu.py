"""Gate: the CPU-only driver computes the same controller as the GPU driver. The GPU driver with the cusadi2 evaluator runs
one-iteration ticks from the default configuration (its samples() returns the last iteration's samples, so the handshake is
per iteration: n_iter = 1 on both sides; the n_iter loop is the same three lines in both drivers); its sampled plans are
handed to the CPU driver (set_samples) started from the same plant state and plan; the applied torque, the final rollout
states, the selected index, the resampled plan and the model plant's next state are compared. Repeated for 3 ticks (each
time re-synchronised). The two evaluate the same CasADi graph through different code (level plans vs C codegen), so fp32
agrees to ~1e-6. Then a 3-iteration tick of each is run on its own samples as a smoke check of the iteration loop (costs of
the selected sample must be non-increasing over the iterations on both).

    python verify_cpu.py [--K 256] [--T 35]
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cpu_driver import CpuPS  # noqa: E402
import numpy as np
from driver import SafePS
from model import NX, N_KNOT
import common
common.ensure_frax()
from frax import load_panda

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=256); ap.add_argument("--T", type=int, default=35)
a = ap.parse_args()
q0 = np.asarray(load_panda().default_configuration)
WP = np.array([[[0.05, 0.55, 0.55]], [[0.55, 0.05, 0.30]], [[0.05, 0.55, 0.30]]])
def rel(p, q): return float(np.abs(p - q).max() / max(1.0, np.abs(q).max()))

g = SafePS("cusadi2", K=a.K, T=a.T, n_iter=1, graph=False)
c = CpuPS(K=a.K, T=a.T, n_iter=1)
print(g.info); print(c.info, flush=True)
x0 = np.zeros((1, NX)); x0[0, :7] = q0
worst = {}; same_ks = True
for k in range(3):
    for d in (g, c): d.set_plant(x0); d.set_plan(np.tile(q0, (1, N_KNOT)))
    ug, sg, _, _ = g.tick(None, WP[k])
    Yg, xTg = g.samples()
    c.set_samples(Yg)
    uc, sc, ms = c.tick(None, WP[k])
    Yc, xTc = c.samples()
    fin = np.isfinite(xTg).all(axis=1) & np.isfinite(xTc).all(axis=1)
    r = dict(samples=rel(Yc, Yg), u=rel(uc, ug), xT=rel(xTc[fin], xTg[fin]), plan=rel(c.plan(), g.plan()), plant=rel(c.plant(), g.plant()), cost_sel=rel(sc[0, 0], sg[0, 0]))
    same_ks = same_ks and int(sc[0, 3]) == int(sg[0, 3])
    for kk, v in r.items(): worst[kk] = max(worst.get(kk, 0.0), v)
    print("tick", k, {kk: f"{v:.2e}" for kk, v in r.items()}, "ks", int(sg[0, 3]), int(sc[0, 3]), f"nonfinite {int((~fin).sum())}", f"cpu tick {ms:.2f} ms", flush=True)
g.close(); c.close()
g3 = SafePS("cusadi2", K=a.K, T=a.T, n_iter=3); c3 = CpuPS(K=a.K, T=a.T, n_iter=3)
mono = {}
for name, d in (("gpu", g3), ("cpu", c3)):
    d.set_plant(x0); d.set_plan(np.tile(q0, (1, N_KNOT)))
    for _ in range(5): d.tick(None, WP[0])
    cs = d.iter_stats()[:, 0, 0]; mono[name] = bool(np.all(np.diff(cs) <= 1e-6)); print(name, "selected costs over the 3 iterations of tick 5:", np.round(cs, 4).tolist())
ok = worst["u"] < 1e-5 and worst["xT"] < 1e-5 and worst["plan"] < 1e-5 and worst["plant"] < 1e-5 and same_ks and all(mono.values())
print("worst:", {k: f"{v:.2e}" for k, v in worst.items()}, "same selected index:", same_ks, "monotone:", mono, "->", "PASS" if ok else "FAIL")
(HERE / "results").mkdir(exist_ok=True)
json.dump(dict(K=a.K, T=a.T, worst=worst, same_selected_index=same_ks, monotone=mono, gpu_info=g3.info, cpu_info=c3.info, ok=ok), open(HERE / "results" / "verify_cpu.json", "w"), indent=1)
