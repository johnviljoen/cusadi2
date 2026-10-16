"""Feasibility gate of the cupboard scene: is the pre-grasp pose reachable with clearance, and is a straight-in path?

    python feasibility.py [--n-path 16] [--x-start 0.27]

For each TCP target (the pre-grasp, then a straight line along +x from x_start to it, warm-started in sequence) solves
the max-min-clearance IK

    max_{q, s} s   s.t.  p_tcp(q) = p*,  e_rot(q) = 0 (tool axis +x, fingers along +-y),
                         h2_geo(q) >= s  (self-collision, floor, slab, cup rows),
                         joint-limit rows >= 0, singularity row >= 0

and reports s* (the best achievable clearance at that pose) and the rows within 1 mm of it. The gate: s* >= 1 cm at
the pre-grasp and along the whole path. Writes results/feasibility.json.
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "panda_backup_cbf")); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import numpy as np
import casadi as ca
from backup_staged import _Model
import scene

ap = argparse.ArgumentParser()
ap.add_argument("--n-path", type=int, default=16)
ap.add_argument("--x-start", type=float, default=scene.X_OPEN - 0.15)
ap.add_argument("--flip", action="store_true", help="the other finger sense (tool rotated 180 deg about its axis)")
ap.add_argument("--tag", default="")
ap.add_argument("--inplane", action="store_true", help="restrict q1, q3, q5 to [-0.05, 0.05]: the in-plane (elbow-up) arm")
a = ap.parse_args()

m = _Model()
q = ca.SX.sym("q", 7)
K = m.kin(q, ca.SX.zeros(7))
h2 = scene.barriers_casadi(m, q, K)
sl = scene.group_slices()
R_des = scene.R_PREGRASP.copy()
if a.flip:
    R_des[:, 0] *= -1; R_des[:, 1] *= -1
rot, pos = K["rot"], K["pos"]
e_rot = -0.5 * sum(ca.cross(rot[:, i], ca.DM(R_des[:, i])) for i in range(3))
f_h2 = ca.Function("h2", [q], [h2]); f_pose = ca.Function("pose", [q], [pos, rot])

geo = list(range(sl["self"].start, scene.N_ROWS))
p_t = ca.SX.sym("p", 3); s = ca.SX.sym("s"); qr = ca.SX.sym("qr", 7)
w = ca.vertcat(q, s)
align = ca.vertcat(*[ca.dot(rot[:, i], ca.DM(R_des[:, i])) for i in range(3)])   # excludes the flipped solutions of e_rot = 0
g = ca.vertcat(pos - p_t, e_rot, h2[geo] - s, h2[sl["joint"]], h2[sl["singularity"]], align)
lbg = np.r_[np.zeros(6), np.zeros(len(geo)), np.zeros(15), np.full(3, 0.5)]; ubg = np.r_[np.zeros(6), np.full(len(geo) + 18, np.inf)]
# seeds: unconstrained pose IK (position + trace orientation error) from random configurations
seed_solver = ca.nlpsol("seed", "ipopt", {"x": q, "p": p_t, "f": 3 - ca.trace(ca.DM(R_des).T @ rot), "g": pos - p_t},
                        {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-12})
solver = ca.nlpsol("ik", "ipopt", {"x": w, "p": ca.vertcat(p_t, qr), "f": -s + 1e-4 * ca.sumsqr(q - qr), "g": g},
                   {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 3000, "ipopt.tol": 1e-10})

NAMES = ([f"q{i}_max" for i in range(7)] + [f"q{i}_min" for i in range(7)] + ["singularity"] + [f"self{i}" for i in range(scene.N_SC)]
         + [f"ground:s{i}" for i in range(scene.N_SPHERES)] + [f"{n}:s{i}" for i in range(scene.N_SPHERES) for n in scene.SLAB_NAMES]  # slab rows: "<slab>:s<sphere>"
         + [f"cup:s{i}" for i in scene.CUP_SPHERES])
assert len(NAMES) == scene.N_ROWS


def solve(p, q0):
    lbq, ubq = np.asarray(m.q_min).ravel().copy(), np.asarray(m.q_max).ravel().copy()
    if a.inplane:
        lbq[[0, 2, 4]] = -0.05; ubq[[0, 2, 4]] = 0.05
    r = solver(x0=np.r_[q0, 0.0], p=np.r_[p, q0], lbx=np.r_[lbq, -1.0], ubx=np.r_[ubq, 1.0],
               lbg=lbg, ubg=ubg)
    st = solver.stats()
    qs = np.asarray(r["x"]).ravel()[:7]; sv = float(r["x"][7])
    hv = np.asarray(f_h2(qs)).ravel(); P, R = (np.asarray(v) for v in f_pose(qs))
    hg = hv[geo]
    tight = [(NAMES[geo[i]], round(float(hg[i]), 4)) for i in np.argsort(hg)[:6] if hg[i] < sv + 1e-3]
    return dict(ok=st["success"], status=st["return_status"], s=sv, q=qs.tolist(), pos_err=float(np.linalg.norm(P.ravel() - p)),
                rot_err=float(np.arccos(np.clip((np.trace(R_des.T @ R) - 1) / 2, -1, 1))), min_joint=float(hv[sl["joint"]].min()),
                sing=float(hv[sl["singularity"]][0]), tight=tight)


rng = np.random.default_rng(0)
lo, hi = np.asarray(m.q_min).ravel(), np.asarray(m.q_max).ravel()
best = None
for k in range(24):
    q0 = lo + (hi - lo) * rng.uniform(0.15, 0.85, 7) if k else np.array([0.0, 0.3, 0.0, -1.6, 0.0, 1.9, 0.785])
    rs = seed_solver(x0=q0, p=scene.P_PREGRASP, lbx=lo, ubx=hi, lbg=0, ubg=0)
    if not seed_solver.stats()["success"] or float(rs["f"]) > 1e-8:
        continue
    r = solve(scene.P_PREGRASP, np.asarray(rs["x"]).ravel())
    if r["ok"] and (best is None or r["s"] > best["s"]):
        best = r
    print(f"  seed {k:2d}: s* = {r['s']*100:6.2f} cm ({r['status'][:24]})", flush=True)
if best is None:
    sys.exit("GATE FAIL: the pre-grasp pose is not reachable (no seed reaches it exactly)")
pre = best
print(f"pre-grasp: s* = {pre['s']*100:.2f} cm  ({pre['status']}, pos err {pre['pos_err']:.1e}, rot err {pre['rot_err']:.1e}, "
      f"joint margin {pre['min_joint']:.3f}, sing {pre['sing']:.4f})")
print("  tight:", pre["tight"]); print("  q:", np.round(pre["q"], 3).tolist())
path = []
xs = np.linspace(a.x_start, scene.P_PREGRASP[0], a.n_path)
qw = np.array(pre["q"])
for x in xs[::-1]:                      # from the pre-grasp outwards, warm-started: the in-cupboard end is the hard one
    p = scene.P_PREGRASP.copy(); p[0] = x
    r = solve(p, qw); qw = np.array(r["q"]); r["x"] = float(x); path.append(r)
    print(f"  x = {x:.3f}: s* = {r['s']*100:6.2f} cm  {r['status'][:20]:20s} tight {r['tight'][:3]}")
path = path[::-1]
s_path = min(r["s"] for r in path)
gate = pre["s"] >= 0.01 and s_path >= 0.01 and all(r["ok"] for r in path) and pre["ok"]
print(f"GATE {'PASS' if gate else 'FAIL'}: pre-grasp {pre['s']*100:.2f} cm, path min {s_path*100:.2f} cm")
out = HERE / "results"; out.mkdir(exist_ok=True)
(out / f"feasibility{'_flip' if a.flip else ''}{'_inplane' if a.inplane else ''}{a.tag}.json").write_text(json.dumps(dict(
    gate=gate, n_rows=scene.N_ROWS, cupboard=dict(W=scene.W, H=scene.H, D=scene.D, x_open=scene.X_OPEN, z_floor=scene.Z_FLOOR),
    cup=dict(x=scene.CUP_X, r=scene.CUP_R, h=scene.CUP_H), p_pregrasp=scene.P_PREGRASP.tolist(), pregrasp=pre, path=path), indent=1))
