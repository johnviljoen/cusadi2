"""Diagnostics for the closed-loop rollout written by mpc_rollout.cu: per instance, base height and tilt,
tracking of the commanded forward speed and yaw rate, and the constraint residuals of the executed stage
(stage 0 of each step's plan x+, evaluated with the nonlinear g; it is what the plant then follows).

usage: python rollout_report.py <build_dir>      -> results/rollout_report.csv, pass/fail on stdout
Pass = every state finite, base height within [0.5, 0.8] m and tilt below 0.35 rad for the whole rollout.
"""
import csv, json, os, sys, numpy as np, casadi as ca
BUILD = sys.argv[1]; R = os.path.join(BUILD, "rollout"); HERE = os.path.dirname(os.path.abspath(__file__))
meta, rm = json.load(open(os.path.join(BUILD, "meta.json"))), json.load(open(os.path.join(R, "meta.json")))
N, NS, B, T, dt0 = meta["N"], meta["NS"], rm["B"], rm["n_steps"], rm["dt0"]
state = np.fromfile(os.path.join(R, "state.f64")).reshape(T + 1, B, -1)
plan = np.fromfile(os.path.join(R, "plan.f64")).reshape(T, B, N, NS)
p_stage = np.fromfile(os.path.join(R, "p_stage.f64")).reshape(T, B, N, -1)
lat = np.fromfile(os.path.join(R, "latency_us.f64"))
g_fn = ca.Function.load(os.path.join(BUILD, "fns", "mpc_stage_g.casadi"))
eq = np.zeros(len(meta["lbg"]), bool); eq[meta["eq_rows"]] = True
lbg, ubg = np.array(meta["lbg"], float), np.array(meta["ubg"], float)
GROUPS = [("integration", slice(0, 24), ""), ("dynamics", slice(24, 30), "N, Nm"), ("swing_z", slice(30, 34), "m"),
          ("contact_vel", slice(50, 58), "m/s")]
q, t = state[:, :, :25], np.arange(T + 1) * dt0
yaw = np.unwrap(2 * np.arctan2(q[:, :, 5], q[:, :, 6]), axis=0)
tilt = np.arccos(np.clip(1 - 2 * (q[:, :, 3] ** 2 + q[:, :, 4] ** 2), -1, 1))            # angle between body z and world z
dp = np.diff(q[:, :, :2], axis=0) / dt0
vx_fd = dp[:, :, 0] * np.cos(yaw[:-1]) + dp[:, :, 1] * np.sin(yaw[:-1]); wz_fd = np.diff(yaw, axis=0) / dt0
rows, ok = [], bool(np.all(np.isfinite(state)))
print(f"{'i':>2} {'vx cmd':>6} {'vx':>6} {'wz cmd':>6} {'wz':>6} {'z min':>6} {'z max':>6} {'tilt':>5} | executed-stage residual, median / max after step 10"
      f" ({', '.join(n for n, _, _ in GROUPS)}, ineq)")
for b in range(B):
    c = rm["commands"][b]; steady = t[:-1] >= c["start"] + rm["RAMP"]
    g = np.stack([np.asarray(g_fn(np.zeros(NS), plan[s, b, 0], dt0, 1.0, p_stage[s, b, 0], state[s, b])).reshape(-1) for s in range(T)])
    viol = np.maximum(np.maximum(lbg - g, g - ubg), 0)[:, ~eq].max(axis=1)
    res = {n: np.abs(g[:, sl]).max(axis=1) for n, sl, _ in GROUPS}; res["ineq"] = viol
    good = 0.5 <= q[:, b, 2].min() and q[:, b, 2].max() <= 0.8 and tilt[:, b].max() < 0.35; ok = ok and good
    row = dict(instance=b, vx_cmd=c["vx"], vx_mean=vx_fd[steady, b].mean(), wz_cmd=c["wz"], wz_mean=wz_fd[steady, b].mean(),
               z_min=q[:, b, 2].min(), z_max=q[:, b, 2].max(), tilt_max=tilt[:, b].max(), upright=int(good))
    for n, r in res.items(): row[n + "_median"], row[n + "_max"], row[n + "_max_step0_9"] = np.median(r), r[10:].max(), r[:10].max()
    rows.append(row)
    print(f"{b:2d} {c['vx']:6.2f} {row['vx_mean']:6.3f} {c['wz']:6.2f} {row['wz_mean']:6.3f} {row['z_min']:6.3f} {row['z_max']:6.3f} {row['tilt_max']:5.3f} | "
          + "  ".join(f"{np.median(r):.1e}/{r[10:].max():.1e}" for r in res.values()))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "rollout_report.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
    for r in rows: w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
print(f"step latency (B = {B}): median {np.median(lat) / 1e3:.2f} ms, max {lat.max() / 1e3:.2f} ms; control period {dt0 * 1e3:.2f} ms")
print("PASS" if ok else "FAIL", f"({T} steps, {T * dt0:.2f} s)")
sys.exit(0 if ok else 1)
