"""Per-robot outcome of a closed-loop run written by mppi_rollout.cu.

    python report.py <build_dir> <run_dir> [out.csv]

Pass = finite, upright (base z-axis within 35 deg of vertical over the last second), parked within
GOAL_TOL of its spot at the end, and never fallen (base less than MIN_CLEAR above the terrain).
"""
import json
import os
import sys

import numpy as np

import course
import go2_model as G

GOAL_TOL, MIN_CLEAR, MAX_TILT = 0.15, 0.05, np.deg2rad(35)


def analyse(build, run):
    meta = json.load(open(os.path.join(build, "meta.json")))
    dtc = meta["dt"] * meta["substeps"]
    rob = np.fromfile(os.path.join(run, "robots.f64")).reshape(-1, 6)
    R = len(rob)
    X = np.fromfile(os.path.join(run, "state.f64")).reshape(-1, R, G.NX)
    lat = np.fromfile(os.path.join(run, "latency_us.f64"))
    n = X.shape[0] - 1
    import pinocchio as pin
    _, pm, _, fids = G.load(meta["dt"])
    data = pm.createData()
    rows = []
    for r in range(R):
        x = X[:, r]
        feet = np.zeros((n + 1, 4))                                            # foot clearance above the terrain
        for s in range(n + 1):
            if np.isfinite(x[s, :G.NQ]).all():
                pin.framesForwardKinematics(pm, data, x[s, :G.NQ])
                feet[s] = [data.oMf[f].translation[2] - G.FOOT_R - course.height(*data.oMf[f].translation[:2]) for f in fids[:4]]
        finite = bool(np.isfinite(x).all())
        qx, qy = x[:, 3], x[:, 4]
        tilt = np.arccos(np.clip(1 - 2 * (qx ** 2 + qy ** 2), -1, 1))          # angle of the base z-axis from vertical
        clear = x[:, 2] - course.height(x[:, 0], x[:, 1])
        dist = np.hypot(x[:, 0] - rob[r, 3], x[:, 1] - rob[r, 4])
        arrived = np.nonzero(dist < GOAL_TOL)[0]
        last = slice(-int(1.0 / dtc), None)
        ok = finite and dist[-1] < GOAL_TOL and tilt[last].max() < MAX_TILT and clear.min() > MIN_CLEAR
        rows.append(dict(robot=r, passed=int(ok), finite=int(finite), final_dist=dist[-1], t_arrive=arrived[0] * dtc if len(arrived) else np.nan,
                         min_clear=clear.min(), max_tilt_deg=np.rad2deg(tilt.max()), end_tilt_deg=np.rad2deg(tilt[last].max()),
                         mean_speed=np.hypot(np.diff(x[:, 0]), np.diff(x[:, 1])).sum() / (n * dtc), end_z=x[-1, 2],
                         swing_front=np.percentile(feet[:, :2], 95, axis=0).mean(), swing_rear=np.percentile(feet[:, 2:], 95, axis=0).mean()))
    return rows, dict(steps=n, sim_s=n * dtc, median_ms=float(np.median(lat)) / 1e3, rtf=dtc * 1e3 / (float(np.median(lat)) / 1e3))


if __name__ == "__main__":
    rows, info = analyse(sys.argv[1], sys.argv[2])
    keys = list(rows[0])
    print(" ".join(f"{k:>12s}" for k in keys))
    for row in rows:
        print(" ".join(f"{row[k]:12.3f}" if isinstance(row[k], float) else f"{row[k]:12d}" for k in keys))
    print(f"{sum(r['passed'] for r in rows)}/{len(rows)} pass | {info['steps']} control steps, {info['sim_s']:.2f} s | "
          f"median control step {info['median_ms']:.1f} ms ({info['rtf']:.3f}x real time)")
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(f"{row[k]:.6g}" for k in keys) + "\n")
