"""Diagnostics of a closed-loop log (results/closed_loop_ps{tag}.npz): where the barriers dip, how fast the arm moves,
how often the iterations improve the plan, per waypoint attempt."""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else ""
L = np.load(HERE / "results" / f"closed_loop_ps{tag}.npz"); S = json.load(open(HERE / "results" / f"closed_loop_ps{tag}.json"))
t, wp, dist, h, v, un, ni, cs, cm = [L[k] for k in ("t", "wp", "dist", "h_min_all", "v_max", "u_norm", "n_improved", "cost_start", "cost_min")]
print(f"seed {S['seed']}: reached {S['waypoints_reached']} timed out {S['waypoints_timed_out']}  h_min {h.min():.4f}  v_max {v.max():.2f}")
print(f"ticks with h < -0.01: {(h < -0.01).sum()} of {len(h)}  (t = {np.round(t[h < -0.01][:12], 2).tolist()}...)")
print(f"|v|max per tick: median {np.median(v):.2f}  p90 {np.percentile(v, 90):.2f}  max {v.max():.2f};   |u|: median {np.median(un):.1f} p90 {np.percentile(un, 90):.1f}")
print(f"improving iterations per tick: mean {ni.mean():.2f}, histogram {np.bincount(ni.astype(int), minlength=5).tolist()}  (0..4)")
print(f"plan cost at tick start -> after the iterations: mean {cs.mean():.3f} -> {cm.mean():.3f}; cost rises between ticks (resample + plant mismatch): "
      f"{np.mean(cs[1:] > cm[:-1]):.2f} of ticks, mean rise {np.mean(np.maximum(cs[1:] - cm[:-1], 0)):.3f}")
for e in S["events"]:
    m = (t >= e["t_start"]) & (t < e["t_end"])
    print(f"  wp {e['wp']} {'reached' if e['reached'] else 'TIMEOUT'} in {e['t_end'] - e['t_start']:.2f} s: dist end {dist[m][-1]:.3f} min {dist[m].min():.3f}  "
          f"h_min {h[m].min():.4f}  |v|max {v[m].max():.2f}  improving it/tick {ni[m].mean():.2f}  cost {cm[m].mean():.2f}")
