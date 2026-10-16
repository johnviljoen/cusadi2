"""Test inputs for verification and timing: states from profiling/sampling.py (joint-limit and collision
states included), random plan clocks, knots around the current configuration, waypoints from the
forward kinematics of random configurations, default parameters."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
from model import DT, N_KNOT, NV, NX, ND, PAR_DEFAULT  # noqa: E402


def make_dataset(problem, n, seed, T, knot_rms=0.3):
    from sampling import build_dataset
    rng = np.random.default_rng(seed)
    ds = build_dataset(problem, n, seed)
    x = np.zeros((n, NX)); x[:, :2 * NV] = ds["z"]
    x[:, 2 * NV] = rng.uniform(0, T * DT, n)
    lo, hi = np.asarray(problem.robot.actuated_joint_lower_limits), np.asarray(problem.robot.actuated_joint_upper_limits)
    Y = np.clip(np.repeat(ds["z"][:, :NV], N_KNOT, axis=0).reshape(n, ND) + rng.normal(0, knot_rms, (n, ND)), np.tile(lo, N_KNOT), np.tile(hi, N_KNOT))
    rp = ds["z_des"][:, :3]
    par = np.tile(PAR_DEFAULT, (n, 1))
    return dict(x=x, Y=Y, rp=rp, par=par, kind=ds.get("kind"))
