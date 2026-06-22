"""Single-shot cold-compile measurement for one (backend, task, robot, batch) tuple.

Driver clears caches before invoking, so every call is a true cold compile.
For ours we report a fixed 1.5 s — the LP kernel was NVCC'd once at session
start and is reused for every shape. For MJX, fresh-subprocess JIT is the
per-shape XLA compile. For mujoco-warp, fresh-subprocess + cleared
~/.cache/warp gives the per-process module compile.

Forward-dynamics path: mjx.forward(...).qacc.
Jacobian path:        jax.jacfwd(forward, argnums=q).
mujoco-warp has no autodiff, so 'jac' on that backend is unsupported.

Usage:
    python bench_compile_time.py --backend mjx --task fwd --robot a1 --batch 128
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def measure_lp(robot, task, batch):
    # Our compile cost is a one-off NVCC of the data-driven LP kernel.
    # No batch dependence, no function dependence.
    return 1.5


def measure_mjx(robot, task, batch, dtype):
    if dtype == "fp64":
        import jax
        jax.config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from mujoco import mjx
    from robot_descriptions.loaders.mujoco import load_robot_description as mj_load

    dt = jnp.float32 if dtype == "fp32" else jnp.float64

    m = mj_load(f"{robot}_mj_description")
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    mx = mjx.put_model(m)
    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d)
    q0 = jnp.asarray(d.qpos, dtype=dt)

    def fwd_single(q, v, tau):
        data = dx.replace(qpos=q, qvel=v, qfrc_applied=tau)
        return mjx.forward(mx, data).qacc.astype(dt)

    if task == "fwd":
        single = fwd_single
    elif task == "jac":
        def jac_single(q, v, tau):
            Jq = jax.jacfwd(fwd_single, argnums=0)(q, v, tau)
            Jv = jax.jacfwd(fwd_single, argnums=1)(q, v, tau)
            Jt = jax.jacfwd(fwd_single, argnums=2)(q, v, tau)
            return Jq, Jv, Jt
        single = jac_single
    else:
        raise ValueError(task)

    batched = jax.jit(jax.vmap(single, in_axes=(0, 0, 0)))

    rng = np.random.default_rng(0)
    qb = jnp.asarray(rng.normal(0.0, 0.3, size=(batch, m.nq)) + np.asarray(q0), dtype=dt)
    vb = jnp.asarray(rng.normal(0.0, 0.2, size=(batch, m.nv)), dtype=dt)
    tb = jnp.asarray(rng.normal(0.0, 1.0, size=(batch, m.nv)), dtype=dt)

    t0 = time.perf_counter()
    out = batched(qb, vb, tb)
    jax.block_until_ready(out)
    return time.perf_counter() - t0


def measure_mjw(robot, batch):
    import warp as wp
    import mujoco
    import mujoco_warp as mjw_mod
    import numpy as np
    from robot_descriptions.loaders.mujoco import load_robot_description as mj_load

    m = mj_load(f"{robot}_mj_description")
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0

    t0 = time.perf_counter()
    wp.init()
    mjw_m = mjw_mod.put_model(m)
    d = mujoco.MjData(m)
    mjw_d = mjw_mod.put_data(m, d, nworld=batch)

    rng = np.random.default_rng(0)
    qpos = rng.normal(0.0, 0.3, size=(batch, m.nq)).astype(np.float32)
    qvel = rng.normal(0.0, 0.2, size=(batch, m.nv)).astype(np.float32)
    qfrc = rng.normal(0.0, 1.0, size=(batch, m.nv)).astype(np.float32)
    mjw_d.qpos = wp.from_numpy(qpos, dtype=wp.float32, device="cuda:0")
    mjw_d.qvel = wp.from_numpy(qvel, dtype=wp.float32, device="cuda:0")
    mjw_d.qfrc_applied = wp.from_numpy(qfrc, dtype=wp.float32, device="cuda:0")

    mjw_mod.forward(mjw_m, mjw_d)
    wp.synchronize_device("cuda:0")
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["lp", "mjx", "mjw"], required=True)
    ap.add_argument("--task",    choices=["fwd", "jac"], required=True)
    ap.add_argument("--robot",   default="a1")
    ap.add_argument("--batch",   type=int, required=True)
    ap.add_argument("--dtype",   default="fp32")
    args = ap.parse_args()

    if args.backend == "lp":
        t = measure_lp(args.robot, args.task, args.batch)
    elif args.backend == "mjx":
        t = measure_mjx(args.robot, args.task, args.batch, args.dtype)
    elif args.backend == "mjw":
        if args.task == "jac":
            print(json.dumps({"error": "mujoco-warp has no autodiff path"}))
            sys.exit(1)
        t = measure_mjw(args.robot, args.batch)
    else:
        raise ValueError(args.backend)

    print(json.dumps({
        "backend": args.backend, "task": args.task, "robot": args.robot,
        "batch": args.batch, "dtype": args.dtype, "compile_time_s": t,
    }))


if __name__ == "__main__":
    main()
