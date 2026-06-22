"""
Forward-dynamics (ABA) benchmark across three backends:
  - Level-parallel CusADi (fp64) on compiled analytical ABA kernels
  - Sequential CusADi (fp64) on the same
  - MJX (fp32 default; fp64 via jax_enable_x64)
  - mujoco-warp (Warp backend) when available

Unlike bench_mjx_jacobian.py, this is forward dynamics only — no autodiff —
so it matches MPPI's workload and is the natural place for MJX to look good.

Run one backend per process because JAX x64 / CUDA allocator state can't be
toggled cleanly. Produces per-backend JSON, joined downstream in compare.py.

Usage:
    # LP / Sequential (both fp64, torch-based)
    python experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
        --robot a1 --backend level_parallel --out results/a1_fwd_lp.json
    python experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
        --robot a1 --backend sequential --out results/a1_fwd_seq.json

    # MJX
    python experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
        --robot a1 --backend mjx --dtype fp32 --out results/a1_fwd_mjx_fp32.json

    # mujoco-warp
    python experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
        --robot a1 --backend mujoco_warp --out results/a1_fwd_mjw.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

N_WARMUP = 10
N_EVALS = 100
BATCH_SIZES = [1, 8, 32, 128, 256, 512]

ROBOT_TO_FN = {"a1": "a1_aba", "h1": "h1_aba"}


def bench_cusadi(robot: str, method: str, batches: list[int], n_warmup: int,
                 n_evals: int, out_path: str) -> None:
    import casadi
    import numpy as np
    import torch

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from cusadi import CUSADI_LP_BUILD_DIR, CUSADI_PYTORCH_BUILD_DIR, CUSADI_FUNCTION_DIR  # noqa: E402
    from cusadi.CusadiFunction import CusadiFunction  # noqa: E402
    from cusadi.level_parallel.function import CusadiLevelParallelFunction  # noqa: E402

    assert torch.cuda.is_available()
    fn_name = ROBOT_TO_FN[robot]
    fn = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, f"{fn_name}.casadi"))
    print(f"[cusadi-{method}] {fn_name}: n_instr={fn.n_instructions():,}  sz_w={fn.sz_w()}")

    if method == "level_parallel":
        so = os.path.join(CUSADI_LP_BUILD_DIR, "liblevel_parallel_kernel.so")
    else:
        so = os.path.join(CUSADI_PYTORCH_BUILD_DIR, f"lib{fn_name}.so")
    assert os.path.exists(so), f"missing kernel: {so}"

    results: dict[int, dict] = {}
    for bs in batches:
        print(f"\n[cusadi-{method}] batch={bs}")
        if method == "level_parallel":
            ev = CusadiLevelParallelFunction(fn, bs, lib_name="level_parallel_kernel")
        else:
            ev = CusadiFunction(fn, bs)
        gen = torch.Generator(device="cuda").manual_seed(0)
        inputs = []
        for i in range(fn.n_in()):
            nm = fn.name_in(i).lower()
            scale = 0.3 if nm.startswith("q") else 0.2 if nm.startswith("v") else 1.0
            inputs.append(scale * torch.randn(
                bs, fn.nnz_in(i), device="cuda", dtype=torch.double, generator=gen
            ).contiguous())
        for _ in range(n_warmup):
            ev.evaluate(inputs)
        torch.cuda.synchronize()
        s = np.empty(n_evals, dtype=np.float64)
        for i in range(n_evals):
            ev.evaluate(inputs)
            s[i] = ev.eval_time
        torch.cuda.synchronize()
        med = float(np.median(s))
        p25 = float(np.percentile(s, 25))
        p75 = float(np.percentile(s, 75))
        print(f"[cusadi-{method}]   median = {med * 1e6:.1f} us  IQR [{p25 * 1e6:.1f}, {p75 * 1e6:.1f}]")
        results[bs] = {"batch": bs, "median_s": med, "mean_s": float(np.mean(s)),
                       "p25_s": p25, "p75_s": p75, "min_s": float(np.min(s)),
                       "n_samples": int(len(s))}

    out = {"robot": robot, "backend": method, "dtype": "fp64",
           "n_instr": int(fn.n_instructions()), "sz_w": int(fn.sz_w()),
           "device": torch.cuda.get_device_name(0),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "batch_sizes": batches, "results": results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[cusadi-{method}] wrote {out_path}")


def bench_mjx(robot: str, dtype: str, batches: list[int], n_warmup: int,
              n_evals: int, out_path: str) -> None:
    if dtype == "fp64":
        import jax
        jax.config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from mujoco import mjx
    from robot_descriptions.loaders.mujoco import load_robot_description as mj_load

    d_jnp = jnp.float64 if dtype == "fp64" else jnp.float32
    m = mj_load(f"{robot}_mj_description")
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    mx = mjx.put_model(m)
    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d)

    def fwd_single(q, v, tau):
        data = dx.replace(qpos=q, qvel=v, qfrc_applied=tau)
        data = mjx.forward(mx, data)
        return data.qacc.astype(d_jnp)

    batched = jax.jit(jax.vmap(fwd_single, in_axes=(0, 0, 0)))

    rng = np.random.default_rng(0)
    results: dict[int, dict] = {}
    for bs in batches:
        print(f"\n[mjx-{dtype}] batch={bs}")
        qb = jnp.asarray(rng.normal(0.0, 0.3, size=(bs, m.nq)), dtype=d_jnp)
        vb = jnp.asarray(rng.normal(0.0, 0.2, size=(bs, m.nv)), dtype=d_jnp)
        tb = jnp.asarray(rng.normal(0.0, 1.0, size=(bs, m.nv)), dtype=d_jnp)

        t0 = time.perf_counter()
        out = batched(qb, vb, tb)
        jax.block_until_ready(out)
        t_jit = time.perf_counter() - t0
        print(f"[mjx-{dtype}]   first-call (JIT + run) = {t_jit * 1000:.1f} ms")

        for _ in range(n_warmup):
            out = batched(qb, vb, tb)
            jax.block_until_ready(out)
        s = np.empty(n_evals, dtype=np.float64)
        for i in range(n_evals):
            t0 = time.perf_counter()
            out = batched(qb, vb, tb)
            jax.block_until_ready(out)
            s[i] = time.perf_counter() - t0
        med = float(np.median(s))
        print(f"[mjx-{dtype}]   median = {med * 1e6:.1f} us")
        results[bs] = {"batch": bs, "jit_compile_plus_run_s": t_jit,
                       "median_s": med, "mean_s": float(np.mean(s)),
                       "p25_s": float(np.percentile(s, 25)),
                       "p75_s": float(np.percentile(s, 75)),
                       "min_s": float(np.min(s)), "n_samples": int(len(s))}

    out = {"robot": robot, "backend": "mjx", "dtype": dtype,
           "nq": int(m.nq), "nv": int(m.nv),
           "jax_version": jax.__version__, "device": str(jax.devices()[0]),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "batch_sizes": batches, "results": results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[mjx-{dtype}] wrote {out_path}")


def bench_mujoco_warp(robot: str, batches: list[int], n_warmup: int,
                      n_evals: int, out_path: str) -> None:
    import warp as wp
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    from robot_descriptions.loaders.mujoco import load_robot_description as mj_load

    wp.init()
    print(f"[mjw] warp devices: {wp.get_devices()}")
    m = mj_load(f"{robot}_mj_description")
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0

    results: dict[int, dict] = {}
    for bs in batches:
        print(f"\n[mjw] batch={bs}")
        mjw_m = mjw.put_model(m)
        d = mujoco.MjData(m)
        mjw_d = mjw.put_data(m, d, nworld=bs)

        # Randomize state
        rng = np.random.default_rng(0)
        qpos = rng.normal(0.0, 0.3, size=(bs, m.nq)).astype(np.float32)
        qvel = rng.normal(0.0, 0.2, size=(bs, m.nv)).astype(np.float32)
        qfrc = rng.normal(0.0, 1.0, size=(bs, m.nv)).astype(np.float32)
        mjw_d.qpos = wp.from_numpy(qpos, dtype=wp.float32, device="cuda:0")
        mjw_d.qvel = wp.from_numpy(qvel, dtype=wp.float32, device="cuda:0")
        mjw_d.qfrc_applied = wp.from_numpy(qfrc, dtype=wp.float32, device="cuda:0")

        for _ in range(n_warmup):
            mjw.forward(mjw_m, mjw_d)
        wp.synchronize_device("cuda:0")
        s = np.empty(n_evals, dtype=np.float64)
        for i in range(n_evals):
            t0 = time.perf_counter()
            mjw.forward(mjw_m, mjw_d)
            wp.synchronize_device("cuda:0")
            s[i] = time.perf_counter() - t0
        med = float(np.median(s))
        print(f"[mjw]   median = {med * 1e6:.1f} us")
        results[bs] = {"batch": bs, "median_s": med, "mean_s": float(np.mean(s)),
                       "p25_s": float(np.percentile(s, 25)),
                       "p75_s": float(np.percentile(s, 75)),
                       "min_s": float(np.min(s)), "n_samples": int(len(s))}

    out = {"robot": robot, "backend": "mujoco_warp", "dtype": "fp32",
           "nq": int(m.nq), "nv": int(m.nv), "warp_version": wp.__version__,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "batch_sizes": batches, "results": results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[mjw] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["a1", "h1"], required=True)
    ap.add_argument("--backend", choices=["level_parallel", "sequential", "mjx", "mujoco_warp"],
                    required=True)
    ap.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32",
                    help="MJX only (LP is always fp64; warp is fp32).")
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES)
    ap.add_argument("--n_warmup", type=int, default=N_WARMUP)
    ap.add_argument("--n_evals", type=int, default=N_EVALS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.backend in ("level_parallel", "sequential"):
        bench_cusadi(args.robot, args.backend, args.batch_sizes,
                     args.n_warmup, args.n_evals, args.out)
    elif args.backend == "mjx":
        bench_mjx(args.robot, args.dtype, args.batch_sizes,
                  args.n_warmup, args.n_evals, args.out)
    elif args.backend == "mujoco_warp":
        bench_mujoco_warp(args.robot, args.batch_sizes,
                          args.n_warmup, args.n_evals, args.out)


if __name__ == "__main__":
    main()
