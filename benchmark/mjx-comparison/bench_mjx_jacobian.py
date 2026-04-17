"""
MJX autodiff ABA-Jacobian benchmark.

For a given robot and dtype, computes ddq/dq, ddq/dv, ddq/dtau via
jax.jacobian of mjx.forward dynamics. Measures JIT compile time (first call)
and per-evaluation time (median of N_EVALS after N_WARMUP warmups) across a
sweep of batch sizes.

Run as a SEPARATE PROCESS per dtype — jax.config x64 must be toggled before
any jnp operation is traced, so a single process can't easily run both.

Usage:
    CUDA_VISIBLE_DEVICES=5 python benchmark/mjx-comparison/bench_mjx_jacobian.py \
        --robot a1 --dtype fp64 --out results/mjx_a1_fp64.json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["a1", "h1"], required=True)
    ap.add_argument("--dtype", choices=["fp32", "fp64"], required=True)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_warmup", type=int, default=N_WARMUP)
    ap.add_argument("--n_evals", type=int, default=N_EVALS)
    ap.add_argument("--with_contact", action="store_true",
                    help="Enable MJX contact dynamics. Swaps unsupported CYLINDER geoms "
                         "(MJX rejects cylinder-box pairs) to CAPSULE and keeps contype/"
                         "conaffinity active. Default: contacts disabled.")
    args = ap.parse_args()

    if args.dtype == "fp64":
        import jax
        jax.config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from mujoco import mjx
    from robot_descriptions.loaders.mujoco import load_robot_description as mj_load

    dtype = jnp.float64 if args.dtype == "fp64" else jnp.float32

    print(f"[mjx] robot={args.robot}  dtype={args.dtype}  jax {jax.__version__}  "
          f"devices={jax.devices()}")

    # Load model
    if args.robot == "a1":
        m = mj_load("a1_mj_description")
    else:
        m = mj_load("h1_mj_description")
    if args.with_contact:
        # MJX rejects cylinder-box collisions. Swap CYLINDER(5) -> CAPSULE(3)
        # (same radius + half-length parametrisation so the swap is drop-in),
        # keep contype/conaffinity active.
        n_swapped = int((m.geom_type == 5).sum())
        m.geom_type[m.geom_type == 5] = 3
        n_active = int(((m.geom_contype != 0) | (m.geom_conaffinity != 0)).sum())
        print(f"[mjx] contact ENABLED: swapped {n_swapped} cylinders->capsules, "
              f"{n_active} contact-active geoms")
    else:
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        print(f"[mjx] contact DISABLED (geom_contype = geom_conaffinity = 0)")
    print(f"[mjx] model: nq={m.nq}  nv={m.nv}  nu={m.nu}")

    mx = mjx.put_model(m)
    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d)

    # Nominal state as template
    q0 = jnp.asarray(d.qpos, dtype=dtype)
    v0 = jnp.asarray(d.qvel, dtype=dtype)
    tau0 = jnp.zeros(m.nv, dtype=dtype)

    def dyn_single(q, v, tau):
        data = dx.replace(qpos=q, qvel=v, qfrc_applied=tau)
        data = mjx.forward(mx, data)
        return data.qacc.astype(dtype)

    def aba_jacs_single(q, v, tau):
        # Reverse-mode is more efficient when output dim <= input dim; qacc is nv.
        # q is nq, but nq ≈ nv+1 (quaternion). We choose forward-mode (jacfwd) since
        # jacobian() defaults to reverse-mode but forward-mode is simpler for ABA.
        # Use jacrev to match autodiff literature ("adjoint dynamics"). Either is OK;
        # both produce the full dense nv × nv (or nv × nq) matrix.
        Jq = jax.jacfwd(dyn_single, argnums=0)(q, v, tau)
        Jv = jax.jacfwd(dyn_single, argnums=1)(q, v, tau)
        Jt = jax.jacfwd(dyn_single, argnums=2)(q, v, tau)
        return Jq, Jv, Jt

    batched = jax.jit(jax.vmap(aba_jacs_single, in_axes=(0, 0, 0)))

    results_per_batch = {}
    rng = np.random.default_rng(0)

    for bs in args.batch_sizes:
        print(f"\n[mjx] batch={bs}")
        # Physically plausible-ish (q, v, tau) — small perturbations around nominal
        qb = jnp.asarray(rng.normal(0.0, 0.3, size=(bs, m.nq)) + np.asarray(q0), dtype=dtype)
        vb = jnp.asarray(rng.normal(0.0, 0.2, size=(bs, m.nv)), dtype=dtype)
        tb = jnp.asarray(rng.normal(0.0, 1.0, size=(bs, m.nv)), dtype=dtype)

        # First call -> JIT compile + execute
        t0 = time.perf_counter()
        out = batched(qb, vb, tb)
        jax.block_until_ready(out)
        t_jit = time.perf_counter() - t0
        print(f"[mjx]   first-call (JIT + run) = {t_jit * 1000:.1f} ms")
        Jq_shape = tuple(out[0].shape)
        Jv_shape = tuple(out[1].shape)
        Jt_shape = tuple(out[2].shape)
        print(f"[mjx]   shapes: Jq={Jq_shape}  Jv={Jv_shape}  Jt={Jt_shape}")

        # Warmup runs (measure subsequent timing only)
        for _ in range(args.n_warmup):
            out = batched(qb, vb, tb)
            jax.block_until_ready(out)

        # Timed runs
        samples = np.empty(args.n_evals, dtype=np.float64)
        for i in range(args.n_evals):
            t0 = time.perf_counter()
            out = batched(qb, vb, tb)
            jax.block_until_ready(out)
            samples[i] = time.perf_counter() - t0

        med = float(np.median(samples))
        mean = float(np.mean(samples))
        p25 = float(np.percentile(samples, 25))
        p75 = float(np.percentile(samples, 75))
        mn = float(np.min(samples))
        print(f"[mjx]   median = {med * 1e6:.1f} us    mean = {mean * 1e6:.1f} us    "
              f"min = {mn * 1e6:.1f} us    IQR [{p25 * 1e6:.1f}, {p75 * 1e6:.1f}] us")

        results_per_batch[bs] = {
            "batch": bs,
            "jit_compile_plus_run_s": t_jit,
            "median_s": med, "mean_s": mean,
            "p25_s": p25, "p75_s": p75, "min_s": mn,
            "n_samples": len(samples),
            "jac_shapes": {"Jq": Jq_shape, "Jv": Jv_shape, "Jt": Jt_shape},
        }

    out_dict = {
        "robot": args.robot,
        "dtype": args.dtype,
        "with_contact": args.with_contact,
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
        "n_warmup": args.n_warmup,
        "n_evals": args.n_evals,
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batch_sizes": args.batch_sizes,
        "results": results_per_batch,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2, default=str)
    print(f"\n[mjx] wrote {args.out}")


if __name__ == "__main__":
    main()
