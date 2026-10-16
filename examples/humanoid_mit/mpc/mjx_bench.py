"""MJX baseline for the MPC QP benchmark: the stage assembly in JAX on MuJoCo MJX (mjx_stage.py), handed on the
device to the same CUDA QP solver the cusadi2 pipeline uses (mpc_qp_lib.cu through mpc_bridge.py).

One solve = assemble (one jitted XLA call, vmapped over instances and stages, forward-mode AD for the
Jacobians) + Ruiz, cuDSS factorization and 25 ADMM iterations (one CUDA graph). Same QP, parameters and
reference as mpc_cudss_bench.cu; every instance is checked against inputs/dx_ref.f64.

usage: python mjx_bench.py <build_dir> <n_warm> <n_timed> <B...>      (CSV on stdout)
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")       # leave device memory for cuDSS
import numpy as np, jax, jax.numpy as jp
import mjx_stage
from mpc_bridge import QPSolver
BUILD, n_warm, n_timed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rd = lambda n: np.fromfile(os.path.join(BUILD, "inputs", n + ".f64"))
asm, meta = mjx_stage.make_assemble(BUILD)
N, NS = meta["N"], meta["NS"]
x_pad, p_stage, p_shared, dx_ref = rd("x_pad").reshape(N + 2, NS), rd("p_stage").reshape(N, -1), rd("p_shared"), rd("dx_ref").reshape(N, NS)
print("dtype,system,batch,assembly_us,solver_us,total_us,solver_alone_us,max_abs_err_ref,jit_s", flush=True)
for B in (int(b) for b in sys.argv[4:]):
    try:
        args = [jp.asarray(np.broadcast_to(a, (B,) + a.shape).copy()) for a in (x_pad, p_stage, p_shared)]
        qp = QPSolver(BUILD, B)
        t0 = time.perf_counter(); out = jax.block_until_ready(asm(*args)); jit_s = time.perf_counter() - t0
        for _ in range(n_warm): out = jax.block_until_ready(asm(*args)); qp.launch(*out)
        err = float(np.abs(qp.dx() - dx_ref).max())
        t_asm, t_qp = [], []
        for _ in range(n_timed):
            t0 = time.perf_counter(); out = jax.block_until_ready(asm(*args)); t1 = time.perf_counter(); qp.launch(*out); t2 = time.perf_counter()
            t_asm.append((t1 - t0) * 1e6); t_qp.append((t2 - t1) * 1e6)
        tot = np.median(np.array(t_asm) + np.array(t_qp)); t_alone = []
        for _ in range(n_timed):                                         # the solver back to back, no XLA call in between
            t1 = time.perf_counter(); qp.launch(*out); t_alone.append((time.perf_counter() - t1) * 1e6)
        print(f"f64,mjx,{B},{np.median(t_asm):.1f},{np.median(t_qp):.1f},{tot:.1f},{np.median(t_alone):.1f},{err:.3e},{jit_s:.1f}", flush=True)
        qp.close(); del out, args
    except Exception as e:                                               # out of device memory at large B: record and go on
        print(f"f64,mjx,{B},nan,nan,nan,nan,nan,nan", flush=True); print(f"[B={B}] {type(e).__name__}: {str(e)[:200]}", file=sys.stderr, flush=True)
    jax.clear_caches()
