"""Assembly swap benchmark for the MPC QP: the same solve with the stage derivatives evaluated either by cusadi2
(the plan `mpc_assemble`) or by JAX on MuJoCo MJX (mjx_stage.py), and everything else identical.

Both systems go through one harness and one solver instance:

    assembly   inputs on the device -> blk, q, z_lb, z_ub on the device, complete on return
               cusadi2: one level-kernel launch;  mjx: one jitted XLA call (vmap over instances and stages, forward-mode AD)
    solver     mpc_qp_lib.cu through mpc_bridge.py: Ruiz (5 launches + 2 reductions), gather, cuDSS factorization,
               25 x (cuDSS solve + ADMM step), one CUDA graph

so `assembly_us` is the derivative evaluation alone, Ruiz is in `solver_us` for both, and the two are timed with
the same wall clock around synchronous calls. Every instance is checked against inputs/dx_ref.f64.

usage: python bridge_bench.py <build_dir> <n_warm> <n_timed> <B...>      (CSV on stdout)
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")       # leave device memory for cuDSS
import numpy as np, jax, jax.numpy as jp
import mjx_stage
from mpc_bridge import QPSolver, Cusadi2Assembler
BUILD, n_warm, n_timed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rd = lambda n: np.fromfile(os.path.join(BUILD, "inputs", n + ".f64"))
mjx_asm, meta = mjx_stage.make_assemble(BUILD)
N, NS = meta["N"], meta["NS"]
x_pad, p_stage, p_shared, dx_ref = rd("x_pad").reshape(N + 2, NS), rd("p_stage").reshape(N, -1), rd("p_shared"), rd("dx_ref").reshape(N, NS)
med = lambda v: float(np.median(v))
print("dtype,system,batch,assembly_us,solver_us,total_us,max_abs_err_ref,setup_s", flush=True)
for B in (int(b) for b in sys.argv[4:]):
    args = [jax.block_until_ready(jp.asarray(np.broadcast_to(a, (B,) + a.shape).copy())) for a in (x_pad, p_stage, p_shared)]
    qp = QPSolver(BUILD, B)
    for system in ("cusadi2", "mjx"):
        try:
            t0 = time.perf_counter()
            if system == "cusadi2": asm = Cusadi2Assembler(BUILD, B, qp); assemble = asm
            else: assemble = lambda *a: jax.block_until_ready(mjx_asm(*a))
            out = assemble(*args); setup_s = time.perf_counter() - t0          # plan upload, or trace + XLA compile
            for _ in range(n_warm): out = assemble(*args); qp.launch(*out)
            err = float(np.abs(qp.dx() - dx_ref).max())
            t_asm, t_qp = [], []
            for _ in range(n_timed):
                t0 = time.perf_counter(); out = assemble(*args); t1 = time.perf_counter(); qp.launch(*out); t2 = time.perf_counter()
                t_asm.append((t1 - t0) * 1e6); t_qp.append((t2 - t1) * 1e6)
            print(f"f64,{system},{B},{med(t_asm):.1f},{med(t_qp):.1f},{med(np.array(t_asm) + np.array(t_qp)):.1f},{err:.3e},{setup_s:.2f}", flush=True)
            del out
        except Exception as e:                                           # out of device memory at large B: record and go on
            print(f"f64,{system},{B},nan,nan,nan,nan,nan", flush=True); print(f"[{system} B={B}] {type(e).__name__}: {str(e)[:200]}", file=sys.stderr, flush=True)
    qp.close(); del args
    jax.clear_caches()
