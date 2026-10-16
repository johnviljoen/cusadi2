"""JAX front end for the cusadi2-compiled backup-CBF model side (one XLA FFI custom call).

    m = Cusadi2Backup(N=8, max_live=0)
    u_nom, P, q, G, h = m.model_part(z, z_des)      # z (B,14), z_des (B,18) on the GPU, jit-able
    m.time_kernel(batch, n)                         # bare kernel, CUDA events, no XLA

Reuses profiling/cusadi2_ffi.cu unchanged (2 inputs, 5 outputs, pointer ring); one shared
library per (N, max_live) because the handler holds one plan.
"""
import ctypes, json, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROFILING = HERE.parent.parent / "profiling"
FFI_CU = PROFILING / "cusadi2_ffi.cu"


def build(N, dtype="double", max_live=0, force=False):
    import jax, cusadi2
    out = HERE / "build" / dtype
    plan = out / f"backup_N{N}_ml{max_live}.plan"
    srcs = [HERE / "backup_casadi.py", PROFILING / "panda_casadi.py"]
    if not plan.exists() or plan.stat().st_mtime < max(s.stat().st_mtime for s in srcs):
        subprocess.run([sys.executable, str(HERE / "build_plan.py"), "--N", str(N), "--dtype", dtype, "--max-live", str(max_live)], check=True)
    so = out / f"libbackup_N{N}_ml{max_live}.so"
    if force or not so.exists() or so.stat().st_mtime < max(FFI_CU.stat().st_mtime, (out / "kernel.cu").stat().st_mtime):
        nvcc = Path(sys.executable).parent / "nvcc"
        cc = jax.devices()[0].compute_capability.replace(".", "")
        subprocess.run([str(nvcc), "-std=c++17", "-O3", f"-arch=sm_{cc}", f"-DCUSADI2_SCALAR={dtype}",
                        f"-I{cusadi2.INCLUDE_DIR}", f"-I{jax.ffi.include_dir()}", str(FFI_CU), str(out / "kernel.cu"),
                        "-shared", "-Xcompiler", "-fPIC", "-o", str(so)], check=True)
    return out, plan, so


class Cusadi2Backup:
    def __init__(self, N, dtype="double", max_live=0, mode="level"):
        import jax, jax.numpy as jnp
        assert dtype == "double"
        out, plan, so = build(N, dtype, max_live)
        self.N, self.max_live, self.mode = N, max_live, mode
        self.lib = lib = ctypes.CDLL(str(so))
        lib.oscbf_init.restype = ctypes.c_int; lib.oscbf_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.oscbf_shared_bytes.restype = ctypes.c_longlong
        lib.oscbf_time_kernel.restype = ctypes.c_int
        lib.oscbf_time_kernel.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.block_size = lib.oscbf_init(str(plan).encode(), 1 if mode == "serial" else 0)
        if self.block_size < 0:
            raise RuntimeError(f"cusadi2 plan rejected by oscbf_init: {self.block_size}")
        self.shared_bytes = int(lib.oscbf_shared_bytes())
        self.plan_stats = json.loads((out / plan.name.replace(".plan", "_stats.json")).read_text())
        n_rows = self.plan_stats["n_rows"]
        self.n_rows = n_rows
        self.target = f"backup_cusadi2_N{N}_{mode}_ml{max_live}"
        jax.ffi.register_ffi_target(self.target, jax.ffi.pycapsule(lib.OscbfCusadi2), platform="CUDA")
        self.shapes = [(7, 1), (7, 7), (7, 1), (n_rows, 7), (n_rows, 1)]
        self.nnz, self.dense_idx = [], []
        for a, (r, c) in enumerate(self.shapes):
            n = lib.oscbf_out_nnz(a); self.nnz.append(n)
            ri = (ctypes.c_int * n)(); lib.oscbf_out_rowidx(a, ri)
            cp = (ctypes.c_int * (c + 1))(); lib.oscbf_out_colptr(a, cp)
            cols = np.repeat(np.arange(c), np.diff(np.asarray(cp[:])))
            self.dense_idx.append(np.asarray(ri[:]) * c + cols)

    def model_part(self, z, z_des):
        import jax, jax.numpy as jnp
        B = z.shape[0]
        outs = jax.ffi.ffi_call(self.target, [jax.ShapeDtypeStruct((B, n), jnp.float64) for n in self.nnz])(z, z_des)
        dense = []
        for o, (r, c), n, idx in zip(outs, self.shapes, self.nnz, self.dense_idx):
            if n == r * c and c == 1:
                dense.append(o)
            else:
                dense.append(jnp.zeros((B, r * c), jnp.float64).at[:, idx].set(o).reshape(B, r, c))
        return tuple(dense)

    def time_kernel(self, batch, n=50):
        med, mn = ctypes.c_double(), ctypes.c_double()
        assert self.lib.oscbf_time_kernel(batch, n, ctypes.byref(med), ctypes.byref(mn)) == 0
        return med.value, mn.value
