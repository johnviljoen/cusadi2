"""JAX front end for the cusadi2-compiled Panda OSC+CBF model side (see panda_casadi.py).

    from cusadi2_oscbf import Cusadi2Oscbf
    m = Cusadi2Oscbf()                      # loads build/double/{oscbf_panda.plan, liboscbf_ffi.so}
    u_nom, P, q, G, h = m.model_part(z, z_des)   # jit-able; z (B,14), z_des (B,18) on the GPU

The FFI call is a single cusadi2 kernel launch on XLA's stream (no host sync); the
dense P (B,7,7) and G (B,78,7) are assembled from cusadi2's column-major nonzeros
with static gathers inside the same jit.
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TARGET = "oscbf_cusadi2"


def build(dtype="double", force=False, max_live=0):
    """Build plan + FFI library (one .so serves both execution modes)."""
    import jax, cusadi2
    out = HERE / "build" / dtype
    so = out / "liboscbf_ffi.so"
    plan = out / (f"oscbf_panda_ml{max_live}.plan" if max_live else "oscbf_panda.plan")
    if not plan.exists() or plan.stat().st_mtime < (HERE / "panda_casadi.py").stat().st_mtime:
        subprocess.run([__import__("sys").executable, str(HERE / "build_cusadi2_plan.py"), "--dtype", dtype, "--max-live", str(max_live)], check=True)
    if force or not so.exists() or so.stat().st_mtime < (HERE / "cusadi2_ffi.cu").stat().st_mtime:
        nvcc = Path(__import__("sys").executable).parent / "nvcc"
        cc = jax.devices()[0].compute_capability.replace(".", "")
        subprocess.run([str(nvcc), "-std=c++17", "-O3", f"-arch=sm_{cc}", f"-DCUSADI2_SCALAR={dtype}",
                        f"-I{cusadi2.INCLUDE_DIR}", f"-I{jax.ffi.include_dir()}", str(HERE / "cusadi2_ffi.cu"),
                        str(out / "kernel.cu"), "-shared", "-Xcompiler", "-fPIC", "-o", str(so)], check=True)
    return out, plan


class Cusadi2Oscbf:
    def __init__(self, dtype="double", mode="level", max_live=0):
        import jax, jax.numpy as jnp
        assert dtype == "double", "the FFI handler is compiled for float64"
        out, plan = build(dtype, max_live=max_live)
        self.max_live = max_live
        self.lib = lib = ctypes.CDLL(str(out / "liboscbf_ffi.so"))
        assert mode in ("level", "serial")
        self.mode = mode
        lib.oscbf_init.restype = ctypes.c_int; lib.oscbf_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.oscbf_shared_bytes.restype = ctypes.c_longlong
        self.block_size = lib.oscbf_init(str(plan).encode(), 1 if mode == "serial" else 0)
        if self.block_size < 0:
            raise RuntimeError(f"cusadi2 plan rejected by oscbf_init: {self.block_size}")
        self.shared_bytes = int(lib.oscbf_shared_bytes())
        self.target = f"{TARGET}_{mode}_ml{max_live}"
        jax.ffi.register_ffi_target(self.target, jax.ffi.pycapsule(lib.OscbfCusadi2), platform="CUDA")
        # output sparsity (CCS) -> static scatter indices into row-major dense
        self.shapes = [(7, 1), (7, 7), (7, 1), (78, 7), (78, 1)]
        self.nnz, self.dense_idx = [], []
        for a, (r, c) in enumerate(self.shapes):
            n = lib.oscbf_out_nnz(a); self.nnz.append(n)
            ri = (ctypes.c_int * n)(); lib.oscbf_out_rowidx(a, ri)
            cp = (ctypes.c_int * (c + 1))(); lib.oscbf_out_colptr(a, cp)
            cols = np.repeat(np.arange(c), np.diff(np.asarray(cp[:])))
            self.dense_idx.append(np.asarray(ri[:]) * c + cols)   # row-major flat index
        self.plan_stats = __import__("json").loads((out / plan.name.replace(".plan", "_stats.json")).read_text())

    def model_part(self, z, z_des):
        """(B,14),(B,18) -> u_nom (B,7), P (B,7,7), q (B,7), G (B,78,7), h (B,78)"""
        import jax, jax.numpy as jnp
        B = z.shape[0]
        outs = jax.ffi.ffi_call(self.target, [jax.ShapeDtypeStruct((B, n), jnp.float64) for n in self.nnz])(z, z_des)
        dense = []
        for o, (r, c), n, idx in zip(outs, self.shapes, self.nnz, self.dense_idx):
            if n == r * c and c == 1:
                dense.append(o)
            else:
                d = jnp.zeros((B, r * c), jnp.float64).at[:, idx].set(o)
                dense.append(d.reshape(B, r, c))
        return tuple(dense)
