"""JAX front end for the CUDA elastic-QP interior-point solver (cbf_ipm.cu).

    ipm = CudaIpm(m=n_rows, max_batch=4096, penalty=pen)   # builds build/double/libcbf_ipm.so if needed
    x, iters, conv = ipm.solve(P, q, G, h)                 # P (B,7,7), q (B,7), G (B,m,7), h (B,m); jit-able
"""
import ctypes, os, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "double"


def build_lib(force=False):
    import jax
    so = OUT / "libcbf_ipm.so"; cu = HERE / "cbf_ipm.cu"
    if so.exists() and so.stat().st_mtime > cu.stat().st_mtime and not force:
        return so
    OUT.mkdir(parents=True, exist_ok=True)
    nvcc = Path(sys.executable).parent / "nvcc"
    cc = jax.devices()[0].compute_capability.replace(".", "")
    subprocess.run([str(nvcc), "-std=c++17", "-O3", f"-arch=sm_{cc}", f"-I{jax.ffi.include_dir()}", str(cu),
                    "-shared", "-Xcompiler", "-fPIC", "-o", str(so)], check=True)
    return so


class CudaIpm:
    _registered = False

    def __init__(self, m, max_batch=4096, penalty=None):
        import jax, jax.numpy as jnp
        so = Path(os.environ["CBF_IPM_SO"]) if os.environ.get("CBF_IPM_SO") else build_lib()
        self.lib = lib = ctypes.CDLL(str(so))
        lib.ipm_init.restype = ctypes.c_int; lib.ipm_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.ipm_set_screen.argtypes = [ctypes.c_int]
        rc = lib.ipm_init(max_batch, m)
        if rc:
            raise RuntimeError(f"ipm_init failed: {rc}")
        self.m, self.max_batch = m, max_batch
        self.penalty = jnp.asarray(penalty, dtype=jnp.float64)
        assert self.penalty.shape == (m,)
        self.target = "cbf_ipm_solve"
        if not CudaIpm._registered:
            jax.ffi.register_ffi_target(self.target, jax.ffi.pycapsule(lib.IpmSolve), platform="CUDA")
            CudaIpm._registered = True

    def set_screen(self, on: bool):
        """Drop CBF rows that cannot be active inside the torque box (default on)."""
        self.lib.ipm_set_screen(1 if on else 0)

    def solve(self, P, q, G, h):
        import jax, jax.numpy as jnp
        B = G.shape[0]
        out = [jax.ShapeDtypeStruct((B, 7), jnp.float64), jax.ShapeDtypeStruct((B,), jnp.int32), jax.ShapeDtypeStruct((B,), jnp.int32)]
        return jax.ffi.ffi_call(self.target, out)(P, q, G, h, self.penalty)
