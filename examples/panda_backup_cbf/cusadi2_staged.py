"""JAX front end for the staged cusadi2 backup-CBF model (head + N step launches in one FFI call).

    m = Cusadi2Staged(N=16, max_batch=4096)
    u_nom, P, q, G, h = m.model_part(z, z_des)     # dense (B,7), (B,7,7), (B,7), (B,n_rows,7), (B,n_rows); jit-able
    m.time_kernel(batch, n)                        # bare head + N steps, CUDA events

Row order of G / h is the monolithic one: [k = 0 rows (64); k = 1..N rows (64 each); u <= u_max; -u <= u_max].
One shared library serves every N (the plan is N-agnostic); the handler holds one (N, max_batch), so one
Cusadi2Staged per process.
"""
import ctypes, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Cusadi2Staged:
    N_CBF = 64
    N_BOUND = 14

    def __init__(self, N, max_batch=4096, max_live_head=0, max_live_step=0):
        import jax, jax.numpy as jnp
        from build_staged import build_plans, build_lib, plan_paths
        build_plans(max_live_head, max_live_step)
        so = build_lib()
        head_plan, step_plan = plan_paths(max_live_head, max_live_step)
        self.N, self.max_batch = N, max_batch
        self.lib = lib = ctypes.CDLL(str(so))
        lib.backup_init.restype = ctypes.c_int; lib.backup_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.backup_shared_bytes.restype = ctypes.c_longlong
        lib.backup_time.restype = ctypes.c_int
        lib.backup_time.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.POINTER(ctypes.c_double)] * 3
        rc = lib.backup_init(str(head_plan).encode(), str(step_plan).encode(), N, max_batch)
        if rc < 0:
            raise RuntimeError(f"backup_init failed: {rc}")
        bh, bs = ctypes.c_int(), ctypes.c_int(); lib.backup_block_sizes(ctypes.byref(bh), ctypes.byref(bs))
        self.block_sizes = (bh.value, bs.value)
        self.shared_bytes = (int(lib.backup_shared_bytes(0)), int(lib.backup_shared_bytes(1)))
        self.stats = {w: json.loads(p.with_name(p.name.replace(".plan", "_stats.json")).read_text()) for w, p in (("head", head_plan), ("step", step_plan))}
        self.n_rows = self.N_BOUND + self.N_CBF * (N + 1)
        self.target = f"backup_cusadi2_staged_N{N}"
        jax.ffi.register_ffi_target(self.target, jax.ffi.pycapsule(lib.BackupCusadi2), platform="CUDA")
        # output sparsity -> row-major scatter indices: a = 0..4 head (u, P, q, G0, h0), 5, 6 step (G, h)
        self.shapes = [(7, 1), (7, 7), (7, 1), (78, 7), (78, 1), (64, 7), (64, 1)]
        self.nnz, self.idx = [], []
        for a, (r, c) in enumerate(self.shapes):
            n = lib.backup_out_nnz(a); self.nnz.append(n)
            ri = (ctypes.c_int * n)(); lib.backup_out_rowidx(a, ri)
            cp = (ctypes.c_int * (c + 1))(); lib.backup_out_colptr(a, cp)
            cols = np.repeat(np.arange(c), np.diff(np.asarray(cp[:])))
            self.idx.append(np.asarray(ri[:]) * c + cols)

    def _dense(self, o, a, lead):
        import jax.numpy as jnp
        r, c = self.shapes[a]; n = self.nnz[a]
        if n == r * c and c == 1:
            return o.reshape(*lead, r)
        flat = o.reshape(-1, n)
        d = jnp.zeros((flat.shape[0], r * c), jnp.float64).at[:, self.idx[a]].set(flat)
        return d.reshape(*lead, r, c) if c > 1 else d.reshape(*lead, r)

    def model_part(self, z, z_des):
        import jax, jax.numpy as jnp
        B, N = z.shape[0], self.N
        out_types = [jax.ShapeDtypeStruct((B, self.nnz[a]), jnp.float64) for a in range(5)] + \
                    [jax.ShapeDtypeStruct((N, B, self.nnz[5]), jnp.float64), jax.ShapeDtypeStruct((N, B, self.nnz[6]), jnp.float64)]
        u, P, q, G0, h0, Gp, hp = jax.ffi.ffi_call(self.target, out_types)(z, z_des)
        u = self._dense(u, 0, (B,)); P = self._dense(P, 1, (B,)); q = self._dense(q, 2, (B,))
        G0 = self._dense(G0, 3, (B,)); h0 = self._dense(h0, 4, (B,))
        Gp = self._dense(Gp, 5, (N, B)).transpose(1, 0, 2, 3).reshape(B, N * 64, 7)
        hp = self._dense(hp, 6, (N, B)).transpose(1, 0, 2).reshape(B, N * 64)
        G = jnp.concatenate([G0[:, :64], Gp, G0[:, 64:]], axis=1)
        h = jnp.concatenate([h0[:, :64], hp, h0[:, 64:]], axis=1)
        return u, P, q, G, h

    def time_kernel(self, batch, n=50):
        med, mn, hd = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        assert self.lib.backup_time(batch, n, ctypes.byref(med), ctypes.byref(mn), ctypes.byref(hd)) == 0
        return med.value, mn.value, hd.value
