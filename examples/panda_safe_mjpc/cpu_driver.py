"""ctypes front end of the CPU-only driver (safe_ps_cpu.cpp -> build/libsafe_ps_cpu_f32.so): the "CPU idealized" row,
the whole safe predictive-sampling controller on the CPU with no GPU transfer (same defaults as driver.SafePS: TUNED).

    d = CpuPS(K=256, T=35, n_iter=3, threads=16)
    u, stat, ms = d.tick(x_plant (1,16) or None, wp (1,3))

Builds the library on first use (g++ -O3 -march=native -fopenmp). The CasADi C code of the two model functions comes from
the GPU driver's build (build/cpu/libcpu_T{T}_{dtype}.so, `python build.py cpu --T ...`).
"""
import ctypes, os, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
from model import NX, ND, NV, N_KNOT, NPAR, PAR_DEFAULT, PARAMS, DT, PROJ_ITERS  # noqa: E402
from driver import joint_limits, TUNED, N_ITER, NOISE  # noqa: E402


def build_lib(dtype="float", force=False):
    out = HERE / "build" / f"libsafe_ps_cpu_{'f32' if dtype == 'float' else 'f64'}.so"
    src = HERE / "safe_ps_cpu.cpp"
    if out.exists() and not force and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    out.parent.mkdir(exist_ok=True)
    cmd = [os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared", f"-DSCALAR={dtype}", str(src), "-o", str(out), "-ldl"]
    subprocess.run(cmd, check=True); print("built", out, flush=True)
    return out


class CpuPS:
    def __init__(self, K, T, n_iter=N_ITER, dtype="float", threads=16, proj_iters=PROJ_ITERS, sigma="tuned", knot_scale="tuned", noise=NOISE, seed=1, par=None,
                 ylim=None, umax=None, keep_nominal=True, pin=True, cpu_lib=None):
        assert dtype == "float", "this example is fp32 only"
        if isinstance(sigma, str) and sigma == "tuned": sigma = TUNED["sigma"]
        if isinstance(knot_scale, str) and knot_scale == "tuned": knot_scale = TUNED["knot_scale"] if sigma is not None else None
        if par is None:
            par = PAR_DEFAULT.copy(); names = [n for n, _ in PARAMS]
            for k_, v_ in TUNED["par"].items(): par[names.index(k_)] = v_
        self.lib = ctypes.CDLL(str(build_lib(dtype)))
        L = self.lib; dp = ctypes.POINTER(ctypes.c_double)
        L.spc_create.restype = ctypes.c_void_p; L.spc_create.argtypes = [ctypes.c_char_p]
        L.spc_tick.restype = ctypes.c_int; L.spc_tick.argtypes = [ctypes.c_void_p, dp, dp, dp, dp, dp]
        for n in ("spc_time_iteration", "spc_time_rollout", "spc_time_model", "spc_time_integrate", "spc_time_project"):
            getattr(L, n).restype = ctypes.c_int; getattr(L, n).argtypes = [ctypes.c_void_p, ctypes.c_int, dp, dp]
        for n in ("spc_set_samples", "spc_get_plan", "spc_set_plan", "spc_get_plant", "spc_set_plant", "spc_get_iter_stats"):
            getattr(L, n).restype = ctypes.c_int; getattr(L, n).argtypes = [ctypes.c_void_p, dp]
        L.spc_get_samples.restype = ctypes.c_int; L.spc_get_samples.argtypes = [ctypes.c_void_p, dp, dp]
        L.spc_n_iter.restype = ctypes.c_int; L.spc_n_iter.argtypes = [ctypes.c_void_p]
        L.spc_info.restype = ctypes.c_char_p; L.spc_info.argtypes = [ctypes.c_void_p]
        L.spc_error.restype = ctypes.c_char_p; L.spc_error.argtypes = []
        L.spc_destroy.restype = None; L.spc_destroy.argtypes = [ctypes.c_void_p]
        self.K, self.T, self.n_iter, self.dtype, self.threads = K, T, n_iter, dtype, threads
        if ylim is None or umax is None:
            lo, hi, um = joint_limits(); ylim = np.concatenate([lo, hi]) if ylim is None else ylim; umax = um if umax is None else umax
        self.ylim, self.umax, self.par = np.asarray(ylim), np.asarray(umax), np.asarray(par)
        if sigma is None:                                   # MJPC: N(0, (noise x half the control range)^2) on every spline parameter
            sigma = noise * 0.5 * (self.ylim[NV:] - self.ylim[:NV])
        sigma = np.asarray(sigma, dtype=np.float64)
        if sigma.shape == (): sigma = np.full(NV, float(sigma))
        if sigma.shape == (NV,):
            ks = np.ones(N_KNOT) if knot_scale is None else np.asarray(knot_scale, dtype=np.float64)
            sigma = (ks[:, None] * sigma[None, :]).reshape(-1)
        self.sigma = sigma; assert self.sigma.shape == (ND,), self.sigma.shape
        cpu_lib = Path(cpu_lib) if cpu_lib else HERE / "build" / "cpu" / f"libcpu_T{T}_{dtype}.so"
        assert cpu_lib.exists(), cpu_lib
        cfg = dict(cpu_lib=str(cpu_lib), K=K, T=T, n_iter=n_iter, threads=threads, seed=seed, dt=DT, proj_iters=proj_iters, keep_nominal=int(keep_nominal), pin=int(pin),
                   sigma=",".join(repr(float(s)) for s in self.sigma), ylim=",".join(repr(float(v)) for v in self.ylim),
                   umax=",".join(repr(float(v)) for v in self.umax), par=",".join(repr(float(v)) for v in self.par))
        self.h = L.spc_create(";".join(f"{k}={v}" for k, v in cfg.items()).encode())
        if not self.h:
            raise RuntimeError("spc_create failed: " + L.spc_error().decode())
        self.info = L.spc_info(self.h).decode()
        self._u = np.zeros((1, NV)); self._stat = np.zeros((1, 4)); self._ms = ctypes.c_double()

    @staticmethod
    def _p(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def tick(self, x_plant, wp):
        """x_plant (1, 16) with q, v or None for the internal model plant; wp (1, 3). Returns (u (1, 7), stat (1, 4), ms)."""
        wp = np.ascontiguousarray(wp, dtype=np.float64).reshape(-1)
        xp = None if x_plant is None else np.ascontiguousarray(x_plant, dtype=np.float64).reshape(-1)
        rc = self.lib.spc_tick(self.h, None if xp is None else self._p(xp), self._p(wp), self._p(self._u), self._p(self._stat), ctypes.byref(self._ms))
        if rc: raise RuntimeError(self.lib.spc_error().decode())
        return self._u.copy(), self._stat.copy(), self._ms.value

    def _timed(self, name, n_calls):
        m, mn = ctypes.c_double(), ctypes.c_double()
        getattr(self.lib, name)(self.h, n_calls, ctypes.byref(m), ctypes.byref(mn)); return m.value, mn.value

    def time_iteration(self, n_calls=20): return self._timed("spc_time_iteration", n_calls)
    def time_rollout(self, n_calls=20): return self._timed("spc_time_rollout", n_calls)
    def time_model(self, n_calls=50): return self._timed("spc_time_model", n_calls)
    def time_integrate(self, n_calls=50): return self._timed("spc_time_integrate", n_calls)
    def time_project(self, n_calls=50): return self._timed("spc_time_project", n_calls)

    def iter_stats(self):
        o = np.zeros((self.n_iter, 1, 7)); self.lib.spc_get_iter_stats(self.h, self._p(o)); return o

    def set_samples(self, Y):
        Y = np.ascontiguousarray(Y, dtype=np.float64).reshape(-1); assert Y.size == self.K * ND; self.lib.spc_set_samples(self.h, self._p(Y))

    def samples(self):
        Y = np.zeros((self.K, ND)); xT = np.zeros((self.K, NX)); self.lib.spc_get_samples(self.h, self._p(Y), self._p(xT)); return Y, xT

    def plan(self):
        y = np.zeros((1, ND)); self.lib.spc_get_plan(self.h, self._p(y)); return y

    def set_plan(self, y):
        y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1); self.lib.spc_set_plan(self.h, self._p(y))

    def plant(self):
        x = np.zeros((1, NX)); self.lib.spc_get_plant(self.h, self._p(x)); return x

    def set_plant(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1); self.lib.spc_set_plant(self.h, self._p(x))

    def close(self):
        if self.h: self.lib.spc_destroy(self.h); self.h = None

    def __del__(self):
        try: self.close()
        except Exception: pass
