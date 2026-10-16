"""ctypes front end of the native driver (safe_mppi.cu -> build/libsafe_mppi_{f32,f64}.so).

    d = SafeMppi(evaluator="cusadi2"|"frax"|"mjx"|"cusadi"|"cpu", K=256, T=20, dtype="double", graph=True, ...)
    u, stat, gpu_ms, wall_ms = d.tick(x_plant (1,16) or None, wp (1,3))

The same binary serves every evaluator; `evaluator` only selects which one the driver constructs:
    cusadi2  the two CasADi functions as level-parallel plans
    frax     frax + cbfpy in JAX, jax.export'ed, run through PJRT          (the demo's own stack)
    mjx      MuJoCo XLA + cbfpy in JAX, jax.export'ed, run through PJRT    (the paper's "MuJoCo XLA" baseline)
    cusadi   the original CusADi's unrolled one-thread-per-instance kernels of the same CasADi functions
    cpu      CasADi's C code generation of the same functions on the CPU (cpu_threads, default 16), with the GPU <-> CPU
             transfer of every call (cpu_transfer=True) or without it (compute-only timing)
"""
import ctypes, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
from model import NX, ND, NV, N_KNOT, NPAR, PAR_DEFAULT, DT, PROJ_ITERS  # noqa: E402

PLUGIN = Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "jax_plugins" / "xla_cuda13" / "xla_cuda_plugin.so"


def joint_limits():
    import common
    common.ensure_frax()
    from frax import load_panda
    r = load_panda()
    return np.asarray(r.actuated_joint_lower_limits), np.asarray(r.actuated_joint_upper_limits), np.asarray(r.actuated_joint_max_forces)


class SafeMppi:
    def __init__(self, evaluator, K, T, R=1, dtype="double", graph=True, proj_iters=PROJ_ITERS, sigma=(0.1, 0.1, 0.1, 0.1), temp=0.3,
                 seed=1, par=PAR_DEFAULT, ylim=None, umax=None, record=32, build=HERE / "build", cpu_threads=16, cpu_transfer=True, block=256,
                 rollout_proj_iters=None, lib=None):
        tag = "f32" if dtype == "float" else "f64"
        self.lib = ctypes.CDLL(str(lib or build / f"libsafe_mppi_{tag}.so"))
        L = self.lib
        L.sm_create.restype = ctypes.c_void_p; L.sm_create.argtypes = [ctypes.c_char_p]
        dp = ctypes.POINTER(ctypes.c_double)
        L.sm_tick.restype = ctypes.c_int; L.sm_tick.argtypes = [ctypes.c_void_p, dp, dp, dp, dp, dp, dp]
        for n in ("sm_time_rollout", "sm_time_model", "sm_time_integrate", "sm_time_project"):
            getattr(L, n).restype = ctypes.c_int; getattr(L, n).argtypes = [ctypes.c_void_p, ctypes.c_int, dp, dp]
        L.sm_get_filter.restype = ctypes.c_int; L.sm_get_filter.argtypes = [ctypes.c_void_p, dp, dp, dp, dp]
        L.sm_get_traj.restype = ctypes.c_int; L.sm_get_traj.argtypes = [ctypes.c_void_p, dp]
        L.sm_get_weights.restype = ctypes.c_int; L.sm_get_weights.argtypes = [ctypes.c_void_p, dp]
        L.sm_n_rec.restype = ctypes.c_int; L.sm_n_rec.argtypes = [ctypes.c_void_p]
        for n in ("sm_get_plan", "sm_set_plan", "sm_get_plant", "sm_set_plant"):
            getattr(L, n).restype = ctypes.c_int; getattr(L, n).argtypes = [ctypes.c_void_p, dp]
        L.sm_get_samples.restype = ctypes.c_int; L.sm_get_samples.argtypes = [ctypes.c_void_p, dp, dp]
        L.sm_info.restype = ctypes.c_char_p; L.sm_info.argtypes = [ctypes.c_void_p]
        L.sm_error.restype = ctypes.c_char_p; L.sm_error.argtypes = []
        L.sm_last_pjrt_exec_ms.restype = ctypes.c_double; L.sm_last_pjrt_exec_ms.argtypes = [ctypes.c_void_p]
        L.sm_destroy.restype = None; L.sm_destroy.argtypes = [ctypes.c_void_p]
        self.R, self.K, self.T, self.B, self.dtype, self.evaluator = R, K, T, R * K, dtype, evaluator
        if ylim is None or umax is None:
            lo, hi, um = joint_limits(); ylim = np.concatenate([lo, hi]) if ylim is None else ylim; umax = um if umax is None else umax
        self.ylim, self.umax = np.asarray(ylim), np.asarray(umax)
        cfg = dict(R=R, K=K, T=T, graph=int(graph), temp=temp, seed=seed, dt=DT, proj_iters=proj_iters, record=record, sigma=",".join(repr(float(s)) for s in sigma),
                   rollout_proj_iters=proj_iters if rollout_proj_iters is None else rollout_proj_iters,
                   ylim=",".join(repr(float(v)) for v in self.ylim), umax=",".join(repr(float(v)) for v in self.umax),
                   par=",".join(repr(float(v)) for v in np.asarray(par)))
        if evaluator == "cusadi2":
            cfg["evaluator"] = "cusadi2"; cfg["model_plan"] = str(build / "plans" / f"safe_model_T{T}_{dtype}.plan")
            cfg["integrate_plan"] = str(build / "plans" / f"safe_integrate_{dtype}.plan")
        elif evaluator in ("frax", "mjx"):
            stem = "safe" if evaluator == "frax" else "mjx"
            cfg["evaluator"] = "pjrt"; cfg["plugin"] = str(PLUGIN)
            cfg["model_mlir"] = str(build / "programs" / f"{stem}_model_T{T}_B{R * K}_{dtype}.mlir")
            cfg["integrate_mlir"] = str(build / "programs" / f"{stem}_integrate_B{R * K}_{dtype}.mlir")
            cfg["compile_options"] = str(build / "programs" / "compile_options.pb")
        elif evaluator == "cusadi":
            cfg["evaluator"] = "cusadi"; cfg["block"] = block
            cfg["model_lib"] = str(build / "unrolled" / f"libunrolled_model_T{T}_{dtype}.so")
            cfg["integrate_lib"] = str(build / "unrolled" / f"libunrolled_integrate_{dtype}.so")
        elif evaluator == "cpu":
            cfg["evaluator"] = "cpu"; cfg["cpu_lib"] = str(build / "cpu" / f"libcpu_T{T}_{dtype}.so")
            cfg["cpu_threads"] = cpu_threads; cfg["cpu_transfer"] = int(cpu_transfer)
        else:
            raise ValueError(f"unknown evaluator {evaluator!r}")
        for v in cfg.values():
            if isinstance(v, str) and v.endswith((".plan", ".mlir", ".pb", ".so")):
                assert os.path.exists(v), v
        s = ";".join(f"{k}={v}" for k, v in cfg.items())
        self.h = L.sm_create(s.encode())
        if not self.h:
            raise RuntimeError("sm_create failed: " + L.sm_error().decode())
        self.info = L.sm_info(self.h).decode()
        self._u = np.zeros((R, NV)); self._stat = np.zeros((R, 4)); self._g = ctypes.c_double(); self._w = ctypes.c_double()

    @staticmethod
    def _p(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def _ck(self, rc):
        if rc:
            raise RuntimeError(self.lib.sm_error().decode())

    def tick(self, x_plant, wp):
        """x_plant (R, 16) with q, v (tau and cost ignored) or None for the internal model plant; wp (R, 3) waypoints."""
        wp = np.ascontiguousarray(wp, dtype=np.float64)
        xp = None if x_plant is None else np.ascontiguousarray(x_plant, dtype=np.float64)
        self._ck(self.lib.sm_tick(self.h, None if xp is None else self._p(xp), self._p(wp), self._p(self._u), self._p(self._stat),
                                  ctypes.byref(self._g), ctypes.byref(self._w)))
        return self._u.copy(), self._stat.copy(), self._g.value, self._w.value

    def _timed(self, name, n_calls):
        m, mn = ctypes.c_double(), ctypes.c_double()
        self._ck(getattr(self.lib, name)(self.h, n_calls, ctypes.byref(m), ctypes.byref(mn)))
        return m.value, mn.value

    def time_rollout(self, n_calls=20):
        return self._timed("sm_time_rollout", n_calls)

    def time_model(self, n_calls=50):
        return self._timed("sm_time_model", n_calls)

    def time_integrate(self, n_calls=50):
        return self._timed("sm_time_integrate", n_calls)

    def time_project(self, n_calls=50):
        return self._timed("sm_time_project", n_calls)

    def filter_data(self):
        """(u_nom, G (B, 64, 7), h, u) of the last projection on the sample buffers (G from cusadi2's column-major layout)."""
        B = self.B; un = np.zeros((B, NV)); G = np.zeros((B, 7, 64)); h = np.zeros((B, 64)); u = np.zeros((B, NV))
        self._ck(self.lib.sm_get_filter(self.h, self._p(un), self._p(G), self._p(h), self._p(u)))
        return un, np.transpose(G, (0, 2, 1)), h, u

    def plan(self):
        y = np.zeros((self.R, ND)); self._ck(self.lib.sm_get_plan(self.h, self._p(y))); return y

    def set_plan(self, y):
        y = np.ascontiguousarray(y, dtype=np.float64); self._ck(self.lib.sm_set_plan(self.h, self._p(y)))

    def plant(self):
        x = np.zeros((self.R, NX)); self._ck(self.lib.sm_get_plant(self.h, self._p(x))); return x

    def set_plant(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64); self._ck(self.lib.sm_set_plant(self.h, self._p(x)))

    def samples(self):
        Y = np.zeros((self.B, ND)); xT = np.zeros((self.B, NX)); self._ck(self.lib.sm_get_samples(self.h, self._p(Y), self._p(xT))); return Y, xT

    def traj(self):
        """(T+1, n_rec, 16): the state trajectories of the first n_rec samples of the last tick (sample 0 = the mean plan)."""
        n = self.lib.sm_n_rec(self.h); out = np.zeros((self.T + 1, n, NX)); self._ck(self.lib.sm_get_traj(self.h, self._p(out))); return out

    def weights(self):
        w = np.zeros(self.B); self._ck(self.lib.sm_get_weights(self.h, self._p(w))); return w

    def last_pjrt_exec_ms(self):
        return self.lib.sm_last_pjrt_exec_ms(self.h)

    def close(self):
        if self.h:
            self.lib.sm_destroy(self.h); self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
