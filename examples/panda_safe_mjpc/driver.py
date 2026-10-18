"""ctypes front end of the native driver (safe_mppi.cu -> build/libsafe_mppi_f32.so; this example is fp32 only).

    d = SafePS(evaluator="cusadi2"|"frax"|"mjx"|"cusadi"|"cpu", K=64, T=25, n_iter=N_ITER, noise=0.1, graph=True, ...)
    u, stat, gpu_ms, wall_ms = d.tick(x_plant (1,16) or None, wp (1,3))     stat = (cost min, mean, std, selected index), last iteration
    d.iter_stats()  (n_iter, R, 7): per iteration (cost of the selected, mean, std, selected index, cost of the nominal = sample 0,
                    number of feasible candidates, worst CBF violation of the selected along its rollout)
    d.recompute()   the gate: the selected plan's filtered torque recomputed at the tick's plant state; must equal u

MJPC-faithful predictive sampling: n_iter sequential iterations per tick, each K samples around the nominal (sigma =
noise x half the joint range per joint on every knot of the cubic 7-knot spline, MJPC's AddNoiseToPolicy), the best
rollout becomes the nominal; the applied torque is the step-0 filtered torque of the final nominal's rollout, no
weights, no temperature, no plant-side re-filter; the plan is then resampled one period later (MJPC's UpdateNominalPolicy).

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
from model import NX, ND, NV, N_KNOT, NPAR, PAR_DEFAULT, PARAMS, DT, PROJ_ITERS, knot_dt  # noqa: E402

N_ITER = 3                      # iterations per 20 ms tick at the tuned horizon (T = 35: 5.3 ms per iteration, tick 15.5 ms)
NOISE = 0.1                     # MJPC's sampling_exploration: knot noise std = NOISE x half the joint range (the MJPC-faithful run)
# The tuned configuration (closed_loop.py defaults; sweep.py grids 1-6, results/closed_loop_ps_<tag>_seed<s>.json): 0.7 s
# horizon, 3 iterations of 256 samples, 0.10 rad knot noise with knot 0 frozen (the applied torque stays continuous),
# kp 20, w_pos 100. MJPC-faithful phase-1 settings: T 25, K 64, n_iter 4, noise 0.1 x half range on every knot, kp 50, w_pos 50.
TUNED = dict(T=35, K=256, n_iter=3, sigma=0.10, knot_scale=(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), par=dict(kp=20.0, w_pos=100.0))

PLUGIN = Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "jax_plugins" / "xla_cuda13" / "xla_cuda_plugin.so"


def joint_limits():
    import common
    common.ensure_frax()
    from frax import load_panda
    r = load_panda()
    return np.asarray(r.actuated_joint_lower_limits), np.asarray(r.actuated_joint_upper_limits), np.asarray(r.actuated_joint_max_forces)


class SafePS:
    def __init__(self, evaluator, K, T, R=1, n_iter=N_ITER, noise=NOISE, dtype="float", graph=True, proj_iters=PROJ_ITERS, sigma="tuned", knot_scale="tuned",
                 seed=1, par=None, ylim=None, umax=None, record=32, build=HERE / "build", cpu_threads=16, cpu_transfer=True, block=256, keep_nominal=True, select="cost", feas_tol=0.0):
        assert dtype == "float", "this example is fp32 only"
        # Defaults are the tuned controller (TUNED): 0.10 rad knot noise with knot 0 frozen, kp 20, w_pos 100, 3 iterations.
        # sigma=None selects MJPC's noise scaling (noise x half the joint range on every knot), knot_scale=None every knot.
        if isinstance(sigma, str) and sigma == "tuned": sigma = TUNED["sigma"]
        if isinstance(knot_scale, str) and knot_scale == "tuned": knot_scale = TUNED["knot_scale"] if sigma is not None else None
        if par is None:
            par = PAR_DEFAULT.copy(); names = [n for n, _ in PARAMS]
            for k_, v_ in TUNED["par"].items(): par[names.index(k_)] = v_
        self.lib = ctypes.CDLL(str(build / "libsafe_mppi_f32.so"))
        L = self.lib
        L.sm_create.restype = ctypes.c_void_p; L.sm_create.argtypes = [ctypes.c_char_p]
        dp = ctypes.POINTER(ctypes.c_double)
        L.sm_tick.restype = ctypes.c_int; L.sm_tick.argtypes = [ctypes.c_void_p, dp, dp, dp, dp, dp, dp]
        for n in ("sm_time_rollout", "sm_time_iteration", "sm_time_model", "sm_time_integrate", "sm_time_project"):
            getattr(L, n).restype = ctypes.c_int; getattr(L, n).argtypes = [ctypes.c_void_p, ctypes.c_int, dp, dp]
        L.sm_get_filter.restype = ctypes.c_int; L.sm_get_filter.argtypes = [ctypes.c_void_p, dp, dp, dp, dp]
        L.sm_get_traj.restype = ctypes.c_int; L.sm_get_traj.argtypes = [ctypes.c_void_p, dp]
        L.sm_recompute.restype = ctypes.c_int; L.sm_recompute.argtypes = [ctypes.c_void_p, dp]
        L.sm_n_rec.restype = ctypes.c_int; L.sm_n_rec.argtypes = [ctypes.c_void_p]
        L.sm_n_iter.restype = ctypes.c_int; L.sm_n_iter.argtypes = [ctypes.c_void_p]
        L.sm_get_iter_stats.restype = ctypes.c_int; L.sm_get_iter_stats.argtypes = [ctypes.c_void_p, dp]
        L.sm_get_viol.restype = ctypes.c_int; L.sm_get_viol.argtypes = [ctypes.c_void_p, dp]
        for n in ("sm_get_plan", "sm_set_plan", "sm_get_plant", "sm_set_plant"):
            getattr(L, n).restype = ctypes.c_int; getattr(L, n).argtypes = [ctypes.c_void_p, dp]
        L.sm_get_samples.restype = ctypes.c_int; L.sm_get_samples.argtypes = [ctypes.c_void_p, dp, dp]
        L.sm_info.restype = ctypes.c_char_p; L.sm_info.argtypes = [ctypes.c_void_p]
        L.sm_error.restype = ctypes.c_char_p; L.sm_error.argtypes = []
        L.sm_last_pjrt_exec_ms.restype = ctypes.c_double; L.sm_last_pjrt_exec_ms.argtypes = [ctypes.c_void_p]
        L.sm_destroy.restype = None; L.sm_destroy.argtypes = [ctypes.c_void_p]
        self.R, self.K, self.T, self.B, self.dtype, self.evaluator, self.n_iter, self.noise = R, K, T, R * K, dtype, evaluator, n_iter, noise
        if ylim is None or umax is None:
            lo, hi, um = joint_limits(); ylim = np.concatenate([lo, hi]) if ylim is None else ylim; umax = um if umax is None else umax
        self.ylim, self.umax = np.asarray(ylim), np.asarray(umax)
        if sigma is None:                                   # MJPC: N(0, (noise x half the control range)^2) on every spline parameter
            sigma = noise * 0.5 * (self.ylim[NV:] - self.ylim[:NV])
        sigma = np.asarray(sigma, dtype=np.float64)
        if sigma.shape == ():                               # one absolute std (rad) for every knot and joint
            sigma = np.full(NV, float(sigma))
        if sigma.shape == (NV,):                            # per joint -> per knot x joint, optionally scaled per knot (knot_scale[j])
            ks = np.ones(N_KNOT) if knot_scale is None else np.asarray(knot_scale, dtype=np.float64)
            sigma = (ks[:, None] * sigma[None, :]).reshape(-1)
        self.sigma = sigma; assert self.sigma.shape == (ND,), self.sigma.shape
        self.par = np.asarray(par, dtype=np.float64)
        cfg = dict(R=R, K=K, T=T, n_iter=n_iter, graph=int(graph), seed=seed, keep_nominal=int(keep_nominal), select=select, feas_tol=repr(float(feas_tol)), dt=DT, proj_iters=proj_iters, record=record, sigma=",".join(repr(float(s)) for s in self.sigma),
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
        """x_plant (R, 16) with q, v (tau and cost ignored) or None for the internal model plant; wp (R, 3) waypoints.
        Returns u (R, 7), stat (R, 4) = (cost min, cost mean, cost std, selected sample index), gpu ms, wall ms."""
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

    def time_iteration(self, n_calls=20):
        """One planner iteration (sample, init, T-step rollout, select) alone, median and min ms."""
        return self._timed("sm_time_iteration", n_calls)

    def iter_stats(self):
        """(n_iter, R, 5) of the last tick: per iteration (cost min, mean, std, selected index, cost of sample 0 = the nominal)."""
        out = np.zeros((self.n_iter, self.R, 7)); self._ck(self.lib.sm_get_iter_stats(self.h, self._p(out))); return out

    def violations(self):
        """(B,) the worst signed distance of each rollout's torque to the CBF rows over the last iteration's horizon."""
        out = np.zeros(self.B); self._ck(self.lib.sm_get_viol(self.h, self._p(out))); return out

    @property
    def kdt(self):
        return knot_dt(self.T)

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
        """(T+1, n_rec, 16): the state trajectories of the first n_rec samples of the last tick (sample 0 = the current
        plan; record=K keeps every rollout, so the selected one is traj[:, stat[0, 3]])."""
        n = self.lib.sm_n_rec(self.h); out = np.zeros((self.T + 1, n, NX)); self._ck(self.lib.sm_get_traj(self.h, self._p(out))); return out

    def costs(self):
        """(B,) the final rollout costs of the last tick."""
        return self.samples()[1][:, NX - 1]

    def recompute(self):
        """(7,) the selected plan's torque recomputed as model + projection at the tick's plant state (external-plant
        ticks only, R = 1): the gate that the gathered step-0 torque is the plant-side filtered torque."""
        u = np.zeros(NV); self._ck(self.lib.sm_recompute(self.h, self._p(u))); return u

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
