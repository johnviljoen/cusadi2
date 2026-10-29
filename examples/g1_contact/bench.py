"""Latency of the G1 step vs batch size: cusadi2 (this example's step library) against mujoco-warp, both
from this process, both as one CUDA graph of N chained steps from the same per-instance initial states
(build/inputs), timed as graph launch + device synchronize on the host wall clock (mujoco-warp's own
testspeed protocol), medians over repeats. The cusadi2 arm also records CUDA-event time of the same graph.

    python bench.py [--phase1 | --phase3] [--scenario standing|drop] [--B 1,4,...] [--steps N] [--runs 3] [--out results]

--phase1: the contact-free premise comparison. MuJoCo contacts and the constraint solver are disabled on
the warp side (contype/conaffinity zeroed, mjDSBL_CONTACT | mjDSBL_CONSTRAINT: the same flags as the
Menagerie campaign's baselines) and the cusadi2 step is smooth -> integrate with qfrc_constraint = 0,
once with the smooth_nogeom plan (the equivalent workload) and once with the full smooth plan (the geom
placements and Jacobians the contact step will need). A MuJoCo C rollout of the first instances gives
the reference final state so all three agree on the trajectory. Rows go to <out>/phase1_<scenario>.csv:
  arm,variant,dtype,scenario,B,n_steps,median_ms,min_ms,event_median_ms,mean_z,mean_speed,max_dz_vs_C

--phase3: contacts on. cusadi2 = smooth -> fused contact kernel -> integrate (variant 1); warp = mjw.step with
its row caps from build/pilot.json (njmax = the largest nefc seen in C with margin, nconmax likewise) and the
warm start zeroed at the start of every launch, as ours and as MuJoCo's reset. Episodes: 200 steps (standing),
300 (drop) unless --steps. Rows go to <out>/phase3_<scenario>.csv with two more columns:
  ...,mean_nefc,mean_niter   (over the instances at the last step; warp: nefc only)
"""
import argparse
import ctypes
import json
import os
import sys
import time

import mujoco
import numpy as np

import g1

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
N_WARMUP, N_TIMED = 5, 20
N_REF = 8          # instances rolled out in MuJoCo C for the agreement column


class CusadiStep:
    def __init__(self, dtype, variant, B, n_steps, q, v, u):
        self.lib = ctypes.CDLL(os.path.join(BUILD, f"libg1step_{dtype}.so"))
        self.lib.g1_create.restype = ctypes.c_void_p
        self.lib.g1_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.POINTER(ctypes.c_double)] * 3
        self.lib.g1_launch.argtypes = [ctypes.c_void_p]
        self.lib.g1_time_events.restype = ctypes.c_double
        self.lib.g1_time_events.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.lib.g1_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.lib.g1_destroy.argtypes = [ctypes.c_void_p]
        self.B, self.nq, self.nv, self.variant = B, q.shape[1], v.shape[1], variant
        P = lambda a: np.ascontiguousarray(a, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._keep = [np.ascontiguousarray(a, dtype=np.float64) for a in (q, v, u)]
        self.h = self.lib.g1_create(BUILD.encode(), variant, B, n_steps, *[P(a) for a in self._keep])

    def launch(self):
        self.lib.g1_launch(self.h)

    def time_events(self, n):
        ms = np.zeros(n)
        med = self.lib.g1_time_events(self.h, n, ms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return med, ms

    def state(self):
        q, v = np.zeros((self.B, self.nq)), np.zeros((self.B, self.nv))
        self.lib.g1_state(self.h, q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return q, v

    def stats(self):
        """(mean nefc, mean niter) over the instances at the last step; nan without the contact kernel."""
        if self.variant == 0:
            return float("nan"), float("nan")
        self.lib.g1_solver.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
        qa, qf = np.zeros((self.B, self.nv)), np.zeros((self.B, self.nv)); st = np.zeros((self.B, 4), dtype=np.int32)
        self.lib.g1_solver(self.h, qa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), qf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           st.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        if st[:, 3].any():
            print(f"  WARNING: solver overflow flags on {int((st[:, 3] != 0).sum())} instances (rows {int((st[:, 3] & 1 != 0).sum())}, "
                  f"iterations {int((st[:, 3] & 2 != 0).sum())}, linesearch {int((st[:, 3] & 4 != 0).sum())})", flush=True)
        return float(st[:, 0].mean()), float(st[:, 2].mean())

    def close(self):
        self.lib.g1_destroy(self.h)


class WarpStep:
    def __init__(self, m, B, n_steps, q, v, u, njmax=None, nconmax=None):
        import warp as wp
        import mujoco_warp as mjw
        self.wp = wp
        self.mw = mjw.put_model(m)
        self.mw.opt.warn_overflow = False          # no device printf when the model's iteration caps are hit (it is their default and ours too)
        self.d = mjw.put_data(m, mujoco.MjData(m), nworld=B, njmax=njmax, nconmax=nconmax)
        self.q0 = wp.array(np.ascontiguousarray(q, dtype=np.float32), dtype=wp.float32)
        self.v0 = wp.array(np.ascontiguousarray(v, dtype=np.float32), dtype=wp.float32)
        self.d.ctrl = wp.array(np.ascontiguousarray(u, dtype=np.float32), dtype=wp.float32)
        with wp.ScopedCapture() as cap:
            wp.copy(self.d.qpos, self.q0)
            wp.copy(self.d.qvel, self.v0)
            self.d.qacc_warmstart.zero_()
            for _ in range(n_steps):
                mjw.step(self.mw, self.d)
        self.graph = cap.graph
        self.launch()

    def launch(self):
        self.wp.capture_launch(self.graph)
        self.wp.synchronize()

    def state(self):
        return self.d.qpos.numpy().astype(np.float64), self.d.qvel.numpy().astype(np.float64)

    def stats(self):
        return float(self.d.nefc.numpy().mean()), float("nan")

    def close(self):
        del self.graph, self.d, self.mw


def median_wall(fn, warm, n):
    for _ in range(warm):
        fn()
    s = np.empty(n)
    for i in range(n):
        t = time.perf_counter()
        fn()
        s[i] = time.perf_counter() - t
    return float(np.median(s)) * 1e3, float(s.min()) * 1e3


def c_reference(m, q, v, u, n_steps):
    d = mujoco.MjData(m)
    out = np.zeros_like(q)
    for b in range(q.shape[0]):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q[b], v[b], u[b]
        for _ in range(n_steps):
            mujoco.mj_step(m, d)
        out[b] = d.qpos
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", action="store_true")
    ap.add_argument("--phase3", action="store_true")
    ap.add_argument("--scenario", default="standing,drop")
    ap.add_argument("--B", default="1,4,16,64,256,1024,4096,16384")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--dtypes", default="f32")
    ap.add_argument("--arms", default="cusadi2,warp")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    if not (a.phase1 or a.phase3):
        sys.exit("pass --phase1 or --phase3")
    phase = 1 if a.phase1 else 3
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(BUILD, "meta.json")))
    Bs = [int(x) for x in a.B.split(",")]
    caps = {}
    if phase == 3:
        pilot = json.load(open(os.path.join(BUILD, "pilot.json")))
        max_nefc = max(p["max_nefc"] for p in pilot.values()); max_ncon = max(p["max_ncon"] for p in pilot.values())
        caps = dict(njmax=meta["njmax"], nconmax=int(np.ceil(max_ncon * 1.5 / 8) * 8))
        if max_nefc > meta["njmax"]:
            sys.exit(f"pilot max nefc {max_nefc} exceeds the built G1_NJMAX {meta['njmax']}: rebuild with a larger cap")
        print(f"row caps: njmax {caps['njmax']} (pilot max nefc {max_nefc}), nconmax {caps['nconmax']} (pilot max ncon {max_ncon})", flush=True)
    for scen in a.scenario.split(","):
        n_steps = a.steps or {"standing": 200, "drop": 300}[scen]
        q = np.fromfile(os.path.join(BUILD, "inputs", f"{scen}_qpos.f64")).reshape(-1, meta["nq"])
        v = np.fromfile(os.path.join(BUILD, "inputs", f"{scen}_qvel.f64")).reshape(-1, meta["nv"])
        u = np.fromfile(os.path.join(BUILD, "inputs", f"{scen}_ctrl.f64")).reshape(-1, meta["nu"])
        m = mujoco.MjModel.from_xml_path(g1.XML)
        if phase == 1:
            g1.disable_contacts(m)
        qref = c_reference(m, q[:N_REF], v[:N_REF], u[:N_REF], n_steps)
        path = os.path.join(a.out, f"phase{phase}_{scen}.csv")
        rows = []
        if os.path.exists(path):                 # arms can be run in separate processes: merge by (arm, variant, dtype, B)
            import csv
            with open(path) as f:
                rows = [{k: (v if k in ("arm", "variant", "dtype", "scenario") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
                for r in rows:
                    r["B"] = int(r["B"]); r["n_steps"] = int(r["n_steps"])
        print(f"[{scen}] N={n_steps} steps, reference z after N (C, first {N_REF}): {np.round(qref[:, 2], 3).tolist()}", flush=True)

        def record(arm, variant, dtype, B, step):
            meds, mins = zip(*[median_wall(step.launch, N_WARMUP, N_TIMED) for _ in range(a.runs)])
            ev = step.time_events(N_TIMED)[0] if hasattr(step, "time_events") else float("nan")
            qf, vf = step.state()
            n = min(B, N_REF)
            dz = float(np.abs(qf[:n, 2] - qref[:n, 2]).max())
            nefc, niter = step.stats()
            row = dict(arm=arm, variant=variant, dtype=dtype, scenario=scen, B=B, n_steps=n_steps, median_ms=float(np.median(meds)),
                       min_ms=float(min(mins)), event_median_ms=ev, mean_z=float(qf[:, 2].mean()),
                       mean_speed=float(np.linalg.norm(vf[:, :3], axis=1).mean()), max_dz_vs_C=dz, mean_nefc=nefc, mean_niter=niter)
            rows[:] = [r for r in rows if not (r["arm"] == arm and r["variant"] == variant and r["dtype"] == dtype and r["B"] == B)] + [row]
            rows.sort(key=lambda r: (r["arm"], r["variant"], r["dtype"], r["B"]))
            print(f"  {arm:8s} {variant:8s} {dtype} B={B:6d}: {row['median_ms']:9.3f} ms (min {row['min_ms']:.3f}, events {ev:.3f}) "
                  f"z {row['mean_z']:.3f} |dz vs C| {dz:.2e} nefc {nefc:.1f} niter {niter:.2f}", flush=True)
            with open(path, "w") as f:
                f.write(",".join(rows[0].keys()) + "\n")
                for r in rows:
                    f.write(",".join(str(x) for x in r.values()) + "\n")

        for arm in a.arms.split(","):
            if arm == "cusadi2":
                for dtype in a.dtypes.split(","):
                    variants = (("nogeom", 0), ("geom", 1)) if phase == 1 else (("fused", 1),)
                    for variant, vid in variants:
                        for B in Bs:
                            step = CusadiStep(dtype, vid, B, n_steps, q[:B], v[:B], u[:B])
                            record("cusadi2", variant, dtype, B, step)
                            step.close()
            elif arm == "warp":
                import warp as wp
                wp.init()
                for B in Bs:
                    try:
                        step = WarpStep(m, B, n_steps, q[:B], v[:B], u[:B], **caps)
                    except Exception as e:
                        print(f"  warp B={B}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                        continue
                    record("warp", "step", "f32", B, step)
                    step.close()


if __name__ == "__main__":
    main()
