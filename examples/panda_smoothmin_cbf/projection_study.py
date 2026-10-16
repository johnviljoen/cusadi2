"""A merged smooth-min CBF (one HOCBF row, exact projection onto half-space ∩ box) against the shared greedy-48
filter and cbfpy's elastic QP, on the data and metrics of examples/panda_safe_mppi/projection_study.py.
Study only: nothing in panda_safe_mppi (driver, plans, committed results) is touched; this directory imports from it.

Composite barrier (cbfpy's 64 relative-degree-2 barriers h2_i of the frax demo scene, S_SING below):
    H(q) = -(1/kappa) log sum_i exp(-kappa s_i h2_i(q))             H <= min_i s_i h2_i, H >= min_i s_i h2_i - log(64)/kappa
HOCBF row, the same cascade cbfpy applies to every h2 row (alpha = alpha_2 = 10 in the demo config):
    h = dH/dt + alpha H = grad_q H . v + alpha H,   a = -L_g h,  b = alpha h + L_f h,   row  a u <= b
with L_f h, L_g h by forward mode (jvp with f(z) and the columns of g(z)), exactly as cbfpy's h_and_Lfh / Lgh.
Filter: the exact projection of u_nom onto {a u <= b} ∩ box,  u(lam) = clip(u_nom - lam a),  lam >= 0 the root of the
monotone piecewise-linear phi(lam) = a . u(lam) - b by 40 fixed bisection steps on [0, lam_sat] (lam_sat saturates every
coordinate with a non-negligible a_j) plus one exact linear polish on the final segment; lam = 0 if phi(0) <= 0.
If phi(lam_sat) > 0 the row is infeasible inside the box and the filter returns the saturated torque (reported). Because that
saturates every joint with a non-negligible a_j (first run: wrist joints at full torque, 50-100 rad/s within a few steps, the
Euler plant diverging), a second variant caps lam at LAM_CAP = 1000, cbfpy's cbf_relaxation_penalty: with P = I that is exactly
the bound on the elastic QP's multiplier of a barrier row, so an infeasible row is violated instead of chased into the corner.

Scales: joint limits (rad) and spheres (m) s = 1. Singularity: s = S_SING = 4, chosen from the measured gradient norms
on the 2048 test states (median |d h2/dq|: joint limits 1.00 /rad, obstacle/floor/self-collision 0.22-0.29 /rad,
singularity 0.068 /rad): 4 x 0.068 = 0.27 /rad, so a 0.01 margin of the scaled singularity barrier is about the same
joint motion to the boundary (0.04 rad) as a 0.01 m sphere margin. (Matching the joint-limit rows instead would be
s = 14.6.)

Sweep kappa in {25, 50, 100, 200, 400, 800} at dt = 20 ms. Metrics as results/projection_study.json of panda_safe_mppi
(same 2048 states, seed 0; the 344 initially safe benign states, 20-step rollouts, raw h2 minimum over the rollout;
instantaneous violation of the 64 original rows after the filter on all states), plus the mean |u - u_nom| (random
states, by kind, and along the filter's own rollouts), the fraction of states the composite already counts as unsafe
(H(x0) < 0 or h(x0) < 0) and the cancellation diagnostics (|a| ~ 0 while an original row is violated, and
rho = |sum_i w_i s_i grad h_i| / sum_i w_i s_i |grad h_i| < 0.1 with the softmin weights w).
A rollout whose joint speed passes 50 rad/s or whose state turns non-finite is 'diverged' (counted as unsafe, left out of
the h2 minimum; the baselines never diverge). The three baselines are re-run here so every row has every metric; their rollout numbers must reproduce the committed
panda_safe_mppi/results/projection_study.json (checked and printed).

Success criterion, fixed before running: some kappa reaches 0 % of rollouts below -0.01 with worst h2 >= -0.0045
(greedy-48 / QP level) and mean |u - u_nom| on the random states no worse than 1.5 x the elastic QP's.
Output: results/projection_study_smoothmin.json, then report.py writes results/report.md."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
SAFE = HERE.parent / "panda_safe_mppi"
sys.path.insert(0, str(SAFE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common
common.configure("gpu", x64=True)
import jax, jax.numpy as jnp, numpy as np, qpax
from frax_model import FraxStep
from model import NV, PARAMS, N_KNOT, DT
from dataset import make_dataset

T = 20; n = 2048
S_SING = float(os.environ.get("SMOOTHMIN_S_SING", 4.0))
ALPHA = 10.0
KAPPAS = [25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
BISECT_ITERS = 40
SUCCESS = {"frac_below_-0.01": 0.0, "worst_h2": -0.0045, "dev_ratio_vs_qp": 1.5}

fs = FraxStep(T); ds = make_dataset(fs.base, n, 0, T)
X, Y, RP, PAR = [jnp.asarray(ds[k]) for k in ("x", "Y", "rp", "par")]
kind = np.asarray(ds["kind"]); ncbf = fs.cfg.num_cbf; m = NV; u_max = fs.u_max; cfg = fs.cfg
groups = fs.base.barrier_groups
I_SING = groups["joint_limits"]                    # row 14
S = np.ones(ncbf); S[I_SING] = S_SING; S = jnp.asarray(S)

# ---------------------------------------------------------------- the pieces of panda_safe_mppi/projection_study.py
def nominal(x, Y, par):
    q, v, tau = x[:m], x[m:2 * m], x[2 * m]
    P = {k: par[i] for i, (k, _) in enumerate(PARAMS)}
    M = fs.robot.mass_matrix(q); g = fs.robot.gravity_vector(q); c = fs.robot.centrifugal_coriolis_vector(q, v)
    w = fs.weights(tau); q_des = sum(w[j] * Y[j * m:(j + 1) * m] for j in range(N_KNOT))
    return jnp.clip(M @ (P["kp"] * (q_des - q) - P["kd"] * v) + g + c, -u_max, u_max), M, g, c
def rows(z):
    return fs.cbf.G_qp(z, jnp.zeros(m))[:ncbf], fs.cbf.h_qp(z, jnp.zeros(m))[:ncbf]
def proj_greedy(u, G, h, iters):
    gg = jnp.sum(G * G, axis=1) + 1e-12
    u = jnp.clip(u, -u_max, u_max)
    for _ in range(iters):
        r = (G @ u - h) / jnp.sqrt(gg)
        i = jnp.argmax(r)
        u = u - jnp.where(r[i] > 0, (G[i] @ u - h[i]) / gg[i], 0.0) * G[i]
        u = jnp.clip(u, -u_max, u_max)
    return u
def qp(u_nom, G, h):
    Gf = jnp.concatenate([G, jnp.eye(m), -jnp.eye(m)]); hf = jnp.concatenate([h, u_max, u_max])
    return qpax.solve_qp_elastic_primal(jnp.eye(m), -u_nom, Gf, hf, penalty=jnp.asarray(fs.base.penalty), solver_tol=fs.base.solver_tol)[:m]

# ---------------------------------------------------------------- the composite barrier, its HOCBF row and the exact one-row projection
def H_comp(z, kappa):
    return -(1.0 / kappa) * jax.nn.logsumexp(-kappa * S * cfg.h_2(z))

def comp_row(z, kappa):
    """(a, b, h, H): the row a u <= b of the composite barrier, built like cbfpy's rows (h = jvp of H along f + alpha H,
    then L_f h and L_g h by jvp along f and the columns of g; f and g are the config's, evaluated at the differentiated state)."""
    def h1(s):
        H, dH = jax.jvp(lambda t: H_comp(t, kappa), (s,), (cfg.f(s),))
        return dH + ALPHA * H
    h, Lfh = jax.jvp(h1, (z,), (cfg.f(z),))
    Lgh = jax.vmap(lambda gc: jax.jvp(h1, (z,), (gc,))[1], in_axes=1)(cfg.g(z))
    return -Lgh, ALPHA * h + Lfh, h, H_comp(z, kappa)

LAM_CAP = 1000.0   # cbfpy's cbf_relaxation_penalty: with P = I the elastic QP's multiplier of a barrier row never exceeds it

def proj_row_box(u_nom, a, b, lam_max=None):
    """Exact projection of u_nom onto {a u <= b} ∩ [-u_max, u_max]: u(lam) = clip(u_nom - lam a), lam by bisection.
    With lam_max the multiplier is capped (the elastic QP's treatment of an infeasible row: its slack absorbs the rest).
    Returns (u, lam, infeasible)."""
    u_nom = jnp.clip(u_nom, -u_max, u_max)
    phi = lambda lam: a @ jnp.clip(u_nom - lam * a, -u_max, u_max) - b   # noqa: E731  non-increasing, piecewise linear
    an = jnp.abs(a); amax = jnp.max(an)
    lam_sat = jnp.max(jnp.where(an > 1e-6 * amax, (jnp.abs(u_nom) + u_max) / jnp.maximum(an, 1e-300), 0.0))
    need = (phi(0.0) > 0) & (amax > 1e-8)
    lo = jnp.zeros(()); hi = lam_sat
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi); viol = phi(mid) > 0
        lo = jnp.where(viol, mid, lo); hi = jnp.where(viol, hi, mid)
    lam = hi
    # one exact linear step on the segment the bisection landed in (phi is linear between saturation points)
    w = u_nom - lam * a; free = (jnp.abs(w) < u_max).astype(a.dtype); sat = jnp.clip(w, -u_max, u_max) * (1 - free)
    den = jnp.sum(free * a * a)
    lam_lin = (jnp.sum(free * a * u_nom) + a @ sat - b) / jnp.where(den > 0, den, 1.0)
    lam_lin = jnp.clip(lam_lin, lo, hi)
    lam = jnp.where((den > 0) & (jnp.abs(phi(lam_lin)) <= jnp.abs(phi(lam))), lam_lin, lam)
    infeasible = need & (phi(lam_sat) > 0)
    if lam_max is not None: lam = jnp.minimum(lam, lam_max)
    lam = jnp.where(need, lam, 0.0)
    return jnp.clip(u_nom - lam * a, -u_max, u_max), lam, infeasible

def smoothmin_filter(kappa, lam_max=None):
    def filt(u, z, G, h):
        a, b, _, _ = comp_row(z, kappa)
        return proj_row_box(u, a, b, lam_max)[0]
    return filt

FILTERS = {"none (box only)": lambda u, z, G, h: jnp.clip(u, -u_max, u_max),
           "greedy 48 (the kernel)": lambda u, z, G, h: proj_greedy(u, G, h, 48),
           "elastic QP (cbfpy / qpax)": lambda u, z, G, h: qp(u, G, h)}
SMOOTH = {}                                            # name -> (kappa, lam_max)
for k in KAPPAS:
    SMOOTH[f"smooth-min kappa={k:g}"] = (k, None)
for k in KAPPAS:
    SMOOTH[f"smooth-min kappa={k:g}, lam<={LAM_CAP:g}"] = (k, LAM_CAP)
for name, (k, cap) in SMOOTH.items():
    FILTERS[name] = smoothmin_filter(k, cap)

def step(filt, x, Y, par):
    u0, M, g, c = nominal(x, Y, par); z = x[:2 * m]; G, h = rows(z)
    u = filt(u0, z, G, h)
    a = jnp.linalg.solve(M, u - g - c); v1 = x[m:2 * m] + DT * a; q1 = x[:m] + DT * v1
    return jnp.concatenate([q1, v1, jnp.array([x[2 * m] + DT, 0.0])]), u, u0

# ---------------------------------------------------------------- data shared by every filter
U0 = jax.jit(jax.vmap(lambda x, Y, par: nominal(x, Y, par)[0]))(X, Y, PAR)
Z = X[:, :2 * m]
G, H = jax.jit(jax.vmap(rows))(Z)
h2_j = jax.jit(jax.vmap(lambda x: cfg.h_2(x[:2 * m])))
H2 = np.asarray(h2_j(X)); h2min0 = H2.min(1)
sel = (kind == 0) & (h2min0 > 0.02)                   # benign, initially safe states for the rollout metric
nsel = int(sel.sum()); print("rollout metric on", nsel, "initially safe random states", flush=True)
Xr, Yr, Pr = X[sel], Y[sel], PAR[sel]
Xr = Xr.at[:, 2 * m].set(0.0)
# per-row gradients w.r.t. q for the cancellation diagnostic (independent of kappa)
JH2 = np.asarray(jax.jit(jax.vmap(jax.jacobian(lambda q: cfg.h_2(jnp.concatenate([q, jnp.zeros(m)])))))(X[:, :m]))   # (n, 64, 7)
GN = np.linalg.norm(JH2, axis=2)                                                                                        # (n, 64)
Gn, Hn, U0n = np.asarray(G), np.asarray(H), np.asarray(U0)

V_DIVERGED = 50.0   # rad/s: far beyond the Panda's limits (2.6 rad/s) and anything the baselines reach (~10 rad/s); past it the Euler plant explodes

GROUP_OF_ROW = np.array(sum([[g] * k for g, k in groups.items()], []))
A_SMALL = 0.05      # |a| below this: the composite row can hardly move a u inside the box (median |a| on the random states is ~1)
KICK = 100.0        # N m: |u - u_nom| above this is a bang on (nearly) every joint

def rollout_metrics(filt, kappa=None, cap=None):
    """20-step rollouts of the 344 safe states under `filt`. A rollout whose state turns non-finite or whose joint speed
    exceeds V_DIVERGED is 'diverged': it counts as unsafe in the fractions and is left out of the h2 minimum / p05.
    For the composite filters, per-step diagnostics: which barrier carries the largest softmin weight, |a| < A_SMALL,
    the row infeasible, |u - u_nom| > KICK, and for the failing rollouts the group of the first barrier to go below -0.01."""
    st = jax.jit(jax.vmap(lambda x, Y, par: step(filt, x, Y, par)))
    diag = None
    if kappa is not None:
        def _diag(x, Y, par):
            z = x[:2 * m]; a, b, _, _ = comp_row(z, kappa); u, lam, inf = proj_row_box(nominal(x, Y, par)[0], a, b, cap)
            return jnp.argmin(S * cfg.h_2(z)), jnp.linalg.norm(a), inf, lam
        diag = jax.jit(jax.vmap(_diag))
    x = Xr; hmin = np.full(nsel, np.inf); diverged = np.zeros(nsel, bool); dev = []; allsat = []; infeas = []; top = []; asmall = []; kick = []; lamcap = []
    first_group = np.full(nsel, "", dtype=object)
    for s in range(T):
        if diag is not None:
            tr, an, inf, lam = [np.asarray(t) for t in diag(x, Yr, Pr)]
            infeas.append(inf & ~diverged); top.append(GROUP_OF_ROW[tr][~diverged]); asmall.append((an < A_SMALL) & ~diverged)
            if cap is not None: lamcap.append((lam >= cap) & ~diverged)
        x, u, u0 = st(x, Yr, Pr); xn = np.asarray(x); un = np.asarray(u)
        ok = np.isfinite(xn).all(1) & (np.abs(xn[:, m:2 * m]).max(1) <= V_DIVERGED)
        h2 = np.asarray(h2_j(x)); h2 = np.where(np.isfinite(h2), h2, np.inf); h2m = h2.min(1)
        newfail = (h2m < -0.01) & (first_group == "") & ~diverged
        first_group[newfail] = GROUP_OF_ROW[h2.argmin(1)[newfail]]
        diverged |= ~ok
        hmin = np.where(diverged, hmin, np.minimum(hmin, h2m))
        d = np.linalg.norm(un - np.asarray(u0), axis=1)
        dev.append(np.where(diverged, np.nan, d)); kick.append((d > KICK) & ~diverged)
        allsat.append(((np.abs(un) >= np.asarray(u_max) - 1e-9).all(1)) & ~diverged)
    keep = ~diverged
    r = {"rollout_hmin_frac_below_-0.01": float(((hmin < -0.01) | diverged).mean()), "rollout_hmin_frac_below_0": float(((hmin < 0) | diverged).mean()),
         "rollout_hmin_min": float(hmin[keep].min()), "rollout_hmin_p05": float(np.percentile(hmin[keep], 5)), "rollout_diverged_frac": float(diverged.mean()),
         "dev_mean_rollout": float(np.nanmean(dev)), "rollout_allsat_step_frac": float(np.mean(allsat)), "rollout_kick_step_frac": float(np.mean(kick))}
    if diag is not None:
        r["rollout_infeasible_step_frac"] = float(np.mean(infeas)); r["rollout_a_small_step_frac"] = float(np.mean(asmall))
        tops = np.concatenate(top); r["rollout_top_weight_frac"] = {g: float((tops == g).mean()) for g in groups}
        if lamcap: r["rollout_lam_capped_step_frac"] = float(np.mean(lamcap))
    fails = (hmin < -0.01) & keep
    r["rollout_first_violated_group"] = {g: int((first_group[fails] == g).sum()) for g in groups}
    return r

res = {}
for name, filt in FILTERS.items():
    U = np.asarray(jax.jit(jax.vmap(lambda u, z, G, h: filt(u, z, G, h)))(U0, Z, G, H))
    viol = (np.einsum("bij,bj->bi", Gn, U) - Hn) / np.maximum(1.0, np.abs(Hn))
    r = {f"kind{k}_inst_frac": float((viol[kind == k] > 1e-6).any(1).mean()) for k in (0, 1, 2)}
    r.update({f"kind{k}_mean_pos": float(np.maximum(viol[kind == k], 0).mean()) for k in (0, 1, 2)})
    d = np.linalg.norm(U - U0n, axis=1)
    r["dev_mean_random"] = float(d.mean()); r.update({f"dev_mean_kind{k}": float(d[kind == k].mean()) for k in (0, 1, 2)})
    kappa, cap = SMOOTH.get(name, (None, None))
    r.update(rollout_metrics(filt, kappa, cap))
    if kappa is not None:
        r["kappa"] = kappa; r["lam_max"] = cap if cap is not None else 0.0
        a, b, h0, H0 = [np.asarray(t) for t in jax.jit(jax.vmap(lambda z: comp_row(z, kappa)))(Z)]
        _, lam, inf = [np.asarray(t) for t in jax.jit(jax.vmap(lambda u, a, b: proj_row_box(u, a, b, cap)))(U0, a, b)]
        anorm = np.linalg.norm(a, axis=1); viol_row = (viol > 1e-6).any(1)
        w = np.exp(-kappa * np.asarray(S) * H2 - (-kappa * np.asarray(S) * H2).max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        rho = np.linalg.norm(np.einsum("bi,bij->bj", w * np.asarray(S), JH2), axis=1) / np.maximum(1e-300, (w * np.asarray(S) * GN).sum(1))
        r.update({"H0_neg_frac_safe": float((H0[sel] < 0).mean()), "h0_neg_frac_safe": float((h0[sel] < 0).mean()),
                  "H0_or_h0_neg_frac_safe": float(((H0[sel] < 0) | (h0[sel] < 0)).mean()),
                  "H0_or_h0_neg_frac_all": float(((H0 < 0) | (h0 < 0)).mean()),
                  "infeasible_frac_all": float(inf.mean()), "infeasible_frac_safe": float(inf[sel].mean()),
                  "lam_active_frac_all": float((lam > 0).mean()), "lam_active_frac_safe": float((lam[sel] > 0).mean()),
                  "a_norm_median": float(np.median(anorm)), "a_norm_min": float(anorm.min()),
                  "b_median_all": float(np.median(b)), "b_p05_all": float(np.percentile(b, 5)), "b_min_all": float(b.min()),
                  "b_median_safe": float(np.median(b[sel])), "b_p05_safe": float(np.percentile(b[sel], 5)), "b_min_safe": float(b[sel].min()),
                  "b_below_min_original_row_frac_all": float((b < Hn.min(1)).mean()),
                  "cancel_a_small_and_row_violated_frac": float(((anorm < 1e-3 * np.median(anorm)) & viol_row).mean()),
                  "cancel_rho_lt_0.1_and_row_violated_frac": float(((rho < 0.1) & viol_row).mean()),
                  "rho_min": float(rho.min()), "rho_min_row_violated": float(rho[viol_row].min()) if viol_row.any() else 1.0,
                  "rho_lt_0.1_frac_all": float((rho < 0.1).mean())})
    res[name] = r; print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

# the baselines must reproduce the committed study
ref_path = SAFE / "results" / "projection_study.json"
check = {}
if ref_path.exists():
    ref = json.loads(ref_path.read_text())
    for name in ("none (box only)", "greedy 48 (the kernel)", "elastic QP (cbfpy / qpax)"):
        check[name] = max(abs(res[name][k] - ref[name][k]) for k in ref[name])
    print("max |difference| to the committed panda_safe_mppi study per baseline:", check, flush=True)

qp_dev = res["elastic QP (cbfpy / qpax)"]["dev_mean_random"]
passing = [name for name in SMOOTH if (r := res[name])["rollout_hmin_frac_below_-0.01"] <= SUCCESS["frac_below_-0.01"]
           and r["rollout_hmin_min"] >= SUCCESS["worst_h2"] and r["dev_mean_random"] <= SUCCESS["dev_ratio_vs_qp"] * qp_dev]
res["meta"] = {"n_states": n, "n_safe": nsel, "T": T, "dt": DT, "alpha": ALPHA, "s_sing": S_SING, "s_other": 1.0, "kappas": KAPPAS,
               "bisect_iters": BISECT_ITERS, "lam_cap": LAM_CAP, "success_criterion": SUCCESS, "passing": passing,
               "baseline_check_vs_committed": check, "groups": groups}
print("success criterion met by:", passing if passing else "none", flush=True)
(HERE / "results" / "projection_study_smoothmin.json").write_text(json.dumps(res, indent=2))
print("wrote", HERE / "results" / "projection_study_smoothmin.json")
