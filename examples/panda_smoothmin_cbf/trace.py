"""Step-by-step trace of the rollouts that fail under the smooth-min filter (results/trace.txt): the raw barrier minimum and its
group, the composite H, its HOCBF value h, the row bound b, |a|, the multiplier, infeasibility, the residual, saturated joints,
the next joint speed and the filter's move. Run after projection_study.py: `python trace.py [--kappa 400] [--cap 1000]`."""
import argparse, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--kappa", type=float, nargs="+", default=[400.0, 50.0]); ap.add_argument("--cap", type=float, default=1000.0)
ap.add_argument("--max-rollouts", type=int, default=3); args = ap.parse_args()
src = (HERE / "projection_study.py").read_text()
src = src[:src.index("GROUP_OF_ROW")]                     # the model, the data, the filters; not the sweep
__file__ = str(HERE / "projection_study.py")
exec(compile(src, __file__, "exec"))
import numpy as np, jax, jax.numpy as jnp  # noqa: E402
names = np.array(sum([[g] * k for g, k in groups.items()], [])); um = np.asarray(u_max)
out = []
for kappa in args.kappa:
    cap = args.cap if args.cap > 0 else None
    filt = smoothmin_filter(kappa, cap)
    st = jax.jit(jax.vmap(lambda x, Y, par: step(filt, x, Y, par)))
    rowj = jax.jit(jax.vmap(lambda z: comp_row(z, kappa)))
    pj = jax.jit(jax.vmap(lambda u, a, b: proj_row_box(u, a, b, cap)))
    nomj = jax.jit(jax.vmap(lambda x, Y, par: nominal(x, Y, par)[0]))
    x = Xr; hist = []
    for s in range(T):
        a, b, h, Hc = [np.asarray(t) for t in rowj(x[:, :2 * m])]
        u0 = np.asarray(nomj(x, Yr, Pr)); u, lam, inf = [np.asarray(t) for t in pj(jnp.asarray(u0), jnp.asarray(a), jnp.asarray(b))]
        h2 = np.asarray(h2_j(x)); x1, uu, _ = st(x, Yr, Pr); x1 = np.asarray(x1)
        hist.append(dict(h2min=h2.min(1), row=h2.argmin(1), H=Hc, h=h, b=b, an=np.linalg.norm(a, axis=1), lam=lam, inf=inf, res=np.einsum("bj,bj->b", a, u) - b,
                         nsat=(np.abs(u) >= um - 1e-9).sum(1), vmax=np.abs(x1[:, m:2 * m]).max(1), dev=np.linalg.norm(u - u0, axis=1)))
        x = jnp.asarray(x1)
    hmin = np.min([hh["h2min"] for hh in hist], axis=0); bad = np.where(~np.isfinite(hmin) | (hmin < -0.01))[0]
    out.append(f"kappa {kappa:g}, lam cap {cap}: {len(bad)} of {nsel} rollouts below -0.01 or diverged: {bad.tolist()}")
    for i in bad[:args.max_rollouts]:
        out.append(f"rollout {i}: step | h2 min (group) | H | h | b | |a| | lam | infeasible | a.u-b | saturated joints | next max|v| | |u-u_nom|")
        for s, hh in enumerate(hist):
            out.append(f"  {s:2d} | {hh['h2min'][i]:9.4f} {names[hh['row'][i]][:6]:6s} | {hh['H'][i]:9.4f} | {hh['h'][i]:9.3f} | {hh['b'][i]:10.2f} | {hh['an'][i]:6.3f} | "
                       f"{hh['lam'][i]:8.1f} | {int(hh['inf'][i])} | {hh['res'][i]:9.2e} | {hh['nsat'][i]} | {hh['vmax'][i]:9.2f} | {hh['dev'][i]:7.1f}")
    out.append("")
(HERE / "results" / "trace.txt").write_text("\n".join(out) + "\n"); print("\n".join(out))
