"""The quadcopter navigation NMPC of `quadcopter_nav` written stage-wise, for
`cusadi2.jax.ipm.StagedNLP`: n_stages = N - 1 intervals, stage arguments
x_k / x_{k+1} (one knot buffer read twice), u_k, the initial state p (shared
parameter) and a per-stage parameter s_k = [obstacle multiplier, first-stage
flag]. The knot-major global x is [X (N x 13), U ((N-1) x 4)], the layout of
`quadcopter_nav`'s z, and the rows are the same constraints in stage-major order
(the initial-state rows exist at stage 0 only, via `rows_active`)."""
import jax.numpy as jnp
import numpy as np

from cusadi2.jax.ipm import rows_active
from tests.problems.quadcopter_nav import NU, NX, OBSTACLES, Q, f_jnp, quad_params, quadcopter_nav


def quadcopter_nav_staged(N=30, Ts=0.1):
    """-> stage_f, stage_c, stage_d, n_stages, specs, c_active, d_active, x0, p0 (on the StagedNLP layout)."""
    n_stages = N - 1
    specs = [dict(name="x_k", size=NX, mode="coupled", offset=0, span=1),
             dict(name="x_k1", size=NX, mode="coupled", offset=1, span=1, alias="x_k"),
             dict(name="u_k", size=NU, mode="stage"),
             dict(name="p", size=NX, mode="shared", param=True),
             dict(name="s_k", size=2, mode="stage", param=True)]
    r__ = [r + OBSTACLES["quad_radius"] for r in OBSTACLES["r_"]]

    def stage_f(x_k, x_k1, u_k, p, s_k):
        return jnp.sum(Q * x_k1 ** 2) + s_k[1] * jnp.sum(Q * x_k ** 2)        # the knot-0 term once, at stage 0

    def stage_c(x_k, x_k1, u_k, p, s_k):
        return jnp.concatenate([x_k1 - (x_k + f_jnp(x_k, u_k, quad_params) * Ts), x_k - p])

    def stage_d(x_k, x_k1, u_k, p, s_k):
        return jnp.stack([(x_k[0] - xc) ** 2 + (x_k[1] - yc) ** 2 - r ** 2 * s_k[0]
                          for xc, yc, r in zip(OBSTACLES["xc_"], OBSTACLES["yc_"], r__)])

    c_active = rows_active(n_stages, [(NX, "all"), (NX, "first")])
    d_active = rows_active(n_stages, [(4, "all")])
    _, _, _, z0, p_init, _ = quadcopter_nav(N=N, Ts=Ts)
    s = np.stack([1 + np.arange(n_stages) * Ts * 0.1, (np.arange(n_stages) == 0).astype(float)], axis=1)
    return stage_f, stage_c, stage_d, n_stages, specs, c_active, d_active, np.asarray(z0), np.asarray(p_init), s
