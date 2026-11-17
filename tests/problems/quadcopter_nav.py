"""Quadcopter navigation NMPC as plain JAX code, copied from jaxipm
(src/problems/mpc.py::quadcopter_nav, src/problems/quadcopter/dynamics.py::f_jnp)
so the cusadi2 tests are self-contained. One change: the initial state is the
parameter `p` of c(z, p) instead of a closed-over constant, matching cuipm's
partition (codegen/generate.py): c = all equalities, d = one-sided inequalities
d(z, p) >= 0; variable bounds live outside the symbolic NLP.

Decision vector z = [X (N x 13) row-major, U ((N-1) x 4) row-major].
"""
import jax.numpy as jnp
import numpy as np

NX, NU = 13, 4

quad_params = {
    "mB": 1.2,
    "dxm": 0.16, "dym": 0.16, "dzm": 0.01,
    "IB": np.array([[0.0123, 0, 0], [0, 0.0123, 0], [0, 0, 0.0224]]),
    "IRzz": 2.7e-5,
    "Cd": 0.1,
    "kTh": 1.076e-5,
    "kTo": 1.632e-7,
    "minWmotor": 75, "maxWmotor": 925,
    "usePrecession": True,
}
quad_params["x_lb"] = np.array([*[-10] * 3, *[-np.inf] * 4, *[-10] * 3, *[-10] * 3])
quad_params["x_ub"] = np.array([*[10] * 3, *[np.inf] * 4, *[10] * 3, *[10] * 3])

OBSTACLES = dict(r_=[0.5, 0.5, 0.5, 0.75], xc_=[1., 1., 3., 3.], yc_=[1., 3., 1., 3.], quad_radius=0.2)
Q = jnp.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


def f_jnp(x, u, qp, d=(0., 0., 0.), g=9.81):
    """Rigid-body quadrotor dynamics (rotor dynamics removed), verbatim from jaxipm."""
    th = qp['kTh'] * u ** 2
    to = qp['kTo'] * u ** 2
    xd = jnp.stack([
        x[7], x[8], x[9],
        - 0.5 * x[10] * x[4] - 0.5 * x[11] * x[5] - 0.5 * x[6] * x[12],
        0.5 * x[10] * x[3] - 0.5 * x[11] * x[6] + 0.5 * x[5] * x[12],
        0.5 * x[10] * x[6] + 0.5 * x[11] * x[3] - 0.5 * x[4] * x[12],
        - 0.5 * x[10] * x[5] + 0.5 * x[11] * x[4] + 0.5 * x[3] * x[12],
        (qp["Cd"] * jnp.sign(d[0] * jnp.cos(d[1]) * jnp.cos(d[2]) - x[7])
         * (d[0] * jnp.cos(d[1]) * jnp.cos(d[2]) - x[7]) ** 2
         - 2 * (x[3] * x[5] + x[4] * x[6]) * (th[0] + th[1] + th[2] + th[3])) / qp["mB"],
        (qp["Cd"] * jnp.sign(d[0] * jnp.sin(d[1]) * jnp.cos(d[2]) - x[8])
         * (d[0] * jnp.sin(d[1]) * jnp.cos(d[2]) - x[8]) ** 2
         + 2 * (x[3] * x[4] - x[5] * x[6]) * (th[0] + th[1] + th[2] + th[3])) / qp["mB"],
        (-qp["Cd"] * jnp.sign(d[0] * jnp.sin(d[2]) + x[9]) * (d[0] * jnp.sin(d[2]) + x[9]) ** 2
         - (th[0] + th[1] + th[2] + th[3]) * (x[3] ** 2 - x[4] ** 2 - x[5] ** 2 + x[6] ** 2)
         + g * qp["mB"]) / qp["mB"],
        ((qp["IB"][1, 1] - qp["IB"][2, 2]) * x[11] * x[12]
         - qp["usePrecession"] * qp["IRzz"] * (u[0] - u[1] + u[2] - u[3]) * x[11]
         + (th[0] - th[1] - th[2] + th[3]) * qp["dym"]) / qp["IB"][0, 0],
        ((qp["IB"][2, 2] - qp["IB"][0, 0]) * x[10] * x[12]
         + qp["usePrecession"] * qp["IRzz"] * (u[0] - u[1] + u[2] - u[3]) * x[10]
         + (th[0] + th[1] - th[2] - th[3]) * qp["dxm"]) / qp["IB"][1, 1],
        ((qp["IB"][0, 0] - qp["IB"][1, 1]) * x[10] * x[11] - to[0] + to[1] - to[2] + to[3]) / qp["IB"][2, 2],
    ])
    return xd


def quadcopter_nav(N=30, Ts=0.1):
    """-> f(z), c(z, p), d(z), z_init, p_init, dims."""
    def xu_to_z(x, u):
        return jnp.hstack([x.flatten(), u.flatten()])

    def z_to_xu(z):
        return z[:N * NX].reshape(N, NX), z[N * NX:].reshape(N - 1, NU)

    xyz = jnp.linspace(jnp.array([4, 4, 0]), jnp.array([0, 0, -2]), N)
    state0 = jnp.hstack([xyz, jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * N)])
    input0 = jnp.array([[522.9847140714692] * 4] * (N - 1))
    z_init = xu_to_z(state0, input0)
    p_init = state0[0]

    def f(z):
        x, u = z_to_xu(z)
        return jnp.sum(Q * x ** 2)

    def c(z, p):
        x, u = z_to_xu(z)
        constraints = [x[0] - p]
        for k in range(N - 1):
            x_next = x[k] + f_jnp(x[k], u[k], quad_params) * Ts     # Euler
            constraints.append(x[k + 1] - x_next)
        return jnp.concatenate(constraints).flatten()

    r__ = [r + OBSTACLES["quad_radius"] for r in OBSTACLES["r_"]]

    def d(z):
        x, u = z_to_xu(z)
        cl = []
        for k in range(N - 1):
            multiplier = 1 + k * Ts * 0.1
            for (xc, yc, r) in zip(OBSTACLES["xc_"], OBSTACLES["yc_"], r__):
                cl.append((x[k, 0] - xc) ** 2 + (x[k, 1] - yc) ** 2 - r ** 2 * multiplier)
        return jnp.hstack(cl)

    dims = dict(nx=int(z_init.size), np=NX, nyc=N * NX, nyd=4 * (N - 1), N=N, Ts=Ts)
    return f, c, d, z_init, p_init, dims
