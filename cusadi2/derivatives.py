"""Exact-sparsity derivatives of SX expressions, the way cusadi2's plans want them.

    from cusadi2 import gradient, jacobian, hessian, weighted_hessian
    g = gradient(f, x)                 # sparse column (nx, 1): the structural gradient pattern
    J = jacobian(y, x)                 # (ny, nx), structural nonzeros
    H = hessian(f, x)                  # (nx, nx)
    W = weighted_hessian(y, x, lam)    # Hessian of lam . y (a Lagrangian Hessian)

`cse=True` (the default) applies common subexpression elimination (`casadi.cse`) to the
expression BEFORE differentiating. CasADi does not merge structurally identical subgraphs built
separately (a helper traced twice on the same input, e.g. forward kinematics under both the foot
positions and the foot Jacobians), and differentiating the duplicated graph duplicates its
derivative too; CSE after differentiation (`cse_function`, which `compile` also applies by
default) merges only part of that. Go2 hopscotch window, Lagrangian Hessian of the constraints:
106k nodes / 241 levels without CSE, 99k / 212 with CSE after differentiation only, 90k / 170
with CSE before and after; 2-6 % faster cuipm solves at B = 1-32 (before and after) against
about 1 % (after only). `cse=False` differentiates the expression as traced.
"""
from __future__ import annotations

import casadi as ca


def _prep(y: ca.SX, cse: bool) -> ca.SX:
    return ca.cse(y) if cse and y.nnz() else y


def gradient(f: ca.SX, x: ca.SX, *, cse: bool = True) -> ca.SX:
    """Gradient of scalar `f` as a sparse (nx, 1) column (ca.gradient would densify it)."""
    return ca.sparsify(ca.jacobian(_prep(f, cse), x).T)


def jacobian(y: ca.SX, x: ca.SX, *, cse: bool = True) -> ca.SX:
    """d y / d x, (ny, nx), structural nonzeros only."""
    return ca.sparsify(ca.jacobian(_prep(y, cse), x))


def hessian(f: ca.SX, x: ca.SX, *, cse: bool = True) -> ca.SX:
    """Hessian of scalar `f`, (nx, nx)."""
    return ca.sparsify(ca.hessian(_prep(f, cse), x)[0])


def weighted_hessian(y: ca.SX, x: ca.SX, lam: ca.SX, *, cse: bool = True) -> ca.SX:
    """Hessian of lam . y w.r.t. x, (nx, nx): the multiplier-weighted (Lagrangian) Hessian."""
    return ca.sparsify(ca.hessian(ca.dot(lam, _prep(y, cse)), x)[0])


def vector_hessian(y: ca.SX, x: ca.SX, *, cse: bool = True) -> ca.SX:
    """True Hessian of every entry of a vector y, as (ny * nx, nx): row k + ny * i is d^2 y_k / dx_i dx."""
    return ca.sparsify(ca.jacobian(ca.jacobian(_prep(y, cse), x), x))
