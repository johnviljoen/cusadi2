"""Parity test for cusadi JAX-FFI backend.

Validates the C++ install_plan + evaluate_lp_stream path against CasADi's own
host-side numeric evaluation (ground truth) at 1e-12 tolerance.

Covers code-time check #2 from plan (C++ upload port parity). Since torch is not
available in this env, we compare against CasADi directly instead of the
PyTorch wrapper — same correctness goal.
"""

import os
import sys

# Make src.* and the built extension importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "build_jax_ffi"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import casadi as ca  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from cusadi.jax_ffi.function import CusadiJaxFunction  # noqa: E402


def _make_test_function():
    """Small deterministic CasADi Function with nontrivial DAG (ops across levels)."""
    x = ca.MX.sym("x", 3)
    y = ca.vertcat(
        x[0] * x[1] + x[2],                       # mul + add
        ca.sin(x[1]) - x[0] * x[0],               # sin, sq, sub
        ca.exp(x[2]) + x[0] + x[1],               # exp + add chain
        (x[0] + x[1]) * (x[1] + x[2]),            # nested ops
    )
    return ca.Function("parity_test", [x], [y])


def test_parity():
    fn = _make_test_function()
    gpu = CusadiJaxFunction(fn, plan_id=1001)

    x_host = np.array([0.3, -0.7, 1.2], dtype=np.float64)
    # CasADi host-side ground truth
    gt = np.asarray(fn(x_host)).flatten()
    assert gt.shape == (4,), gt.shape

    # GPU evaluation (single instance, batch = 1 via 1-D input of shape (3,))
    x_jax = jnp.asarray(x_host)
    y = gpu(x_jax)
    y = jax.block_until_ready(y)
    y_host = np.asarray(y).flatten()
    assert y_host.shape == (4,), y_host.shape

    err = np.max(np.abs(y_host - gt))
    print(f"[parity] max abs error = {err:.3e}")
    assert err < 1e-12, f"parity failed: max err = {err}"
    print("[parity] PASS at 1e-12")


if __name__ == "__main__":
    test_parity()
