"""JAX-FFI backend for cusadi level-parallel kernels.

A CusadiJaxFunction instance wraps a CasADi Function, extracts a LevelPlan,
uploads the plan to the GPU via the compiled _cusadi_jax_ffi extension, and
exposes a JAX-traceable __call__ that dispatches via jax.ffi.ffi_call on the
XLA CUDA stream (zero-copy).

Design notes:
- plan_id is an FFI Attr<int64_t>, i.e. a compile-time constant baked into
  the jit'd program. It is batch-invariant under vmap: all B elements in a
  vmapped batch share the same plan.
- Vmap mapped axis becomes position 0 of the input buffer (vmap_method
  "broadcast_all") and of the output buffer. The kernel launches grid=(B,)
  so the vmap dimension becomes the kernel's native batch dimension.
- Current arity is 1 input / 1 output. Multi-input (e.g. passing parameters
  alongside x) needs a RemainingArgs handler — deferred.
"""

from __future__ import annotations

import threading

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

from cusadi.level_parallel.codegen import pick_block_size
from cusadi.level_parallel.dag import LevelPlan, extract_level_plan
from cusadi.level_parallel.serialize import flatten_plan

# Force JAX to initialize its CUDA context/GPU plugin BEFORE importing the
# nanobind extension below (which links cudart and touches CUDA on import). If
# the extension initializes CUDA first, JAX's plugin comes up afterwards with a
# half-configured context and cuDNN init fails (RET_CHECK dnn_support != nullptr
# at first GPU op). Touching jax.devices() here orders it correctly.
jax.devices()

try:
    # Installed in-package by scikit-build-core (cusadi/jax_ffi/_cusadi_jax_ffi*.so).
    from . import _cusadi_jax_ffi as _ext  # type: ignore
except ImportError:
    _ext = None  # we will raise a clearer error on first use


_REGISTERED = False
_REGISTER_LOCK = threading.Lock()


def _ensure_ffi_target_registered() -> None:
    """Register the C++ handler with jax.ffi on first use."""
    global _REGISTERED
    if _REGISTERED:
        return
    with _REGISTER_LOCK:
        if _REGISTERED:
            return
        if _ext is None:
            raise RuntimeError(
                "cusadi._cusadi_jax_ffi extension is not importable. "
                "Build it via the CMake target on branch feature/jax-ffi-backend."
            )
        jax.ffi.register_ffi_target(
            "cusadi_generic_f64", _ext.handler(), platform="CUDA"
        )
        _REGISTERED = True


def _flatten_plan_for_upload(plan: LevelPlan) -> dict[str, np.ndarray]:
    """Flatten via the shared serialize.flatten_plan into host numpy arrays.

    Same source of truth as the torch/native backends — the only difference
    here is dtype (int32 / float64) and that empty maps stay length-0 (the JAX
    extension handles zero-length plan arrays directly, unlike the torch path's
    [0] sentinel)."""
    flat = flatten_plan(plan)
    i32 = lambda xs: np.asarray(xs, dtype=np.int32)
    f64 = lambda xs: np.asarray(xs, dtype=np.float64)
    return dict(
        const_slots=i32(flat.const_slots),
        const_vals=f64(flat.const_vals),
        input_slots=i32(flat.input_slots),
        input_args=i32(flat.input_args),
        input_elems=i32(flat.input_elems),
        nnz_in=i32(flat.nnz_in),
        level_offsets=i32(flat.level_offsets),
        level_widths=i32(flat.level_widths),
        node_ops=i32(flat.node_ops),
        node_outs=i32(flat.node_outs),
        node_in_a=i32(flat.node_in_a),
        node_in_b=i32(flat.node_in_b),
        out_slots=i32(flat.out_slots),
        out_args=i32(flat.out_args),
        out_elems=i32(flat.out_elems),
        nnz_out=i32(flat.nnz_out),
    )


class CusadiJaxFunction:
    """JAX-callable wrapper around a cusadi level-parallel CasADi kernel.

    plan_id is an `Attr<int64_t>` in the XLA FFI binding — i.e. a compile-time
    constant. This means every element of a vmapped batch shares the same plan;
    per-batch-element plan variation would require a different design.

    Current arity support: 1 input / 1 output. The NMPC f/c/d/jac/hess functions
    all take z as a single flat argument and emit a flat nnz vector, so the fixed
    arity is sufficient.
    """

    _ARITY_LIMIT = 1

    def __init__(self, fn_casadi: ca.Function, plan_id: int):
        _ensure_ffi_target_registered()

        if fn_casadi.n_in() != self._ARITY_LIMIT or fn_casadi.n_out() != self._ARITY_LIMIT:
            raise ValueError(
                f"CusadiJaxFunction currently supports n_in=n_out=1 only; got "
                f"n_in={fn_casadi.n_in()}, n_out={fn_casadi.n_out()} for {fn_casadi.name()!r}."
            )

        # extract_level_plan needs an SX-backed Function; expand MX if needed.
        if fn_casadi.is_a("MXFunction"):
            fn_casadi = fn_casadi.expand()
        self.fn_casadi = fn_casadi
        self.plan_id = int(plan_id)

        plan = extract_level_plan(fn_casadi)
        self._plan = plan
        self._block_size = int(pick_block_size(plan))
        self._shared_bytes = int(plan.peak_live_slots * 8)

        flat = _flatten_plan_for_upload(plan)

        n_in = fn_casadi.n_in()
        n_out = fn_casadi.n_out()
        assert _ext is not None
        _ext.install_plan(
            self.plan_id,
            n_in, n_out,
            self._block_size, self._shared_bytes,
            flat["const_slots"], flat["const_vals"],
            flat["input_slots"], flat["input_args"], flat["input_elems"], flat["nnz_in"],
            flat["level_offsets"], flat["level_widths"],
            flat["node_ops"], flat["node_outs"], flat["node_in_a"], flat["node_in_b"],
            flat["out_slots"], flat["out_args"], flat["out_elems"], flat["nnz_out"],
        )

        # Cache sparsity triplets for COO accessors.
        self._nnz_in = [int(fn_casadi.nnz_in(i)) for i in range(n_in)]
        self._nnz_out = [int(fn_casadi.nnz_out(i)) for i in range(n_out)]
        self._coo_rows = []
        self._coo_cols = []
        for i in range(n_out):
            rows, cols = fn_casadi.sparsity_out(i).get_triplet()
            self._coo_rows.append(jnp.asarray(np.asarray(rows, dtype=np.int32)))
            self._coo_cols.append(jnp.asarray(np.asarray(cols, dtype=np.int32)))

    @property
    def nnz_in(self) -> list[int]:
        return list(self._nnz_in)

    @property
    def nnz_out(self) -> list[int]:
        return list(self._nnz_out)

    def get_coo(self, out_idx: int = 0) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._coo_rows[out_idx], self._coo_cols[out_idx]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate on a single (or vmap-unrolled) input of shape (nnz_in[0],).

        Under jax.vmap with vmap_method="broadcast_all", the mapped axis becomes
        position 0 of the buffer seen by the FFI handler, and the kernel launches
        grid=(B,). Output shape under vmap is (B, nnz_out[0]); outside vmap it is
        (nnz_out[0],).
        """
        assert _ext is not None  # ensured by __init__
        out_shape = (self._nnz_out[0],)
        out_spec = jax.ShapeDtypeStruct(out_shape, jnp.float64)
        call = jax.ffi.ffi_call(
            "cusadi_generic_f64",
            out_spec,
            vmap_method="broadcast_all",
        )
        # plan_id is an FFI Attr: must be hashable Python scalar, not jnp.int64.
        return call(x, plan_id=self.plan_id)

    def close(self) -> None:
        """Explicit cleanup (optional). Not called from __del__ because
        re-using a plan_id (e.g. reconstructing the same problem in a loop)
        would otherwise race with GC of the prior instance and silently break
        the newer one."""
        if _ext is not None:
            _ext.free_plan(self.plan_id)
