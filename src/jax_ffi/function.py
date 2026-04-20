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

from src.level_parallel.codegen import pick_block_size
from src.level_parallel.dag import LevelPlan, extract_level_plan

try:
    import _cusadi_jax_ffi as _ext  # type: ignore
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
    """Mirror CusadiLevelParallelFunction._upload_plan but as host numpy arrays."""
    if plan.constants:
        const_slots = np.asarray([c.ssa_slot for c in plan.constants], dtype=np.int32)
        const_vals = np.asarray([c.value for c in plan.constants], dtype=np.float64)
    else:
        const_slots = np.zeros(0, dtype=np.int32)
        const_vals = np.zeros(0, dtype=np.float64)

    if plan.input_mappings:
        input_slots = np.asarray([m.ssa_slot for m in plan.input_mappings], dtype=np.int32)
        input_args = np.asarray([m.input_arg for m in plan.input_mappings], dtype=np.int32)
        input_elems = np.asarray([m.element_idx for m in plan.input_mappings], dtype=np.int32)
    else:
        input_slots = np.zeros(0, dtype=np.int32)
        input_args = np.zeros(0, dtype=np.int32)
        input_elems = np.zeros(0, dtype=np.int32)

    nnz_in = np.asarray(plan.nnz_in, dtype=np.int32)

    all_ops: list[int] = []
    all_outs: list[int] = []
    all_in_a: list[int] = []
    all_in_b: list[int] = []
    level_offsets: list[int] = []
    level_widths: list[int] = []
    offset = 0
    for level in plan.compute_levels:
        level_offsets.append(offset)
        level_widths.append(level.width)
        for node in level.nodes:
            all_ops.append(node.op_id)
            all_outs.append(node.ssa_out)
            all_in_a.append(node.ssa_inputs[0] if len(node.ssa_inputs) >= 1 else 0)
            all_in_b.append(node.ssa_inputs[1] if len(node.ssa_inputs) >= 2 else 0)
        offset += level.width

    if plan.output_mappings:
        out_slots = np.asarray([m.ssa_slot for m in plan.output_mappings], dtype=np.int32)
        out_args = np.asarray([m.output_arg for m in plan.output_mappings], dtype=np.int32)
        out_elems = np.asarray([m.element_idx for m in plan.output_mappings], dtype=np.int32)
    else:
        out_slots = np.zeros(0, dtype=np.int32)
        out_args = np.zeros(0, dtype=np.int32)
        out_elems = np.zeros(0, dtype=np.int32)

    nnz_out = np.asarray(plan.nnz_out, dtype=np.int32)

    return dict(
        const_slots=const_slots,
        const_vals=const_vals,
        input_slots=input_slots,
        input_args=input_args,
        input_elems=input_elems,
        nnz_in=nnz_in,
        level_offsets=np.asarray(level_offsets, dtype=np.int32),
        level_widths=np.asarray(level_widths, dtype=np.int32),
        node_ops=np.asarray(all_ops, dtype=np.int32),
        node_outs=np.asarray(all_outs, dtype=np.int32),
        node_in_a=np.asarray(all_in_a, dtype=np.int32),
        node_in_b=np.asarray(all_in_b, dtype=np.int32),
        out_slots=out_slots,
        out_args=out_args,
        out_elems=out_elems,
        nnz_out=nnz_out,
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
