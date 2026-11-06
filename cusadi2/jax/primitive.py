"""The `cusadi2_eval` JAX primitive: one installed plan evaluated on a batch.

Inputs: one array of shape (batch, nnz_in[a]) per plan input, all of the plan's
dtype (float64 or float32; each dtype is its own extension module, `_ffi_f64`
/ `_ffi_f32`, with its own plan registry). Outputs: one array of shape
(batch, nnz_out[k]) per plan output in the same dtype. `plan_id`, `batch`,
`dtype`, `nnz_out` and `scratch` are static; a plan with reduces (a composed
plan, cusadi2.compose) needs `scratch` scalars of scratch stream per instance,
which the call passes as one extra XLA-owned result buffer and drops.

Transformations: jit lowers to the XLA custom call (the spineax pattern);
vmap flattens the mapped axis into the plan's batch axis, broadcasting
arguments that are not mapped. There is no JVP rule: derivatives are
separate plans built symbolically (cusadi2.jax.function).
"""
from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive
from jax.interpreters import batching, mlir

# JAX must own the CUDA context before the extensions (which link cudart) are
# imported, otherwise the GPU plugin initialises against a half-configured
# context; touching the devices first orders it correctly (spineax does the same).
jax.devices()

EXT: dict = {}        # dtype -> extension module
TARGET: dict = {}     # dtype -> FFI target name
for _dtype, _mod in ((np.dtype("float64"), "_ffi_f64"), (np.dtype("float32"), "_ffi_f32")):
    try:
        _ext = importlib.import_module(f"cusadi2.jax.{_mod}")
    except ImportError:
        continue
    _name = f"cusadi2_eval_{_mod[-3:]}"
    # The handler's state type must be registered before the handler itself.
    if hasattr(jax.ffi, "register_ffi_type"):          # jax >= 0.9
        jax.ffi.register_ffi_type(_name + "_state", {"type_id": _ext.type_id(), "type_info": _ext.type_info()},
                                  platform="CUDA")
    else:                                              # jax 0.8
        jax.ffi.register_ffi_type_id(_name + "_state", _ext.type_id(), platform="CUDA")
    jax.ffi.register_ffi_target(_name, _ext.handler(), platform="CUDA")
    EXT[_dtype] = _ext
    TARGET[_dtype] = _name
if not EXT:
    raise ImportError("cusadi2.jax: no FFI extension built (_ffi_f64 / _ffi_f32)")

MAX_ARGS = next(iter(EXT.values())).MAX_ARGS


def ext_for(dtype) -> object:
    dtype = np.dtype(dtype)
    if dtype not in EXT:
        raise TypeError(f"cusadi2.jax: no kernel built for {dtype}; available: {sorted(map(str, EXT))}")
    return EXT[dtype]


eval_p = Primitive("cusadi2_eval")
eval_p.multiple_results = True


def cusadi2_eval(args, *, plan_id: int, batch: int, dtype, nnz_out: tuple, scratch: int = 0) -> list:
    return eval_p.bind(*args, plan_id=int(plan_id), batch=int(batch), dtype=np.dtype(dtype),
                       nnz_out=tuple(int(n) for n in nnz_out), scratch=int(scratch))


@eval_p.def_impl
def _impl(*args, plan_id, batch, dtype, nnz_out, scratch):
    shapes = [jax.ShapeDtypeStruct((batch, n), dtype) for n in nnz_out]
    if scratch:
        shapes.append(jax.ShapeDtypeStruct((batch, scratch), dtype))
    call = jax.ffi.ffi_call(TARGET[dtype], shapes)
    return list(call(*args, plan_id=plan_id, batch=batch))[:len(nnz_out)]


@eval_p.def_abstract_eval
def _abstract(*args, plan_id, batch, dtype, nnz_out, scratch):
    for a in args:
        if a.dtype != dtype:
            raise TypeError(f"cusadi2_eval: plan dtype is {dtype}, got an input of {a.dtype}")
        if a.ndim != 2 or a.shape[0] != batch:
            raise ValueError(f"cusadi2_eval: input shape {a.shape} is not (batch={batch}, nnz)")
    return [jax.core.ShapedArray((batch, n), dtype) for n in nnz_out]


mlir.register_lowering(eval_p, mlir.lower_fun(_impl, multiple_results=True))


def _batch(args, dims, *, plan_id, batch, dtype, nnz_out, scratch):
    """vmap: fold the mapped axis into the plan's batch axis."""
    size = next(a.shape[d] for a, d in zip(args, dims) if d is not None)
    flat = []
    for a, d in zip(args, dims):
        if d is None:
            a = jnp.broadcast_to(a, (size,) + a.shape)
        else:
            a = jnp.moveaxis(a, d, 0)
        flat.append(a.reshape(size * batch, a.shape[-1]))
    outs = cusadi2_eval(flat, plan_id=plan_id, batch=size * batch, dtype=dtype, nnz_out=nnz_out, scratch=scratch)
    return [o.reshape(size, batch, o.shape[-1]) for o in outs], [0] * len(outs)


batching.primitive_batchers[eval_p] = _batch
