"""cusadi2.jax: JAX front end for cusadi2.

``trace_to_sx`` turns a JAX function into CasADi SX (one scalar per array
element) so Jacobians and Hessians can be formed symbolically with exact
sparsity before plan extraction. ``jax_function`` (requires the CUDA FFI
extension, built with the ``[jax]`` extra) wraps the resulting plans as JAX
primitives that work under ``jit`` and ``vmap``. ``rbd`` gives rigid-body
dynamics (RNEA, ABA, kinematics) in JAX from a MuJoCo-parsed MJCF / URDF.
``to_mx`` translates a JAX function into a CasADi MX graph that keeps its
``lax.map`` / ``vmap`` as ``F.map(n)`` calls, the input of ``cusadi2.staging``.
"""
from cusadi2.jax import rbd
from cusadi2.jax.interpret import Unsupported, eval_jaxpr, trace_to_sx
from cusadi2.jax.to_mx import MXGraph, to_mx, trace, vmap_as_map

__all__ = ["Unsupported", "eval_jaxpr", "trace_to_sx", "rbd", "MXGraph", "to_mx", "vmap_as_map"]

try:  # the FFI extension is optional (``pip install cusadi2[jax]``)
    from cusadi2.jax.function import GpuFunction, gpu_functions, jax_function  # noqa: F401
    __all__ += ["GpuFunction", "gpu_functions", "jax_function"]
except ImportError:  # pragma: no cover
    pass
