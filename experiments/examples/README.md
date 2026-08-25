# Examples

End-to-end walkthroughs of the cusadi interface, starting from a CasADi `Opti`
problem.

## `opti_to_cusadi` — Opti → cusadi

A double-integrator OCP is built with the Opti stack; its sparse constraint
Jacobian `dg/dz` is extracted as a fixed-structure `Function` and evaluated for
thousands of parameter instances at once on the GPU.

The pipeline is three steps:

| Step | What | `.py` (PyTorch) | `.cpp` | `_jax.py` (JAX) |
|------|------|:---:|:---:|:---:|
| 1 | **Formulate** the OCP with `Opti` | ✅ | ✅ | ✅ |
| 2 | **Extract** `dg/dz` as a `Function`, save `.casadi` | ✅ | ✅ | ✅ |
| 3 | **Batch** it on the GPU | ✅ `CusadiFunction2` | — (Python only) | ✅ `jax.vmap` + FFI |

Three implementations of the same example:

- **`opti_to_cusadi.py`** — PyTorch backend (`CusadiFunction2`). Hand-marshalled
  ctypes; multi-input; no build step.
- **`opti_to_cusadi.cpp`** — C++ does the modeling side (steps 1–2) and writes
  the `.casadi`; batching is invoked from Python (see below).
- **`opti_to_cusadi_jax.py`** — JAX backend (`CusadiJaxFunction`). Batching is
  `jax.vmap` + `jax.ffi`; XLA owns the buffer plumbing. Two consequences:
  the binding is **1-input/1-output**, so the example collapses `(z, p)` into
  one vector `w = [z; p]` via `ca.substitute`; and fp64 requires
  `jax_enable_x64`. Needs the `_cusadi_jax_ffi` extension built (below).

**Why no C++ for step 3.** The GPU batching depends on the level-plan
extraction in `cusadi/level_parallel/dag.py`, which is pure Python with no C++
counterpart. The LP kernel `.so` does expose an `extern "C" evaluate_lp(...)`
entry, but feeding it requires that plan. So the `.casadi` file is the
**language-agnostic handoff**: either language produces it, Python batches it.

> The key boundary: cusadi batches a *fixed-structure* `Function`, **not** the
> solver. You cannot batch `opti.to_function()` of the whole NLP solve — it
> embeds an `nlpsol`/IPOPT call as one instruction, which the level-parallel
> interpreter has no opcode for. Batch the building blocks (dynamics,
> constraint Jacobians, Gauss-Newton `JᵀJ` Hessians) instead.

### Run (PyTorch backend, full pipeline)

```bash
conda activate cusadi2
CUDA_VISIBLE_DEVICES=1 python experiments/examples/opti_to_cusadi.py
```

### Run (JAX backend, full pipeline)

The JAX FFI extension is a compiled nanobind module, built by scikit-build-core
as part of the (uv-native, editable) install — no manual cmake step:

```bash
uv sync                                                   # builds cusadi editable + extension
CUDA_VISIBLE_DEVICES=1 uv run python experiments/examples/opti_to_cusadi_jax.py
```

`uv sync` builds cusadi **without isolation** (`[tool.uv] no-build-isolation-package`),
so the extension compiles against the *same* CUDA `jax[cuda13]` pulled into the
venv — including jax's bundled `nvcc` (this matters: building with a system nvcc
newer than the driver emits PTX the driver rejects). `editable.rebuild = true`
means editing `cusadi/jax_ffi/cusadi_jax_ffi.cpp` recompiles on the next import,
and the `.so` always matches the venv's Python (`cpython-3XX`). The extension
installs in-package as `cusadi.jax_ffi._cusadi_jax_ffi`.

### Run (C++ producer + Python batcher)

```bash
conda activate cusadi2
cd experiments/examples

# Build. CasADi's CMake config lives under the conda prefix.
cmake -B build -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build

# Step 1-2 in C++: writes build/casadi_functions/opti_con_jac.casadi
./build/opti_to_cusadi ../../build/casadi_functions

# Step 3: batch the C++-produced function on the GPU
CUDA_VISIBLE_DEVICES=1 python opti_to_cusadi.py
```

(If `find_package(casadi)` fails, point `CMAKE_PREFIX_PATH` at wherever
`casadiConfig.cmake` lives — under conda-forge's casadi that's
`$CONDA_PREFIX/lib/cmake/casadi` or `$CONDA_PREFIX`.)
