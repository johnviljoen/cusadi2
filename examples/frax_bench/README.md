# cusadi2 vs frax: latency head-to-head

Batched rigid-body dynamics on one RTX 4080, cusadi2 against
[frax](https://github.com/danielpmorton/frax) 0.0.4 (Morton & Pavone,
arXiv 2604.04310; JAX 0.11.1 on CUDA 13).

```bash
python examples/frax_bench/bench.py            # codegen, nvcc, both sweeps, report (~30 min)
python examples/frax_bench/bench.py --plot     # rebuild CSV, check and figures from build/
```

Needs `pip install frax` next to the packages `examples/paper_results` uses.

## What is compared

| function | frax | cusadi2 |
|---|---|---|
| `rnea`  inverse dynamics tau(q, qd, qdd), gravity on | `jax.vmap(robot.rnea)` | `cpin.rnea` |
| `crba`  mass matrix M(q), full symmetric | `jax.vmap(robot.crba)` | `cpin.crba`, mirrored |
| `jac`  d tau / d q | `jax.vmap(jax.jacfwd(rnea))` | `ca.jacobian` |
| `hess`  d² tau / d q² | `jax.vmap(jax.hessian(rnea))` | `ca.jacobian` of the Jacobian |

Robots are the two frax ships: Franka Panda (7 dof) and Unitree G1 (35 dof). The
cusadi2 functions are built with Pinocchio-CasADi from frax's own URDF files, in
frax's joint order, and both libraries are fed the same inputs.

**The G1 base is not a quaternion.** frax models the floating base as six virtual
joints (x, y, z prismatic, then roll, pitch, yaw revolute), so q has 35 entries.
The cusadi2 side uses the same composite root joint instead of Pinocchio's
free-flyer, which makes all four functions the same functions of the same
coordinates. Inputs are joint angles uniform inside the URDF limits, base position
within ±1 m, base angles within ±0.5 rad (clear of the pitch singularity), qd and
qdd standard normal.

**The Hessian is a Jacobian of the Jacobian**, not `ca.hessian` per output. The
per-output form is larger on the Panda (62 k instructions against 42 k) and its G1
plan does not fit the device's shared memory in fp64 at any windowing.

cusadi2 writes CasADi's structural nonzeros only; frax returns dense arrays. For
the G1 that is 542 of 1225 Jacobian entries and 6920 of 42875 Hessian entries.

## Protocol

- Batches 1, 4, 16, 64, 256, 1024, 4096; fp64 primary, fp32 secondary.
- Both sides are timed from Python with `time.perf_counter` around one batched
  call (cusadi2 through ctypes, `driver.cu`). Warm-up first, then 20 to 2000 timed
  calls (about 2 s per case); the CSV has the median and 90th percentile.
- `includes_io=0`: inputs and outputs stay on the GPU; the call returns when the
  result is ready (`block_until_ready` / stream sync).
- `includes_io=1`: inputs start as host numpy arrays and the result ends as one.
  frax: `np.asarray(f(numpy...))`. cusadi2: `cudaMemcpy` in, launch, `cudaMemcpy`
  out into a preallocated pageable numpy buffer. This is the variant frax's Fig. 2
  reports.
- cusadi2 plans use `max_live=device_max_live(dtype)` and the runtime block size;
  the level-parallel kernel only.
- frax is jit-compiled and warmed per batch shape; compile time is reported apart.

## Results

All files are in `results/`: `latency.csv`, one figure per function and dtype
(`latency_<function>_<fp>.png`), `check.json`, `compile.json`.

Median latency per batch in µs, fp64, data on the GPU (cusadi2 / frax):

| | B = 1 | B = 64 | B = 4096 |
|---|---|---|---|
| Panda RNEA | 31 / 97 | 32 / 110 | 131 / 598 |
| Panda CRBA | 31 / 87 | 31 / 99 | 124 / 590 |
| Panda Jacobian | 33 / 157 | 34 / 234 | 239 / 4 612 |
| Panda Hessian | 40 / 273 | 42 / 1 230 | 1 950 / 60 988 |
| G1 RNEA | 44 / 249 | 45 / 259 | 313 / 3 343 |
| G1 CRBA | 47 / 234 | 49 / 243 | 354 / 3 752 |
| G1 Jacobian | 59 / 500 | 62 / 1 795 | 2 977 / 108 721 |
| G1 Hessian | 1 392 / 2 647 | 1 439 / 100 224 | 75 358 / out of memory |

**Where cusadi2 wins.** Every fp64 point, with and without transfer, and every
fp32 point but one.

- RNEA and CRBA: 3 to 9 times faster up to batch 1024, mostly because one launch
  costs 30 to 50 µs against 90 to 250 µs for a jitted JAX call. At batch 4096 the
  fp64 gap is 5 times (Panda) and 11 times (G1).
- Derivatives: the gap grows with batch and model size. At batch 4096 in fp64 the
  Jacobian is 19 times (Panda) and 37 times (G1) faster, the Panda Hessian 31
  times. The G1 Hessian is 70 times faster at batch 64, the last point frax
  reaches.
- Memory: frax's G1 Hessian runs out of the 16 GB at batch 256 in fp64 (a 17 GiB
  allocation) and at batch 1024 in fp32 (35 GiB). cusadi2 runs all seven batches.

**Where frax wins or comes close.**

- G1 CRBA in fp32 with host transfer at batch 4096: frax 2.10 ms, cusadi2 2.38 ms.
  Copying the dense 35 × 35 matrices back dominates both, and the same case in
  fp64 is the narrowest fp64 margin (1.6 to 1.8 times at batch 1024 and above).
- fp32 RNEA and CRBA narrow at the top of the sweep: 1.6 to 1.9 times on the Panda
  and 2.3 to 2.9 times on the G1 at batch 4096, down from roughly 3 to 4 times at small
  batch. Batches above 4096 were not measured, so a crossover there is open.
- The G1 Hessian at batch 1 is cusadi2's weakest point: 1.39 ms against 2.65 ms in
  fp64. Its plan is windowed into 3285 levels to fit shared memory (528 in fp32,
  where it takes 0.37 ms).

**Output check** at batch 64: the two libraries agree to 4e-15 or better in fp64
on all eight functions (max abs difference over max abs value). In fp32 they
differ by 3e-4 to 2e-3.

**Compile time.**

| | cusadi2 | frax |
|---|---|---|
| once per dtype | nvcc 3.2 s (kernel + driver, function-independent) | none |
| per function | codegen 0.01 to 0.35 s; G1 Hessian 15 s (fp64), 10 s (fp32) | none |
| per function and batch shape | none | first call 0.3 to 2.6 s (RNEA, CRBA), 1.1 to 5.3 s (Jacobian), 3.2 to 12 s (Hessian) |

frax recompiles for every new batch size; the seven shapes of one function cost
3 to 30 s in total, the G1 Hessian 43 s for the four shapes that fit. A cusadi2
plan serves every batch size.

## Caveats

- One run; the 90th percentile is within 21 % of the median everywhere (typically
  0.5 % for cusadi2, 3 % for frax).
- frax's paper used an RTX 4090 and its README recommends CPU for single
  instances; everything here is on the GPU as specified.
- The cusadi2 transfer numbers use pageable host buffers and synchronous
  `cudaMemcpy`; pinned buffers were not measured.
