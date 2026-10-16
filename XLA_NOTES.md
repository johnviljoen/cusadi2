# Why MJX (XLA:GPU) behaves differently from cusadi2

Source-grounded notes on the XLA:GPU mechanisms behind the MJX baseline's
behaviour in `examples/menagerie_bench`, `examples/frax_bench` and
`examples/humanoid_mit/mpc`. XLA citations are `file:line` in the openxla/xla
checkout at `~/code/xla`, HEAD `82e6c3119a` (2026-10-06); the measurements
below were taken on this machine (RTX 4080, JAX 0.11.1, MuJoCo 3.12.0) with the
benchmark's own G1 fk/jac/hess functions, dumped with `--xla_dump_to` and
profiled with `jax.profiler` (scripts in the session scratchpad). Note the
benchmark's "B" is B·N_STAGES = 8B configurations; the numbers here are per B
configurations, and they agree with the committed results once divided by 8.

## 1. Execution model, side by side

| | XLA:GPU (plain `jax.jit`) | cusadi2 |
|---|---|---|
| unit of a kernel | one *fusion*: a connected producer/consumer subgraph, at most 96 operands+outputs (`gpu_fusible.h:88`), about 10k emitted ops (`gpu_hlo_cost_analysis.h:43`) | the whole function DAG, one interpreter kernel |
| in-kernel parallelism | the output elements of the fusion's array shape, one thread per element (`launch_dimensions.cc:62-82`, 128 threads per block) | batch × stages × level width |
| where intermediates live | registers inside a fusion; a full-shape HBM buffer between fusions (kernel args are `BufferAllocation::Slice`s, `thunk_emitter.cc:1779-1812`; shared memory only inside reduction/transpose emitters) | shared memory for the whole evaluation |
| sync between dependent ops | kernel boundary on one stream (`thunk_executor.cc:87-124`) | `__syncthreads` per level |
| overlap of independent ops | none by default: one compute stream, extra streams only for async collectives/copies or the opt-in `_xla_stream_annotation` (`execution_stream_assignment.cc:223-266`); CUDA-graph commands are chained in `LHS` mode (`command_buffer_cmd_emitter.cc:69-90`); horizontal fusion was deleted in July 2025 (commit `50f16da8fb`) | same level, same block |
| grows with function size | number of kernels | levels and shared-memory footprint |
| grows with batch size | grid of every kernel; kernel count fixed | number of blocks |

Fusion decisions: `priority_fusion.cc` fuses a producer into a consumer only if
it can fuse into *all* users (`:1166-1215`), duplicating it (`:1423-1453`);
dot, sort, custom-call, while, conditional and scatter outputs are hard
boundaries (`:98-139`, `gpu_fusible.cc:553-577`); gather, slice, concatenate,
transpose and small reductions fuse. Every unfused op gets wrapped as its own
kernel (`gpu_compiler.cc:2021`). Dots: matrix-vector and vector-vector dots are
strength-reduced to multiply+reduce (`dot_strength_reduction.cc:202-254`), so
MJX's `rotmat.T @ v` style products never reach cuBLAS; the dumped FK/jac/hess
modules contain zero `dot` and zero cuBLAS calls. (Caveat for other code: the
"too small for GEMM rewriting" exemption, threshold 100, is gated on
f16/bf16/f32 (`matmul_utils.cc:1207-1222`), so an fp64 *matrix-matrix* 3×3
product would go to Triton or cuBLAS regardless of size.)

## 2. What the benchmark's FK actually compiles to

MJX's `scan.body_tree` groups bodies by tree depth and joint type and calls a
vmapped body function per group in a Python loop (`mjx/_src/scan.py:337`), with
`take`/`.at[].set` between groups. Each group boundary is a gather or scatter
through HBM, so the kernel count is set by tree depth × joint-type groups, not
by the amount of arithmetic:

| G1 function | kernels (fusions) at B=1 | at B=16384 | fp64 B=16384 | fp32 B=16384 |
|---|---|---|---|---|
| fk | 42 | 92 | 3.3 ms | 0.44 ms |
| jac | 114 | 141 | 2.9 ms | 1.37 ms |
| hess (B=256) | 208 | 297 | 2.2 ms | 0.95 ms |

All of them are captured into one CUDA graph by default
(`thunk_sequence_after_thunk_passes.txt`: one `kCommandBuffer` holding every
kernel; defaults at `debug_options_flags.cc:307-321`, min graph size 5).

## 3. The batch-1 floor (190 µs round trip, 90 µs on device, vs 30-60 µs)

- On device: FK at B=1 is 42 dependent kernels launched with 2 to 64 threads
  on a single block each (grid `[1,1,1]`), i.e. a serial chain of tiny kernels;
  profiled kernel time sums to about 100 µs, with the final concatenate fusion
  alone 32 µs (one long per-thread chain in fp64) and each scatter 5-7 µs.
  The graph launch removes host launch cost but not the serial dependency chain
  or each kernel's fill/drain.
- Host: every call does a BFC allocation of the temp and output buffers
  (`gpu_executable_buffer_allocator.cc:174-184`), a `cuGraphExecNodeSetParams`
  update of every node whenever buffer addresses change (`ALWAYS_UPDATE`,
  `command_buffer_thunk.cc:109-134`, `debug_options_flags.cc:627`), and the
  completion chain `cuEventRecord` → callback stream → `cuLaunchHostFunc` →
  worker thread → Python (`local_device_state.cc:252-292, 386-412`), which is
  what `block_until_ready` waits on. That is the ~100 µs between the
  on-device and the Python-timed numbers.
- cusadi2's one launch per evaluation has none of the per-kernel serialisation
  and no per-call allocation.

## 4. The fp64 cliff (XLA 7× slower in fp64, cusadi2 1.7×)

Per-kernel profile of FK at B=16384:

| | fp64 | fp32 | ratio |
|---|---|---|---|
| final `input_concatenate_fusion` (the ROOT fusion) | 2467 µs | 207 µs | 12× |
| the other 91/95 kernels together | 660 µs | 169 µs | 3.9× |
| total device busy | 3127 µs | 376 µs | 8.3× |

One kernel is 79% of the fp64 time. It is the ROOT fusion that assembles the
`(B,360)` output from the 30 bodies' `xpos`/`xmat`: a concatenate hero with
81 parameters (just under the 96 cap), 50 nested concatenates and 831 (fp64)
to 1113 (fp32) fused instructions, with the last tree level's quaternion and
rotation arithmetic fused in as producers. The concatenate emitter sizes the
grid by the largest operand and emits one loop per operand in which each
thread evaluates that operand's entire producer subgraph for its one element
(`concatenate_kernel_emitter.cc:210-248`, `ProvideParameter`); nothing is
shared across the nine elements of a 3×3 block or across the nested
concatenates, so the per-body arithmetic is recomputed per output element. The
final LLVM IR of that kernel has 457 fp64 loads and 563 fp64 arithmetic
instructions (static, over 195 basic blocks and 44 switches), no FMA calls at
the IR level (contraction is left to the NVPTX backend; XLA sets no
`contract` flag on GPU, `ir_compiler.cc:484` is CPU-only), no shared memory,
no register cap beyond `nvvm.reqntid` (`fusion_emitter.cc:125-137`).

Reading: at 1/64-rate fp64 (about 0.76 TFLOPS on sm_89) that kernel is
fp64-ALU-bound (2.5 ms corresponds to roughly 2×10⁹ fp64 ops, i.e. a few
hundred per output element, consistent with per-element recomputation),
whereas the fp32 version of the same work is bound by its L2 loads. Nothing
else between the two builds differs by more than 2× (bytes moved). cusadi2
evaluates each DAG node once (about 2000 nodes per instance for G1 FK) and is
barrier/latency-bound, so only its arithmetic share grows in fp64. This is an
inference from the measurements and the emitter source; an Nsight Compute
pipe-utilisation read-out would confirm it. Disabling Triton GEMM changes
nothing (no GEMMs exist); the `__triton` fusions in the dump are transpose
fusions the per-fusion autotuner picked (`xla_gpu_experimental_enable_fusion_autotuner`
default true, `debug_options_flags.cc:554`).

Why cusadi2 is only 3× ahead in fp32: with the fp64 ALU wall gone, XLA's
remaining cost is kernel count, partial waves (a `(B,3)` output at B=16384 is
384 blocks, 42% of one wave) and HBM/L2 round trips between fusions.

## 5. Hessian out-of-memory (17 GiB at B=256 fp64)

- All non-parameter, non-output values are packed into one temp allocation
  whose size is fixed at compile time (`buffer_assignment.cc:2194-2208,
  2786-2801`) and requested as one contiguous `Allocate` per call
  (`gpu_executable_buffer_allocator.cc:56-62, 172-182`); the request must fit
  or the call fails with `ResourceExhausted` (`gpu_executable.cc:1246-1251`).
- `vmap` and `jacfwd` add the batch and the tangent dimension to every HLO
  value, so each materialised intermediate is B × tangents × |x|. The
  scheduler minimises peak memory only by reordering whole instructions
  (`gpu_hlo_schedule.cc:839-858`); rematerialization runs on GPU but only logs
  when it cannot reach the limit (`hlo_rematerialization.cc:3416-3423`); no
  pass tiles the batch dimension to stream instances. The measured hess at
  B=256 already has a 500 MB temp buffer and 297 kernels.
- cusadi2 keeps the per-instance live set (85 KB for the G1 Hessian) in shared
  memory and the only HBM traffic is inputs and outputs.

## 6. Recompile per batch size (2-10 s MJX, 31-48 s MPC)

Buffer sizes, launch grids and index maps are compile-time constants
(`buffer_assignment.h:177`, `launch_dimensions.cc:63-80`), so a new B is a new
module. Bounded dynamic shapes exist (`DynamicPadder`, runtime shape checks)
but pay for the bound. Compile time is the HLO pipeline plus LLVM at O3 plus
ptxas per kernel (92-297 kernels here) plus the per-fusion autotuner
(`xla_gpu_autotune_level` 4). Kernel and autotune caches
(`xla_gpu_kernel_cache_file`, `xla_gpu_per_fusion_autotune_cache_dir`) key on
fusion identity, which changes with B. cusadi2's plan carries no shape.

## 7. Rollouts (scan of `mjx.step`, 3 ms at B=1)

A `lax.scan` becomes an HLO while. With a known trip count the host loops the
body without a sync (`while_thunk.cc:284-297`); otherwise each iteration does a
device→host copy of the predicate plus `BlockHostUntilDone` (`:316-327`).
WHILE is not a default command-buffer type (`debug_options_flags.cc:307-321`),
so the body becomes its own CUDA graph relaunched per iteration
(`command_buffer_conversion_pass.cc:865-883`); loop unrolling only applies to
loops with collectives (`double_buffer_loop_unrolling.cc:580-593`). The step
body (kinematics + RNE + integration) is a few hundred kernels, so 8 steps at
B=1 are 8 × (graph update + launch + a serial chain of tiny kernels).

## 8. Numerics

GPU math goes to libdevice (`lower_to_llvm_gpu.cc:308`); fp32 `tanh` and `erf`
use XLA polynomial/rational approximations (`expand_float_ops.cc:61-113,
710-711`); no fast-math, no FTZ, full-precision fp32 division
(`nvptx_backend.cc:131-145, 214`); `xla_gpu_enable_fast_min_max` false. The
3e-4 to 2e-3 fp32 differences in `frax_bench/results/check.json` are
therefore accumulation order and conditioning, not approximate intrinsics.

## Not verified

Per-kernel fp64 pipe utilisation (no Nsight on this machine); whether
`WhileLoopTripCountAnnotator` matches JAX's scan in the rollout case; the exact
XLA commit inside jaxlib 0.11.1 (the dumped launch dimensions and thunk layout
match HEAD).
