# HANDOFF — cusadi defense codebase

You are picking up an in-progress defense-prep project on John Viljoen's
forked `cusadi` codebase. This document tells you what's been built,
where it lives, and what's left. Defense is **2026-05-04**.

If you only read three things, read these and the rest of this doc is
just context:

1. `ALGORITHM.md` (this directory) — canonical 11-section technical doc
   with ASCII diagrams. **Source of truth for what the contributions are.**
2. `~/.claude/projects/-home-john-code-manim/memory/project_cusadi_stage_system.md`
   — running state of the work across sessions, key benchmark findings,
   remaining gaps. Updated as work progressed.
3. `figures/lp_ablation.png` — the headline defense figure (2×2 ablation
   with the missing corner annotated). Rendered from
   `figures/plot_lp_ablation.py`.

## 1. What's in this fork (what we built)

Three orthogonal contributions on top of the prior level-parallel (LP)
work, all auto-dispatched from one entry point:

| contribution | what it closes | where |
|---|---|---|
| Stage compilation (chunk kernel + hybrid wrapper) | per-block smem grows with horizon → flat at chunk size; cross-stage parallelism axis | `src/level_parallel/codegen.py:CHUNK_KERNEL_TEMPLATE`, `src/level_parallel/stage.py:CusadiHybridStageLevelParallelFunction` |
| Tiled-stage (composes tiling with stage axis) | per-stage > device smem regime | `src/level_parallel/codegen.py:STAGE_TILED_KERNEL_TEMPLATE`, `src/level_parallel/stage.py:CusadiTiledStageLevelParallelFunction` |
| Setup-time engineering (algorithmic fix + analytic chunk replication + per_stage_plan dedup) | per-session setup grows with horizon → constant | `src/level_parallel/dag.py:_compute_liveness` (one-line fix), `src/level_parallel/stage.py:_build_chunk_plan_analytic`, `validation/_helpers.py:build_smart_stage_lp_fn` |

Plus a unified entry point with smart auto-dispatch:

| | LP off | LP on |
|---|---|---|
| **staging off** | (NVCC-infeasible at MPC horizons; we measured this) | unrolled-LP path (`build_lp_fn`) |
| **staging on** | original cusadi @ B×N (deployable) | **stage + LP** (our default; `build_smart_stage_lp_fn`) |

The smart dispatcher picks chunk_size from {device smem, per-stage size,
COUPLED-input presence} via four rules in `src/level_parallel/stage.py:smart_pick_chunk_size`.

## 2. Key file map

```
cusadi/
├── ALGORITHM.md                               ← canonical algorithm doc (READ FIRST)
├── HANDOFF.md                                 ← this file
├── src/
│   ├── level_parallel/
│   │   ├── codegen.py                         ← 5 kernel templates (LP, tiled, stage, chunk, stage_tiled)
│   │   ├── dag.py                             ← extract_level_plan, extract_tiled_level_plan
│   │   │                                        (quadratic fix at _compute_liveness)
│   │   ├── stage.py                           ← StageMode, StageSpec, hybrid + tiled-stage wrappers,
│   │   │                                        smart dispatch, analytic chunk plan
│   │   ├── function.py                        ← legacy LP wrappers (untouched, latent smem-attr bug)
│   │   └── unified.py                         ← auto-builds the .so chain
│   ├── casadi_functions/                      ← .casadi fixtures
│   │   ├── a1_aba_jac_analytical.casadi       ← used in LP ablation
│   │   ├── a1_aba_jac_analytical_unrolled_N12.casadi  ← built for stage-OFF LP-OFF corner
│   │   ├── a1_constrained_aba_jac.casadi      ← used in apples-to-apples LP repro
│   │   └── h1_constrained_aba_jac.casadi      ← used in tiled-stage validation
│   └── generateCUDACode.py                    ← original cusadi codegen (used for unrolled .so build)
├── codegen/
│   ├── level_parallel_kernel.cu               ← LP kernel (existing)
│   ├── level_parallel_tiled_kernel.cu         ← tiled LP kernel (existing)
│   ├── level_parallel_stage_kernel.cu         ← NEW stage kernel
│   ├── level_parallel_chunk_kernel.cu         ← NEW chunk kernel (the main contribution)
│   ├── level_parallel_stage_tiled_kernel.cu   ← NEW tiled-stage kernel
│   └── a1_aba_jac_analytical_unrolled_N12.cu  ← 35.4 MB single-fn .cu — does not NVCC-compile in 2 h
├── build/
│   └── liblevel_parallel_*.so                 ← all kernel libs (built once, function-independent)
├── validation/
│   ├── _helpers.py                            ← build_lp_fn, build_smart_stage_lp_fn,
│   │                                            make_random_inputs (with normalize_freeflyer_quat=...)
│   ├── _section_lp_ablation.py                ← 4-curve LP ablation (N=12 default)
│   ├── _section_lp_ablation_unrolled.py       ← stage-OFF LP-OFF corner attempt
│   ├── _section_stage.py                      ← unicycle MPC stage benchmark
│   ├── _section_stage_robotics.py             ← A1 / H1 stage benchmark
│   ├── _section_stage_hybrid.py               ← chunk_size sweep
│   ├── _section_VID.py                        ← MJX comparison
│   └── _section_VIB.py                        ← NVCC compile-time table
├── figures/
│   ├── plot_lp_ablation.py                    ← renders the headline figure
│   ├── lp_ablation.png / .pdf                 ← rendered figure
│   └── lp_ablation_unrolled_cusadi.json       ← unrolled-cusadi compile-attempt result
└── benchmark/                                 ← original LP benchmark scripts (existing)
```

## 3. Key benchmark data (already measured, persisted)

### Setup time (per Python session, end-to-end build_smart_stage_lp_fn)

| problem | N | before our work | now |
|---|---|---|---|
| unicycle (small, COUPLED) | 120 | 28 ms | **16 ms** |
| A1 ABA jac (heavy, STAGE-only) | 12 | 10.4 s | **0.68 s** |
| A1 ABA jac | 120 | 24.3 s | **0.68 s** |
| H1 ABA jac (very heavy) | 12 | 74.5 s | **0.83 s** |
| H1 ABA jac | 120 | 83.9 s | **1.26 s** |

Source: see `project_cusadi_stage_system.md` and the timing logs from
`build_smart_stage_lp_fn` calls.

### LP-vs-no-LP ablation on A1 ABA Jac, n_stages=12

Cached in `figures/plot_lp_ablation.py:DEFAULT_DATA`:

| B | stage+LP | unrolled+LP | original cusadi @ B×N | chunk @ bs=1 |
|---|---|---|---|---|
| 1 | 161 µs | 2,429 µs | **599 µs** | 5,717 µs |
| 8 | 163 µs | 2,815 µs | 1,601 µs | 6,140 µs |
| 32 | 316 µs | 3,882 µs | 4,336 µs | 12,129 µs |
| 128 | 940 µs | 5,430 µs | 4,338 µs | 36,663 µs |
| 256 | 1,756 µs | 10,845 µs | 4,302 µs | 67,780 µs |
| 1024 | 7,588 µs | 44,672 µs | 8,881 µs | 299,362 µs |
| 4096 | 29,838 µs | 190,259 µs | 157,753 µs | 1,172,010 µs |

LP gain over deployable original cusadi: 1.2× (B=256 saturation point) to
13.7× (B=32, real-time MPC regime).

### Unrolled-cusadi compile attempt (the missing 2×2 corner)

`figures/lp_ablation_unrolled_cusadi.json`:

```json
{
  "fn_name": "a1_aba_jac_analytical_unrolled_N12",
  "n_instructions": 490638,
  "cu_size_mb": 35.4,
  "codegen_s": 3.13,
  "nvcc_s": 7200.0,           ← TIMED OUT
  "compile_success": false,
  "outcome": "cicc still in PTX phase at 2h kill; estimated >3h total, OOM risk"
}
```

This is the headline: original cusadi on the unrolled-horizon function
**cannot be compiled in defense-prep time**. By contrast, the LP kernel
NVCC-compiles in 1.5 s once and is function-independent.

### Numerical correctness

| function depth | abs err vs CasADi | comment |
|---|---|---|
| ≤ 108 levels (synthetic) | **0.00** (bit-exact) | level order matches CasADi to fp64 precision |
| 170-281 levels (a1/h1 ABA jac) | 1e-13 to 1e-9 | order-dependent rounding |
| 538 levels (h1_constrained, tiled) | ~7e-7 | same depth/ordering story; tiling itself is bit-exact |

Relative error stays at fp64 unit roundoff (~10⁻¹⁵) in every regime.
Chunk and tiled-stage kernels are bit-identical to their non-stage
equivalents at the corresponding configuration.

The `1.24` "catastrophic error" red herring is documented in
`ALGORITHM.md §8.1` — random inputs to free-flyer Jacobians produce
non-unit quaternions → output magnitudes blow up to 10¹³, making
fp64 unit-roundoff look like absolute errors of 1.24. Use
`make_random_inputs(..., normalize_freeflyer_quat=True)` for clean
numbers.

## 4. Known gaps (designed and deferred)

1. **`n_stages` must divide chunk_size**. Padding/remainder kernel would
   close it. Minor.
2. **Auto-tuner is memory-only**. Misses by ~5% on A1 N=12 (picks c=2
   when c=1 is empirically optimal). Throughput-aware micro-benchmark
   would close it.
3. **`cudaFuncSetAttribute` per-kernel-symbol bug**. Fixed in the new
   wrappers (chunk, stage_tiled) by re-asserting the smem attribute on
   every `evaluate()` call. Latent in the legacy wrappers
   (`CusadiLevelParallelFunction`, `CusadiTiledLevelParallelFunction`,
   `CusadiStageLevelParallelFunction`). Manifests only when sibling
   wrappers with different smem budgets coexist in one process. If you
   touch their `evaluate()`, port the same fix.
4. **Tiled-extract bug on very wide unrolled DAGs**. `extract_tiled_level_plan`
   produces chunks with peak > `max_smem_slots` when the per-level width
   itself exceeds the budget. Fails at A1 unrolled N=24 (max_width=2781
   per stage × 24 stages). Workaround: keep N ≤ 12 for unrolled cases,
   or use stage compilation instead.
5. **Plan extraction is pure Python**. ~0.9 s on H1 after the quadratic
   fix. C++ port via existing nanobind infrastructure would shave another
   order of magnitude — but no current workload needs it.

## 5. Defense narrative (for context, see HANDOFF in `manim/qualifying_exam_05_04_2026/HANDOFF.md` for slide-by-slide)

10-minute arc, post-jaxipm:

```
1 min   KKT block + sparsity-pattern reveal     ← visual hook
2 min   "MJX is too slow; cusadi is too coarse"  ← problem framing
2 min   Contribution 1 — level-parallel         ← intra-stage win
2 min   Contribution 2 — stage compilation      ← cross-stage win
1 min   Contribution 3 — setup-time engineering ← buildability win
1 min   Tiling + smart dispatch (composition)   ← pre-empt questions
1 min   Headline numbers + jaxipm-on-H1 close   ← the payoff
```

Closing punchline:

> Three orthogonal cost axes — NVCC compile, per-session setup, per-block
> shared memory — collapse into one system: a JAX-callable, auto-
> dispatching, sparse-derivative evaluator that takes a URDF and a horizon
> and gives you GPU-throughput-class evaluation at sampling rates jaxipm
> needs. The contribution isn't any single kernel — it's that the four
> operating regimes all map to one entry point with bounded setup time.

## 6. Working preferences (from John, captured across sessions)

- **GPU pinning**: always set `CUDA_VISIBLE_DEVICES=1`. GPU 0 is reserved.
  Saved in memory as `feedback_gpu_selection.md`.
- **Conda env**: `cusadi2` for cusadi work. Activate via
  `source /opt/Miniconda/etc/profile.d/conda.sh && conda activate cusadi2`.
- **Honest framing over rhetorical claims**. John explicitly prefers
  "designed and deferred" over "complete" when there are real gaps.
- **Don't auto-detect things**. Opt-in flags are fine; magic detection
  is fragile.
- **Pyright import errors are project-wide**. The cusadi project uses
  `sys.path.insert(0, ...)` rather than installable packages. Every file
  that imports from `src.*` or `validation.*` will trip Pyright. Ignore.

## 7. How to extend the figure / regenerate data

### Re-run the LP ablation (4 curves, B sweep)

```bash
source /opt/Miniconda/etc/profile.d/conda.sh && conda activate cusadi2
CUDA_VISIBLE_DEVICES=1 cd /home/john/code/cusadi
python -m validation._section_lp_ablation
```

The script prints a table; copy values into `figures/plot_lp_ablation.py:DEFAULT_DATA`,
then re-render with:

```bash
python figures/plot_lp_ablation.py
# → writes figures/lp_ablation.png and .pdf
```

### Re-attempt the unrolled-cusadi compile (the missing corner)

```bash
CUDA_VISIBLE_DEVICES=1 python -m validation._section_lp_ablation_unrolled
```

Note: this kicks off a 2-hour-bounded NVCC compile that is expected to
fail (it timed out at the cicc PTX-generation phase last time). The
script writes `figures/lp_ablation_unrolled_cusadi.json` with the result.

### Smart-dispatch sanity sweep on all three benchmarks

```bash
CUDA_VISIBLE_DEVICES=1 python -m validation._section_stage_hybrid
```

### Numerical equivalence check (chunk c=1 vs original LP)

This was the apples-to-apples reproduction proving we don't regress the
prior contribution. To re-verify:

```python
from validation._helpers import build_lp_fn, build_hybrid_stage_lp_fn, load_casadi_fn
from cusadi.pytorch.lp_stage import StageMode, StageSpec
import torch

f = load_casadi_fn('a1_constrained_aba_jac')
n_in = f.n_in()
in_specs = [StageSpec(StageMode.STAGE)] * n_in
out_specs = [StageSpec(StageMode.STAGE)] * f.n_out()

# Original LP
lp_orig, _, _ = build_lp_fn(f, B=256, max_smem_slots=12000)
# Chunk c=1 with N=1
lp_chunk, _, _ = build_hybrid_stage_lp_fn(f, 256, 1, in_specs, out_specs, chunk_size=1)
# Compare runtimes — should be 1.00× ratio across all batch sizes.
```

Last measurement: 1.00× ratio at every batch size (991 µs at B=1; 1963 µs
at B=256), exactly reproducing the original LP-vs-MJX paper numbers.

## 8. Memory system (cross-session continuity)

Updated as work progressed. Read these to come up to speed:

- `~/.claude/projects/-home-john-code-manim/memory/MEMORY.md` — index
- `project_cusadi_stage_system.md` — current state, key findings,
  what was closed and what remains
- `project_cusadi_audit.md` — historical pre-work audit (now superseded
  for contribution #2; flagged as historical)
- `feedback_gpu_selection.md` — GPU pinning rule
- `user_role.md` — John Viljoen, Berkeley PhD, defense 2026-05-04
- `feedback_manim_render_workflow.md` — manim-specific (don't `rm -rf`
  between renders)

Keep the memory updated as you discover things future sessions should
know. Use the format in the MEMORY.md template; one-line index entries.

## 9. Companion document

Slide-by-slide animation production happens in
`/home/john/code/manim/qualifying_exam_05_04_2026/HANDOFF.md` — that's a
separate codebase / handoff. They're related: `ALGORITHM.md` is the
technical source of truth for both, and the LP-ablation figure here is
slide-ready.

---

**Defense in 5 days.** Most of the contribution work is done; remaining
priorities are slide animations and the closing narrative. If you find
yourself wanting to add a new feature, ask first whether it's load-
bearing for the defense — most of the "designed and deferred" gaps
above are explicitly fine to keep deferred.
