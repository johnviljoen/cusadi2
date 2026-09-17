# cusadi2 Menagerie campaign — per-robot latency vs batch size

The **exact benchmark matrix of `examples/paper_results`** — a latency-vs-batch
saturation sweep of four kernel configurations (`full`/`staged`/`lp`/`batch`)
× two dtypes (double/float), timed from C++ **and** from Python via ctypes, plus
MuJoCo **MJX** baselines (fp32/fp64; batched / rollout / marginal-GPU variants),
a **mujoco-warp** FK baseline, and compile-time tracking — run **once per robot**
for every robot in the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
checkout at `examples/mujoco_menagerie/` (a separate read-only clone, treated as
pure data).

```bash
CUSADI2_GPU=5 python examples/menagerie_bench/run.py                 # full campaign
python examples/menagerie_bench/run.py --robot unitree_a1            # one robot
python examples/menagerie_bench/run.py --robot unitree_a1 --force    # re-run it
python examples/menagerie_bench/run.py --summarize                   # cross-robot outputs only
```

Read `examples/paper_results/README.md` first — everything it says about the
four configurations, the stage axis, timing symmetry, the MJX/warp baselines and
the compile-time panels holds here verbatim. This README documents only what the
per-robot campaign adds or changes.

## Model loading: same MJCF on both sides

Unlike paper_results (which loads Pinocchio from URDF via `robot_descriptions`),
**both backends here load the SAME Menagerie MJCF XML**: MuJoCo via
`MjModel.from_xml_path`, Pinocchio via `pin.buildModelFromMJCF` (Pinocchio ≥ 4.0)
then `cpin.Model(model)`. Base handling comes from the MJCF itself — a
`<freejoint>` becomes a Pinocchio FreeFlyer, its absence a fixed base; a root
joint is never forced.

**Frame set**: the **named** MuJoCo bodies excluding `world`, in MuJoCo body
order, matched 1:1 by name to Pinocchio BODY frames. Unnamed bodies
(`mj_id2name` → `None`) cannot be matched by name, so they are excluded from
the **output** set on both sides — but they are still *computed* internally by
both backends: MuJoCo/MJX kinematics always covers every body regardless of
what is extracted, and on the Pinocchio side unnamed intermediate bodies remain
part of the kinematic tree (its MJCF parser keeps them under auto-generated
names such as `Bis`/`torsoBis`), so the FK/Jacobian *work* is identical — only
the extracted outputs shrink. In the current Menagerie this is load-bearing for
exactly two robots: `google_barkour_vb` (an unnamed 0-dof decorative weld on
the torso) and `umi_gripper` (its unnamed 6-dof articulated root). The gate
records the excluded count as `n_bodies_unnamed`, and its dof-equality check
runs *before* frame matching, so an unnamed articulated body whose joints were
dropped by either parser is caught as `dof_mismatch` rather than silently
shrinking the workload. FK = each link's world placement (translation +
flattened rotation); Jacobian = each link's 6×nv world-aligned
(`LOCAL_WORLD_ALIGNED`) geometric Jacobian.

### Pinocchio 4.0 MJCF parser defects (and the repaired load path)

Loading the Menagerie through `pin.buildModelFromMJCF` exposed **three defects
in Pinocchio 4.0's MJCF support**, each producing a *silently wrong or partial*
model (no exception). `model.load_pinocchio` — the single Pinocchio load path,
used by both the gate and the benchmark codegen, so what is benchmarked is what
was gated — repairs all three:

```
flatten <include>s (A) → wrap multi-tree <worldbody> (B)
    → pin.buildModelFromMJCF → fix dropped joint anchors (C)
```

MuJoCo always loads the **original** XML; only Pinocchio sees the rewritten
temp file, and each repair is a numeric no-op wherever parsing was already
correct. The gate's random-config FK cross-check (tolerance 1e-8) verifies the
repaired model per robot, and its JSON records which repairs were load-bearing
as `"preproc": {"includes_flattened", "multitree_wrapped", "anchors_fixed"}`.

- **(A) `<include>` elements are silently ignored.** MuJoCo's include semantics
  are literal textual insertion of the included file's root-element contents
  (paths relative to the including file); Pinocchio skips the element entirely,
  yielding a partial model (or a confusing failure, e.g. `ms_human_700`'s
  "Keyframe size does not match model size" — the keyframe in the main file no
  longer matches the include-stripped model). Repair: recursively inline the
  includes textually before the parse — exact numerics preserved,
  byte-identical no-op for include-free files. Affected: `ms_human_700`,
  `aloha` (whose two arms arrive via includes).
- **(B) Only the FIRST kinematic tree under `<worldbody>` is built.** Sibling
  root bodies are silently dropped: `aloha` lost one arm (`dof_mismatch`),
  `shadow_dexee` lost everything but a static camera stand (`pin nq=0`),
  `apptronik_apollo` lost a world-welded link (`missing_frames`). Repair: wrap
  *all* worldbody root bodies in a single static dummy body
  (`__pin_multitree_root__`, with an explicit zero `<inertial>` so
  `inertiafromgeom="false"` models like the crazyflie still parse); no
  renaming or reordering, so name matching and qpos order are unaffected. Side
  effect: the wrapper also preserves the root body's world placement, so
  `root_correction` (below) is now usually inactive — it is kept as a verified
  fallback and remains correct either way.
- **(C) Joint anchors inherited from a `<default>` class are dropped.** A
  joint's `pos="..."` is honored inline but ignored when it comes from a
  default class, silently moving the joint axis (`robotiq_2f85` /
  `stanford_tidybot`: FK off by ~1e-2 m). Repair: post-load, using MuJoCo's
  compiled `m.jnt_pos` as ground truth — for each non-free joint matched by
  name, the anchor Pinocchio actually applied is read off the child BODY frame
  placement; any nonzero delta is applied to `jointPlacements[jid]` and
  conjugated out of the joint's inertia, attached frames and child joints, so
  only the anchor moves. Delta-detection makes it a no-op where parsing was
  correct (`anchors_fixed: 0`).

These are publishable observations about Pinocchio 4.0's MJCF support: with
the three repairs, **all 66 benchmarkable Menagerie robots pass the FK gate at
machine precision** (max error < 3e-14).

> **Provenance note (a1/h1).** The Menagerie MJCF and the URDFs used by
> paper_results are separate model files: link poses, joint ordering and inertial
> frames can differ slightly, so the `unitree_a1`/`unitree_h1` numbers here may
> differ slightly from paper_results' URDF-based a1/h1 numbers. paper_results
> remains the deep-dive reference; this campaign is the breadth study.

## The compatibility gate

Before any benchmarking, each robot passes a gate (`model.run_gate`, run in its
own subprocess so parser crashes cannot kill the campaign). **Every failure is a
machine-recorded skip reason in `status.json`, never a crash**:

1. `pin.buildModelFromMJCF` parses without error (`pin_mjcf_parse_failed`).
2. Pinocchio nq/nv **exactly equal** MuJoCo nq/nv (`dof_mismatch`) — catches
   silently-partial Pinocchio builds, e.g. `aloha`, where only one arm survives.
3. Every **named** MuJoCo body (excl. `world`) exists as a Pinocchio BODY frame
   (`missing_frames`); unnamed bodies are excluded from the frame set (counted
   in `n_bodies_unnamed`, see the frame-set paragraph above).
4. **Numerical FK cross-check**: at 5 random configurations the CasADi FK is
   evaluated and compared against `mj_kinematics` `xpos`/`xmat` of the named
   bodies, tolerance `1e-8` (fp64 CasADi on CPU vs MuJoCo double) —
   `fk_mismatch`.
5. `nq == 0` → `zero_dof` (static props, sensors, ...).

Anything `run_gate` itself does not catch is trapped by the `--gate` entrypoint
and written to the gate JSON as `{"status": "skip", "reason": "gate_error:
<ExcType>: <msg>", "traceback": ...}` with exit code 0 — the gate subprocess
never dies without a recorded result.

Two MJCF↔Pinocchio convention traps are handled inside the gate and builders,
and *verified* by check 4:

- **qpos quaternion order.** MuJoCo stores free- and ball-joint quaternions as
  `(w,x,y,z)`; Pinocchio FreeFlyer/Spherical expect `(x,y,z,w)`. A per-joint
  qpos index map (`model.qpos_mj_to_pin_index`) reorders them; random samples
  are quaternion-normalized.
- **Root placement.** Pinocchio's MJCF parser drops the root body's world
  placement (e.g. ur5e's `<body name="base" quat="0 0 0 -1">`), leaving its
  whole model expressed in the root-body frame. `model.root_correction` derives
  the constant rigid offset numerically and the builders fold it into the CasADi
  functions as a constant per-frame transform, so the benchmark workload stays
  "world placements", identical to what MJX computes (`root_corr: true` in the
  gate record). The random-config cross-check runs on the *corrected* functions,
  so any mismatch that is not a constant offset still fails the gate.

## Manifest

The robot list is **manifest-driven**: `manifest.py` in this directory defines

```python
MENAGERIE_COMMIT = "<full sha>"
ROBOTS = {
    "unitree_a1": {"xml": "unitree_a1/a1.xml"},            # path relative to menagerie root
    "realsense_d435i": {"xml": None, "skip": "zero_dof_camera"},
    ...
}
```

- Keys are arbitrary unique names; they become the `results/<robot>/` directory
  names. `xml` is relative to the menagerie root.
- **Variant rules:** pick each robot's canonical standalone robot XML — *not*
  `scene.xml` (which adds floor/light/prop world bodies that would pollute the
  frame set) and *not* MJX-tuned variants (`mjx_*.xml`), which alter solver
  options for MJX and would break the "same model both sides" premise. Robots
  with genuinely distinct variants (e.g. with/without hands) may appear as
  separate keys pointing at different XMLs.
- Entries with `"skip": <reason>` (or `xml: None`) are recorded as manifest
  skips without touching the model.
- `run.py` warns **loudly** if the menagerie checkout is dirty or its HEAD
  differs from `MENAGERIE_COMMIT`, and records both hashes in every
  `status.json`.

An alternate manifest can be supplied with `--manifest PATH` or
`CUSADI2_MANIFEST` (used e.g. for throwaway smoke manifests kept in `build/`).

## Orchestration, resumability, timeouts

Robots run **sequentially**; each robot's stages run as **subprocesses** so a
segfault/OOM in one stage is a recorded failure, not a campaign abort:

| stage             | subprocess                                   | writes |
|-------------------|----------------------------------------------|--------|
| `gate`            | `run.py --gate --robot R`                    | `build/<R>/gate.json` → `status.json` |
| `cusadi_<dtype>`  | `run.py --cusadi --robot R --dtype D`        | plans + `latency` binary in `build/<R>/<D>/`, `latency_<D>.csv`, `compile_cusadi2_<D>.json` |
| `pylib_<dtype>`   | `run.py --pylib --dtype fpXX --lib ... `     | `latency_py_<D>.csv` (ctypes-hosted twin sweep) |
| `cpu_compile_<dtype>` | `run.py --cpu-compile --robot R --dtype D` | CasADi C codegen + gcc of the paper's CPU driver (`<stem>`, `<stem>_transfer`) in `build/<R>/<D>/cpu/`, `cpu_compile_<D>.json` |
| `cpu_runs`        | in-process, sequential, per run k            | `runs/<k>/cpu_<D>.json` (paper_results' `run_cpu` schema) |
| `mjx_<fpXX>`      | `run.py --mjx --robot R --dtype fpXX --runs K` | `runs/<k>/mjx_<fpXX>.json` (K files, one per timing pass) |
| `warp`            | `run.py --warp --robot R --runs K`           | `runs/<k>/warp.json` |
| `plot`            | in-process, best-effort                      | per-robot figures |

- `results/<robot>/status.json` records per-stage `pass`/`fail`/`skip`/`timeout`
  + timings, the gate result, and the menagerie commit hashes. It is rewritten
  after every stage; `"complete": true` only once all stages were attempted.
- **Resumability:** on restart, robots whose `status.json` is complete are
  skipped. `--robot X` runs one robot; `--force` re-runs even if complete.
- **Timeouts:** generous per-stage limits (15 min gate, 2 h each for
  cusadi/pylib/mjx/warp), recorded as status `"timeout"` on expiry.
- **`mjx_unavailable` / `warp_unavailable`:** if `mjx.put_model` (or warp's
  `put_model`) rejects a model outright, the subprocess writes
  `{"error": "mjx_unavailable: ..."}` and the stage is recorded as a skip —
  the cusadi2 + warp (resp. MJX) results are kept.
- **MJX / warp memory failures:** the baselines run one GPU process at a time,
  smallest robot first; every shape (series × batch) is JIT-compiled once and
  timed `--runs` times, and the per-run files are rewritten after every shape.
  A shape that raises (VRAM out of memory) is `null` with its reason in the
  file's `failed` map. A process that dies (earlyoom) or stops making progress
  for `CUSADI2_MJX_STALL` s (default 3600; a GPU op hung on VRAM never raises)
  is killed and restarted, and the shape it was on is recorded as failed and
  skipped; `status.json` carries `attempts` and `failed_shapes`.

> **Fairness of the MJX / warp baselines.** Both load the same MJCF with
> contacts masked off (`contype`/`conaffinity` zeroed) and, through the model's
> disable flags, **no contact detection and no constraint solver at all**
> (`mjDSBL_CONTACT | mjDSBL_CONSTRAINT`, `_disable_contacts_and_constraints`).
> In MJX that means the collision driver and every constraint row are gone
> (`ncon = nefc = 0` at `put_data`) and `mjx.forward` returns `qacc_smooth`
> without ever tracing `solver.solve` — the friction cone, joint limits and
> equality constraints enter no code path, so the rollout's `mjx.step` is
> smooth dynamics + kinematics, the same workload class as cusadi2. Derivatives
> are analytical throughout: `mjx.jac` is MuJoCo's body Jacobian from the
> motion subspace (`cdof`), the Hessian is exact forward-mode AD
> (`jax.jacfwd`) of it — no finite differences anywhere. mujoco-warp has no
> autodiff but exposes the same analytical Jacobian (`mjw.jac`, one launch per
> body over all worlds), so it contributes FK **and** Jacobians (fp32), no
> Hessian; both were checked against MuJoCo's `mj_jacBody` (1e-7).

> **Equality constraints & the rollout curve.** Several Menagerie robots carry
> tendons or `<equality>` constraints (parallel grippers, four-bar linkages).
> They do not affect the kinematics workload, and with the constraint solver
> disabled (above) they do not enter the rollout either: the rollout is the
> unconstrained smooth dynamics, deliberately, so that no baseline pays for
> work the cusadi2 side never does.

## Knobs (identical to paper_results)

`N_STAGES = 8`, `BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]`,
`N_WARMUP = 5` / `N_TIMED = 20`, `DTYPES = ("double", "float")`, MJX/warp full
batch sweep to 16384 for every series (paper_results caps MJX at 1024) with
OOM→`null`, warp fp32 FK + analytical Jacobians (no Hessian: no autodiff). Plans are written with
`max_live` set to the GPU's per-block shared-memory limit, so a plan whose
longest-path live set does not fit (the larger Hessians, and most ×8-unrolled
Hessians) is windowed by cusadi2 until it does; `compile_cusadi2_<dtype>.json`
records `windowed` / `windowed_unr` per function. `--no-baselines` runs only
the gate and the cusadi2 stages (MJX / warp recorded as skipped).

Env overrides: `CUSADI2_GPU` (device index, **default `0`**), `NVCC`,
`CUSADI2_MENAGERIE` (menagerie root, default `../mujoco_menagerie` relative to
this dir), `CUSADI2_MANIFEST`. Smoke-test knobs (propagate to subprocesses):
`CUSADI2_BATCHES`, `CUSADI2_N_WARMUP`, `CUSADI2_N_TIMED`, `CUSADI2_RESULTS`,
`CUSADI2_BUILD`.

## Outputs

Per robot, `results/<robot>/` holds the same artifacts as paper_results'
`results/` (per-robot rather than per-campaign):

- `latency_<dtype>.csv` / `latency_py_<dtype>.csv` — the C++- and Python-hosted
  sweeps, schema `dtype,func,config,batch,n_stages,grid_blocks,latency_us,wall_us`
  with `func ∈ {fk, fkjac, fkhess}` (the Hessian, `model.build_fkhess`, is d/dq of the stacked frame Jacobians as in paper_results).
- `runs/<k>/mjx_<fpXX>.json` — run k's MJX medians, keys
  `{fk,jac,hess}_{batched,rollout}` and `{fk,jac,hess}_batched_gpu`, plus the
  per-shape first-call JIT time (`compile`, paid once, copied into every run
  file), `failed` (shape → reason) and `complete` (or an `error` field).
- `runs/<k>/warp.json` — run k's mujoco-warp medians, keys `fk` and `jac`
  (analytical, `mjw.jac`), + `compile_once_s` / `compile_jac_once_s`.
- `runs/<k>/cpu_<dtype>.json` — the CPU reference: `latency_us` (one problem),
  `seq_us[func][threads][B]` (idealized) and `seq_transfer_us[...]` (with the
  GPU→CPU→GPU copies), `compile_s` (codegen + gcc), `skipped` (RAM-guarded or
  failed points); `cpu_compile_<dtype>.json` — the compile record.
- `compile_cusadi2_<dtype>.json` — per-function `codegen_s`, shared `nvcc_s`,
  plus a `meta` block (instruction counts, per-stage/unrolled smem) consumed by
  the summary.
- `status.json` — the per-robot campaign record (see above).
- `latency_<dtype>.png/pdf`, `throughput_<dtype>.png/pdf`,
  `compile_<dtype>.png/pdf` — the paper_results figures as 1×2 panels
  (FK | frame Jacobians) with the robot's name in the titles.

Cross-robot (regenerated by the `summarize` step at campaign end, or standalone
with `--summarize`), in `results/`:

- `summary.csv` — one row per robot × func × dtype: nq/nv/n_bodies, instruction
  counts, per-stage/unrolled smem KB, the best cusadi2 wall latency at the
  saturation batch (largest batch feasible for both sides, from the
  Python-hosted CSV — host-language-symmetric with MJX), MJX batched/rollout
  latency at that matching batch, the speedups, compile times, and gate status
  (gate-skipped robots get one bare row with the reason).
- `status.md` — markdown table robot × stage with pass/fail/skip reasons.
- `summary_speedup.png/pdf` — cusadi2-vs-MJX-batched (and mujoco-warp on the FK
  and Jacobian panels) speedup against model size (nv), one point per robot,
  fp64 and fp32 as two series, one panel per function.
- `summary_speedup_cpu.png/pdf` — the same against the CPU reference (16
  threads; filled = CPU transfer, hollow = CPU idealized), three panels.
  `summary.csv` carries `cpu_batch`, the four CPU latencies (seq/par ×
  idealized/transfer) and their speedups over cusadi2's best config.

## Files

- `dag_shape.py` — DAG shape of every robot's FK / Jacobian / Hessian (depth,
  per-level widths of the plain level plan, and whether cusadi2 windowed the
  plan at this GPU's shared-memory limit), one subprocess per robot, cached in
  `results/dag/<robot>.json`; `paper_results/dag_shape.py` draws it.
- `run.py` — orchestrator + all subprocess stage entrypoints
  (`--gate/--cusadi/--pylib/--mjx/--warp/--summarize`). The MJX / warp / pylib
  benchmark bodies are verbatim from paper_results, parameterized by robot.
- `model.py` — MJCF loading for both backends, qpos index map, root-placement
  correction, CasADi FK / frame-Jacobian / frame-Hessian builders, horizon
  unroll, and the gate.
- the driver is `paper_results/latency.cu` unchanged: each robot's build dir
  holds its `.plan` files, the original-cusadi unrolled kernels and the
  generated `unrolled_table.h` the driver takes its function list from.
- `plot.py` — paper_results' figures per robot (1×3 panels, robot-name titles);
  also runnable standalone: `python plot.py results/<robot> [name]`.
- `summarize.py` — builds `summary.csv`, `status.md` and the aggregate speedup
  figure from whatever `results/<robot>/` directories exist.
- `manifest.py` — the robot list (authored separately; see above).
