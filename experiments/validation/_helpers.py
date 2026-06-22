"""Shared helpers used across all section runners."""
from __future__ import annotations

import ctypes
import os
import sys
from typing import List

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# THIS_DIR is experiments/validation/ → repo root is two levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, REPO_ROOT)


def repo_path(*parts) -> str:
    return os.path.join(REPO_ROOT, *parts)


# -----------------------------------------------------------------------------
# CUDA-event median timing
# -----------------------------------------------------------------------------
def benchmark_evaluate(evaluate_fn, n_warmup: int, n_timed: int) -> float:
    """Returns median time of `evaluate_fn()` in µs, measured with CUDA events
    in Python. Includes Python + ctypes + zero_() overhead (~10-50 µs)."""
    import torch
    for _ in range(n_warmup):
        evaluate_fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(n_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        evaluate_fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(np.median(times_ms) * 1000.0)


def benchmark_kernel_only(kernel_call_fn, get_last_eval_time_s,
                          n_warmup: int, n_timed: int) -> float:
    """Returns median kernel-only time in µs.

    `kernel_call_fn()` invokes the kernel (which internally uses
    cudaEventRecord around the kernel launch and returns elapsed seconds).
    `get_last_eval_time_s()` reads the most recent `.eval_time`.

    This matches the paper's / the existing CusADi benchmark's timing
    methodology — Python dispatch overhead is excluded."""
    for _ in range(n_warmup):
        kernel_call_fn()
    times_s = []
    for _ in range(n_timed):
        kernel_call_fn()
        times_s.append(float(get_last_eval_time_s()))
    return float(np.median(times_s) * 1e6)


# -----------------------------------------------------------------------------
# Sequential CusADi .so wrapper
# -----------------------------------------------------------------------------
class SequentialFn:
    def __init__(self, fn_casadi, num_instances: int, so_dir: str):
        import torch
        lib = os.path.join(so_dir, f"lib{fn_casadi.name()}.so")
        if not os.path.exists(lib):
            raise FileNotFoundError(
                f"Sequential .so missing: {lib}\n"
                f"For fn_1eX: run `python benchmark/run_paper_comparison.py`.\n"
                f"For Pinocchio fns: run "
                f"`python tests/test_level_parallel_cuda.py --fn {fn_casadi.name()}`.")
        self.fn = fn_casadi
        self.N = num_instances
        self._lib = ctypes.CDLL(lib)
        self._lib.evaluate.restype = ctypes.c_float
        n = num_instances
        self._inp = [torch.zeros(n, fn_casadi.nnz_in(i), device="cuda",
                                 dtype=torch.double).contiguous()
                     for i in range(fn_casadi.n_in())]
        self._out = [torch.zeros(n, fn_casadi.nnz_out(i), device="cuda",
                                 dtype=torch.double).contiguous()
                     for i in range(fn_casadi.n_out())]
        self._work = torch.zeros(n, fn_casadi.sz_w(),
                                  device="cuda", dtype=torch.double).contiguous()
        self._in_ptr = torch.zeros(fn_casadi.n_in(), device="cuda",
                                   dtype=torch.int64).contiguous()
        self._out_ptr = torch.zeros(fn_casadi.n_out(), device="cuda",
                                    dtype=torch.int64).contiguous()
        for i in range(fn_casadi.n_in()):
            self._in_ptr[i] = self._inp[i].data_ptr()
        for i in range(fn_casadi.n_out()):
            self._out_ptr[i] = self._out[i].data_ptr()

    def set_inputs(self, inputs: List):
        for i, t in enumerate(inputs):
            self._inp[i] = t
            self._in_ptr[i] = t.data_ptr()

    def evaluate(self):
        import torch  # noqa
        for o in self._out:
            o.zero_()
        self._work.zero_()
        fn_in = ctypes.cast(self._in_ptr.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_out = ctypes.cast(self._out_ptr.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_work = ctypes.cast(self._work.data_ptr(), ctypes.POINTER(ctypes.c_double))
        # Generated signature: evaluate(inputs[], work, outputs[], N) → elapsed (seconds)
        self.eval_time = float(self._lib.evaluate(fn_in, fn_work, fn_out, self.N))


# -----------------------------------------------------------------------------
# CasADi fn loader
# -----------------------------------------------------------------------------
def load_casadi_fn(path_or_name: str):
    """Load a saved CasADi Function by path or by bare name (search common dirs)."""
    import casadi
    if os.path.isabs(path_or_name) and os.path.exists(path_or_name):
        return casadi.Function.load(path_or_name)
    if path_or_name.endswith(".casadi") and os.path.exists(path_or_name):
        return casadi.Function.load(path_or_name)
    # Search default dirs in priority order
    candidates = [
        repo_path("build", "casadi_functions", f"{path_or_name}.casadi"),
        repo_path("build", "casadi_functions", f"{path_or_name}.casadi"),
        path_or_name,
    ]
    for c in candidates:
        if os.path.exists(c):
            return casadi.Function.load(c)
    raise FileNotFoundError(
        f"Could not find CasADi function '{path_or_name}'. Tried:\n  " +
        "\n  ".join(candidates))


def make_random_inputs(fn_casadi, N: int, seed: int = 42, *,
                       normalize_freeflyer_quat: bool = False) -> List:
    """Random batched inputs for a CasADi function.

    `normalize_freeflyer_quat=True` assumes input arg 0 is a Pinocchio
    free-flyer configuration `q` with layout (pos[0:3], quat[3:7], joints[7:nq])
    and rescales each batch row's quaternion to unit length. Use for
    floating-base robot fixtures (a1_*, h1_*, talos_*) — without it, random
    `N(0, 1)` quaternions have norm ~2 and downstream rotation matrices have
    entries ~4× identity, which inflates analytical Jacobians by 10+ orders
    of magnitude. Defense correctness numbers reported on un-normalised
    random inputs look catastrophic in absolute terms even though relative
    error is still at fp64 unit roundoff. See defense notes for details.
    """
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    inputs = [torch.randn(N, fn_casadi.nnz_in(i), device="cuda",
                          dtype=torch.double, generator=g).contiguous()
              for i in range(fn_casadi.n_in())]
    if normalize_freeflyer_quat:
        if not inputs:
            raise ValueError("normalize_freeflyer_quat=True but function has no inputs")
        q = inputs[0]
        if q.shape[1] < 7:
            raise ValueError(
                f"normalize_freeflyer_quat=True but input 0 has only "
                f"{q.shape[1]} elements (need ≥7 for free-flyer quat at [3:7])")
        norms = q[:, 3:7].norm(dim=1, keepdim=True).clamp_min(1e-12)
        q[:, 3:7] = q[:, 3:7] / norms
    return inputs


# -----------------------------------------------------------------------------
# LP / Tiled LP convenience
# -----------------------------------------------------------------------------
def build_lp_fn(fn_casadi, N: int, max_smem_slots: int = 12000):
    """Return (LP_instance, plan, 'lp' | 'tiled')."""
    from cusadi.pytorch.lp_function import (CusadiLevelParallelFunction,
                                             CusadiTiledLevelParallelFunction)
    from cusadi.level_parallel.dag import extract_level_plan, extract_tiled_level_plan

    plan = extract_level_plan(fn_casadi)
    if plan.peak_live_slots <= max_smem_slots:
        lp = CusadiLevelParallelFunction(fn_casadi, N,
                                         lib_name="level_parallel_kernel")
        return lp, plan, "lp"
    tiled_plan = extract_tiled_level_plan(fn_casadi, max_smem_slots=max_smem_slots)
    lp = CusadiTiledLevelParallelFunction(fn_casadi, N, tiled_plan,
                                          lib_name="level_parallel_tiled_kernel")
    return lp, tiled_plan, "tiled"


def build_stage_lp_fn(f_stage, num_instances: int, n_stages: int,
                      input_specs, output_specs):
    """Return (StageLP_instance, plan).

    Stage compilation: f_stage's level plan is extracted once and replicated
    across the (num_instances × n_stages) launch grid. Per-arg `input_specs`
    and `output_specs` declare how each argument is indexed along the stage
    axis (STAGE / SHARED / COUPLED).

    Tiling on the per-stage program is not yet supported — caller should
    ensure peak_live_slots fits in shared memory (which is the typical case
    for MPC/OCP stage residuals; structural repetition is what makes stage
    compilation worthwhile in the first place).
    """
    from cusadi.pytorch.lp_stage import CusadiStageLevelParallelFunction
    from cusadi.level_parallel.dag import extract_level_plan

    plan = extract_level_plan(f_stage)
    lp = CusadiStageLevelParallelFunction(
        f_stage, num_instances, n_stages,
        input_specs=input_specs,
        output_specs=output_specs,
        lib_name="level_parallel_stage_kernel",
    )
    return lp, plan


def build_hybrid_stage_lp_fn(f_stage, num_instances: int, n_stages: int,
                              input_specs, output_specs,
                              chunk_size="auto",
                              smem_budget_bytes=None):
    """Return (HybridLP_instance, chunked_plan, chunk_size).

    Hybrid stage compilation. blockIdx.y indexes a chunk of `chunk_size`
    consecutive stages of f_stage; threads cooperate on the chunked DAG
    inside each block. chunk_size=1 collapses to pure stage-LP;
    chunk_size=n_stages is the unrolled-LP equivalent.

    chunk_size="auto" runs auto_tune_chunk_size, picking the largest
    divisor of n_stages whose chunked plan fits in shared memory.
    """
    from cusadi.pytorch.lp_stage import (CusadiHybridStageLevelParallelFunction,
                                           auto_tune_chunk_size)

    if chunk_size == "auto":
        chunk_size = auto_tune_chunk_size(
            f_stage, n_stages, input_specs,
            smem_budget_bytes=smem_budget_bytes,
        )

    lp = CusadiHybridStageLevelParallelFunction(
        f_stage, num_instances, n_stages,
        input_specs=input_specs,
        output_specs=output_specs,
        chunk_size=chunk_size,
        lib_name="level_parallel_chunk_kernel",
    )
    return lp, lp._plan, chunk_size


def build_smart_stage_lp_fn(f_stage, num_instances: int, n_stages: int,
                             input_specs, output_specs,
                             smem_budget_bytes=None,
                             verbose: bool = False):
    """Return (HybridLP_instance, chunked_plan, decision).

    Smart auto-dispatch entry point. Inspects f_stage and the device,
    then picks a chunk_size that should be near-optimal for the
    (problem, hardware) combination. Decision rules:

      1. f_stage > device smem  → raise (would need tiled-stage path)
      2. has any COUPLED input  → maximize c (operand reuse pays)
      3. tiny per-stage program → maximize c (amortise launch overhead)
      4. otherwise               → c=1 (heavy independent stages)

    The returned DispatchDecision carries the rationale (per-stage
    smem, device smem, has_coupled, auto_max_c, picked c, plain-text
    reason) so callers can log / surface it.
    """
    from cusadi.pytorch.lp_stage import (CusadiHybridStageLevelParallelFunction,
                                           CusadiTiledStageLevelParallelFunction,
                                           smart_pick_chunk_size)
    from cusadi.level_parallel.dag import extract_level_plan

    # Extract f_stage's plan ONCE here and pass it through; otherwise it's
    # re-extracted in smart_pick_chunk_size, again in auto_tune_chunk_size,
    # and again in the wrapper's __init__ — three full passes over a 78k-instr
    # function adds up to ~80 s on H1 jac.
    per_stage_plan = extract_level_plan(f_stage)

    decision = smart_pick_chunk_size(
        f_stage, n_stages, input_specs,
        smem_budget_bytes=smem_budget_bytes,
        per_stage_plan=per_stage_plan,
    )
    if verbose:
        print(f"[smart-dispatch] f_stage={f_stage.name()}, N={n_stages}: "
              f"mode={decision.mode}, chunk_size={decision.chunk_size}")
        print(f"[smart-dispatch] {decision.reason}")

    if decision.mode == "tiled-stage":
        lp = CusadiTiledStageLevelParallelFunction(
            f_stage, num_instances, n_stages,
            input_specs=input_specs,
            output_specs=output_specs,
            lib_name="level_parallel_stage_tiled_kernel",
        )
        return lp, lp._plan, decision

    lp = CusadiHybridStageLevelParallelFunction(
        f_stage, num_instances, n_stages,
        input_specs=input_specs,
        output_specs=output_specs,
        chunk_size=decision.chunk_size,
        lib_name="level_parallel_chunk_kernel",
        per_stage_plan=per_stage_plan,
    )
    return lp, lp._plan, decision


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------
def fmt_hz(hz: float) -> str:
    return f"{hz:.0f}" if np.isfinite(hz) else "N/A"


def fmt_us(us: float) -> str:
    if not np.isfinite(us):
        return "N/A"
    if us < 1000:
        return f"{us:.1f}"
    return f"{us:,.0f}"


def fmt_speedup(s: float) -> str:
    return f"{s:.2f}x" if np.isfinite(s) else "N/A"


def fmt_vs_paper(measured: float, expected: float, fmt=fmt_us) -> str:
    if expected <= 0:
        return f"{fmt(measured)}"
    pct = abs(measured - expected) / expected * 100.0
    tag = "ok" if pct <= 10 else ("WARN" if pct <= 25 else "FAIL")
    return f"{fmt(measured)} (paper={fmt(expected)}, {pct:.0f}% {tag})"


def print_header(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
