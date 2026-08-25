"""Offline emitter for the native (host-free C++/CUDA) backend.

``generate_native_plan(fn, path)`` extracts the level-parallel plan and writes it
as a C header of ``static const`` arrays plus a ``cusadi2::StagePlanHost`` aggregate
that the header-only runtime (``cusadi2.h``) uploads:

    #include "cusadi_native.h"
    #include "myfn_plan.h"             // generated: provides MYFN_plan
    cusadi2::StageProgram prog; prog.upload(MYFN_plan);
    prog.evaluate(d_in, d_out, batch, n_stages);

Three axes of parallelism, and Stage wraps (does not replace) the level axis:
  * level  — within a block: the DAG ops run one topological level at a time with
             a __syncthreads() between levels. This is the LevelPlan node arrays
             and is always present — it is the compute itself.
  * batch  — grid axis: many instances.
  * stage  — grid axis: the horizon. One block per (instance, stage).

So the grid is (batch, n_stages) and every block runs the full level-parallel
evaluation. ``StagePlanHost`` is exactly ``{ LPPlanHost base; + stage-spec
arrays }``: the base *is* the complete level-parallel program; in_mode/in_offset/
in_span/out_mode layer the stage axis on top. Plain batch-only LP is the special
case ``n_stages == 1`` with every arg "stage" — what ``generate_native_plan(fn,
path)`` (no specs) emits. The only true runtime branch is tiled vs non-tiled
(deferred until the tiling scheduler lands).

Stage modes are plain strings ("stage"/"shared"/"coupled"); see ``dag.py``.
``generate_reference(fn, path)`` writes a separate self-check header (random
inputs + their CasADi-evaluated outputs over n_stages knots) for tests that want
to verify the runtime against CasADi ground truth; it is not part of the plan.
Build-time only; nothing here runs at deployment. The plan is already the flat
wire format (see ``dag.LevelPlan``) — there is no separate serializer.
"""
import os
import re

import casadi as ca
import numpy as np

from cusadi2.dag import (STAGE_MODES, extract_level_plan, pick_block_size,
                         stage_count, validate_specs)


def _clit(v):
    """C literal for a float. Non-finite values (a self-check reference sampled
    outside fn's domain) map to the <cmath> macros; bare repr() would emit the
    invalid tokens `nan`/`inf`/`-inf` and the header would not compile."""
    v = float(v)
    if v != v:
        return "NAN"
    if v == float("inf"):
        return "INFINITY"
    if v == float("-inf"):
        return "(-INFINITY)"
    return repr(v)


def _carr(name, vals, ctype="int"):
    is_real = ctype in ("double", "float")
    body = ", ".join((_clit(v) if is_real else str(int(v))) for v in vals) or "0"
    return f"static const {ctype} {name}[] = {{{body}}};\n"


def _sanitize(name):
    return re.sub(r"\W", "_", name).upper()


def _reference(fn, in_mode, in_offset, in_span, nnz_in, n_stages, rng, lo, up):
    """Evaluate fn over `n_stages` knots; return per-arg input buffers (laid out
    exactly as the kernel indexes them along the stage axis) and the concatenated
    per-knot output nonzeros. Reduces to one evaluation when n_stages == 1."""
    ref_in = [rng.uniform(lo, up, stage_count(in_mode[a], in_span[a], n_stages) * nnz_in[a])
              for a in range(len(nnz_in))]
    ref_out = []
    for k in range(n_stages):
        args = []
        for a in range(len(nnz_in)):
            m = in_mode[a]
            idx = (0 if m == "shared"
                   else k + in_offset[a] if m == "coupled" else k)
            args.append(ca.DM(ref_in[a][idx * nnz_in[a]:(idx + 1) * nnz_in[a]]))
        res = fn(*args)
        res = res if isinstance(res, (list, tuple)) else [res]
        for o in res:
            ref_out += list(np.array(ca.DM(o).nonzeros()).reshape(-1))
    return ref_in, ref_out


def generate_native_plan(fn, filepath, in_mode=None, in_offset=None, in_span=None,
                         out_mode=None, prefix=None, dtype="double"):
    """Write a C plan header (``cusadi2::StagePlanHost``) for `fn`.

    Stage specs are parallel lists, one entry per arg: `in_mode` (each "stage" /
    "shared" / "coupled"), `in_offset`, `in_span`, and `out_mode`. When `in_mode`
    is omitted every arg defaults to "stage" — the plain batch-only level-parallel
    case.

    `dtype` ("double" or "float") sets the C scalar type of the constant arrays;
    it must match the kernel's dtype and the CUSADI2_SCALAR the runtime is
    compiled with.

    The header holds the plan only; self-check test data is a separate artifact
    (see `generate_reference`).
    """
    plan = extract_level_plan(fn)
    P = prefix or _sanitize(fn.name())

    staged = in_mode is not None
    if in_mode is None:
        in_mode = ["stage"] * fn.n_in()
    if in_offset is None:
        in_offset = [0] * fn.n_in()
    if in_span is None:
        in_span = [0] * fn.n_in()
    if out_mode is None:
        out_mode = ["stage"] * fn.n_out()
    # the plan is horizon-independent (n_stages is a runtime evaluate() arg);
    # the coupled offset/span check is too, so any representative horizon works.
    validate_specs(fn, 2 if staged else 1, in_mode, in_offset, in_span, out_mode)

    # string modes -> kernel wire value (position in STAGE_MODES)
    in_mode_id = [STAGE_MODES.index(m) for m in in_mode]
    out_mode_id = [STAGE_MODES.index(m) for m in out_mode]

    n_args_in = len(plan.nnz_in)
    n_args_out = len(plan.nnz_out)
    shared_bytes = plan.peak_live_slots * (4 if dtype == "float" else 8)
    block_size = pick_block_size(plan)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as o:
        o.write(f"// AUTO-GENERATED cusadi native plan for '{fn.name()}'. Do not edit.\n")
        o.write("#pragma once\n#include <cmath>\n#include \"cusadi2.h\"\n\n")

        def arr(suffix, vals, dt="int"):
            o.write(_carr(f"{P}_{suffix}", vals or [0], dt))

        arr("nnz_in", plan.nnz_in); arr("nnz_out", plan.nnz_out)
        arr("const_slots", plan.const_slots); arr("const_vals", plan.const_vals, dtype)
        arr("input_slots", plan.input_slots); arr("input_args", plan.input_args); arr("input_elems", plan.input_elems)
        arr("level_offsets", plan.level_offsets); arr("level_widths", plan.level_widths)
        arr("node_ops", plan.node_ops); arr("node_outs", plan.node_outs)
        arr("node_in_a", plan.node_in_a); arr("node_in_b", plan.node_in_b)
        arr("out_slots", plan.out_slots); arr("out_args", plan.out_args); arr("out_elems", plan.out_elems)
        arr("in_mode", in_mode_id); arr("in_offset", in_offset)
        arr("in_span", in_span); arr("out_mode", out_mode_id)

        # base = the complete level-parallel program (LPPlanHost); the stage-spec
        # arrays below wrap it with the horizon axis.
        base = (f"  {n_args_in}, {n_args_out}, {P}_nnz_in, {P}_nnz_out,\n"
                f"  {block_size}, {shared_bytes},\n"
                f"  {len(plan.const_slots)}, {P}_const_slots, {P}_const_vals,\n"
                f"  {len(plan.input_slots)}, {P}_input_slots, {P}_input_args, {P}_input_elems,\n"
                f"  {len(plan.level_offsets)}, {P}_level_offsets, {P}_level_widths,\n"
                f"  {len(plan.node_ops)}, {P}_node_ops, {P}_node_outs, {P}_node_in_a, {P}_node_in_b,\n"
                f"  {len(plan.out_slots)}, {P}_out_slots, {P}_out_args, {P}_out_elems")
        o.write(f"\nstatic const cusadi2::StagePlanHost {P}_plan = {{\n  {{\n{base}\n  }},\n"
                f"  {P}_in_mode, {P}_in_offset, {P}_in_span, {P}_out_mode\n}};\n")

        # per-output sparsity (CasADi CCS): the kernel emits each output as its
        # flat nonzeros in this order; consumers need the pattern to interpret
        # them (same convention as CasADi codegen's casadi_s* arrays).
        o.write("\n// per-output sparsity patterns (CasADi CCS)\n")
        for a in range(fn.n_out()):
            sp = fn.sparsity_out(a)
            arr(f"out{a}_colptr", list(sp.colind()))
            arr(f"out{a}_rowidx", list(sp.row()))
    return filepath


def generate_reference(fn, filepath, in_mode=None, in_offset=None, in_span=None,
                       prefix=None, n_ref=4, dtype="double", lo=-1.0, up=1.0):
    """Write a standalone self-check reference header for `fn`: random inputs
    (uniform in [lo, up], fixed seed) plus their CasADi-evaluated outputs, as
    ``{P}_ref_nstages`` / ``{P}_ref_in*`` / ``{P}_ref_out`` / ``{P}_ref_nout``.

    Companion to `generate_native_plan` — pass the same stage specs, prefix and
    dtype. This is test fixture data (see examples/native/selfcheck.cu), kept
    out of the plan wire format on purpose. Pick `lo`/`up` inside fn's domain:
    the reference is meaningless (NaN) where fn is undefined. Staged specs use
    `n_ref` knots; the plain batch case uses one.
    """
    P = prefix or _sanitize(fn.name())
    staged = in_mode is not None
    if in_mode is None:
        in_mode = ["stage"] * fn.n_in()
    if in_offset is None:
        in_offset = [0] * fn.n_in()
    if in_span is None:
        in_span = [0] * fn.n_in()
    n_stages = n_ref if staged else 1
    nnz_in = [fn.nnz_in(a) for a in range(fn.n_in())]
    ref_in, ref_out = _reference(fn, in_mode, in_offset, in_span, nnz_in,
                                 n_stages, np.random.default_rng(0), lo, up)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as o:
        o.write(f"// AUTO-GENERATED cusadi self-check reference for '{fn.name()}'. Do not edit.\n")
        o.write("#pragma once\n#include <cmath>\n\n")
        o.write(f"static const int {P}_ref_nstages = {n_stages};\n")
        ptrs = []
        for a, buf in enumerate(ref_in):
            o.write(_carr(f"{P}_ref_in_{a}", list(buf) or [0], dtype))
            ptrs.append(f"{P}_ref_in_{a}")
        o.write(f"static const {dtype}* const {P}_ref_in[] = {{{', '.join(ptrs)}}};\n")
        o.write(_carr(f"{P}_ref_out", ref_out or [0], dtype))
        o.write(f"static const int {P}_ref_nout = {len(ref_out)};\n")
    return filepath


if __name__ == "__main__":
    from cusadi2.dag import _demo_function
    out = generate_native_plan(_demo_function(), "demo_plan.h")  # LP = all "stage", n_stages=1
    print("wrote", out)
