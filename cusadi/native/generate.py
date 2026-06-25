"""Offline emitter for the native (host-free C++/CUDA) backend.

`generate_native_plan(fn, path)` extracts the level-parallel plan and writes it
as a C header of `static const` arrays plus a `cusadi::LPPlanHost` aggregate that
the header-only runtime (cusadi_native.h) uploads. Passing `input_specs` emits a
`cusadi::StagePlanHost` for the 2-axis (batch x horizon) stage kernel instead.

A reference (inputs + expected outputs, evaluated through CasADi) is embedded so
the runtime can self-check end-to-end. Build-time only; nothing here runs at
deployment.
"""
import os
import re

import casadi as ca
import numpy as np

from cusadi.level_parallel.dag import extract_level_plan
from cusadi.level_parallel.serialize import (flatten_plan, flatten_stage_specs,
                                             FlatStageSpecs)
from cusadi.level_parallel.stage import StageMode, StageSpec


def _carr(name, vals, dtype="int"):
    ct = "double" if dtype == "double" else "int"
    body = ", ".join((repr(float(v)) if dtype == "double" else str(int(v))) for v in vals) or "0"
    return f"static const {ct} {name}[] = {{{body}}};\n"


def _sanitize(name):
    return re.sub(r"\W", "_", name).upper()


def _reference(fn, fs: FlatStageSpecs, nnz_in, nnz_out, n_stages, rng):
    """Evaluate fn over `n_stages` knots; return per-arg ref input buffers (laid
    out exactly as the kernel indexes them) and the concatenated per-knot output.
    Works for the 1-axis case too (n_stages=1, all-STAGE)."""
    ref_in = [rng.uniform(-1.0, 1.0, fs.in_stage_count(a, n_stages) * nnz_in[a])
              for a in range(len(nnz_in))]
    ref_out = []
    for k in range(n_stages):
        args = []
        for a in range(len(nnz_in)):
            m = fs.in_mode[a]
            s = 0 if m == int(StageMode.SHARED) else (k + fs.in_offset[a] if m == int(StageMode.COUPLED) else k)
            args.append(ca.DM(ref_in[a][s * nnz_in[a]:(s + 1) * nnz_in[a]]))
        res = fn(*args)
        res = res if isinstance(res, (list, tuple)) else [res]
        for o in res:
            ref_out += list(np.array(ca.DM(o).nonzeros()).reshape(-1))
    return ref_in, ref_out


def generate_native_plan(fn, filepath, input_specs=None, output_specs=None,
                         prefix=None, n_ref=4):
    """Write a C plan header for `fn`. If `input_specs` (a list of StageSpec) is
    given, emit a StagePlanHost for the 2-axis stage kernel; else an LPPlanHost."""
    plan = extract_level_plan(fn)
    fp = flatten_plan(plan)
    P = prefix or _sanitize(fn.name())
    stage = input_specs is not None
    if stage:
        fs = flatten_stage_specs(input_specs, output_specs)
        n_stages = n_ref
    else:
        fs = FlatStageSpecs(in_mode=[0] * fp.n_args_in, in_offset=[0] * fp.n_args_in,
                            in_span=[0] * fp.n_args_in, out_mode=[0] * fp.n_args_out)
        n_stages = 1

    ref_in, ref_out = _reference(fn, fs, fp.nnz_in, fp.nnz_out, n_stages, np.random.default_rng(0))

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as o:
        o.write(f"// AUTO-GENERATED cusadi native plan for '{fn.name()}'. Do not edit.\n")
        o.write("#pragma once\n#include \"cusadi_native.h\"\n\n")

        def arr(suffix, vals, dt="int"):
            o.write(_carr(f"{P}_{suffix}", vals or [0], dt))

        arr("nnz_in", fp.nnz_in); arr("nnz_out", fp.nnz_out)
        arr("const_slots", fp.const_slots); arr("const_vals", fp.const_vals, "double")
        arr("input_slots", fp.input_slots); arr("input_args", fp.input_args); arr("input_elems", fp.input_elems)
        arr("level_offsets", fp.level_offsets); arr("level_widths", fp.level_widths)
        arr("node_ops", fp.node_ops); arr("node_outs", fp.node_outs)
        arr("node_in_a", fp.node_in_a); arr("node_in_b", fp.node_in_b)
        arr("out_slots", fp.out_slots); arr("out_args", fp.out_args); arr("out_elems", fp.out_elems)

        base = (f"  {fp.n_args_in}, {fp.n_args_out}, {P}_nnz_in, {P}_nnz_out,\n"
                f"  {fp.block_size}, {fp.shared_bytes},\n"
                f"  {fp.n_consts}, {P}_const_slots, {P}_const_vals,\n"
                f"  {fp.n_input_maps}, {P}_input_slots, {P}_input_args, {P}_input_elems,\n"
                f"  {fp.n_levels}, {P}_level_offsets, {P}_level_widths,\n"
                f"  {fp.n_nodes}, {P}_node_ops, {P}_node_outs, {P}_node_in_a, {P}_node_in_b,\n"
                f"  {fp.n_output_maps}, {P}_out_slots, {P}_out_args, {P}_out_elems")
        if stage:
            arr("in_mode", fs.in_mode); arr("in_offset", fs.in_offset)
            arr("in_span", fs.in_span); arr("out_mode", fs.out_mode)
            o.write(f"\nstatic const cusadi::StagePlanHost {P}_plan = {{\n  {{\n{base}\n  }},\n"
                    f"  {P}_in_mode, {P}_in_offset, {P}_in_span, {P}_out_mode\n}};\n")
        else:
            o.write(f"\nstatic const cusadi::LPPlanHost {P}_plan = {{\n{base}\n}};\n")

        # --- embedded reference for self-check ---
        o.write(f"\nstatic const int {P}_ref_nstages = {n_stages};\n")
        ptrs = []
        for a, buf in enumerate(ref_in):
            arr(f"ref_in_{a}", list(buf), "double"); ptrs.append(f"{P}_ref_in_{a}")
        o.write(f"static const double* const {P}_ref_in[] = {{{', '.join(ptrs)}}};\n")
        arr("ref_out", ref_out, "double")
        o.write(f"static const int {P}_ref_nout = {len(ref_out)};\n")
    return filepath
