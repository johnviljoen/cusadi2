"""Plan files: the binary contract between cusadi2's Python front-end and the
C++ runtime (``cusadi2.h``).

    write_plan(fn, "f.plan")                       # batch-only, n_stages == 1
    write_plan(fn, "f.plan", n_stages=30,          # staged over a horizon
               in_mode=["coupled", "shared"], in_offset=[0, 0], in_span=[1, 0])

The file is one plan for one function at one horizon. Everything the runtime
needs is stored, so the C++ side computes nothing: block size, per-arg stage
counts, the flat level-parallel program, and the CasADi CCS sparsity of every
argument (from which the loader can derive COO). Constants are float64 always;
the runtime casts to its compiled scalar type on upload.

Layout (little-endian; i32 = int32, f64 = float64; arrays are contiguous):

    magic "CS2P", i32 version = 1
    i32 name_len, bytes name
    i32 n_in, n_out, n_stages, block_size, peak_live_slots
    per input  a: i32 rows, cols, nnz, mode, offset, span, count
                  i32 colptr[cols+1], i32 rowidx[nnz]
    per output a: i32 rows, cols, nnz, mode, count
                  i32 colptr[cols+1], i32 rowidx[nnz]
    i32 n_consts;      i32 const_slots[], f64 const_vals[]
    i32 n_input_maps;  i32 input_slots[], input_args[], input_elems[]
    i32 n_levels;      i32 level_offsets[], level_widths[]
    i32 n_nodes;       i32 node_ops[], node_outs[], node_in_a[], node_in_b[]
    i32 n_output_maps; i32 out_slots[], out_args[], out_elems[]

``write_reference`` writes a sibling self-check file (random inputs laid out as
the kernel reads them, plus CasADi's outputs) for the GPU test; it is test data,
not part of the plan contract:

    magic "CS2R", i32 version = 1
    i32 n_in, n_out, n_stages
    per input  a: i32 len, f64 data[len]     (count * nnz, batch of one)
    per output a: i32 len, f64 data[len]     (n_stages * nnz)
"""
import os
import struct

import casadi as ca
import numpy as np

from cusadi2.dag import (STAGE_MODES, LevelPlan, extract_level_plan, pick_block_size,
                         stage_count, validate_specs)

PLAN_MAGIC, REF_MAGIC, VERSION = b"CS2P", b"CS2R", 1


def _specs(fn, n_stages, in_mode, in_offset, in_span, out_mode):
    in_mode = in_mode or ["stage"] * fn.n_in()
    in_offset = in_offset or [0] * fn.n_in()
    in_span = in_span or [0] * fn.n_in()
    out_mode = out_mode or ["stage"] * fn.n_out()
    validate_specs(fn, n_stages, in_mode, in_offset, in_span, out_mode)
    return in_mode, in_offset, in_span, out_mode


class _W:
    def __init__(self, f):
        self.f = f

    def i(self, *vals):
        self.f.write(struct.pack("<%di" % len(vals), *vals))

    def ia(self, arr):
        self.f.write(np.ascontiguousarray(arr, dtype="<i4").tobytes())

    def fa(self, arr):
        self.f.write(np.ascontiguousarray(arr, dtype="<f8").tobytes())


def write_plan(fn: ca.Function, path: str, n_stages: int = 1, in_mode=None,
               in_offset=None, in_span=None, out_mode=None, block_size=None,
               plan: LevelPlan = None) -> str:
    """Extract `fn`'s level-parallel plan and write it, with the stage specs and
    horizon baked in, to `path`. Pass `plan` to reuse an already extracted one."""
    in_mode, in_offset, in_span, out_mode = _specs(fn, n_stages, in_mode, in_offset,
                                                   in_span, out_mode)
    plan = plan or extract_level_plan(fn)
    name = fn.name().encode()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        w = _W(f)
        f.write(PLAN_MAGIC)
        w.i(VERSION, len(name))
        f.write(name)
        w.i(fn.n_in(), fn.n_out(), n_stages, block_size or pick_block_size(plan),
            plan.peak_live_slots)
        for a in range(fn.n_in()):
            sp = fn.sparsity_in(a)
            w.i(sp.size1(), sp.size2(), sp.nnz(), STAGE_MODES.index(in_mode[a]),
                in_offset[a], in_span[a], stage_count(in_mode[a], in_span[a], n_stages))
            w.ia(sp.colind()); w.ia(sp.row())
        for a in range(fn.n_out()):
            sp = fn.sparsity_out(a)
            w.i(sp.size1(), sp.size2(), sp.nnz(), STAGE_MODES.index(out_mode[a]),
                stage_count(out_mode[a], 0, n_stages))
            w.ia(sp.colind()); w.ia(sp.row())
        w.i(len(plan.const_slots)); w.ia(plan.const_slots); w.fa(plan.const_vals)
        w.i(len(plan.input_slots)); w.ia(plan.input_slots); w.ia(plan.input_args); w.ia(plan.input_elems)
        w.i(plan.n_levels); w.ia(plan.level_offsets); w.ia(plan.level_widths)
        w.i(plan.n_nodes); w.ia(plan.node_ops); w.ia(plan.node_outs); w.ia(plan.node_in_a); w.ia(plan.node_in_b)
        w.i(len(plan.out_slots)); w.ia(plan.out_slots); w.ia(plan.out_args); w.ia(plan.out_elems)
    return path


def read_plan(path: str) -> dict:
    """Parse a plan file back into a dict mirroring the layout (for tests and
    inspection; the C++ loader in cusadi2.h is the deployment reader)."""
    with open(path, "rb") as f:
        buf = f.read()
    pos = 0

    def i(n=1):
        nonlocal pos
        vals = struct.unpack_from("<%di" % n, buf, pos)
        pos += 4 * n
        return vals[0] if n == 1 else list(vals)

    def ia(n):
        nonlocal pos
        arr = np.frombuffer(buf, "<i4", n, pos).tolist()
        pos += 4 * n
        return arr

    def fa(n):
        nonlocal pos
        arr = np.frombuffer(buf, "<f8", n, pos).tolist()
        pos += 8 * n
        return arr

    if buf[:4] != PLAN_MAGIC:
        raise ValueError(f"{path}: not a cusadi2 plan file")
    pos = 4
    version, name_len = i(), i()
    if version != VERSION:
        raise ValueError(f"{path}: plan version {version}, expected {VERSION}")
    name = buf[pos:pos + name_len].decode(); pos += name_len
    d = dict(name=name, version=version)
    d["n_in"], d["n_out"], d["n_stages"], d["block_size"], d["peak_live_slots"] = i(5)
    d["inputs"], d["outputs"] = [], []
    for _ in range(d["n_in"]):
        rows, cols, nnz, mode, offset, span, count = i(7)
        d["inputs"].append(dict(rows=rows, cols=cols, nnz=nnz, mode=STAGE_MODES[mode],
                                offset=offset, span=span, count=count,
                                colptr=ia(cols + 1), rowidx=ia(nnz)))
    for _ in range(d["n_out"]):
        rows, cols, nnz, mode, count = i(5)
        d["outputs"].append(dict(rows=rows, cols=cols, nnz=nnz, mode=STAGE_MODES[mode],
                                 count=count, colptr=ia(cols + 1), rowidx=ia(nnz)))
    n = i(); d["const_slots"] = ia(n); d["const_vals"] = fa(n)
    n = i(); d["input_slots"] = ia(n); d["input_args"] = ia(n); d["input_elems"] = ia(n)
    n = i(); d["level_offsets"] = ia(n); d["level_widths"] = ia(n)
    n = i(); d["node_ops"] = ia(n); d["node_outs"] = ia(n); d["node_in_a"] = ia(n); d["node_in_b"] = ia(n)
    n = i(); d["out_slots"] = ia(n); d["out_args"] = ia(n); d["out_elems"] = ia(n)
    if pos != len(buf):
        raise ValueError(f"{path}: {len(buf) - pos} trailing bytes")
    return d


def write_reference(fn: ca.Function, path: str, n_stages: int = 1, in_mode=None,
                    in_offset=None, in_span=None, seed: int = 0,
                    lo: float = -1.0, up: float = 1.0) -> str:
    """Write a self-check reference for `fn` under the same specs as its plan:
    random inputs (uniform in [lo, up], laid out as the kernel indexes them for
    a batch of one) and CasADi's outputs at every stage. Pick lo/up inside fn's
    domain; the reference is meaningless where fn is undefined."""
    in_mode, in_offset, in_span, _ = _specs(fn, n_stages, in_mode, in_offset, in_span, None)
    rng = np.random.default_rng(seed)
    nnz_in = [fn.nnz_in(a) for a in range(fn.n_in())]
    ref_in = [rng.uniform(lo, up, stage_count(in_mode[a], in_span[a], n_stages) * nnz_in[a])
              for a in range(fn.n_in())]
    ref_out = [[] for _ in range(fn.n_out())]
    for k in range(n_stages):
        args = []
        for a in range(fn.n_in()):
            s = 0 if in_mode[a] == "shared" else k + in_offset[a]
            args.append(ca.DM(ref_in[a][s * nnz_in[a]:(s + 1) * nnz_in[a]]))
        res = fn(*args)
        res = res if isinstance(res, (list, tuple)) else [res]
        for o, r in zip(ref_out, res):
            o.extend(np.array(ca.DM(r).nonzeros()).reshape(-1))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        w = _W(f)
        f.write(REF_MAGIC)
        w.i(VERSION, fn.n_in(), fn.n_out(), n_stages)
        for buf in ref_in:
            w.i(len(buf)); w.fa(buf)
        for buf in ref_out:
            w.i(len(buf)); w.fa(buf)
    return path
