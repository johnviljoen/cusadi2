"""Plan files: the binary contract between cusadi2's Python front-end and the
C++ runtime (``cusadi2.h``).

    write_plan(fn, "f.plan")                       # batch-only, n_stages == 1
    write_plan(fn, "f.plan", n_stages=30,          # staged over a horizon
               in_mode=["coupled", "shared"], in_offset=[0, 0], in_span=[1, 0])

The file is one plan for one function at one horizon. Everything the runtime
needs is stored, so the C++ side computes nothing but the block size (0 here
unless the caller overrides it; ``cusadi2.h:pick_block_size`` chooses it at
upload from the level widths and the device's occupancy): per-arg stage
counts, the flat level-parallel program, CasADi's own instruction stream for
the serial kernel, and the CasADi CCS sparsity of every argument (from which
the loader can derive COO). Constants are float64 always; the runtime casts to
its compiled scalar type on upload.

Layout (little-endian; i32 = int32, f64 = float64; arrays are contiguous):

    magic "CS2P", i32 version = 4
    i32 name_len, bytes name
    i32 n_in, n_out, n_stages, block_size, peak_live_slots
    per declared input  a: i32 rows, cols, nnz, mode, offset, span, count
                           i32 colptr[cols+1], i32 rowidx[nnz]
    per declared output a: i32 rows, cols, nnz, mode, count
                           i32 colptr[cols+1], i32 rowidx[nnz]
    i32 n_stage_in;    per function input  a: i32 src (-1: stage_consts), nnz;  i32 table[n_stages * nnz]
    i32 n_stage_out;   per function output k: i32 dst, nnz;  i32 table[n_stages * nnz]
    i32 n_reduce;      per reduce: i32 dst, G; i32 segptr[G+1]; i32 n_seg; i32 segidx[n_seg]
    i32 scratch_per_instance
    i32 n_stage_consts; f64 stage_consts[]
    i32 n_consts;      i32 const_slots[], f64 const_vals[]
    i32 n_levels;      i32 level_offsets[], level_widths[]
    i32 n_nodes;       i32 node_ops[], node_outs[], node_in_a[], node_in_b[]
    i32 sz_w
    i32 n_instr;       i32 serial_ops[], serial_outs[], serial_in_a[], serial_in_b[]
    i32 n_serial_consts; f64 serial_const_vals[]

The declared arguments are the buffers the caller passes; the stage tables say,
per (stage, element), where each function argument reads inside the declared
input's instance block (-1: zero) and where each function output writes inside
the declared output's instance block (-1: discard; dst -1: the scratch stream
that the reduces sum, see ``cusadi2.compose``). Plain stage specs are the
special case where each function argument is its own declared buffer
(``compose.legacy_composition``); a version-3 file (before the tables) is read
back as that case.

The level program (node arrays) is SSA-renamed and hazard-free per level, with
INPUT / OUTPUT as nodes (slot, argument, element) and constants preloaded; the
serial program is CasADi's stream verbatim on CasADi's `sz_w` work slots with
CONST / INPUT / OUTPUT inline (see ``dag.SerialProgram``). Each kernel reads
only its own program.

``write_reference`` writes a sibling self-check file (random inputs laid out as
the kernel reads them, plus CasADi's outputs) for the GPU test; it is test data,
not part of the plan contract:

    magic "CS2R", i32 version = 4        (the plan VERSION; readers skip it)
    i32 n_in, n_out, n_stages
    per declared input  a: i32 len, f64 data[len]     (batch of one)
    per declared output a: i32 len, f64 data[len]
"""
import io
import os
import struct

import casadi as ca
import numpy as np

from cusadi2.compose import Composition, DeclaredArg, Reduce, StageMap, evaluate_composition, legacy_composition
from cusadi2.dag import (STAGE_MODES, LevelPlan, SerialProgram, extract_level_plan, extract_serial_program,
                         validate_specs)

PLAN_MAGIC, REF_MAGIC, VERSION = b"CS2P", b"CS2R", 4


def _specs(fn, n_stages, in_mode, in_offset, in_span, out_mode):
    """The stage specs with defaults filled in and validated (the unrolled emitter
    still works from them)."""
    in_mode = in_mode or ["stage"] * fn.n_in()
    in_offset = in_offset or [0] * fn.n_in()
    in_span = in_span or [0] * fn.n_in()
    out_mode = out_mode or ["stage"] * fn.n_out()
    validate_specs(fn, n_stages, in_mode, in_offset, in_span, out_mode)
    return in_mode, in_offset, in_span, out_mode


def _composition(fn, n_stages, in_mode, in_offset, in_span, out_mode, composition):
    if composition is None:
        return legacy_composition(fn, n_stages, in_mode, in_offset, in_span, out_mode)
    if any(v is not None for v in (in_mode, in_offset, in_span, out_mode)) or n_stages not in (1, composition.n_stages):
        raise ValueError("write_plan: give either stage specs or a composition, not both")
    composition.validate(fn)
    return composition


class _W:
    def __init__(self, f):
        self.f = f

    def i(self, *vals):
        self.f.write(struct.pack("<%di" % len(vals), *vals))

    def ia(self, arr):
        self.f.write(np.ascontiguousarray(arr, dtype="<i4").tobytes())

    def fa(self, arr):
        self.f.write(np.ascontiguousarray(arr, dtype="<f8").tobytes())


def device_max_live(dtype: str = "double", device: int = 0) -> int:
    """Largest `peak_live_slots` of `dtype` the device's per-block opt-in shared
    memory holds (CUDA runtime queried through ctypes)."""
    import ctypes
    import ctypes.util
    cudart = None
    for name in [ctypes.util.find_library("cudart"), "libcudart.so", "libcudart.so.13", "libcudart.so.12"]:
        if name:
            try:
                cudart = ctypes.CDLL(name)
                break
            except OSError:
                pass
    if cudart is None:   # pip CUDA (nvidia-cuda-runtime): versioned .so only, under site-packages/nvidia
        import glob
        from importlib.util import find_spec
        spec = find_spec("nvidia")
        for root in (spec.submodule_search_locations or []) if spec else []:
            for lib in sorted(glob.glob(os.path.join(root, "*", "lib", "libcudart.so*"))):
                try:
                    cudart = ctypes.CDLL(lib)
                    break
                except OSError:
                    pass
            if cudart:
                break
    if cudart is None:
        raise RuntimeError("cusadi2: libcudart not found (system CUDA or the nvidia-cuda-runtime wheel)")
    v = ctypes.c_int()
    if cudart.cudaDeviceGetAttribute(ctypes.byref(v), 97, device):   # cudaDevAttrMaxSharedMemoryPerBlockOptin
        raise RuntimeError(f"cudaDeviceGetAttribute failed for device {device}")
    return v.value // {"double": 8, "float": 4}[dtype]


def plan_bytes(fn: ca.Function, n_stages: int = 1, in_mode=None, in_offset=None,
               in_span=None, out_mode=None, block_size=None, max_live=None,
               plan: LevelPlan = None, serial: SerialProgram = None,
               composition: Composition = None) -> bytes:
    """The plan file for `fn` as bytes (same format as `write_plan`), for consumers
    that upload plans without touching the filesystem (cusadi2.jax)."""
    buf = io.BytesIO()
    _write_plan_stream(buf, fn, n_stages, in_mode, in_offset, in_span, out_mode, block_size,
                       max_live, plan, serial, composition)
    return buf.getvalue()


def write_plan(fn: ca.Function, path: str, n_stages: int = 1, in_mode=None,
               in_offset=None, in_span=None, out_mode=None, block_size=None,
               max_live=None, plan: LevelPlan = None, serial: SerialProgram = None,
               composition: Composition = None) -> str:
    """Extract `fn`'s level-parallel plan and serial program and write them, with
    the stage specs and horizon baked in, to `path`. `max_live` (slots; see
    `device_max_live`) makes the planner window the levels until the plan fits.
    Pass `plan` / `serial` to reuse already extracted ones. A `composition`
    (`cusadi2.compose`) replaces the stage specs: `fn` runs per stage behind
    the composition's declared arguments."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        _write_plan_stream(f, fn, n_stages, in_mode, in_offset, in_span, out_mode, block_size,
                           max_live, plan, serial, composition)
    return path


def _write_plan_stream(f, fn, n_stages, in_mode, in_offset, in_span, out_mode, block_size,
                       max_live, plan, serial, composition=None):
    comp = _composition(fn, n_stages, in_mode, in_offset, in_span, out_mode, composition)
    plan = plan or extract_level_plan(fn, max_live=max_live)
    serial = serial or extract_serial_program(fn)
    name = fn.name().encode()
    if True:
        w = _W(f)
        f.write(PLAN_MAGIC)
        w.i(VERSION, len(name))
        f.write(name)
        w.i(len(comp.inputs), len(comp.outputs), comp.n_stages, block_size or 0, plan.peak_live_slots)
        for d in comp.inputs:
            w.i(d.rows, d.cols, d.nnz, STAGE_MODES.index(d.mode), d.offset, d.span, d.count)
            w.ia(d.colptr); w.ia(d.rowidx)
        for d in comp.outputs:
            w.i(d.rows, d.cols, d.nnz, STAGE_MODES.index(d.mode), d.count)
            w.ia(d.colptr); w.ia(d.rowidx)
        w.i(len(comp.stage_in))
        for m in comp.stage_in:
            w.i(m.arg, m.nnz); w.ia(m.table.reshape(-1))
        w.i(len(comp.stage_out))
        for m in comp.stage_out:
            w.i(m.arg, m.nnz); w.ia(m.table.reshape(-1))
        w.i(len(comp.reduces))
        for r in comp.reduces:
            w.i(r.dst, len(r.segptr) - 1); w.ia(r.segptr); w.i(len(r.segidx)); w.ia(r.segidx)
        w.i(comp.scratch_per_instance)
        w.i(len(comp.stage_consts)); w.fa(comp.stage_consts)
        w.i(len(plan.const_slots)); w.ia(plan.const_slots); w.fa(plan.const_vals)
        w.i(plan.n_levels); w.ia(plan.level_offsets); w.ia(plan.level_widths)
        w.i(plan.n_nodes); w.ia(plan.node_ops); w.ia(plan.node_outs); w.ia(plan.node_in_a); w.ia(plan.node_in_b)
        w.i(serial.sz_w)
        w.i(serial.n_instr); w.ia(serial.ops); w.ia(serial.outs); w.ia(serial.in_a); w.ia(serial.in_b)
        w.i(len(serial.const_vals)); w.fa(serial.const_vals)


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
    if version not in (3, VERSION):
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
    N = d["n_stages"]
    if version >= 4:
        d["stage_in"], d["stage_out"], d["reduces"] = [], [], []
        for _ in range(i()):
            src, nnz = i(2)
            d["stage_in"].append(dict(arg=src, nnz=nnz, table=np.array(ia(N * nnz), np.int32).reshape(N, nnz)))
        for _ in range(i()):
            dst, nnz = i(2)
            d["stage_out"].append(dict(arg=dst, nnz=nnz, table=np.array(ia(N * nnz), np.int32).reshape(N, nnz)))
        for _ in range(i()):
            dst, G = i(2)
            segptr = ia(G + 1)
            d["reduces"].append(dict(dst=dst, segptr=segptr, segidx=ia(i())))
        d["scratch_per_instance"] = i()
        d["stage_consts"] = fa(i())
    else:   # the stage specs as tables (what the C++ loader synthesises too)
        d["stage_in"] = [dict(arg=a, nnz=x["nnz"], table=(np.tile(np.arange(x["nnz"]), (N, 1)) if x["mode"] == "shared"
                              else (np.arange(N)[:, None] + x["offset"]) * x["nnz"] + np.arange(x["nnz"])[None, :]).astype(np.int32))
                         for a, x in enumerate(d["inputs"])]
        d["stage_out"] = [dict(arg=k, nnz=x["nnz"], table=(np.arange(N)[:, None] * x["nnz"] + np.arange(x["nnz"])[None, :]).astype(np.int32))
                          for k, x in enumerate(d["outputs"])]
        d["reduces"], d["scratch_per_instance"], d["stage_consts"] = [], 0, []
    n = i(); d["const_slots"] = ia(n); d["const_vals"] = fa(n)
    n = i(); d["level_offsets"] = ia(n); d["level_widths"] = ia(n)
    n = i(); d["node_ops"] = ia(n); d["node_outs"] = ia(n); d["node_in_a"] = ia(n); d["node_in_b"] = ia(n)
    d["sz_w"] = i()
    n = i(); d["serial_ops"] = ia(n); d["serial_outs"] = ia(n); d["serial_in_a"] = ia(n); d["serial_in_b"] = ia(n)
    n = i(); d["serial_const_vals"] = fa(n)
    if pos != len(buf):
        raise ValueError(f"{path}: {len(buf) - pos} trailing bytes")
    return d


def write_reference(fn: ca.Function, path: str, n_stages: int = 1, in_mode=None,
                    in_offset=None, in_span=None, seed: int = 0,
                    lo: float = -1.0, up: float = 1.0, composition: Composition = None) -> str:
    """Write a self-check reference for `fn` under the same specs (or composition)
    as its plan: random declared inputs (uniform in [lo, up], laid out as the
    kernel indexes them for a batch of one) and the declared outputs evaluated
    on the host (`compose.evaluate_composition`). Pick lo/up inside fn's domain;
    the reference is meaningless where fn is undefined."""
    comp = _composition(fn, n_stages, in_mode, in_offset, in_span, None, composition)
    rng = np.random.default_rng(seed)
    ref_in = [rng.uniform(lo, up, d.length) for d in comp.inputs]
    ref_out = evaluate_composition(fn, comp, ref_in)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        w = _W(f)
        f.write(REF_MAGIC)
        w.i(VERSION, len(comp.inputs), len(comp.outputs), comp.n_stages)
        for buf in ref_in:
            w.i(len(buf)); w.fa(buf)
        for buf in ref_out:
            w.i(len(buf)); w.fa(buf)
    return path
