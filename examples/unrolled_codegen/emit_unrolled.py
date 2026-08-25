"""Resurrected fully-unrolled level-parallel codegen.

This is the design cusadi2's interpreter replaced: commit 2b352bd in the old
cusadi repo ("Phase 2: level-parallel CUDA codegen"), abandoned in e7ea152
("35k-line .cu files with 61-minute compile times"). It is re-emitted here from
the *current* cusadi2 LevelPlan — either a live ``extract_level_plan(fn)`` or a
parsed ``generate_native_plan`` header — so the comparison against the
interpreter is controlled: identical SSA slot assignment, identical level
schedule, identical block size and shared-memory footprint. The ONLY difference
is dispatch — each node is a hardcoded ``case`` with its opcode and operand
slots burned into the instruction stream, instead of read from node arrays at
runtime.

One deliberate improvement over 2b352bd (to avoid strawmanning codegen): the
old emitter used a two-phase read/sync/write pattern per level. The current
slot allocator guarantees no slot is both read and written within a level (the
same guarantee the interpreter's single-sync fast path relies on), so we emit
single-phase nodes and one __syncthreads() per level — the same sync count as
the interpreter.

Emitted extern "C" API (fixed names; one .cu per function, kernel is
``unrolled_kernel`` to avoid clashing with the interpreter's
``evaluate_kernel`` at link time):

    float unr_evaluate(const T* const inputs[], T* outputs[], int B);  // seconds
    int   unr_configure();          // raise dynamic smem cap if needed
"""

import os
import re
from types import SimpleNamespace

from cusadi2.cusadi_operations import OP_EXPR_DOUBLE, OP_EXPR_FLOAT


def _clit(v, dtype):
    v = float(v)
    if v != v:
        return "NAN"
    if v == float("inf"):
        return "INFINITY"
    if v == float("-inf"):
        return "(-INFINITY)"
    return repr(v) + ("f" if dtype == "float" else "")


def parse_plan_header(path, prefix):
    """Reconstruct the LevelPlan arrays + block/smem config from a
    ``generate_native_plan`` header, so the unrolled kernel can be emitted from
    the exact plan an existing interpreter build uses (no CasADi/Pinocchio
    needed)."""
    text = open(path).read()
    arrays = {}
    for ctype, name, body in re.findall(
            r"static const (int|double|float) (\w+)\[\] = \{([^}]*)\};", text):
        vals = [v.strip() for v in body.split(",") if v.strip()]
        arrays[name] = ([int(v) for v in vals] if ctype == "int"
                        else [float(v.replace("NAN", "nan")
                                     .replace("(-INFINITY)", "-inf")
                                     .replace("INFINITY", "inf")) for v in vals])

    def a(suffix):
        return arrays[f"{prefix}_{suffix}"]

    m = re.search(r"StagePlanHost \w+ = \{\s*\{\s*\d+, \d+, [^\n]*\n\s*(\d+), (\d+),",
                  text)
    assert m, f"could not parse block_size/shared_bytes from {path}"
    block_size, shared_bytes = int(m.group(1)), int(m.group(2))

    # generate_native_plan pads empty arrays to [0]; trim input/const/out maps
    # to their true lengths from the struct's count fields
    def count(field):
        cm = re.search(r"(\d+), \w+_%s" % field, text)
        assert cm, f"could not parse count for {field} from {path}"
        return int(cm.group(1))

    n_consts, n_inputs, n_levels, n_nodes, n_outputs = (
        count("const_slots"), count("input_slots"), count("level_offsets"),
        count("node_ops"), count("out_slots"))

    return SimpleNamespace(
        nnz_in=a("nnz_in"), nnz_out=a("nnz_out"),
        const_slots=a("const_slots")[:n_consts], const_vals=a("const_vals")[:n_consts],
        input_slots=a("input_slots")[:n_inputs], input_args=a("input_args")[:n_inputs],
        input_elems=a("input_elems")[:n_inputs],
        level_offsets=a("level_offsets")[:n_levels], level_widths=a("level_widths")[:n_levels],
        node_ops=a("node_ops")[:n_nodes], node_outs=a("node_outs")[:n_nodes],
        node_in_a=a("node_in_a")[:n_nodes], node_in_b=a("node_in_b")[:n_nodes],
        out_slots=a("out_slots")[:n_outputs], out_args=a("out_args")[:n_outputs],
        out_elems=a("out_elems")[:n_outputs],
    ), block_size, shared_bytes


def write_unrolled_source(plan, name, block_size, shared_bytes, filepath,
                          dtype="double"):
    """Emit a fully-unrolled level-parallel .cu from LevelPlan-shaped arrays.

    Returns (filepath, stats dict)."""
    exprs = OP_EXPR_FLOAT if dtype == "float" else OP_EXPR_DOUBLE
    bs = block_size
    nnz_in, nnz_out = plan.nnz_in, plan.nnz_out
    n_nodes = len(plan.node_ops)

    def stmt(g):
        e = (exprs[plan.node_ops[g]]
             .replace("{0}", "w[%d]" % plan.node_in_a[g])
             .replace("{1}", "w[%d]" % plan.node_in_b[g]))
        return "w[%d] = %s;" % (plan.node_outs[g], e)

    L = []
    emit = L.append
    emit("// AUTOMATICALLY GENERATED: fully-unrolled level-parallel kernel")
    emit("// (resurrected cusadi 2b352bd design, re-emitted from cusadi2 LevelPlan)")
    emit("// function: %s   nodes: %d   levels: %d   block: %d   smem: %dB"
         % (name, n_nodes, len(plan.level_offsets), bs, shared_bytes))
    emit("#include <cuda_runtime.h>")
    emit("#include <math.h>")
    emit("#include <cstdio>")
    emit("")
    emit("#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }")
    emit("__host__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {")
    emit('    if (code != cudaSuccess) {')
    emit('        fprintf(stderr,"GPUassert: %s %s %d\\n", cudaGetErrorString(code), file, line);')
    emit('        if (abort) exit(code);')
    emit('    }')
    emit('}')
    emit("")
    emit("__global__ void unrolled_kernel(")
    emit("        const %s * const * __restrict__ inputs," % dtype)
    emit("        %s ** __restrict__ outputs," % dtype)
    emit("        const int total_instances) {")
    emit("")
    emit("    extern __shared__ %s w[];" % dtype)
    emit("    const int inst = blockIdx.x;")
    emit("    const int tid  = threadIdx.x;")
    emit("    if (inst >= total_instances) return;")
    emit("")

    # --- Init: constants + inputs, chunked over the block, hardcoded ---------
    init = ["w[%d] = %s;" % (plan.const_slots[i], _clit(plan.const_vals[i], dtype))
            for i in range(len(plan.const_slots))]
    init += ["w[%d] = inputs[%d][(long long)inst * %d + %d];"
             % (plan.input_slots[i], plan.input_args[i],
                nnz_in[plan.input_args[i]], plan.input_elems[i])
             for i in range(len(plan.input_slots))]
    emit("    // === Init: %d slots (consts + inputs) ===" % len(init))
    for c0 in range(0, len(init), bs):
        for i, s in enumerate(init[c0:c0 + bs]):
            emit("    if (tid == %d) %s" % (i, s))
    emit("    __syncthreads();")
    emit("")

    # --- Compute: one switch(tid) per block-size chunk, one sync per level ---
    for lev in range(len(plan.level_offsets)):
        off, width = plan.level_offsets[lev], plan.level_widths[lev]
        emit("    // level %d: %d nodes" % (lev, width))
        for base in range(0, width, bs):
            cw = min(bs, width - base)
            gs = range(off + base, off + base + cw)
            if cw <= 4:
                for i, g in enumerate(gs):
                    emit("    if (tid == %d) { %s }" % (i, stmt(g)))
            else:
                emit("    switch (tid) {")
                for i, g in enumerate(gs):
                    emit("    case %d: %s break;" % (i, stmt(g)))
                emit("    }")
        emit("    __syncthreads();")
    emit("")

    # --- Outputs -------------------------------------------------------------
    outs = list(zip(plan.out_slots, plan.out_args, plan.out_elems))
    emit("    // === Outputs: %d ===" % len(outs))
    for c0 in range(0, len(outs), bs):
        for i, (slot, arg, elem) in enumerate(outs[c0:c0 + bs]):
            emit("    if (tid == %d) outputs[%d][(long long)inst * %d + %d] = w[%d];"
                 % (i, arg, nnz_out[arg], elem, slot))
    emit("}")
    emit("")

    # --- extern "C" wrappers (mirror the interpreter's evaluate_cusadi) ------
    emit('extern "C" float unr_evaluate(')
    emit("        const %s * const inputs[]," % dtype)
    emit("        %s * outputs[]," % dtype)
    emit("        const int total_instances) {")
    emit("    dim3 grid(total_instances);")
    emit("    dim3 block(%d);" % bs)
    emit("    float milliseconds;")
    emit("    cudaEvent_t start, stop;")
    emit("    cudaEventCreate(&start);")
    emit("    cudaEventCreate(&stop);")
    emit("    cudaEventRecord(start, 0);")
    emit("    unrolled_kernel<<<grid, block, %d>>>(inputs, outputs, total_instances);" % shared_bytes)
    emit("    cudaEventRecord(stop, 0);")
    emit("    cudaEventSynchronize(stop);")
    emit("    cudaEventElapsedTime(&milliseconds, start, stop);")
    emit("    gpuErrchk(cudaPeekAtLastError());")
    emit("    gpuErrchk(cudaDeviceSynchronize());")
    emit("    return milliseconds / 1000.0f;")
    emit("}")
    emit("")
    emit('extern "C" int unr_configure() {')
    if shared_bytes > 49152:
        emit("    cudaFuncSetAttribute(unrolled_kernel,")
        emit("                         cudaFuncAttributeMaxDynamicSharedMemorySize, %d);" % shared_bytes)
    emit("    return 0;")
    emit("}")

    src = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(src)
    return filepath, {
        "n_nodes": n_nodes,
        "n_levels": len(plan.level_offsets),
        "block_size": bs,
        "shared_bytes": shared_bytes,
        "cu_lines": src.count("\n") + 1,
    }


def write_unrolled_from_fn(fn, filepath, dtype="double"):
    """Emit from a live CasADi function (extracts a fresh LevelPlan)."""
    from cusadi2.dag import extract_level_plan, pick_block_size
    plan = extract_level_plan(fn)
    bs = pick_block_size(plan)
    shared_bytes = plan.peak_live_slots * (4 if dtype == "float" else 8)
    return write_unrolled_source(plan, fn.name(), bs, shared_bytes, filepath, dtype)


def write_unrolled_from_header(header_path, prefix, filepath, dtype="double"):
    """Emit from an existing generate_native_plan header — the exact plan an
    existing interpreter build uses."""
    plan, bs, shared_bytes = parse_plan_header(header_path, prefix)
    return write_unrolled_source(plan, prefix.lower(), bs, shared_bytes, filepath, dtype)
