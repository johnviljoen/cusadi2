"""
Level-parallel CUDA codegen using SSA-renamed DAG (Option C).

Emits a kernel where one CUDA block evaluates one function instance.
Threads within a block cooperate across topological levels of the DAG.
Work vector lives in shared memory; __syncthreads() between levels.

Within each level, the read-then-write two-phase pattern is used:
  Phase 1: all threads read their inputs from shared memory into registers.
  __syncthreads()
  Phase 2: all threads compute and write results to shared memory.
  __syncthreads()

Because SSA renaming guarantees no two nodes at the same level write to the
same slot, Phase 2 has no write-write conflicts. Phase 1's snapshot read
(before any writes) prevents read-write conflicts. Together, the two-phase
pattern + SSA renaming makes each level fully data-race free.
"""

import textwrap

from casadi import (OP_SQ, OP_ASSIGN,
                    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_NEG,
                    OP_EXP, OP_LOG, OP_POW, OP_CONSTPOW, OP_SQRT, OP_TWICE,
                    OP_SIN, OP_COS, OP_TAN, OP_ASIN, OP_ACOS, OP_ATAN, OP_ATAN2,
                    OP_SINH, OP_COSH, OP_TANH, OP_ASINH, OP_ACOSH, OP_ATANH,
                    OP_LT, OP_LE, OP_EQ, OP_NE, OP_NOT, OP_AND, OP_OR,
                    OP_FLOOR, OP_CEIL, OP_FMOD, OP_FABS, OP_SIGN, OP_COPYSIGN,
                    OP_IF_ELSE_ZERO, OP_ERF, OP_FMIN, OP_FMAX, OP_INV)

from src.level_parallel.dag import LevelPlan, Level

_OP_WRITE = {
    OP_ASSIGN:       "w[{o}] = ra;",
    OP_ADD:          "w[{o}] = ra + rb;",
    OP_SUB:          "w[{o}] = ra - rb;",
    OP_MUL:          "w[{o}] = ra * rb;",
    OP_DIV:          "w[{o}] = ra / rb;",
    OP_NEG:          "w[{o}] = -ra;",
    OP_EXP:          "w[{o}] = exp(ra);",
    OP_LOG:          "w[{o}] = log(ra);",
    OP_POW:          "w[{o}] = pow(ra, rb);",
    OP_CONSTPOW:     "w[{o}] = pow(ra, rb);",
    OP_SQRT:         "w[{o}] = sqrt(ra);",
    OP_SQ:           "w[{o}] = ra * ra;",
    OP_TWICE:        "w[{o}] = 2.0 * ra;",
    OP_SIN:          "w[{o}] = sin(ra);",
    OP_COS:          "w[{o}] = cos(ra);",
    OP_TAN:          "w[{o}] = tan(ra);",
    OP_ASIN:         "w[{o}] = asin(ra);",
    OP_ACOS:         "w[{o}] = acos(ra);",
    OP_ATAN:         "w[{o}] = atan(ra);",
    OP_ATAN2:        "w[{o}] = atan2(ra, rb);",
    OP_LT:           "w[{o}] = (double)(ra < rb);",
    OP_LE:           "w[{o}] = (double)(ra <= rb);",
    OP_EQ:           "w[{o}] = (double)(ra == rb);",
    OP_NE:           "w[{o}] = (double)(ra != rb);",
    OP_NOT:          "w[{o}] = (double)(!ra);",
    OP_AND:          "w[{o}] = (double)(ra && rb);",
    OP_OR:           "w[{o}] = (double)(ra || rb);",
    OP_FLOOR:        "w[{o}] = floor(ra);",
    OP_CEIL:         "w[{o}] = ceil(ra);",
    OP_FMOD:         "w[{o}] = fmod(ra, rb);",
    OP_FABS:         "w[{o}] = fabs(ra);",
    OP_SIGN:         "w[{o}] = (ra > 0.0) - (ra < 0.0);",
    OP_COPYSIGN:     "w[{o}] = copysign(ra, rb);",
    OP_IF_ELSE_ZERO: "w[{o}] = (ra == 0.0) ? 0.0 : rb;",
    OP_ERF:          "w[{o}] = erf(ra);",
    OP_FMIN:         "w[{o}] = fmin(ra, rb);",
    OP_FMAX:         "w[{o}] = fmax(ra, rb);",
    OP_INV:          "w[{o}] = 1.0 / ra;",
    OP_SINH:         "w[{o}] = sinh(ra);",
    OP_COSH:         "w[{o}] = cosh(ra);",
    OP_TANH:         "w[{o}] = tanh(ra);",
    OP_ASINH:        "w[{o}] = asinh(ra);",
    OP_ACOSH:        "w[{o}] = acosh(ra);",
    OP_ATANH:        "w[{o}] = atanh(ra);",
}


def pick_block_size(plan: LevelPlan) -> int:
    """Smallest power of 2 >= median width, capped at 256."""
    med = plan.median_width
    if med <= 0:
        return 32
    bs = 32
    while bs < med and bs < 256:
        bs *= 2
    return bs


def _emit_read(node) -> str:
    a = node.ssa_inputs[0] if len(node.ssa_inputs) >= 1 else 0
    b = node.ssa_inputs[1] if len(node.ssa_inputs) >= 2 else 0
    if len(node.ssa_inputs) >= 2:
        return f"ra = w[{a}]; rb = w[{b}];"
    elif len(node.ssa_inputs) >= 1:
        return f"ra = w[{a}];"
    return ""


def _emit_write(node) -> str:
    tmpl = _OP_WRITE.get(node.op_id)
    if tmpl is None:
        raise ValueError(f"Unknown op_id {node.op_id} at instruction {node.instr_idx}")
    return tmpl.format(o=node.ssa_out)


def _emit_level(level: Level, block_size: int) -> str:
    """Emit read-sync-write code for one compute level."""
    width = level.width
    lines = []

    def _emit_chunk(nodes, chunk_width, chunk_label):
        has_binary = any(len(n.ssa_inputs) >= 2 for n in nodes)
        lines.append(f"    {{ // L{level.depth} {chunk_label}: {chunk_width} ops")
        if has_binary:
            lines.append(f"    double ra = 0.0, rb = 0.0;")
        else:
            lines.append(f"    double ra = 0.0;")
        if chunk_width <= 4:
            for i, node in enumerate(nodes):
                lines.append(f"    if (tid == {i}) {{ {_emit_read(node)} }}")
        else:
            lines.append(f"    if (tid < {chunk_width}) {{")
            lines.append(f"        switch (tid) {{")
            for i, node in enumerate(nodes):
                lines.append(f"        case {i}: {_emit_read(node)} break;")
            lines.append(f"        }}")
            lines.append(f"    }}")
        lines.append(f"    __syncthreads();")
        if chunk_width <= 4:
            for i, node in enumerate(nodes):
                lines.append(f"    if (tid == {i}) {{ {_emit_write(node)} }}")
        else:
            lines.append(f"    if (tid < {chunk_width}) {{")
            lines.append(f"        switch (tid) {{")
            for i, node in enumerate(nodes):
                lines.append(f"        case {i}: {_emit_write(node)} break;")
            lines.append(f"        }}")
            lines.append(f"    }}")
        lines.append(f"    }} // end L{level.depth} {chunk_label}")

    if width <= block_size:
        _emit_chunk(level.nodes, width, "")
    else:
        n_chunks = (width + block_size - 1) // block_size
        for c in range(n_chunks):
            start = c * block_size
            end = min(start + block_size, width)
            _emit_chunk(level.nodes[start:end], end - start, f"chunk {c}/{n_chunks}")
            if c < n_chunks - 1:
                lines.append(f"    __syncthreads();")

    return "\n".join(lines)


def generate_level_parallel_cuda(plan: LevelPlan, filepath: str,
                                  block_size: int, debug_mode: bool = True):
    """Emit a level-parallel .cu file from an SSA-renamed LevelPlan."""
    shared_bytes = plan.peak_live_slots * 8

    with open(filepath, "w") as out:
        out.write("// AUTOMATICALLY GENERATED CODE FOR CUSADI (level-parallel, Option C)\n")
        out.write(textwrap.dedent("""
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
"""))
        out.write(f"__constant__ int nnz_in[] = {{{','.join(map(str, plan.nnz_in))}}};\n")
        out.write(f"__constant__ int nnz_out[] = {{{','.join(map(str, plan.nnz_out))}}};\n")
        out.write(textwrap.dedent(r"""
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
"""))

        # Kernel
        out.write(f"""
__global__ void evaluate_kernel(
        const double * const * __restrict__ inputs,
        double ** __restrict__ outputs,
        const int total_instances) {{

    extern __shared__ double w[];
    const int inst = blockIdx.x;
    const int tid = threadIdx.x;
    if (inst >= total_instances) return;

""")

        # Init: constants and inputs into shared memory
        init_items = []
        for c in plan.constants:
            init_items.append(('const', c))
        for im in plan.input_mappings:
            init_items.append(('input', im))
        n_init = len(init_items)

        if n_init > 0:
            out.write(f"    // === Init: {n_init} slots (consts + inputs) ===\n")
            for chunk_start in range(0, n_init, block_size):
                chunk_end = min(chunk_start + n_init - chunk_start, n_init)
                actual_end = min(chunk_start + block_size, n_init)
                for i, (kind, item) in enumerate(init_items[chunk_start:actual_end]):
                    tid_idx = i
                    if kind == 'const':
                        out.write(f"    if (tid == {tid_idx}) w[{item.ssa_slot}] = {item.value:.17e};\n")
                    else:
                        out.write(f"    if (tid == {tid_idx}) w[{item.ssa_slot}] = "
                                  f"inputs[{item.input_arg}][inst * nnz_in[{item.input_arg}] + {item.element_idx}];\n")
                if actual_end < n_init:
                    out.write("    __syncthreads();\n")
            out.write("    __syncthreads();\n\n")

        # Compute levels
        for level in plan.compute_levels:
            out.write(_emit_level(level, block_size))
            out.write("\n    __syncthreads();\n\n")

        # Output writes
        out.write("    // === Outputs ===\n")
        for out_arg in range(plan.n_outputs):
            mappings = [om for om in plan.output_mappings if om.output_arg == out_arg]
            n_om = len(mappings)
            for chunk_start in range(0, n_om, block_size):
                chunk_end = min(chunk_start + block_size, n_om)
                for i, om in enumerate(mappings[chunk_start:chunk_end]):
                    out.write(f"    if (tid == {i}) outputs[{out_arg}]"
                              f"[inst * nnz_out[{out_arg}] + {om.element_idx}] = w[{om.ssa_slot}];\n")
                if chunk_end < n_om:
                    out.write("    __syncthreads();\n")
        out.write("}\n")

        # C interface
        out.write(f'\n\nextern "C" {{\n')
        out.write(f"""
float evaluate(const double *inputs[],
               double *work_unused,
               double *outputs[],
               const int total_instances) {{
    const int blockSize = {block_size};
    const int sharedMem = {shared_bytes};
    dim3 grid(total_instances);
    dim3 block(blockSize);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_kernel<<<grid, block, sharedMem>>>(inputs, outputs, total_instances);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
""")
        if debug_mode:
            out.write("    gpuErrchk(cudaPeekAtLastError());\n")
            out.write("    gpuErrchk(cudaDeviceSynchronize());\n")
        out.write("    return milliseconds / 1000.0f;\n")
        out.write("}\n")

        # Set shared memory attribute helper
        out.write(f"""
int configure_shared_memory() {{
    cudaFuncSetAttribute(evaluate_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         {shared_bytes});
    return 0;
}}
""")
        out.write("\n}\n")

    return filepath
