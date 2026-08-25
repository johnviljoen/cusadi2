"""Warp-batched (instances-across-lanes) data-driven interpreter kernel.

The unexplored quadrant of the {interpreted, compiled} x {nodes-across-lanes,
instances-across-lanes} design space: keep cusadi2's data-driven dispatch (one
function-independent kernel, compiled once), but adopt the lane mapping of
warp-specialized codegen (Singe, PPoPP 2014) — each warp processes ONE node at
a time across 32 batch instances in its lanes.

Consequences vs the standard interpreter kernel:
  * node metadata (op, in_a, in_b, out) is loaded at ONE warp-uniform address
    -> a single broadcast transaction per node instead of one per lane;
  * the exec_op switch is warp-convergent (every lane has the same op)
    -> zero divergence regardless of the level's op mix;
  * the shared work vector is transposed to [slot][lane32] and grows 32x
    (peak_live_slots * 32 scalars) — the cost axis. fp64 A1/H1 FK fit the
    ~99KB Ada per-block cap; the frame Jacobians do not;
  * global input/output accesses are strided by nnz across lanes
    (uncoalesced) — the batch-major layout is kept for buffer compatibility
    with the standard interpreter in the same benchmark.

Block = block_warps * 32 threads; grid.x = ceil(B / 32) instance groups.
Within a level, warp W handles nodes W, W+n_warps, ... (no intra-level sync
needed: the slot schedule guarantees no slot is both read and written within a
level — the same guarantee the standard kernel's fast path uses).

Emitted extern "C" API:
    float wb_evaluate(...);      // seconds, cudaEvent-timed like evaluate_cusadi
    int   wb_configure(int shared_bytes);   // only ever raises the smem cap
"""

import os

from cusadi2.kernel_template import _render

WB_TEMPLATE = r"""
// AUTOMATICALLY GENERATED: warp-batched (instances-across-lanes) interpreter.
#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__host__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CasADi op IDs — must match casadi/core/calculus.hpp exactly
enum CasadiOp {
    OP_ASSIGN=0, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_NEG,     // 0-5
    OP_EXP, OP_LOG, OP_POW, OP_CONSTPOW, OP_SQRT, OP_SQ,     // 6-11
    OP_TWICE, OP_SIN, OP_COS, OP_TAN, OP_ASIN, OP_ACOS,      // 12-17
    OP_ATAN,                                                    // 18
    OP_LT, OP_LE, OP_EQ, OP_NE, OP_NOT, OP_AND, OP_OR,       // 19-25
    OP_FLOOR, OP_CEIL, OP_FMOD, OP_FABS, OP_SIGN, OP_COPYSIGN,// 26-31
    OP_IF_ELSE_ZERO, OP_ERF, OP_FMIN, OP_FMAX, OP_INV,        // 32-36
    OP_SINH, OP_COSH, OP_TANH, OP_ASINH, OP_ACOSH, OP_ATANH,  // 37-42
    OP_ATAN2                                                    // 43
};

__EXEC_OP__

__global__ void wb_kernel(
        const {dtype} * const * __restrict__ inputs,
        {dtype} ** __restrict__ outputs,
        const int total_instances,
        const int n_consts,
        const int * __restrict__ const_slots,
        const {dtype} * __restrict__ const_vals,
        const int n_inputs,
        const int * __restrict__ input_slots,
        const int * __restrict__ input_args,
        const int * __restrict__ input_elems,
        const int * __restrict__ nnz_in,
        const int n_levels,
        const int * __restrict__ level_offsets,
        const int * __restrict__ level_widths,
        const int * __restrict__ node_ops,
        const int * __restrict__ node_outs,
        const int * __restrict__ node_in_a,
        const int * __restrict__ node_in_b,
        const int n_outputs,
        const int * __restrict__ out_slots,
        const int * __restrict__ out_args,
        const int * __restrict__ out_elems,
        const int * __restrict__ nnz_out) {

    extern __shared__ {dtype} w[];               // [slot][lane32], transposed
    const int lane    = threadIdx.x & 31;
    const int warp    = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    const long long inst = (long long)blockIdx.x * 32 + lane;
    const bool live = inst < total_instances;

    // --- Init: constants (warp-uniform value, all lanes) ---
    for (int i = warp; i < n_consts; i += n_warps)
        w[const_slots[i] * 32 + lane] = const_vals[i];

    // --- Init: inputs (per-lane instance; strided/uncoalesced by design) ---
    for (int i = warp; i < n_inputs; i += n_warps) {
        int arg = input_args[i];
        w[input_slots[i] * 32 + lane] =
            live ? inputs[arg][inst * nnz_in[arg] + input_elems[i]] : ({dtype})0;
    }
    __syncthreads();

    // --- Compute: warp takes one node at a time across its 32 lane-instances.
    // Metadata addresses are warp-uniform (one broadcast load per node); the
    // exec_op switch is warp-convergent. No intra-level sync: no slot is both
    // read and written within a level.
    for (int lev = 0; lev < n_levels; lev++) {
        int off = level_offsets[lev];
        int width = level_widths[lev];
        for (int i = warp; i < width; i += n_warps) {
            int g = off + i;
            {dtype} ra = w[node_in_a[g] * 32 + lane];
            {dtype} rb = w[node_in_b[g] * 32 + lane];
            w[node_outs[g] * 32 + lane] = exec_op(node_ops[g], ra, rb);
        }
        __syncthreads();
    }

    // --- Outputs (per-lane instance; strided/uncoalesced by design) ---
    for (int i = warp; i < n_outputs; i += n_warps) {
        int arg = out_args[i];
        if (live)
            outputs[arg][inst * nnz_out[arg] + out_elems[i]] = w[out_slots[i] * 32 + lane];
    }
}

extern "C" float wb_evaluate(
        const {dtype} * const * inputs,
        {dtype} ** outputs,
        const int total_instances,
        const int block_warps,
        const int shared_bytes,          // already the 32x (transposed) size
        const int n_consts, const int *const_slots, const {dtype} *const_vals,
        const int n_inputs, const int *input_slots, const int *input_args,
        const int *input_elems, const int *nnz_in,
        const int n_levels, const int *level_offsets, const int *level_widths,
        const int *node_ops, const int *node_outs, const int *node_in_a, const int *node_in_b,
        const int n_outputs, const int *out_slots, const int *out_args,
        const int *out_elems, const int *nnz_out) {

    dim3 grid((total_instances + 31) / 32);
    dim3 block(block_warps * 32);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    wb_kernel<<<grid, block, shared_bytes>>>(
        inputs, outputs, total_instances,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems, nnz_in,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems, nnz_out);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return milliseconds / 1000.0f;
}

extern "C" int wb_configure(int shared_bytes) {
    // same only-ever-raise semantics as configure_cusadi_shared_memory
    static int configured = 49152;
    if (shared_bytes <= configured) return 0;
    configured = shared_bytes;
    cudaFuncSetAttribute(wb_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
    return 0;
}
"""


def write_wb_kernel_source(filepath, dtype="double"):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(_render(WB_TEMPLATE, dtype))
    return filepath
