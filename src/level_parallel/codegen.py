"""
Level-parallel CUDA codegen: data-driven dispatch.

Instead of unrolling one code branch per DAG node (which produces massive .cu
files and 60+ minute compile times), we emit a single fixed-size kernel template
that reads per-node operation metadata from arrays in global memory.

The kernel body is ~80 lines of CUDA regardless of function size. Per-function
data (op types, operand indices, constants, level structure) is computed from
the LevelPlan at setup time and uploaded to GPU once.

Within each level, the two-phase read-then-write pattern is used:
  Phase 1: each thread reads its input operands from shared memory into registers.
  __syncthreads()
  Phase 2: each thread dispatches on op_id, computes, and writes to shared memory.
  __syncthreads()
"""

import os

from src.level_parallel.dag import LevelPlan


KERNEL_TEMPLATE = r"""
// AUTOMATICALLY GENERATED: level-parallel data-driven kernel
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

__device__ __forceinline__ double exec_op(int op, double a, double b) {
    switch (op) {
        case OP_ASSIGN:       return a;
        case OP_ADD:          return a + b;
        case OP_SUB:          return a - b;
        case OP_MUL:          return a * b;
        case OP_DIV:          return a / b;
        case OP_NEG:          return -a;
        case OP_EXP:          return exp(a);
        case OP_LOG:          return log(a);
        case OP_POW:          return pow(a, b);
        case OP_CONSTPOW:     return pow(a, b);
        case OP_SQRT:         return sqrt(a);
        case OP_SQ:           return a * a;
        case OP_TWICE:        return 2.0 * a;
        case OP_SIN:          return sin(a);
        case OP_COS:          return cos(a);
        case OP_TAN:          return tan(a);
        case OP_ASIN:         return asin(a);
        case OP_ACOS:         return acos(a);
        case OP_ATAN:         return atan(a);
        case OP_FMIN:         return fmin(a, b);
        case OP_FMAX:         return fmax(a, b);
        case OP_INV:          return 1.0 / a;
        case OP_SINH:         return sinh(a);
        case OP_COSH:         return cosh(a);
        case OP_TANH:         return tanh(a);
        case OP_ASINH:        return asinh(a);
        case OP_ACOSH:        return acosh(a);
        case OP_ATANH:        return atanh(a);
        case OP_ATAN2:        return atan2(a, b);
        case OP_LT:           return (double)(a < b);
        case OP_LE:           return (double)(a <= b);
        case OP_EQ:           return (double)(a == b);
        case OP_NE:           return (double)(a != b);
        case OP_NOT:          return (double)(!a);
        case OP_AND:          return (double)((bool)a && (bool)b);
        case OP_OR:           return (double)((bool)a || (bool)b);
        case OP_IF_ELSE_ZERO: return (a == 0.0) ? 0.0 : b;
        case OP_FLOOR:        return floor(a);
        case OP_CEIL:         return ceil(a);
        case OP_FMOD:         return fmod(a, b);
        case OP_FABS:         return fabs(a);
        case OP_SIGN:         return (a > 0.0) - (a < 0.0);
        case OP_COPYSIGN:     return copysign(a, b);
        case OP_ERF:          return erf(a);
        default:              return 0.0;
    }
}

__global__ void evaluate_kernel(
        const double * const * __restrict__ inputs,
        double ** __restrict__ outputs,
        const int total_instances,
        // Init data
        const int n_consts,
        const int * __restrict__ const_slots,
        const double * __restrict__ const_vals,
        const int n_inputs,
        const int * __restrict__ input_slots,
        const int * __restrict__ input_args,
        const int * __restrict__ input_elems,
        const int * __restrict__ nnz_in,
        // Compute data
        const int n_levels,
        const int * __restrict__ level_offsets,
        const int * __restrict__ level_widths,
        const int * __restrict__ node_ops,
        const int * __restrict__ node_outs,
        const int * __restrict__ node_in_a,
        const int * __restrict__ node_in_b,
        // Output data
        const int n_outputs,
        const int * __restrict__ out_slots,
        const int * __restrict__ out_args,
        const int * __restrict__ out_elems,
        const int * __restrict__ nnz_out) {

    extern __shared__ double w[];
    const int inst = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_sz = blockDim.x;
    if (inst >= total_instances) return;

    // --- Init: load constants ---
    for (int i = tid; i < n_consts; i += block_sz) {
        w[const_slots[i]] = const_vals[i];
    }
    // --- Init: load external inputs ---
    for (int i = tid; i < n_inputs; i += block_sz) {
        int arg = input_args[i];
        w[input_slots[i]] = inputs[arg][inst * nnz_in[arg] + input_elems[i]];
    }
    __syncthreads();

    // --- Compute levels ---
    // SSA renaming guarantees no slot is both read and written within a level.
    // The read-phase barrier is unnecessary: each thread reads slots written at
    // previous levels (already visible from the previous level's barrier) and
    // writes to a slot no other thread at this level reads.
    // One __syncthreads() per level (after all chunks) suffices.
    for (int lev = 0; lev < n_levels; lev++) {
        int offset = level_offsets[lev];
        int width = level_widths[lev];
        for (int base = 0; base < width; base += block_sz) {
            int idx = base + tid;
            double ra = 0.0, rb = 0.0;
            int my_op = 0, my_out = 0;
            if (idx < width) {
                int g = offset + idx;
                ra = w[node_in_a[g]];
                rb = w[node_in_b[g]];
                my_op = node_ops[g];
                my_out = node_outs[g];
            }
            if (idx < width) {
                w[my_out] = exec_op(my_op, ra, rb);
            }
        }
        __syncthreads();
    }

    // --- Write outputs ---
    for (int i = tid; i < n_outputs; i += block_sz) {
        int arg = out_args[i];
        outputs[arg][inst * nnz_out[arg] + out_elems[i]] = w[out_slots[i]];
    }
}

extern "C" {

float evaluate_lp(
        const double *inputs[],
        double *outputs[],
        const int total_instances,
        const int block_size,
        const int shared_bytes,
        // Init
        const int n_consts,
        const int *const_slots,
        const double *const_vals,
        const int n_inputs,
        const int *input_slots,
        const int *input_args,
        const int *input_elems,
        const int *nnz_in,
        // Compute
        const int n_levels,
        const int *level_offsets,
        const int *level_widths,
        const int *node_ops,
        const int *node_outs,
        const int *node_in_a,
        const int *node_in_b,
        // Outputs
        const int n_outputs,
        const int *out_slots,
        const int *out_args,
        const int *out_elems,
        const int *nnz_out) {

    dim3 grid(total_instances);
    dim3 block(block_size);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_kernel<<<grid, block, shared_bytes>>>(
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

int configure_shared_memory(int shared_bytes) {
    cudaFuncSetAttribute(evaluate_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
    return 0;
}

} // extern "C"
"""


def pick_block_size(plan: LevelPlan) -> int:
    """Smallest power of 2 >= median width, capped at 256."""
    med = plan.median_width
    if med <= 0:
        return 32
    bs = 32
    while bs < med and bs < 256:
        bs *= 2
    return bs


def write_kernel_source(filepath: str):
    """Write the fixed kernel template to a .cu file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(KERNEL_TEMPLATE)
    return filepath
