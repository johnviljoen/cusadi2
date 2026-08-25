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

from cusadi.level_parallel.dag import LevelPlan


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

__EXEC_OP__
__global__ void evaluate_kernel(
        const {dtype} * const * __restrict__ inputs,
        {dtype} ** __restrict__ outputs,
        const int total_instances,
        // Init data
        const int n_consts,
        const int * __restrict__ const_slots,
        const {dtype} * __restrict__ const_vals,
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

    extern __shared__ {dtype} w[];
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
            {dtype} ra = 0.0, rb = 0.0;
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
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int block_size,
        const int shared_bytes,
        // Init
        const int n_consts,
        const int *const_slots,
        const {dtype} *const_vals,
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

// Stream-aware variant for XLA / JAX FFI integration.
// Launches on the caller-supplied stream; no host-side synchronization —
// XLA's stream ordering guarantees downstream ops wait on this launch.
// Returns 0 on success, or the integer value of the cudaError_t on failure.
int evaluate_lp_stream(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int block_size,
        const int shared_bytes,
        // Init
        const int n_consts,
        const int *const_slots,
        const {dtype} *const_vals,
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
        const int *nnz_out,
        // Stream
        cudaStream_t stream) {

    dim3 grid(total_instances);
    dim3 block(block_size);
    evaluate_kernel<<<grid, block, shared_bytes, stream>>>(
        inputs, outputs, total_instances,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems, nnz_in,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems, nnz_out);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

int configure_shared_memory(int shared_bytes) {
    // The attribute caps every later launch of the (shared) kernel, so it must
    // only ever grow: multiple uploaded plans each call this with their own size.
    static int configured = 0;
    if (shared_bytes <= configured) return 0;
    configured = shared_bytes;
    cudaFuncSetAttribute(evaluate_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
    return 0;
}

} // extern "C"
"""


TILED_KERNEL_TEMPLATE = r"""
// AUTOMATICALLY GENERATED: level-parallel data-driven kernel with tiling
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

enum CasadiOp {
    OP_ASSIGN=0, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_NEG,
    OP_EXP, OP_LOG, OP_POW, OP_CONSTPOW, OP_SQRT, OP_SQ,
    OP_TWICE, OP_SIN, OP_COS, OP_TAN, OP_ASIN, OP_ACOS,
    OP_ATAN,
    OP_LT, OP_LE, OP_EQ, OP_NE, OP_NOT, OP_AND, OP_OR,
    OP_FLOOR, OP_CEIL, OP_FMOD, OP_FABS, OP_SIGN, OP_COPYSIGN,
    OP_IF_ELSE_ZERO, OP_ERF, OP_FMIN, OP_FMAX, OP_INV,
    OP_SINH, OP_COSH, OP_TANH, OP_ASINH, OP_ACOSH, OP_ATANH,
    OP_ATAN2
};

__EXEC_OP__
__global__ void evaluate_kernel_tiled(
        const {dtype} * const * __restrict__ inputs,
        {dtype} ** __restrict__ outputs,
        const int total_instances,
        // Init data
        const int n_consts,
        const int * __restrict__ const_slots,
        const {dtype} * __restrict__ const_vals,
        const int n_inputs,
        const int * __restrict__ input_slots,
        const int * __restrict__ input_args,
        const int * __restrict__ input_elems,
        const int * __restrict__ nnz_in,
        // Level data
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
        const int * __restrict__ nnz_out,
        // Tiling data
        {dtype} * __restrict__ spill_buffer,
        const int spill_stride,
        const int n_chunks,
        const int * __restrict__ chunk_lev_start,
        const int * __restrict__ chunk_lev_end,
        // Boundary spill/reload (n_chunks - 1 boundaries)
        const int * __restrict__ bnd_spill_off,
        const int * __restrict__ bnd_spill_cnt,
        const int * __restrict__ bnd_spill_smem,
        const int * __restrict__ bnd_spill_pos,
        const int * __restrict__ bnd_reload_off,
        const int * __restrict__ bnd_reload_cnt,
        const int * __restrict__ bnd_reload_smem,
        const int * __restrict__ bnd_reload_pos,
        // Sub-batch spill (per-level)
        const int * __restrict__ lev_sb_off,
        const int * __restrict__ lev_sb_cnt,
        const int * __restrict__ sb_spill_smem,
        const int * __restrict__ sb_spill_pos) {

    extern __shared__ {dtype} w[];
    const int inst = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_sz = blockDim.x;
    if (inst >= total_instances) return;

    {dtype} * spill = spill_buffer + (long long)inst * spill_stride;

    // --- Init: load constants and inputs (chunk 0) ---
    for (int i = tid; i < n_consts; i += block_sz)
        w[const_slots[i]] = const_vals[i];
    for (int i = tid; i < n_inputs; i += block_sz) {
        int arg = input_args[i];
        w[input_slots[i]] = inputs[arg][inst * nnz_in[arg] + input_elems[i]];
    }
    __syncthreads();

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        // --- Reload phase (skip chunk 0) ---
        if (chunk > 0) {
            int bi = chunk - 1;
            int roff = bnd_reload_off[bi];
            int rcnt = bnd_reload_cnt[bi];
            for (int i = tid; i < rcnt; i += block_sz)
                w[bnd_reload_smem[roff + i]] = spill[bnd_reload_pos[roff + i]];
            __syncthreads();
        }

        // --- Process levels in this chunk ---
        for (int lev = chunk_lev_start[chunk]; lev < chunk_lev_end[chunk]; lev++) {
            int offset = level_offsets[lev];
            int width = level_widths[lev];

            for (int base = 0; base < width; base += block_sz) {
                int idx = base + tid;
                {dtype} ra = 0.0, rb = 0.0;
                int my_op = 0, my_out = 0;
                if (idx < width) {
                    int g = offset + idx;
                    ra = w[node_in_a[g]];
                    rb = w[node_in_b[g]];
                    my_op = node_ops[g];
                    my_out = node_outs[g];
                }
                __syncthreads();
                if (idx < width) {
                    w[my_out] = exec_op(my_op, ra, rb);
                }
                __syncthreads();
            }

            // Sub-batch spill (zero iterations when lev_sb_cnt[lev] == 0)
            int sb_cnt = lev_sb_cnt[lev];
            if (sb_cnt > 0) {
                int sb_off = lev_sb_off[lev];
                for (int i = tid; i < sb_cnt; i += block_sz)
                    spill[sb_spill_pos[sb_off + i]] = w[sb_spill_smem[sb_off + i]];
                __syncthreads();
            }
        }

        // --- Inter-chunk spill (skip last chunk) ---
        if (chunk < n_chunks - 1) {
            int soff = bnd_spill_off[chunk];
            int scnt = bnd_spill_cnt[chunk];
            for (int i = tid; i < scnt; i += block_sz)
                spill[bnd_spill_pos[soff + i]] = w[bnd_spill_smem[soff + i]];
            __syncthreads();
        }
    }

    // --- Write outputs ---
    for (int i = tid; i < n_outputs; i += block_sz) {
        int arg = out_args[i];
        outputs[arg][inst * nnz_out[arg] + out_elems[i]] = w[out_slots[i]];
    }
}

extern "C" {

float evaluate_lp_tiled(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int block_size,
        const int shared_bytes,
        // Init
        const int n_consts, const int *const_slots, const {dtype} *const_vals,
        const int n_inputs, const int *input_slots, const int *input_args,
        const int *input_elems, const int *nnz_in,
        // Levels
        const int n_levels, const int *level_offsets, const int *level_widths,
        const int *node_ops, const int *node_outs,
        const int *node_in_a, const int *node_in_b,
        // Outputs
        const int n_outputs, const int *out_slots, const int *out_args,
        const int *out_elems, const int *nnz_out,
        // Tiling
        {dtype} *spill_buffer, const int spill_stride, const int n_chunks,
        const int *chunk_lev_start, const int *chunk_lev_end,
        const int *bnd_spill_off, const int *bnd_spill_cnt,
        const int *bnd_spill_smem, const int *bnd_spill_pos,
        const int *bnd_reload_off, const int *bnd_reload_cnt,
        const int *bnd_reload_smem, const int *bnd_reload_pos,
        const int *lev_sb_off, const int *lev_sb_cnt,
        const int *sb_spill_smem, const int *sb_spill_pos) {

    dim3 grid(total_instances);
    dim3 block(block_size);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_kernel_tiled<<<grid, block, shared_bytes>>>(
        inputs, outputs, total_instances,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems, nnz_in,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems, nnz_out,
        spill_buffer, spill_stride, n_chunks,
        chunk_lev_start, chunk_lev_end,
        bnd_spill_off, bnd_spill_cnt, bnd_spill_smem, bnd_spill_pos,
        bnd_reload_off, bnd_reload_cnt, bnd_reload_smem, bnd_reload_pos,
        lev_sb_off, lev_sb_cnt, sb_spill_smem, sb_spill_pos);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return milliseconds / 1000.0f;
}

int configure_tiled_shared_memory(int shared_bytes) {
    cudaFuncSetAttribute(evaluate_kernel_tiled,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
    return 0;
}

} // extern "C"
"""


STAGE_KERNEL_TEMPLATE = r"""
// AUTOMATICALLY GENERATED: stage-aware level-parallel data-driven kernel
//
// Compiles a single per-stage program once; instantiates it across both
// batch instances (blockIdx.x) and stages (blockIdx.y). The compute body is
// identical to the untiled level-parallel kernel — only input/output
// indexing changes to honour each argument's stage mode.
//
// Per-input-arg metadata (length n_in, with analogous arrays for outputs):
//   stage_mode[arg]   ∈ {0=STAGE, 1=SHARED, 2=COUPLED}
//   stage_offset[arg] : added to blockIdx.y when mode==COUPLED (e.g. +1 for x_{k+1})
//   spacing[arg]      : distance IN DOUBLES between consecutive instances'
//                       data for this argument. The dense default is
//                       stage_count * nnz (stage_count: STAGE -> n_stages,
//                       SHARED -> 1, COUPLED -> n_stages + span); passing a
//                       larger spacing lets an argument live inside a bigger
//                       per-instance block (e.g. x inside z), and spacing 0
//                       shares one copy across all instances.
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

enum CasadiOp {
    OP_ASSIGN=0, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_NEG,
    OP_EXP, OP_LOG, OP_POW, OP_CONSTPOW, OP_SQRT, OP_SQ,
    OP_TWICE, OP_SIN, OP_COS, OP_TAN, OP_ASIN, OP_ACOS,
    OP_ATAN,
    OP_LT, OP_LE, OP_EQ, OP_NE, OP_NOT, OP_AND, OP_OR,
    OP_FLOOR, OP_CEIL, OP_FMOD, OP_FABS, OP_SIGN, OP_COPYSIGN,
    OP_IF_ELSE_ZERO, OP_ERF, OP_FMIN, OP_FMAX, OP_INV,
    OP_SINH, OP_COSH, OP_TANH, OP_ASINH, OP_ACOSH, OP_ATANH,
    OP_ATAN2
};

__EXEC_OP__
// the argument tables a launch reads: pointer + per-instance spacing per
// in/out argument. Passed by value as a kernel parameter by
// evaluate_stage_kernel_args so a call needs no device-side table at all.
#define CUSADI_MAX_ARGS 8
struct StageArgTable {
    const {dtype}* in[CUSADI_MAX_ARGS];
    {dtype}* out[CUSADI_MAX_ARGS];
    int in_spacing[CUSADI_MAX_ARGS];
    int out_spacing[CUSADI_MAX_ARGS];
};

// the per-stage program (shared by both kernels below); w = dynamic shared memory
__device__ __forceinline__ void evaluate_stage_body(
        const {dtype} * const * __restrict__ inputs,
        {dtype} * const * __restrict__ outputs,
        const int * __restrict__ in_spacing,
        const int * __restrict__ out_spacing,
        const int total_instances,
        const int n_stages,
        // Init data
        const int n_consts,
        const int * __restrict__ const_slots,
        const {dtype} * __restrict__ const_vals,
        const int n_inputs,
        const int * __restrict__ input_slots,
        const int * __restrict__ input_args,
        const int * __restrict__ input_elems,
        const int * __restrict__ nnz_in,
        const int * __restrict__ in_stage_mode,
        const int * __restrict__ in_stage_offset,
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
        const int * __restrict__ nnz_out,
        const int * __restrict__ out_stage_mode,
        {dtype}* w) {

    const int inst  = blockIdx.x;
    const int stage = blockIdx.y;
    const int tid   = threadIdx.x;
    const int block_sz = blockDim.x;
    if (inst >= total_instances || stage >= n_stages) return;

    // --- Init: load constants ---
    for (int i = tid; i < n_consts; i += block_sz) {
        w[const_slots[i]] = const_vals[i];
    }
    // --- Init: load external inputs (stage-aware indexing) ---
    for (int i = tid; i < n_inputs; i += block_sz) {
        int arg  = input_args[i];
        int mode = in_stage_mode[arg];
        int s = (mode == 1) ? 0
              : (mode == 2) ? (stage + in_stage_offset[arg])
              :               stage;
        long long base = (long long)inst * in_spacing[arg] + (long long)s * nnz_in[arg];
        w[input_slots[i]] = inputs[arg][base + input_elems[i]];
    }
    __syncthreads();

    // --- Compute levels (identical to untiled LP kernel) ---
    for (int lev = 0; lev < n_levels; lev++) {
        int offset = level_offsets[lev];
        int width = level_widths[lev];
        for (int base = 0; base < width; base += block_sz) {
            int idx = base + tid;
            {dtype} ra = 0.0, rb = 0.0;
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

    // --- Write outputs (stage-aware indexing) ---
    for (int i = tid; i < n_outputs; i += block_sz) {
        int arg  = out_args[i];
        int mode = out_stage_mode[arg];
        int s = (mode == 1) ? 0 : stage;
        long long base = (long long)inst * out_spacing[arg] + (long long)s * nnz_out[arg];
        outputs[arg][base + out_elems[i]] = w[out_slots[i]];
    }
}

#define CUSADI_STAGE_PARAMS \
        const int total_instances, const int n_stages, \
        const int n_consts, const int * __restrict__ const_slots, const {dtype} * __restrict__ const_vals, \
        const int n_inputs, const int * __restrict__ input_slots, const int * __restrict__ input_args, \
        const int * __restrict__ input_elems, const int * __restrict__ nnz_in, \
        const int * __restrict__ in_stage_mode, const int * __restrict__ in_stage_offset, \
        const int n_levels, const int * __restrict__ level_offsets, const int * __restrict__ level_widths, \
        const int * __restrict__ node_ops, const int * __restrict__ node_outs, \
        const int * __restrict__ node_in_a, const int * __restrict__ node_in_b, \
        const int n_outputs, const int * __restrict__ out_slots, const int * __restrict__ out_args, \
        const int * __restrict__ out_elems, const int * __restrict__ nnz_out, \
        const int * __restrict__ out_stage_mode
#define CUSADI_STAGE_ARGS \
        total_instances, n_stages, n_consts, const_slots, const_vals, \
        n_inputs, input_slots, input_args, input_elems, nnz_in, in_stage_mode, in_stage_offset, \
        n_levels, level_offsets, level_widths, node_ops, node_outs, node_in_a, node_in_b, \
        n_outputs, out_slots, out_args, out_elems, nnz_out, out_stage_mode

// tables in device memory (bind() once, then launch; graph-capturable)
__global__ void evaluate_stage_kernel(
        const {dtype} * const * __restrict__ inputs, {dtype} ** __restrict__ outputs,
        const int * __restrict__ in_spacing, const int * __restrict__ out_spacing,
        CUSADI_STAGE_PARAMS) {
    extern __shared__ {dtype} w[];
    evaluate_stage_body(inputs, outputs, in_spacing, out_spacing, CUSADI_STAGE_ARGS, w);
}

// tables by value (evaluate_lp_stage_args): a pure launch, no prior bind
__global__ void evaluate_stage_kernel_args(const StageArgTable t, CUSADI_STAGE_PARAMS) {
    extern __shared__ {dtype} w[];
    evaluate_stage_body(t.in, t.out, t.in_spacing, t.out_spacing, CUSADI_STAGE_ARGS, w);
}

extern "C" {

float evaluate_lp_stage(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int n_stages,
        const int block_size,
        const int shared_bytes,
        // Init
        const int n_consts,
        const int *const_slots,
        const {dtype} *const_vals,
        const int n_inputs,
        const int *input_slots,
        const int *input_args,
        const int *input_elems,
        const int *nnz_in,
        const int *in_stage_mode,
        const int *in_stage_offset,
        const int *in_spacing,
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
        const int *nnz_out,
        const int *out_stage_mode,
        const int *out_spacing) {

    dim3 grid(total_instances, n_stages);
    dim3 block(block_size);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_stage_kernel<<<grid, block, shared_bytes>>>(
        inputs, outputs, in_spacing, out_spacing, total_instances, n_stages,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems, nnz_in,
        in_stage_mode, in_stage_offset,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems, nnz_out,
        out_stage_mode);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return milliseconds / 1000.0f;
}

int evaluate_lp_stage_stream(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int n_stages,
        const int block_size,
        const int shared_bytes,
        const int n_consts,
        const int *const_slots,
        const {dtype} *const_vals,
        const int n_inputs,
        const int *input_slots,
        const int *input_args,
        const int *input_elems,
        const int *nnz_in,
        const int *in_stage_mode,
        const int *in_stage_offset,
        const int *in_spacing,
        const int n_levels,
        const int *level_offsets,
        const int *level_widths,
        const int *node_ops,
        const int *node_outs,
        const int *node_in_a,
        const int *node_in_b,
        const int n_outputs,
        const int *out_slots,
        const int *out_args,
        const int *out_elems,
        const int *nnz_out,
        const int *out_stage_mode,
        const int *out_spacing,
        cudaStream_t stream) {

    dim3 grid(total_instances, n_stages);
    dim3 block(block_size);
    evaluate_stage_kernel<<<grid, block, shared_bytes, stream>>>(
        inputs, outputs, in_spacing, out_spacing, total_instances, n_stages,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems, nnz_in,
        in_stage_mode, in_stage_offset,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems, nnz_out,
        out_stage_mode);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

// by-value argument tables: a pure launch with no prior bind.
int evaluate_lp_stage_args(
        const {dtype} * const *inputs, {dtype} * const *outputs, const int *in_spacing, const int *out_spacing,
        const int n_args_in, const int n_args_out,
        const int total_instances, const int n_stages, const int block_size, const int shared_bytes,
        const int n_consts, const int *const_slots, const {dtype} *const_vals,
        const int n_inputs, const int *input_slots, const int *input_args, const int *input_elems,
        const int *nnz_in, const int *in_stage_mode, const int *in_stage_offset,
        const int n_levels, const int *level_offsets, const int *level_widths,
        const int *node_ops, const int *node_outs, const int *node_in_a, const int *node_in_b,
        const int n_outputs, const int *out_slots, const int *out_args, const int *out_elems,
        const int *nnz_out, const int *out_stage_mode, cudaStream_t stream) {
    if (n_args_in > CUSADI_MAX_ARGS || n_args_out > CUSADI_MAX_ARGS) return (int)cudaErrorInvalidValue;
    StageArgTable t{};
    for (int a = 0; a < n_args_in; ++a) { t.in[a] = inputs[a]; t.in_spacing[a] = in_spacing[a]; }
    for (int a = 0; a < n_args_out; ++a) { t.out[a] = outputs[a]; t.out_spacing[a] = out_spacing[a]; }
    dim3 grid(total_instances, n_stages);
    dim3 block(block_size);
    evaluate_stage_kernel_args<<<grid, block, shared_bytes, stream>>>(t, CUSADI_STAGE_ARGS);
    return (int)cudaPeekAtLastError();
}

int configure_stage_shared_memory(int shared_bytes) {
    // The attribute caps every later launch of the (shared) kernel, so it must
    // only ever grow: multiple uploaded plans each call this with their own size.
    static int configured = 0;
    if (shared_bytes <= configured) return 0;
    configured = shared_bytes;
    cudaFuncSetAttribute(evaluate_stage_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
    cudaFuncSetAttribute(evaluate_stage_kernel_args,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
    return 0;
}

} // extern "C"
"""


CHUNK_KERNEL_TEMPLATE = r"""
// AUTOMATICALLY GENERATED: chunked stage-aware level-parallel data-driven kernel.
//
// One block processes one chunk of `chunk_size` consecutive stages of f_stage.
// blockIdx.y indexes the chunk; threads cooperate over the (already stamped)
// chunk-DAG inside the block. The level-DAG plan is built on f_chunk =
// stamp(f_stage, chunk_size); only the I/O indexing differs from the stage
// kernel, replacing per-stage stride with per-chunk stride.
//
// Per-input-arg metadata (length n_in of f_chunk, analogous arrays for outputs):
//   in_stage_mode[arg]   ∈ {0=STAGE, 1=SHARED, 2=COUPLED}
//   in_chunk_offset[arg] : added to chunk_idx*chunk_size when mode==COUPLED
//                          (mirrors the original spec.offset, e.g. +1 for x_{k+1})
//   in_total_stages[arg] : full extent of buffer along stage axis:
//                            STAGE   → n_stages
//                            SHARED  → 1
//                            COUPLED → n_stages + span
//   in_nnz_per_stage[arg]: f_stage's per-stage nnz_in[arg]
//                          (NOT f_chunk's nnz_in, which is chunk_size× this)
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

enum CasadiOp {
    OP_ASSIGN=0, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_NEG,
    OP_EXP, OP_LOG, OP_POW, OP_CONSTPOW, OP_SQRT, OP_SQ,
    OP_TWICE, OP_SIN, OP_COS, OP_TAN, OP_ASIN, OP_ACOS,
    OP_ATAN,
    OP_LT, OP_LE, OP_EQ, OP_NE, OP_NOT, OP_AND, OP_OR,
    OP_FLOOR, OP_CEIL, OP_FMOD, OP_FABS, OP_SIGN, OP_COPYSIGN,
    OP_IF_ELSE_ZERO, OP_ERF, OP_FMIN, OP_FMAX, OP_INV,
    OP_SINH, OP_COSH, OP_TANH, OP_ASINH, OP_ACOSH, OP_ATANH,
    OP_ATAN2
};

__EXEC_OP__
__global__ void evaluate_chunk_kernel(
        const {dtype} * const * __restrict__ inputs,
        {dtype} ** __restrict__ outputs,
        const int total_instances,
        const int n_chunks,
        const int chunk_size,
        // Init data
        const int n_consts,
        const int * __restrict__ const_slots,
        const {dtype} * __restrict__ const_vals,
        const int n_inputs,
        const int * __restrict__ input_slots,
        const int * __restrict__ input_args,
        const int * __restrict__ input_elems,
        const int * __restrict__ in_stage_mode,
        const int * __restrict__ in_chunk_offset,
        const int * __restrict__ in_total_stages,
        const int * __restrict__ in_nnz_per_stage,
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
        const int * __restrict__ out_stage_mode,
        const int * __restrict__ out_total_stages,
        const int * __restrict__ out_nnz_per_stage) {

    extern __shared__ {dtype} w[];
    const int inst      = blockIdx.x;
    const int chunk_idx = blockIdx.y;
    const int tid       = threadIdx.x;
    const int block_sz  = blockDim.x;
    if (inst >= total_instances || chunk_idx >= n_chunks) return;

    // --- Init: load constants ---
    for (int i = tid; i < n_consts; i += block_sz) {
        w[const_slots[i]] = const_vals[i];
    }

    // --- Init: load external inputs (chunk-aware indexing) ---
    // For STAGE/COUPLED args, the chunk reads stages
    //   [chunk_idx*chunk_size + offset, chunk_idx*chunk_size + offset + extent)
    // where extent = chunk_size for STAGE, chunk_size + span for COUPLED. The
    // per-mapping element_idx (from f_chunk's plan) already encodes the
    // (inner_stage, elem_within_stage) slot inside that extent.
    for (int i = tid; i < n_inputs; i += block_sz) {
        int arg  = input_args[i];
        int mode = in_stage_mode[arg];
        int chunk_start = (mode == 1) ? 0
                        : (chunk_idx * chunk_size + in_chunk_offset[arg]);
        long long base = ((long long)inst * in_total_stages[arg] + chunk_start)
                          * in_nnz_per_stage[arg];
        w[input_slots[i]] = inputs[arg][base + input_elems[i]];
    }
    __syncthreads();

    // --- Compute levels (identical to untiled LP / stage kernel) ---
    for (int lev = 0; lev < n_levels; lev++) {
        int offset = level_offsets[lev];
        int width = level_widths[lev];
        for (int base = 0; base < width; base += block_sz) {
            int idx = base + tid;
            {dtype} ra = 0.0, rb = 0.0;
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

    // --- Write outputs (chunk-aware indexing; outputs are STAGE only) ---
    for (int i = tid; i < n_outputs; i += block_sz) {
        int arg  = out_args[i];
        int mode = out_stage_mode[arg];
        int chunk_start = (mode == 1) ? 0 : (chunk_idx * chunk_size);
        long long base = ((long long)inst * out_total_stages[arg] + chunk_start)
                          * out_nnz_per_stage[arg];
        outputs[arg][base + out_elems[i]] = w[out_slots[i]];
    }
}

extern "C" {

float evaluate_lp_chunk(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int n_chunks,
        const int chunk_size,
        const int block_size,
        const int shared_bytes,
        const int n_consts,
        const int *const_slots,
        const {dtype} *const_vals,
        const int n_inputs,
        const int *input_slots,
        const int *input_args,
        const int *input_elems,
        const int *in_stage_mode,
        const int *in_chunk_offset,
        const int *in_total_stages,
        const int *in_nnz_per_stage,
        const int n_levels,
        const int *level_offsets,
        const int *level_widths,
        const int *node_ops,
        const int *node_outs,
        const int *node_in_a,
        const int *node_in_b,
        const int n_outputs,
        const int *out_slots,
        const int *out_args,
        const int *out_elems,
        const int *out_stage_mode,
        const int *out_total_stages,
        const int *out_nnz_per_stage) {

    dim3 grid(total_instances, n_chunks);
    dim3 block(block_size);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_chunk_kernel<<<grid, block, shared_bytes>>>(
        inputs, outputs, total_instances, n_chunks, chunk_size,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems,
        in_stage_mode, in_chunk_offset, in_total_stages, in_nnz_per_stage,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems,
        out_stage_mode, out_total_stages, out_nnz_per_stage);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return milliseconds / 1000.0f;
}

int evaluate_lp_chunk_stream(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int n_chunks,
        const int chunk_size,
        const int block_size,
        const int shared_bytes,
        const int n_consts,
        const int *const_slots,
        const {dtype} *const_vals,
        const int n_inputs,
        const int *input_slots,
        const int *input_args,
        const int *input_elems,
        const int *in_stage_mode,
        const int *in_chunk_offset,
        const int *in_total_stages,
        const int *in_nnz_per_stage,
        const int n_levels,
        const int *level_offsets,
        const int *level_widths,
        const int *node_ops,
        const int *node_outs,
        const int *node_in_a,
        const int *node_in_b,
        const int n_outputs,
        const int *out_slots,
        const int *out_args,
        const int *out_elems,
        const int *out_stage_mode,
        const int *out_total_stages,
        const int *out_nnz_per_stage,
        cudaStream_t stream) {

    dim3 grid(total_instances, n_chunks);
    dim3 block(block_size);
    evaluate_chunk_kernel<<<grid, block, shared_bytes, stream>>>(
        inputs, outputs, total_instances, n_chunks, chunk_size,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems,
        in_stage_mode, in_chunk_offset, in_total_stages, in_nnz_per_stage,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems,
        out_stage_mode, out_total_stages, out_nnz_per_stage);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

int configure_chunk_shared_memory(int shared_bytes) {
    cudaFuncSetAttribute(evaluate_chunk_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
    return 0;
}

} // extern "C"
"""


STAGE_TILED_KERNEL_TEMPLATE = r"""
// AUTOMATICALLY GENERATED: stage-aware tiled level-parallel kernel.
//
// Closes the regime where a single f_stage's working set exceeds the
// device's shared memory budget. Each block processes one (instance,
// stage_chunk) pair; chunk_size is implicitly 1 along the stage axis
// (fusing multiple too-big stages would only worsen the smem problem),
// so n_stage_chunks == n_stages here. Within the block, the per-stage
// DAG is tiled over `n_tiles` chunks with spill/reload at boundaries —
// identical mechanism to the standalone tiled kernel.
//
// blockIdx.x : batch instance
// blockIdx.y : stage index (== stage_chunk_idx since chunk_size==1)
//
// Spill buffer layout: (total_instances, n_stages, spill_stride).
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

enum CasadiOp {
    OP_ASSIGN=0, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_NEG,
    OP_EXP, OP_LOG, OP_POW, OP_CONSTPOW, OP_SQRT, OP_SQ,
    OP_TWICE, OP_SIN, OP_COS, OP_TAN, OP_ASIN, OP_ACOS,
    OP_ATAN,
    OP_LT, OP_LE, OP_EQ, OP_NE, OP_NOT, OP_AND, OP_OR,
    OP_FLOOR, OP_CEIL, OP_FMOD, OP_FABS, OP_SIGN, OP_COPYSIGN,
    OP_IF_ELSE_ZERO, OP_ERF, OP_FMIN, OP_FMAX, OP_INV,
    OP_SINH, OP_COSH, OP_TANH, OP_ASINH, OP_ACOSH, OP_ATANH,
    OP_ATAN2
};

__EXEC_OP__
__global__ void evaluate_stage_tiled_kernel(
        const {dtype} * const * __restrict__ inputs,
        {dtype} ** __restrict__ outputs,
        const int total_instances,
        const int n_stages,
        // Init data
        const int n_consts,
        const int * __restrict__ const_slots,
        const {dtype} * __restrict__ const_vals,
        const int n_inputs,
        const int * __restrict__ input_slots,
        const int * __restrict__ input_args,
        const int * __restrict__ input_elems,
        // Stage-axis indexing for inputs (per-arg, length n_in of f_stage)
        const int * __restrict__ in_stage_mode,
        const int * __restrict__ in_chunk_offset,
        const int * __restrict__ in_total_stages,
        const int * __restrict__ in_nnz_per_stage,
        // Level data (per-stage DAG, possibly very large)
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
        const int * __restrict__ out_stage_mode,
        const int * __restrict__ out_total_stages,
        const int * __restrict__ out_nnz_per_stage,
        // Tiling data — same as standalone tiled kernel
        {dtype} * __restrict__ spill_buffer,
        const int spill_stride,
        const int n_tiles,
        const int * __restrict__ chunk_lev_start,
        const int * __restrict__ chunk_lev_end,
        const int * __restrict__ bnd_spill_off,
        const int * __restrict__ bnd_spill_cnt,
        const int * __restrict__ bnd_spill_smem,
        const int * __restrict__ bnd_spill_pos,
        const int * __restrict__ bnd_reload_off,
        const int * __restrict__ bnd_reload_cnt,
        const int * __restrict__ bnd_reload_smem,
        const int * __restrict__ bnd_reload_pos,
        const int * __restrict__ lev_sb_off,
        const int * __restrict__ lev_sb_cnt,
        const int * __restrict__ sb_spill_smem,
        const int * __restrict__ sb_spill_pos) {

    extern __shared__ {dtype} w[];
    const int inst  = blockIdx.x;
    const int stage = blockIdx.y;
    const int tid   = threadIdx.x;
    const int block_sz = blockDim.x;
    if (inst >= total_instances || stage >= n_stages) return;

    // Spill buffer is per-(instance, stage). chunk_size==1 so n_stage_chunks
    // == n_stages and we offset by (inst * n_stages + stage) * spill_stride.
    {dtype} * spill = spill_buffer
                   + ((long long)inst * n_stages + stage) * spill_stride;

    // --- Init: load constants ---
    for (int i = tid; i < n_consts; i += block_sz)
        w[const_slots[i]] = const_vals[i];

    // --- Init: load external inputs (stage-aware indexing) ---
    for (int i = tid; i < n_inputs; i += block_sz) {
        int arg  = input_args[i];
        int mode = in_stage_mode[arg];
        int chunk_start = (mode == 1) ? 0
                        : (stage + in_chunk_offset[arg]);
        long long base = ((long long)inst * in_total_stages[arg] + chunk_start)
                          * in_nnz_per_stage[arg];
        w[input_slots[i]] = inputs[arg][base + input_elems[i]];
    }
    __syncthreads();

    // --- Tiled compute (identical to evaluate_kernel_tiled) ---
    for (int tile = 0; tile < n_tiles; tile++) {
        if (tile > 0) {
            int bi = tile - 1;
            int roff = bnd_reload_off[bi];
            int rcnt = bnd_reload_cnt[bi];
            for (int i = tid; i < rcnt; i += block_sz)
                w[bnd_reload_smem[roff + i]] = spill[bnd_reload_pos[roff + i]];
            __syncthreads();
        }

        for (int lev = chunk_lev_start[tile]; lev < chunk_lev_end[tile]; lev++) {
            int offset = level_offsets[lev];
            int width = level_widths[lev];

            for (int base = 0; base < width; base += block_sz) {
                int idx = base + tid;
                {dtype} ra = 0.0, rb = 0.0;
                int my_op = 0, my_out = 0;
                if (idx < width) {
                    int g = offset + idx;
                    ra = w[node_in_a[g]];
                    rb = w[node_in_b[g]];
                    my_op = node_ops[g];
                    my_out = node_outs[g];
                }
                __syncthreads();
                if (idx < width) {
                    w[my_out] = exec_op(my_op, ra, rb);
                }
                __syncthreads();
            }

            int sb_cnt = lev_sb_cnt[lev];
            if (sb_cnt > 0) {
                int sb_off = lev_sb_off[lev];
                for (int i = tid; i < sb_cnt; i += block_sz)
                    spill[sb_spill_pos[sb_off + i]] = w[sb_spill_smem[sb_off + i]];
                __syncthreads();
            }
        }

        if (tile < n_tiles - 1) {
            int soff = bnd_spill_off[tile];
            int scnt = bnd_spill_cnt[tile];
            for (int i = tid; i < scnt; i += block_sz)
                spill[bnd_spill_pos[soff + i]] = w[bnd_spill_smem[soff + i]];
            __syncthreads();
        }
    }

    // --- Write outputs (stage-aware indexing) ---
    for (int i = tid; i < n_outputs; i += block_sz) {
        int arg  = out_args[i];
        int mode = out_stage_mode[arg];
        int chunk_start = (mode == 1) ? 0 : stage;
        long long base = ((long long)inst * out_total_stages[arg] + chunk_start)
                          * out_nnz_per_stage[arg];
        outputs[arg][base + out_elems[i]] = w[out_slots[i]];
    }
}

extern "C" {

float evaluate_lp_stage_tiled(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int n_stages,
        const int block_size,
        const int shared_bytes,
        const int n_consts,
        const int *const_slots,
        const {dtype} *const_vals,
        const int n_inputs,
        const int *input_slots,
        const int *input_args,
        const int *input_elems,
        const int *in_stage_mode,
        const int *in_chunk_offset,
        const int *in_total_stages,
        const int *in_nnz_per_stage,
        const int n_levels,
        const int *level_offsets,
        const int *level_widths,
        const int *node_ops,
        const int *node_outs,
        const int *node_in_a,
        const int *node_in_b,
        const int n_outputs,
        const int *out_slots,
        const int *out_args,
        const int *out_elems,
        const int *out_stage_mode,
        const int *out_total_stages,
        const int *out_nnz_per_stage,
        {dtype} *spill_buffer,
        const int spill_stride,
        const int n_tiles,
        const int *chunk_lev_start,
        const int *chunk_lev_end,
        const int *bnd_spill_off,
        const int *bnd_spill_cnt,
        const int *bnd_spill_smem,
        const int *bnd_spill_pos,
        const int *bnd_reload_off,
        const int *bnd_reload_cnt,
        const int *bnd_reload_smem,
        const int *bnd_reload_pos,
        const int *lev_sb_off,
        const int *lev_sb_cnt,
        const int *sb_spill_smem,
        const int *sb_spill_pos) {

    dim3 grid(total_instances, n_stages);
    dim3 block(block_size);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_stage_tiled_kernel<<<grid, block, shared_bytes>>>(
        inputs, outputs, total_instances, n_stages,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems,
        in_stage_mode, in_chunk_offset, in_total_stages, in_nnz_per_stage,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems,
        out_stage_mode, out_total_stages, out_nnz_per_stage,
        spill_buffer, spill_stride, n_tiles,
        chunk_lev_start, chunk_lev_end,
        bnd_spill_off, bnd_spill_cnt, bnd_spill_smem, bnd_spill_pos,
        bnd_reload_off, bnd_reload_cnt, bnd_reload_smem, bnd_reload_pos,
        lev_sb_off, lev_sb_cnt, sb_spill_smem, sb_spill_pos);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return milliseconds / 1000.0f;
}

int configure_stage_tiled_shared_memory(int shared_bytes) {
    cudaFuncSetAttribute(evaluate_stage_tiled_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
    return 0;
}

} // extern "C"
"""


# Upstream per-instruction baseline: NOT a data-driven interpreter — one
# hardcoded line per CasADi instruction is spliced into __BODY__, so the .cu
# grows with the instruction count. Parallel over the batch only (one thread per
# instance). Kept as the baseline the level-parallel kernels are measured against.
SEQUENTIAL_KERNEL_TEMPLATE = r"""// AUTOMATICALLY GENERATED CODE FOR CUSADI
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

__constant__ int nnz_in[] = {__NNZ_IN__};
__constant__ int nnz_out[] = {__NNZ_OUT__};
__constant__ int n_w = __N_W__;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void evaluate_kernel(
        const {dtype} *inputs[],
        {dtype} *work,
        {dtype} *outputs[],
        const int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int env_idx = idx * n_w;
    if (idx < batch_size) {__BODY__
    }
}

extern "C" {

float evaluate(const {dtype} *inputs[],
               {dtype} *work,
               {dtype} *outputs[],
               const int batch_size) {
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_kernel<<<gridSize, blockSize>>>(inputs, work, outputs, batch_size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);__DEBUG__
    return milliseconds / 1000;
}

}
"""


# Unrolled 2-axis variant of SEQUENTIAL_KERNEL_TEMPLATE: same per-instruction body
# (__BODY__), but a 2D launch over (batch_size, n_instances) with a flat
# idx = batch_idx*n_instances + pattern_idx. The instance axis is treated as a
# second independent batch axis (no stage-mode SHARED/COUPLED indexing).
TWOD_KERNEL_TEMPLATE = r"""// AUTOMATICALLY GENERATED CODE FOR CUSADI (2D launch)
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

__constant__ int nnz_in[] = {__NNZ_IN__};
__constant__ int nnz_out[] = {__NNZ_OUT__};
__constant__ int n_w = __N_W__;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void evaluate_kernel(
        const {dtype} *inputs[],
        {dtype} *work,
        {dtype} *outputs[],
        const int batch_size,
        const int n_instances) {
    int batch_idx = blockIdx.x;
    int pattern_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || pattern_idx >= n_instances) return;
    int idx = batch_idx * n_instances + pattern_idx;
    int env_idx = idx * n_w;__BODY__
}

extern "C" {

float evaluate(const {dtype} *inputs[],
               {dtype} *work,
               {dtype} *outputs[],
               const int batch_size,
               const int n_instances) {
    int threadsPerBlock = 256;
    if (n_instances < threadsPerBlock) threadsPerBlock = n_instances;
    dim3 grid(batch_size, (n_instances + threadsPerBlock - 1) / threadsPerBlock);
    dim3 block(threadsPerBlock);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_kernel<<<grid, block>>>(inputs, work, outputs, batch_size, n_instances);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);__DEBUG__
    return milliseconds / 1000;
}

}
"""


def _with_dtype(template: str, dtype: str) -> str:
    """Fill the {dtype} placeholder with the kernel's data-scalar C type.

    The templates write {dtype} at every data-scalar position; the timing return
    type is a fixed `float`, so it is unaffected. NOTE: this changes only the
    *kernel* — the caller must match the device buffers it passes and the
    shared-memory byte budget (LevelPlan.shared_mem_bytes assumes 8 bytes) to the
    chosen type, or results are garbage. The validated path is fp64.
    """
    return template.replace("{dtype}", dtype)


def _build_exec_op(dtype: str) -> str:
    """Generate the exec_op device function for the data-driven templates from the
    single-source op->expression map (OP_EXPR_{DOUBLE,FLOAT}) — the same map the
    unrolled OP_CUDA_DICT is derived from. Operand placeholders {0}/{1} become the
    register args a/b. This is what makes a float interpreter use float intrinsics
    (expf/...) rather than promoting through the double ones, and removes the 5x
    hand-copied switch."""
    import casadi as ca
    from cusadi.CusadiOperations import OP_EXPR_DOUBLE, OP_EXPR_FLOAT
    names = {getattr(ca, n): n for n in dir(ca) if n.startswith("OP_")}
    exprs = OP_EXPR_FLOAT if dtype == "float" else OP_EXPR_DOUBLE
    zero = "0.f" if dtype == "float" else "0.0"
    def _rhs(expr):
        return expr.replace("{0}", "a").replace("{1}", "b")
    cases = "\n".join("        case %s: return %s;" % (names[op], _rhs(expr))
                      for op, expr in exprs.items())
    return (f"__device__ __forceinline__ {dtype} exec_op(int op, {dtype} a, {dtype} b) {{\n"
            f"    switch (op) {{\n{cases}\n        default: return {zero};\n    }}\n}}")


def _render_lp(template: str, dtype: str) -> str:
    """Fill {dtype} and inject the generated exec_op into a data-driven template."""
    return _with_dtype(template, dtype).replace("__EXEC_OP__", _build_exec_op(dtype))


def pick_block_size(plan) -> int:
    """Smallest power of 2 >= median width, capped at 256."""
    med = plan.median_width
    if med <= 0:
        return 32
    bs = 32
    while bs < med and bs < 256:
        bs *= 2
    return bs


def write_kernel_source(filepath: str, dtype: str = "double"):
    """Write the fixed kernel template to a .cu file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(_render_lp(KERNEL_TEMPLATE, dtype))
    return filepath


def write_tiled_kernel_source(filepath: str, dtype: str = "double"):
    """Write the tiled kernel template to a .cu file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(_render_lp(TILED_KERNEL_TEMPLATE, dtype))
    return filepath


def write_stage_kernel_source(filepath: str, dtype: str = "double"):
    """Write the stage-aware kernel template to a .cu file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(_render_lp(STAGE_KERNEL_TEMPLATE, dtype))
    return filepath


def write_chunk_kernel_source(filepath: str, dtype: str = "double"):
    """Write the chunked stage kernel template to a .cu file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(_render_lp(CHUNK_KERNEL_TEMPLATE, dtype))
    return filepath


def write_stage_tiled_kernel_source(filepath: str, dtype: str = "double"):
    """Write the stage-aware tiled kernel template to a .cu file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(_render_lp(STAGE_TILED_KERNEL_TEMPLATE, dtype))
    return filepath


def _unrolled_instruction_body(f, dtype: str = "double") -> str:
    """One OP_CUDA_DICT_{DOUBLE,FLOAT} line per CasADi instruction (shared by the
    sequential and 2D unrolled templates). `dtype` selects the op map so the body's
    intrinsics/literals match the kernel's scalar type (double: exp/2.; float:
    expf/2.f). Matches the historical generateCUDACode emission for double."""
    from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ, OP_SIGN
    from cusadi.CusadiOperations import OP_CUDA_DICT_DOUBLE, OP_CUDA_DICT_FLOAT
    op_dict = OP_CUDA_DICT_FLOAT if dtype == "float" else OP_CUDA_DICT_DOUBLE

    in_idx, in_len = [], [0]
    out_idx, out_len = [], [0]
    for k in range(f.n_instructions()):
        in_idx.extend(f.instruction_input(k))
        in_len.append(len(f.instruction_input(k)))
        out_idx.extend(f.instruction_output(k))
        out_len.append(len(f.instruction_output(k)))

    body = ""
    o = i = 0
    for k in range(f.n_instructions()):
        op = f.instruction_id(k)
        tmpl = op_dict[op]
        o_idx, i_idx = out_idx[o], in_idx[i]
        if op == OP_CONST:
            body += tmpl % (o_idx, f.instruction_constant(k))
        elif op == OP_INPUT:
            body += tmpl % (o_idx, i_idx, i_idx, in_idx[i + 1])
        elif op == OP_OUTPUT:
            body += tmpl % (o_idx, o_idx, out_idx[o + 1], i_idx)
        elif op in (OP_SQ, OP_SIGN):
            body += tmpl % (o_idx, i_idx, i_idx)
        elif tmpl.count("%d") == 3:
            body += tmpl % (o_idx, i_idx, in_idx[i + 1])
        elif tmpl.count("%d") == 2:
            body += tmpl % (o_idx, i_idx)
        else:
            raise Exception("Unknown CasADi operation: " + str(op))
        o += out_len[k + 1]
        i += in_len[k + 1]
    return body


def _debug_lines(debug_mode: bool) -> str:
    return ("\n    gpuErrchk(cudaPeekAtLastError());"
            "\n    gpuErrchk(cudaDeviceSynchronize());") if debug_mode else ""


def write_sequential_kernel_source(f, filepath: str, debug_mode: bool = True,
                                   dtype: str = "double"):
    """Write the upstream per-instruction sequential kernel for CasADi function `f`.

    Emits one OP_CUDA_DICT line per instruction into SEQUENTIAL_KERNEL_TEMPLATE,
    matching the historical generateCUDACodeDouble output.
    """
    src = (_with_dtype(SEQUENTIAL_KERNEL_TEMPLATE, dtype)
           .replace("__NNZ_IN__", ",".join(str(f.nnz_in(j)) for j in range(f.n_in())))
           .replace("__NNZ_OUT__", ",".join(str(f.nnz_out(j)) for j in range(f.n_out())))
           .replace("__N_W__", str(f.sz_w()))
           .replace("__BODY__", _unrolled_instruction_body(f, dtype))
           .replace("__DEBUG__", _debug_lines(debug_mode)))

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as out:
        out.write(src)
    return filepath


def write_2d_kernel_source(f, filepath: str, debug_mode: bool = True,
                           dtype: str = "double"):
    """Write the unrolled 2-axis (batch x instance) kernel for CasADi function `f`.

    Same per-instruction body as the sequential kernel, but a 2D launch with
    idx = batch_idx*n_instances + pattern_idx. Matches the historical
    generateCUDACode2D output.
    """
    src = (_with_dtype(TWOD_KERNEL_TEMPLATE, dtype)
           .replace("__NNZ_IN__", ",".join(str(f.nnz_in(j)) for j in range(f.n_in())))
           .replace("__NNZ_OUT__", ",".join(str(f.nnz_out(j)) for j in range(f.n_out())))
           .replace("__N_W__", str(f.sz_w()))
           .replace("__BODY__", _unrolled_instruction_body(f, dtype))
           .replace("__DEBUG__", _debug_lines(debug_mode)))

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as out:
        out.write(src)
    return filepath
