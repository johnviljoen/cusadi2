"""Unified level-parallel CUDA kernel template.

This is the single source of truth for the data-driven level-parallel kernel that
replaces the five separate templates in ``cusadi/level_parallel/codegen.py``
(``KERNEL_TEMPLATE``, ``STAGE_KERNEL_TEMPLATE``, ``CHUNK_KERNEL_TEMPLATE``,
``TILED_KERNEL_TEMPLATE``, ``STAGE_TILED_KERNEL_TEMPLATE``).

Design — collapse on the two axes that fold for free, specialize only on the one
that has a hot-loop cost:

  * **Stage axis (LP / stage / chunk) — always on, runtime-parameterised.**
    The kernel is stage-aware: ``grid = (batch, n_stages)`` with per-arg
    ``STAGE`` / ``SHARED`` / ``COUPLED`` indexing. Pure batch level-parallel is
    the ``n_stages == 1``, all-``STAGE`` special case (indexing reduces exactly
    to ``inst * nnz + elem``). "Chunked stages" is the same kernel run on a
    stamped DAG with chunk-stride indexing — a serializer concern, not a kernel
    variant. The extra stage arithmetic lives only in the cold init/output
    loops, never in the per-level compute loop, so folding it in is ~free.

  * **Tiling axis (untiled / tiled) — a compile-time switch (``CUSADI_TILED``).**
    This is the *only* axis with hot-loop cost. The untiled body exploits the
    SSA-renaming guarantee (no slot is both read and written within a level) to
    emit a single ``__syncthreads()`` per level. The tiled body recycles shared
    memory slots (pool reset at chunk boundaries, sub-batch reuse), which breaks
    that guarantee, so it must read-all -> barrier -> write-all -> barrier — ~2x
    the barriers per level. We therefore refuse to pay that on the untiled fast
    path: ``#if CUSADI_TILED`` selects the body at compile time.

One source file, compiled twice:

    nvcc -DCUSADI_TILED=0 ...   -> exports evaluate_cusadi{,_stream},
                                   configure_cusadi_shared_memory
    nvcc -DCUSADI_TILED=1 ...   -> exports evaluate_cusadi_tiled{,_stream},
                                   configure_cusadi_tiled_shared_memory

Both variants share the kernel signature (the tiling pointers are simply unused
when ``CUSADI_TILED == 0``; the untiled wrapper passes nulls for them), the
op enum, the generated ``exec_op``, and the stage-aware init/output loops.

The validated data type is fp64; ``dtype="float"`` is supported for the kernel
body but the caller must match device buffers and the shared-memory byte budget
to the chosen type.
"""

import os

from cusadi.level_parallel.codegen import _render_lp


# ---------------------------------------------------------------------------
# Unified CUDA template.  {dtype} and __EXEC_OP__ are filled by render_kernel();
# CUSADI_TILED is resolved by nvcc at compile time (-DCUSADI_TILED=0|1).
# ---------------------------------------------------------------------------
UNIFIED_KERNEL_TEMPLATE = r"""
// AUTOMATICALLY GENERATED: unified level-parallel data-driven kernel.
//
// Stage-aware (grid = batch x n_stages) and tiling-capable. The compute body is
// selected at compile time by CUSADI_TILED; everything else is shared.
//
//   -DCUSADI_TILED=0  -> single-barrier-per-level fast path (LP / stage)
//   -DCUSADI_TILED=1  -> chunked spill/reload + sub-batch path (tiled / stage-tiled)
//
// Stage modes (per arg): 0=STAGE, 1=SHARED, 2=COUPLED.
//   STAGE   -> leading stride n_stages,        knot s = stage
//   SHARED  -> leading stride 1,               knot s = 0
//   COUPLED -> leading stride n_stages+span,   knot s = stage + offset
// The unified index is  ((inst * stage_count[arg] + s) * nnz[arg] + elem),
// with s = (mode==SHARED) ? 0 : (stage + stage_offset[arg])   (offset 0 unless
// COUPLED). Pure batch LP is n_stages=1, all-STAGE, stage_count=1 -> inst*nnz+elem.
#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>

#ifndef CUSADI_TILED
#define CUSADI_TILED 0
#endif

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
        const int * __restrict__ in_stage_count,
        // Compute data (per-stage level DAG)
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
        const int * __restrict__ out_stage_count,
        // Tiling data (unused / null when CUSADI_TILED == 0)
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

    // --- Init: load constants ---
    for (int i = tid; i < n_consts; i += block_sz)
        w[const_slots[i]] = const_vals[i];

    // --- Init: load external inputs (stage-aware indexing) ---
    for (int i = tid; i < n_inputs; i += block_sz) {
        int arg  = input_args[i];
        int mode = in_stage_mode[arg];
        int s = (mode == 1) ? 0 : (stage + in_stage_offset[arg]);
        long long base = ((long long)inst * in_stage_count[arg] + s) * nnz_in[arg];
        w[input_slots[i]] = inputs[arg][base + input_elems[i]];
    }
    __syncthreads();

#if CUSADI_TILED
    // ===================== tiled compute =====================
    // Spill buffer is per-(instance, stage). For 1-axis tiling n_stages==1 and
    // stage==0, so this reduces to inst * spill_stride.
    {dtype} * spill = spill_buffer
                   + ((long long)inst * n_stages + stage) * spill_stride;

    for (int tile = 0; tile < n_tiles; tile++) {
        // --- Reload phase (skip first tile) ---
        if (tile > 0) {
            int bi = tile - 1;
            int roff = bnd_reload_off[bi];
            int rcnt = bnd_reload_cnt[bi];
            for (int i = tid; i < rcnt; i += block_sz)
                w[bnd_reload_smem[roff + i]] = spill[bnd_reload_pos[roff + i]];
            __syncthreads();
        }

        // --- Process levels in this tile ---
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
                __syncthreads();                 // slot recycling -> read/write barrier
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

        // --- Inter-tile spill (skip last tile) ---
        if (tile < n_tiles - 1) {
            int soff = bnd_spill_off[tile];
            int scnt = bnd_spill_cnt[tile];
            for (int i = tid; i < scnt; i += block_sz)
                spill[bnd_spill_pos[soff + i]] = w[bnd_spill_smem[soff + i]];
            __syncthreads();
        }
    }
#else
    // ===================== untiled compute (LP fast path) =====================
    // SSA renaming guarantees no slot is both read and written within a level,
    // so the read/write barrier is unnecessary — one __syncthreads() per level.
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
#endif

    // --- Write outputs (stage-aware indexing) ---
    for (int i = tid; i < n_outputs; i += block_sz) {
        int arg  = out_args[i];
        int mode = out_stage_mode[arg];
        int s = (mode == 1) ? 0 : stage;
        long long base = ((long long)inst * out_stage_count[arg] + s) * nnz_out[arg];
        outputs[arg][base + out_elems[i]] = w[out_slots[i]];
    }
}

extern "C" {

#if CUSADI_TILED

float evaluate_cusadi_tiled(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int n_stages,
        const int block_size,
        const int shared_bytes,
        const int n_consts, const int *const_slots, const {dtype} *const_vals,
        const int n_inputs, const int *input_slots, const int *input_args,
        const int *input_elems, const int *nnz_in,
        const int *in_stage_mode, const int *in_stage_offset, const int *in_stage_count,
        const int n_levels, const int *level_offsets, const int *level_widths,
        const int *node_ops, const int *node_outs, const int *node_in_a, const int *node_in_b,
        const int n_outputs, const int *out_slots, const int *out_args,
        const int *out_elems, const int *nnz_out,
        const int *out_stage_mode, const int *out_stage_count,
        {dtype} *spill_buffer, const int spill_stride, const int n_tiles,
        const int *chunk_lev_start, const int *chunk_lev_end,
        const int *bnd_spill_off, const int *bnd_spill_cnt,
        const int *bnd_spill_smem, const int *bnd_spill_pos,
        const int *bnd_reload_off, const int *bnd_reload_cnt,
        const int *bnd_reload_smem, const int *bnd_reload_pos,
        const int *lev_sb_off, const int *lev_sb_cnt,
        const int *sb_spill_smem, const int *sb_spill_pos) {

    dim3 grid(total_instances, n_stages);
    dim3 block(block_size);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_kernel<<<grid, block, shared_bytes>>>(
        inputs, outputs, total_instances, n_stages,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems, nnz_in,
        in_stage_mode, in_stage_offset, in_stage_count,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems, nnz_out,
        out_stage_mode, out_stage_count,
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

int evaluate_cusadi_tiled_stream(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int n_stages,
        const int block_size,
        const int shared_bytes,
        const int n_consts, const int *const_slots, const {dtype} *const_vals,
        const int n_inputs, const int *input_slots, const int *input_args,
        const int *input_elems, const int *nnz_in,
        const int *in_stage_mode, const int *in_stage_offset, const int *in_stage_count,
        const int n_levels, const int *level_offsets, const int *level_widths,
        const int *node_ops, const int *node_outs, const int *node_in_a, const int *node_in_b,
        const int n_outputs, const int *out_slots, const int *out_args,
        const int *out_elems, const int *nnz_out,
        const int *out_stage_mode, const int *out_stage_count,
        {dtype} *spill_buffer, const int spill_stride, const int n_tiles,
        const int *chunk_lev_start, const int *chunk_lev_end,
        const int *bnd_spill_off, const int *bnd_spill_cnt,
        const int *bnd_spill_smem, const int *bnd_spill_pos,
        const int *bnd_reload_off, const int *bnd_reload_cnt,
        const int *bnd_reload_smem, const int *bnd_reload_pos,
        const int *lev_sb_off, const int *lev_sb_cnt,
        const int *sb_spill_smem, const int *sb_spill_pos,
        cudaStream_t stream) {

    dim3 grid(total_instances, n_stages);
    dim3 block(block_size);
    evaluate_kernel<<<grid, block, shared_bytes, stream>>>(
        inputs, outputs, total_instances, n_stages,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems, nnz_in,
        in_stage_mode, in_stage_offset, in_stage_count,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems, nnz_out,
        out_stage_mode, out_stage_count,
        spill_buffer, spill_stride, n_tiles,
        chunk_lev_start, chunk_lev_end,
        bnd_spill_off, bnd_spill_cnt, bnd_spill_smem, bnd_spill_pos,
        bnd_reload_off, bnd_reload_cnt, bnd_reload_smem, bnd_reload_pos,
        lev_sb_off, lev_sb_cnt, sb_spill_smem, sb_spill_pos);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

int configure_cusadi_tiled_shared_memory(int shared_bytes) {
    cudaFuncSetAttribute(evaluate_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
    return 0;
}

#else  // ----------------------------- untiled -----------------------------

float evaluate_cusadi(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int n_stages,
        const int block_size,
        const int shared_bytes,
        const int n_consts, const int *const_slots, const {dtype} *const_vals,
        const int n_inputs, const int *input_slots, const int *input_args,
        const int *input_elems, const int *nnz_in,
        const int *in_stage_mode, const int *in_stage_offset, const int *in_stage_count,
        const int n_levels, const int *level_offsets, const int *level_widths,
        const int *node_ops, const int *node_outs, const int *node_in_a, const int *node_in_b,
        const int n_outputs, const int *out_slots, const int *out_args,
        const int *out_elems, const int *nnz_out,
        const int *out_stage_mode, const int *out_stage_count) {

    dim3 grid(total_instances, n_stages);
    dim3 block(block_size);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_kernel<<<grid, block, shared_bytes>>>(
        inputs, outputs, total_instances, n_stages,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems, nnz_in,
        in_stage_mode, in_stage_offset, in_stage_count,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems, nnz_out,
        out_stage_mode, out_stage_count,
        // tiling params unused on this path
        nullptr, 0, 1, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return milliseconds / 1000.0f;
}

int evaluate_cusadi_stream(
        const {dtype} *inputs[],
        {dtype} *outputs[],
        const int total_instances,
        const int n_stages,
        const int block_size,
        const int shared_bytes,
        const int n_consts, const int *const_slots, const {dtype} *const_vals,
        const int n_inputs, const int *input_slots, const int *input_args,
        const int *input_elems, const int *nnz_in,
        const int *in_stage_mode, const int *in_stage_offset, const int *in_stage_count,
        const int n_levels, const int *level_offsets, const int *level_widths,
        const int *node_ops, const int *node_outs, const int *node_in_a, const int *node_in_b,
        const int n_outputs, const int *out_slots, const int *out_args,
        const int *out_elems, const int *nnz_out,
        const int *out_stage_mode, const int *out_stage_count,
        cudaStream_t stream) {

    dim3 grid(total_instances, n_stages);
    dim3 block(block_size);
    evaluate_kernel<<<grid, block, shared_bytes, stream>>>(
        inputs, outputs, total_instances, n_stages,
        n_consts, const_slots, const_vals,
        n_inputs, input_slots, input_args, input_elems, nnz_in,
        in_stage_mode, in_stage_offset, in_stage_count,
        n_levels, level_offsets, level_widths,
        node_ops, node_outs, node_in_a, node_in_b,
        n_outputs, out_slots, out_args, out_elems, nnz_out,
        out_stage_mode, out_stage_count,
        nullptr, 0, 1, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

int configure_cusadi_shared_memory(int shared_bytes) {
    cudaFuncSetAttribute(evaluate_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
    return 0;
}

#endif  // CUSADI_TILED

} // extern "C"
"""


def render_kernel(dtype: str = "double") -> str:
    """Fill {dtype} and inject the generated exec_op into the unified template.

    The result still contains the ``#if CUSADI_TILED`` preprocessor switch; pick
    the variant at compile time with ``-DCUSADI_TILED=0`` (untiled / LP fast
    path) or ``-DCUSADI_TILED=1`` (tiled). Both share this single source file.
    """
    return _render_lp(UNIFIED_KERNEL_TEMPLATE, dtype)


def write_kernel_source(filepath: str, dtype: str = "double") -> str:
    """Write the unified kernel template to a .cu file.

    Compile it twice to get both variants, e.g.::

        nvcc -DCUSADI_TILED=0 -o liblp.so        --shared -Xcompiler -fPIC kernel.cu
        nvcc -DCUSADI_TILED=1 -o liblp_tiled.so  --shared -Xcompiler -fPIC kernel.cu
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(render_kernel(dtype))
    return filepath
