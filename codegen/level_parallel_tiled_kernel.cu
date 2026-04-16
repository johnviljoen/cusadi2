
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

__global__ void evaluate_kernel_tiled(
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
        double * __restrict__ spill_buffer,
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

    extern __shared__ double w[];
    const int inst = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_sz = blockDim.x;
    if (inst >= total_instances) return;

    double * spill = spill_buffer + inst * spill_stride;

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
                double ra = 0.0, rb = 0.0;
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
        const double *inputs[],
        double *outputs[],
        const int total_instances,
        const int block_size,
        const int shared_bytes,
        // Init
        const int n_consts, const int *const_slots, const double *const_vals,
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
        double *spill_buffer, const int spill_stride, const int n_chunks,
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
