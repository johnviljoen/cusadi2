// cusadi2 — host-free C++/CUDA runtime for the function-independent level-parallel
// stage kernel. No torch, no jax, no Python at runtime.
//
// Pair a generated plan header (see cusadi2.generate.generate_native_plan) with the
// kernel source emitted by cusadi2.kernel_template:
//
//   #include "cusadi2.h"
//   #include "myfn_plan.h"           // generated: provides MYFN_plan (StagePlanHost)
//   cusadi2::StageProgram prog; prog.upload(MYFN_plan);
//   prog.evaluate(d_in, d_out, batch, n_stages);   // d_in/d_out: device pointer arrays
//
// Stage is the one path: the kernel runs one block per (instance, stage) over a
// (batch x n_stages) grid, and every block runs the full level-parallel evaluation.
// Plain batch-only level parallelism is just n_stages == 1 with every arg STAGE.
// Inputs/outputs are caller-owned device buffers laid out batch-major (instance *
// nnz + element); the pointer arrays themselves live in device memory.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#ifndef CUSADI2_CHECK
#define CUSADI2_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "cusadi2 CUDA error '%s' at %s:%d\n",              \
                cudaGetErrorString(_e), __FILE__, __LINE__);                   \
            abort();                                                           \
        }                                                                      \
    } while (0)
#endif

// Scalar precision: must match the dtype the kernel (.cu) was emitted with.
// Compile a float build with -DCUSADI2_SCALAR=float; defaults to double.
#ifndef CUSADI2_SCALAR
#define CUSADI2_SCALAR double
#endif

// --- kernel entry points (defined in the cusadi2.kernel_template .cu) ---------
// Names/signature match the wrappers write_kernel_source emits exactly.
extern "C" float evaluate_cusadi(
    const CUSADI2_SCALAR* inputs[], CUSADI2_SCALAR* outputs[], int total_instances, int n_stages,
    int block_size, int shared_bytes,
    int n_consts, const int* const_slots, const CUSADI2_SCALAR* const_vals,
    int n_inputs, const int* input_slots, const int* input_args, const int* input_elems,
    const int* nnz_in, const int* in_stage_mode, const int* in_stage_offset,
    const int* in_stage_count,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    int n_outputs, const int* out_slots, const int* out_args, const int* out_elems,
    const int* nnz_out, const int* out_stage_mode, const int* out_stage_count);
// Stream-aware sibling: identical, launches on `stream` and does NOT synchronize
// (no event timing, no cudaDeviceSynchronize). Use this for single-stream pipelines
// and CUDA-graph capture. Returns cudaPeekAtLastError() as int.
extern "C" int evaluate_cusadi_stream(
    const CUSADI2_SCALAR* inputs[], CUSADI2_SCALAR* outputs[], int total_instances, int n_stages,
    int block_size, int shared_bytes,
    int n_consts, const int* const_slots, const CUSADI2_SCALAR* const_vals,
    int n_inputs, const int* input_slots, const int* input_args, const int* input_elems,
    const int* nnz_in, const int* in_stage_mode, const int* in_stage_offset,
    const int* in_stage_count,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    int n_outputs, const int* out_slots, const int* out_args, const int* out_elems,
    const int* nnz_out, const int* out_stage_mode, const int* out_stage_count,
    cudaStream_t stream);
extern "C" int configure_cusadi_shared_memory(int shared_bytes);

namespace cusadi2 {

// --- host-side plan descriptors (emitted by generate_native_plan) ------------
// LPPlanHost is the level-parallel program: the compute graph (level/node arrays)
// plus the constant/input/output slot maps. StagePlanHost wraps it with the
// per-arg stage-axis overlay (mode/offset/span). One descriptor, one program.
struct LPPlanHost {
    int n_args_in, n_args_out;
    const int *nnz_in, *nnz_out;
    int block_size, shared_bytes;
    int n_consts;       const int *const_slots; const CUSADI2_SCALAR *const_vals;
    int n_input_maps;   const int *input_slots, *input_args, *input_elems;
    int n_levels;       const int *level_offsets, *level_widths;
    int n_nodes;        const int *node_ops, *node_outs, *node_in_a, *node_in_b;
    int n_output_maps;  const int *out_slots, *out_args, *out_elems;
};

struct StagePlanHost {
    LPPlanHost base;
    const int *in_stage_mode, *in_stage_offset, *in_stage_span;  // length n_args_in
    const int *out_stage_mode;                                   // length n_args_out
};

namespace detail {
inline const int* dev_i(const int* h, int n) {
    int* d = nullptr; n = n > 0 ? n : 1;
    CUSADI2_CHECK(cudaMalloc(&d, size_t(n) * sizeof(int)));
    CUSADI2_CHECK(cudaMemcpy(d, h, size_t(n) * sizeof(int), cudaMemcpyHostToDevice));
    return d;
}
inline const CUSADI2_SCALAR* dev_d(const CUSADI2_SCALAR* h, int n) {
    CUSADI2_SCALAR* d = nullptr; n = n > 0 ? n : 1;
    CUSADI2_CHECK(cudaMalloc(&d, size_t(n) * sizeof(CUSADI2_SCALAR)));
    CUSADI2_CHECK(cudaMemcpy(d, h, size_t(n) * sizeof(CUSADI2_SCALAR), cudaMemcpyHostToDevice));
    return d;
}
}  // namespace detail

// 2-axis (batch x horizon) stage program. grid = (batch, n_stages); one block per
// (instance, stage), each running the full level-parallel evaluation. Batch-only
// level parallelism is this with n_stages == 1 and every arg STAGE.
struct StageProgram {
    StagePlanHost h{};
    // device copies of the base (level-parallel) program arrays
    const int *d_nnz_in, *d_nnz_out;
    const int *d_const_slots; const CUSADI2_SCALAR *d_const_vals;
    const int *d_input_slots, *d_input_args, *d_input_elems;
    const int *d_level_offsets, *d_level_widths;
    const int *d_node_ops, *d_node_outs, *d_node_in_a, *d_node_in_b;
    const int *d_out_slots, *d_out_args, *d_out_elems;
    // device copies of the stage-axis arrays
    const int *d_in_mode, *d_in_offset, *d_out_mode;
    // host copies for the per-n_stages count recompute
    std::vector<int> in_mode_, in_span_, out_mode_;
    int *d_in_count = nullptr, *d_out_count = nullptr;   // recomputed per n_stages
    // persistent count cache for the stream path: the per-arg stage counts depend only
    // on n_stages, so we upload them once and skip the memcpy on subsequent same-n_stages
    // calls (keeps the hot path a pure kernel launch -> single-stream, graph-capturable).
    std::vector<int> ic_, oc_;
    int count_stages_ = -1;

    void upload(const StagePlanHost& p) {
        using namespace detail;
        h = p;
        const LPPlanHost& b = p.base;
        d_nnz_in = dev_i(b.nnz_in, b.n_args_in);   d_nnz_out = dev_i(b.nnz_out, b.n_args_out);
        d_const_slots = dev_i(b.const_slots, b.n_consts);
        d_const_vals  = dev_d(b.const_vals, b.n_consts);
        d_input_slots = dev_i(b.input_slots, b.n_input_maps);
        d_input_args  = dev_i(b.input_args, b.n_input_maps);
        d_input_elems = dev_i(b.input_elems, b.n_input_maps);
        d_level_offsets = dev_i(b.level_offsets, b.n_levels);
        d_level_widths  = dev_i(b.level_widths, b.n_levels);
        d_node_ops  = dev_i(b.node_ops, b.n_nodes);  d_node_outs = dev_i(b.node_outs, b.n_nodes);
        d_node_in_a = dev_i(b.node_in_a, b.n_nodes); d_node_in_b = dev_i(b.node_in_b, b.n_nodes);
        d_out_slots = dev_i(b.out_slots, b.n_output_maps);
        d_out_args  = dev_i(b.out_args, b.n_output_maps);
        d_out_elems = dev_i(b.out_elems, b.n_output_maps);

        const int ni = b.n_args_in, no = b.n_args_out;
        d_in_mode = dev_i(p.in_stage_mode, ni);
        d_in_offset = dev_i(p.in_stage_offset, ni);
        d_out_mode = dev_i(p.out_stage_mode, no);
        in_mode_.assign(p.in_stage_mode, p.in_stage_mode + ni);
        in_span_.assign(p.in_stage_span, p.in_stage_span + ni);
        out_mode_.assign(p.out_stage_mode, p.out_stage_mode + no);
        CUSADI2_CHECK(cudaMalloc(&d_in_count, size_t(ni) * sizeof(int)));
        CUSADI2_CHECK(cudaMalloc(&d_out_count, size_t(no) * sizeof(int)));
        configure_cusadi_shared_memory(b.shared_bytes);
    }

    // Leading-axis stride per arg given the horizon length: STAGE->H, SHARED->1,
    // COUPLED->H+span. (mode: 0=STAGE, 1=SHARED, 2=COUPLED)
    int in_stage_count(int arg, int n_stages) const {
        int m = in_mode_[arg];
        return m == 1 ? 1 : (m == 2 ? n_stages + in_span_[arg] : n_stages);
    }
    int out_stage_count(int arg, int n_stages) const {
        return out_mode_[arg] == 1 ? 1 : n_stages;
    }

    float evaluate(const CUSADI2_SCALAR** d_in, CUSADI2_SCALAR** d_out, int batch, int n_stages) {
        const LPPlanHost& b = h.base;
        const int ni = b.n_args_in, no = b.n_args_out;
        std::vector<int> ic(ni), oc(no);
        for (int a = 0; a < ni; ++a) ic[a] = in_stage_count(a, n_stages);
        for (int a = 0; a < no; ++a) oc[a] = out_stage_count(a, n_stages);
        CUSADI2_CHECK(cudaMemcpy(d_in_count, ic.data(), size_t(ni) * sizeof(int), cudaMemcpyHostToDevice));
        CUSADI2_CHECK(cudaMemcpy(d_out_count, oc.data(), size_t(no) * sizeof(int), cudaMemcpyHostToDevice));
        return evaluate_cusadi(d_in, d_out, batch, n_stages, b.block_size, b.shared_bytes,
            b.n_consts, d_const_slots, d_const_vals,
            b.n_input_maps, d_input_slots, d_input_args, d_input_elems, d_nnz_in,
            d_in_mode, d_in_offset, d_in_count,
            b.n_levels, d_level_offsets, d_level_widths,
            d_node_ops, d_node_outs, d_node_in_a, d_node_in_b,
            b.n_output_maps, d_out_slots, d_out_args, d_out_elems, d_nnz_out,
            d_out_mode, d_out_count);
    }

    // Stream-aware, sync-free evaluate. All work (count upload + kernel) is ordered on
    // `stream`; the count upload is skipped whenever n_stages is unchanged, so a steady
    // run is a single kernel launch per call with no host sync -> capturable in a graph.
    int evaluate(const CUSADI2_SCALAR** d_in, CUSADI2_SCALAR** d_out, int batch, int n_stages,
                 cudaStream_t stream) {
        const LPPlanHost& b = h.base;
        const int ni = b.n_args_in, no = b.n_args_out;
        if (n_stages != count_stages_) {                 // one-time (or on n_stages change)
            ic_.resize(ni); oc_.resize(no);
            for (int a = 0; a < ni; ++a) ic_[a] = in_stage_count(a, n_stages);
            for (int a = 0; a < no; ++a) oc_[a] = out_stage_count(a, n_stages);
            CUSADI2_CHECK(cudaMemcpyAsync(d_in_count, ic_.data(), size_t(ni) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream));
            CUSADI2_CHECK(cudaMemcpyAsync(d_out_count, oc_.data(), size_t(no) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream));
            count_stages_ = n_stages;
        }
        return evaluate_cusadi_stream(d_in, d_out, batch, n_stages, b.block_size, b.shared_bytes,
            b.n_consts, d_const_slots, d_const_vals,
            b.n_input_maps, d_input_slots, d_input_args, d_input_elems, d_nnz_in,
            d_in_mode, d_in_offset, d_in_count,
            b.n_levels, d_level_offsets, d_level_widths,
            d_node_ops, d_node_outs, d_node_in_a, d_node_in_b,
            b.n_output_maps, d_out_slots, d_out_args, d_out_elems, d_nnz_out,
            d_out_mode, d_out_count, stream);
    }
};

}  // namespace cusadi2
