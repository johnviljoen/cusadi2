// cusadi native backend — host-free C++/CUDA runtime for the function-independent
// level-parallel kernels. No torch, no jax, no Python at runtime.
//
// Pair a generated plan header (see cusadi.native.generate_native_plan) with one
// of the kernel sources from cusadi/lp_kernels/:
//   * level_parallel_kernel.cu        -> CusadiLPProgram     (batch axis)
//   * level_parallel_stage_kernel.cu  -> CusadiStageProgram  (batch x horizon)
//
//   #include "cusadi_native.h"
//   #include "myfn_plan.h"            // generated: provides MYFN_plan
//   CusadiLPProgram prog; prog.upload(MYFN_plan);
//   prog.evaluate(d_in, d_out, batch);   // d_in/d_out: device pointer arrays
//
// Inputs/outputs are caller-owned device buffers laid out batch-major
// (instance * nnz + element); the pointer arrays themselves live in device
// memory. See cusadi/native/example for a complete, self-contained program.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#ifndef CUSADI_NATIVE_CHECK
#define CUSADI_NATIVE_CHECK(call)                                              \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "cusadi_native CUDA error '%s' at %s:%d\n",        \
                cudaGetErrorString(_e), __FILE__, __LINE__);                   \
            abort();                                                           \
        }                                                                      \
    } while (0)
#endif

// --- kernel entry points (defined in cusadi/lp_kernels/*.cu) -----------------
extern "C" float evaluate_lp(
    const double* inputs[], double* outputs[], int total_instances,
    int block_size, int shared_bytes,
    int n_consts, const int* const_slots, const double* const_vals,
    int n_inputs, const int* input_slots, const int* input_args, const int* input_elems,
    const int* nnz_in,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    int n_outputs, const int* out_slots, const int* out_args, const int* out_elems,
    const int* nnz_out);
extern "C" int configure_shared_memory(int shared_bytes);

extern "C" float evaluate_lp_stage(
    const double* inputs[], double* outputs[], int total_instances, int n_stages,
    int block_size, int shared_bytes,
    int n_consts, const int* const_slots, const double* const_vals,
    int n_inputs, const int* input_slots, const int* input_args, const int* input_elems,
    const int* nnz_in, const int* in_stage_mode, const int* in_stage_offset,
    const int* in_spacing,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    int n_outputs, const int* out_slots, const int* out_args, const int* out_elems,
    const int* nnz_out, const int* out_stage_mode, const int* out_spacing);
extern "C" int evaluate_lp_stage_stream(
    const double* inputs[], double* outputs[], int total_instances, int n_stages,
    int block_size, int shared_bytes,
    int n_consts, const int* const_slots, const double* const_vals,
    int n_inputs, const int* input_slots, const int* input_args, const int* input_elems,
    const int* nnz_in, const int* in_stage_mode, const int* in_stage_offset,
    const int* in_spacing,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    int n_outputs, const int* out_slots, const int* out_args, const int* out_elems,
    const int* nnz_out, const int* out_stage_mode, const int* out_spacing,
    cudaStream_t stream);
extern "C" int evaluate_lp_stage_args(
    const double* const* inputs, double* const* outputs, const int* in_spacing, const int* out_spacing,
    int n_args_in, int n_args_out, int total_instances, int n_stages,
    int block_size, int shared_bytes,
    int n_consts, const int* const_slots, const double* const_vals,
    int n_inputs, const int* input_slots, const int* input_args, const int* input_elems,
    const int* nnz_in, const int* in_stage_mode, const int* in_stage_offset,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    int n_outputs, const int* out_slots, const int* out_args, const int* out_elems,
    const int* nnz_out, const int* out_stage_mode, cudaStream_t stream);
extern "C" int configure_stage_shared_memory(int shared_bytes);

namespace cusadi {

// --- host-side plan descriptors (emitted by generate_native_plan) -----------
struct LPPlanHost {
    int n_args_in, n_args_out;
    const int *nnz_in, *nnz_out;
    int block_size, shared_bytes;
    int n_consts;       const int *const_slots; const double *const_vals;
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

// One row of a generated plan registry (cusadi.native.generate_plan_registry):
// a named stage plan, its intended horizon, and — only when the plans were
// generated with emit_reference=True — an embedded CasADi self-check
// reference. ref_in/ref_out are nullptr (ref_nout 0) otherwise; consumers
// must skip the self-check then.
struct PlanEntry {
    const char* name;
    const StagePlanHost* plan;
    int n_stages;                 // horizon the plan is meant to run at
    const double* const* ref_in;  // per-arg reference inputs, kernel layout
    const double* ref_out;        // knot-major expected outputs
    int ref_nout;
};

namespace detail {
inline const int* dev_i(const int* h, int n) {
    int* d = nullptr; n = n > 0 ? n : 1;
    CUSADI_NATIVE_CHECK(cudaMalloc(&d, size_t(n) * sizeof(int)));
    CUSADI_NATIVE_CHECK(cudaMemcpy(d, h, size_t(n) * sizeof(int), cudaMemcpyHostToDevice));
    return d;
}
inline const double* dev_d(const double* h, int n) {
    double* d = nullptr; n = n > 0 ? n : 1;
    CUSADI_NATIVE_CHECK(cudaMalloc(&d, size_t(n) * sizeof(double)));
    CUSADI_NATIVE_CHECK(cudaMemcpy(d, h, size_t(n) * sizeof(double), cudaMemcpyHostToDevice));
    return d;
}
}  // namespace detail

// 1-axis (batch) level-parallel program. One CUDA block per instance.
struct LPProgram {
    LPPlanHost h{};
    const int *d_nnz_in, *d_nnz_out;
    const int *d_const_slots; const double *d_const_vals;
    const int *d_input_slots, *d_input_args, *d_input_elems;
    const int *d_level_offsets, *d_level_widths;
    const int *d_node_ops, *d_node_outs, *d_node_in_a, *d_node_in_b;
    const int *d_out_slots, *d_out_args, *d_out_elems;

    void upload(const LPPlanHost& p) {
        h = p;
        using namespace detail;
        d_nnz_in = dev_i(p.nnz_in, p.n_args_in);   d_nnz_out = dev_i(p.nnz_out, p.n_args_out);
        d_const_slots = dev_i(p.const_slots, p.n_consts);
        d_const_vals  = dev_d(p.const_vals, p.n_consts);
        d_input_slots = dev_i(p.input_slots, p.n_input_maps);
        d_input_args  = dev_i(p.input_args, p.n_input_maps);
        d_input_elems = dev_i(p.input_elems, p.n_input_maps);
        d_level_offsets = dev_i(p.level_offsets, p.n_levels);
        d_level_widths  = dev_i(p.level_widths, p.n_levels);
        d_node_ops  = dev_i(p.node_ops, p.n_nodes);  d_node_outs = dev_i(p.node_outs, p.n_nodes);
        d_node_in_a = dev_i(p.node_in_a, p.n_nodes); d_node_in_b = dev_i(p.node_in_b, p.n_nodes);
        d_out_slots = dev_i(p.out_slots, p.n_output_maps);
        d_out_args  = dev_i(p.out_args, p.n_output_maps);
        d_out_elems = dev_i(p.out_elems, p.n_output_maps);
        configure_shared_memory(p.shared_bytes);
    }

    // d_in / d_out: device arrays of device pointers (length n_args_in/out).
    float evaluate(const double** d_in, double** d_out, int batch) const {
        return evaluate_lp(d_in, d_out, batch, h.block_size, h.shared_bytes,
            h.n_consts, d_const_slots, d_const_vals,
            h.n_input_maps, d_input_slots, d_input_args, d_input_elems, d_nnz_in,
            h.n_levels, d_level_offsets, d_level_widths,
            d_node_ops, d_node_outs, d_node_in_a, d_node_in_b,
            h.n_output_maps, d_out_slots, d_out_args, d_out_elems, d_nnz_out);
    }
};

// 2-axis (batch x horizon) stage program. grid = (batch, n_stages).
//
// spacing[arg] = distance IN DOUBLES between consecutive instances' data for
// that argument. The dense default is stage_count * nnz — each argument its
// own tightly-packed (instance, stage, nnz) buffer. Custom spacings let an
// argument live inside a larger per-instance block (e.g. the states inside a
// full primal vector), and spacing 0 shares one copy across all instances.
//
// Two ways to run:
//  * bind(...) once, then evaluate_async(batch, stream): pointers and
//    spacings are uploaded at bind time, so evaluate_async is a pure kernel
//    launch — no events, no syncs, nothing host-side — and is safe to record
//    into a CUDA graph. bind() may be called again to rebind.
//  * evaluate_async(ins, outs, n_stages, batch, stream, ...): a plain
//    callable -- the pointer tables travel by value as a kernel parameter,
//    so there is no bind, no device-side table, and nothing host-side but
//    the launch itself. Use it when the same function is evaluated at
//    different arrays over time (e.g. a line search's trial points).
//  * evaluate(d_in, d_out, batch, n_stages): synchronous convenience with
//    dense default spacings; returns seconds.
struct StageProgram {
    LPProgram lp;
    StagePlanHost h{};
    const int *d_in_mode, *d_in_offset;
    const int *d_out_mode;
    std::vector<int> in_mode_, in_span_, out_mode_;
    int *d_in_spacing = nullptr, *d_out_spacing = nullptr;
    const double** d_in = nullptr;   // argument pointer tables, set by bind()
    double** d_out = nullptr;
    int n_stages_ = 0;               // set by bind()

    void upload(const StagePlanHost& p) {
        h = p; lp.upload(p.base);
        const int ni = p.base.n_args_in, no = p.base.n_args_out;
        d_in_mode = detail::dev_i(p.in_stage_mode, ni);
        d_in_offset = detail::dev_i(p.in_stage_offset, ni);
        d_out_mode = detail::dev_i(p.out_stage_mode, no);
        in_mode_.assign(p.in_stage_mode, p.in_stage_mode + ni);
        in_span_.assign(p.in_stage_span, p.in_stage_span + ni);
        out_mode_.assign(p.out_stage_mode, p.out_stage_mode + no);
        CUSADI_NATIVE_CHECK(cudaMalloc(&d_in_spacing, size_t(ni) * sizeof(int)));
        CUSADI_NATIVE_CHECK(cudaMalloc(&d_out_spacing, size_t(no) * sizeof(int)));
        CUSADI_NATIVE_CHECK(cudaMalloc(&d_in, size_t(ni) * sizeof(double*)));
        CUSADI_NATIVE_CHECK(cudaMalloc(&d_out, size_t(no) * sizeof(double*)));
        configure_stage_shared_memory(p.base.shared_bytes);
    }

    // Dense default spacing per arg: stage_count * nnz, with stage_count
    // STAGE->H, SHARED->1, COUPLED->H+span. (mode: 0=STAGE, 1=SHARED, 2=COUPLED)
    int default_in_spacing(int arg, int n_stages) const {
        int m = in_mode_[arg];
        int count = m == 1 ? 1 : (m == 2 ? n_stages + in_span_[arg] : n_stages);
        return count * h.base.nnz_in[arg];
    }
    int default_out_spacing(int arg, int n_stages) const {
        return (out_mode_[arg] == 1 ? 1 : n_stages) * h.base.nnz_out[arg];
    }

    // Upload argument pointers + spacings to the device once. `ins`/`outs`
    // are host arrays of device pointers, lengths n_args_in/n_args_out.
    // nullptr spacing arrays select the dense defaults. May be called again
    // to rebind.
    void bind(const double* const* ins, double* const* outs, int n_stages,
              const int* in_spacing = nullptr, const int* out_spacing = nullptr) {
        const int ni = h.base.n_args_in, no = h.base.n_args_out;
        n_stages_ = n_stages;
        CUSADI_NATIVE_CHECK(cudaMemcpy(d_in, ins, size_t(ni) * sizeof(double*), cudaMemcpyHostToDevice));
        CUSADI_NATIVE_CHECK(cudaMemcpy(d_out, outs, size_t(no) * sizeof(double*), cudaMemcpyHostToDevice));
        std::vector<int> isp(ni), osp(no);
        for (int a = 0; a < ni; ++a) isp[a] = in_spacing ? in_spacing[a] : default_in_spacing(a, n_stages);
        for (int a = 0; a < no; ++a) osp[a] = out_spacing ? out_spacing[a] : default_out_spacing(a, n_stages);
        CUSADI_NATIVE_CHECK(cudaMemcpy(d_in_spacing, isp.data(), size_t(ni) * sizeof(int), cudaMemcpyHostToDevice));
        CUSADI_NATIVE_CHECK(cudaMemcpy(d_out_spacing, osp.data(), size_t(no) * sizeof(int), cudaMemcpyHostToDevice));
    }
    void bind(std::initializer_list<const double*> ins,
              std::initializer_list<double*> outs,
              int n_stages,
              const int* in_spacing = nullptr,
              const int* out_spacing = nullptr) {
        if ((int)ins.size() != h.base.n_args_in || (int)outs.size() != h.base.n_args_out) {
            fprintf(stderr, "cusadi StageProgram::bind: %zu in / %zu out args given, plan has %d/%d\n",
                    ins.size(), outs.size(), h.base.n_args_in, h.base.n_args_out);
            abort();
        }
        bind(ins.begin(), outs.begin(), n_stages, in_spacing, out_spacing);
    }

    // Pure kernel launch on `stream` (requires a prior bind). Returns the
    // cudaError_t of the launch as int; no events, no synchronization.
    int evaluate_async(int batch, cudaStream_t stream) const {
        const LPPlanHost& b = h.base;
        return evaluate_lp_stage_stream(d_in, d_out, batch, n_stages_, b.block_size, b.shared_bytes,
            b.n_consts, lp.d_const_slots, lp.d_const_vals,
            b.n_input_maps, lp.d_input_slots, lp.d_input_args, lp.d_input_elems, lp.d_nnz_in,
            d_in_mode, d_in_offset, d_in_spacing,
            b.n_levels, lp.d_level_offsets, lp.d_level_widths,
            lp.d_node_ops, lp.d_node_outs, lp.d_node_in_a, lp.d_node_in_b,
            b.n_output_maps, lp.d_out_slots, lp.d_out_args, lp.d_out_elems, lp.d_nnz_out,
            d_out_mode, d_out_spacing, stream);
    }

    // Pure kernel launch with the argument tables given per call (host
    // arrays of device pointers; nullptr spacings select the dense defaults).
    // Returns the cudaError_t of the launch as int.
    int evaluate_async(const double* const* ins, double* const* outs, int n_stages, int batch,
                       cudaStream_t stream, const int* in_spacing = nullptr,
                       const int* out_spacing = nullptr) const {
        const int ni = h.base.n_args_in, no = h.base.n_args_out;
        int isp[16], osp[16];
        for (int a = 0; a < ni; ++a) isp[a] = in_spacing ? in_spacing[a] : default_in_spacing(a, n_stages);
        for (int a = 0; a < no; ++a) osp[a] = out_spacing ? out_spacing[a] : default_out_spacing(a, n_stages);
        const LPPlanHost& b = h.base;
        return evaluate_lp_stage_args(ins, outs, isp, osp, ni, no, batch, n_stages, b.block_size, b.shared_bytes,
            b.n_consts, lp.d_const_slots, lp.d_const_vals,
            b.n_input_maps, lp.d_input_slots, lp.d_input_args, lp.d_input_elems, lp.d_nnz_in,
            d_in_mode, d_in_offset,
            b.n_levels, lp.d_level_offsets, lp.d_level_widths,
            lp.d_node_ops, lp.d_node_outs, lp.d_node_in_a, lp.d_node_in_b,
            b.n_output_maps, lp.d_out_slots, lp.d_out_args, lp.d_out_elems, lp.d_nnz_out,
            d_out_mode, stream);
    }
    int evaluate_async(std::initializer_list<const double*> ins, std::initializer_list<double*> outs,
                       int n_stages, int batch, cudaStream_t stream,
                       const int* in_spacing = nullptr, const int* out_spacing = nullptr) const {
        if ((int)ins.size() != h.base.n_args_in || (int)outs.size() != h.base.n_args_out) {
            fprintf(stderr, "cusadi StageProgram::evaluate_async: %zu in / %zu out args given, plan has %d/%d\n",
                    ins.size(), outs.size(), h.base.n_args_in, h.base.n_args_out);
            abort();
        }
        return evaluate_async(ins.begin(), outs.begin(), n_stages, batch, stream, in_spacing, out_spacing);
    }

    // Synchronous convenience: dense default spacings, caller-built device
    // pointer tables, blocks until done. Returns seconds.
    float evaluate(const double** in_tbl, double** out_tbl, int batch, int n_stages) {
        const int ni = h.base.n_args_in, no = h.base.n_args_out;
        std::vector<int> isp(ni), osp(no);
        for (int a = 0; a < ni; ++a) isp[a] = default_in_spacing(a, n_stages);
        for (int a = 0; a < no; ++a) osp[a] = default_out_spacing(a, n_stages);
        CUSADI_NATIVE_CHECK(cudaMemcpy(d_in_spacing, isp.data(), size_t(ni) * sizeof(int), cudaMemcpyHostToDevice));
        CUSADI_NATIVE_CHECK(cudaMemcpy(d_out_spacing, osp.data(), size_t(no) * sizeof(int), cudaMemcpyHostToDevice));
        const LPPlanHost& b = h.base;
        return evaluate_lp_stage(in_tbl, out_tbl, batch, n_stages, b.block_size, b.shared_bytes,
            b.n_consts, lp.d_const_slots, lp.d_const_vals,
            b.n_input_maps, lp.d_input_slots, lp.d_input_args, lp.d_input_elems, lp.d_nnz_in,
            d_in_mode, d_in_offset, d_in_spacing,
            b.n_levels, lp.d_level_offsets, lp.d_level_widths,
            lp.d_node_ops, lp.d_node_outs, lp.d_node_in_a, lp.d_node_in_b,
            b.n_output_maps, lp.d_out_slots, lp.d_out_args, lp.d_out_elems, lp.d_nnz_out,
            d_out_mode, d_out_spacing);
    }
};

}  // namespace cusadi
