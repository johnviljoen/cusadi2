// XLA FFI integration for cusadi level-parallel kernel.
//
// Exposes a single FFI handler `cusadi_generic_f64` that jax.ffi.ffi_call can
// target. Plan data (DAG metadata for a specific CasADi function) is uploaded
// once via install_plan() and looked up by plan_id: Attr<int64_t>. Kernel runs
// on the caller-supplied XLA CUDA stream (zero-copy, no host sync).
//
// Arity is currently fixed at 1 input (x) / 1 output (sparse values). This
// matches the NMPC problem's f/c/d/jac_*/hess_* functions which only depend
// on z. Extending to variadic arity is a future RemainingArgs/RemainingRets
// refactor.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_runtime.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

// ---------------------------------------------------------------------------
// C-level kernel entry provided by level_parallel_kernel.cu (same translation
// unit when built together). Mirrors the extern "C" declaration in that file.
// ---------------------------------------------------------------------------
extern "C" {
int evaluate_lp_stream(
    const double *inputs[],
    double *outputs[],
    const int total_instances,
    const int block_size,
    const int shared_bytes,
    const int n_consts,
    const int *const_slots,
    const double *const_vals,
    const int n_inputs,
    const int *input_slots,
    const int *input_args,
    const int *input_elems,
    const int *nnz_in,
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
    cudaStream_t stream);

int configure_shared_memory(int shared_bytes);
}

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t _err = (call);                                              \
    if (_err != cudaSuccess) {                                              \
      throw std::runtime_error(std::string("CUDA error at " __FILE__ ":")   \
          + std::to_string(__LINE__) + ": " + cudaGetErrorString(_err));    \
    }                                                                       \
  } while (0)

#define CUDA_CHECK_FFI(call)                                                \
  do {                                                                      \
    cudaError_t _err = (call);                                              \
    if (_err != cudaSuccess) {                                              \
      return ffi::Error::Internal(std::string("CUDA error: ")               \
          + cudaGetErrorString(_err));                                      \
    }                                                                       \
  } while (0)

// ---------------------------------------------------------------------------
// Device plan storage (all arrays cudaMalloc'd, freed in dtor).
// ---------------------------------------------------------------------------
struct DevicePlan {
    int n_in = 0;
    int n_out = 0;
    int n_consts = 0;
    int n_input_maps = 0;
    int n_levels = 0;
    int n_output_maps = 0;
    int block_size = 32;
    int shared_bytes = 0;

    int *const_slots = nullptr;
    double *const_vals = nullptr;
    int *input_slots = nullptr;
    int *input_args = nullptr;
    int *input_elems = nullptr;
    int *nnz_in = nullptr;
    int *level_offsets = nullptr;
    int *level_widths = nullptr;
    int *node_ops = nullptr;
    int *node_outs = nullptr;
    int *node_in_a = nullptr;
    int *node_in_b = nullptr;
    int *out_slots = nullptr;
    int *out_args = nullptr;
    int *out_elems = nullptr;
    int *nnz_out = nullptr;

    // Per-call scratchpads sized for arity n_in / n_out.
    double **d_input_ptrs = nullptr;
    double **d_output_ptrs = nullptr;

    ~DevicePlan() {
        auto safe_free = [](void *p) { if (p) cudaFree(p); };
        safe_free(const_slots); safe_free(const_vals);
        safe_free(input_slots); safe_free(input_args);
        safe_free(input_elems); safe_free(nnz_in);
        safe_free(level_offsets); safe_free(level_widths);
        safe_free(node_ops); safe_free(node_outs);
        safe_free(node_in_a); safe_free(node_in_b);
        safe_free(out_slots); safe_free(out_args);
        safe_free(out_elems); safe_free(nnz_out);
        safe_free(d_input_ptrs); safe_free(d_output_ptrs);
    }
};

class PlanRegistry {
 public:
    static PlanRegistry &instance() { static PlanRegistry r; return r; }
    DevicePlan *get(int64_t plan_id) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = plans_.find(plan_id);
        return it == plans_.end() ? nullptr : it->second.get();
    }
    DevicePlan *create(int64_t plan_id) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto p = std::make_unique<DevicePlan>();
        DevicePlan *raw = p.get();
        plans_[plan_id] = std::move(p);
        return raw;
    }
    void remove(int64_t plan_id) {
        std::lock_guard<std::mutex> lk(mtx_);
        plans_.erase(plan_id);
    }
 private:
    std::mutex mtx_;
    std::unordered_map<int64_t, std::unique_ptr<DevicePlan>> plans_;
};

// Allocate on device and copy from host numpy array. Empty arrays get a
// 1-element dummy to keep the pointer non-null (kernel short-circuits on
// n==0 loops so contents are never read, but passing nullptr is UB).
template <typename T>
static T *alloc_and_copy_1d(const T *host_data, size_t n) {
    T *d = nullptr;
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d), n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d, host_data, n * sizeof(T), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d), sizeof(T)));
        T zero{};
        CUDA_CHECK(cudaMemcpy(d, &zero, sizeof(T), cudaMemcpyHostToDevice));
    }
    return d;
}

using np_i32 = nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using np_f64 = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Size-0 arrays pass through as zero-length; alloc_and_copy_1d allocates a
// 1-element dummy in that case.
static void install_plan(
    int64_t plan_id,
    int n_in,
    int n_out,
    int block_size,
    int shared_bytes,
    np_i32 const_slots, np_f64 const_vals,
    np_i32 input_slots, np_i32 input_args, np_i32 input_elems, np_i32 nnz_in,
    np_i32 level_offsets, np_i32 level_widths,
    np_i32 node_ops, np_i32 node_outs, np_i32 node_in_a, np_i32 node_in_b,
    np_i32 out_slots, np_i32 out_args, np_i32 out_elems, np_i32 nnz_out) {

    // Configure kernel shared memory attribute. The kernel is a single global
    // symbol shared across all plans, so attempting to SHRINK this attribute
    // (later plan with smaller smem need than earlier plan) would make later
    // launches of the larger plan fail with cudaErrorInvalidValue. Track the
    // running max and only grow.
    {
        static std::mutex smem_mtx;
        static int s_max_smem = 0;
        std::lock_guard<std::mutex> lg(smem_mtx);
        if (shared_bytes > s_max_smem) {
            s_max_smem = shared_bytes;
            configure_shared_memory(s_max_smem);
        }
    }

    DevicePlan *p = PlanRegistry::instance().create(plan_id);
    p->n_in = n_in;
    p->n_out = n_out;
    p->n_consts = static_cast<int>(const_slots.size());
    p->n_input_maps = static_cast<int>(input_slots.size());
    p->n_levels = static_cast<int>(level_offsets.size());
    p->n_output_maps = static_cast<int>(out_slots.size());
    p->block_size = block_size;
    p->shared_bytes = shared_bytes;

    p->const_slots = alloc_and_copy_1d<int32_t>(const_slots.data(), const_slots.size());
    p->const_vals = alloc_and_copy_1d<double>(const_vals.data(), const_vals.size());
    p->input_slots = alloc_and_copy_1d<int32_t>(input_slots.data(), input_slots.size());
    p->input_args = alloc_and_copy_1d<int32_t>(input_args.data(), input_args.size());
    p->input_elems = alloc_and_copy_1d<int32_t>(input_elems.data(), input_elems.size());
    p->nnz_in = alloc_and_copy_1d<int32_t>(nnz_in.data(), nnz_in.size());
    p->level_offsets = alloc_and_copy_1d<int32_t>(level_offsets.data(), level_offsets.size());
    p->level_widths = alloc_and_copy_1d<int32_t>(level_widths.data(), level_widths.size());
    p->node_ops = alloc_and_copy_1d<int32_t>(node_ops.data(), node_ops.size());
    p->node_outs = alloc_and_copy_1d<int32_t>(node_outs.data(), node_outs.size());
    p->node_in_a = alloc_and_copy_1d<int32_t>(node_in_a.data(), node_in_a.size());
    p->node_in_b = alloc_and_copy_1d<int32_t>(node_in_b.data(), node_in_b.size());
    p->out_slots = alloc_and_copy_1d<int32_t>(out_slots.data(), out_slots.size());
    p->out_args = alloc_and_copy_1d<int32_t>(out_args.data(), out_args.size());
    p->out_elems = alloc_and_copy_1d<int32_t>(out_elems.data(), out_elems.size());
    p->nnz_out = alloc_and_copy_1d<int32_t>(nnz_out.data(), nnz_out.size());

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&p->d_input_ptrs),
                          std::max(1, n_in) * sizeof(double *)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&p->d_output_ptrs),
                          std::max(1, n_out) * sizeof(double *)));
}

static void free_plan(int64_t plan_id) {
    PlanRegistry::instance().remove(plan_id);
}

// ---------------------------------------------------------------------------
// FFI handler: 1 input (x or flat CasADi inputs) and 1 output (nnz vector).
// For n_in > 1 or n_out > 1 we will need a RemainingArgs/RemainingRets
// variant; deferred.
// ---------------------------------------------------------------------------
static ffi::Error CusadiExecute(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> input_buf,
    ffi::ResultBuffer<ffi::F64> output_buf,
    int64_t plan_id) {

    DevicePlan *plan = PlanRegistry::instance().get(plan_id);
    if (plan == nullptr) {
        return ffi::Error::Internal(
            std::string("cusadi plan_id not installed: ") + std::to_string(plan_id));
    }
    if (plan->n_in != 1 || plan->n_out != 1) {
        return ffi::Error::Internal(
            "cusadi handler is fixed-arity (1 in, 1 out); plan is " +
            std::to_string(plan->n_in) + " in, " + std::to_string(plan->n_out) + " out");
    }

    // Infer batch from leading dim of input; if input is 1-D we treat as batch=1.
    auto in_dims = input_buf.dimensions();
    int batch = 1;
    if (in_dims.size() >= 2) {
        batch = static_cast<int>(in_dims[0]);
    }

    // Assemble host pointer arrays (1 each here) and stage to device scratchpad.
    const double *h_in_ptrs[1] = { input_buf.typed_data() };
    double *h_out_ptrs[1] = { output_buf->typed_data() };

    CUDA_CHECK_FFI(cudaMemcpyAsync(plan->d_input_ptrs, h_in_ptrs,
                                   sizeof(double *), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_FFI(cudaMemcpyAsync(plan->d_output_ptrs, h_out_ptrs,
                                   sizeof(double *), cudaMemcpyHostToDevice, stream));

    int rc = evaluate_lp_stream(
        const_cast<const double **>(plan->d_input_ptrs),
        plan->d_output_ptrs,
        batch,
        plan->block_size,
        plan->shared_bytes,
        plan->n_consts, plan->const_slots, plan->const_vals,
        plan->n_input_maps, plan->input_slots, plan->input_args,
        plan->input_elems, plan->nnz_in,
        plan->n_levels, plan->level_offsets, plan->level_widths,
        plan->node_ops, plan->node_outs, plan->node_in_a, plan->node_in_b,
        plan->n_output_maps, plan->out_slots, plan->out_args,
        plan->out_elems, plan->nnz_out,
        stream);
    if (rc != 0) {
        return ffi::Error::Internal(
            std::string("evaluate_lp_stream CUDA error: ") +
            cudaGetErrorString(static_cast<cudaError_t>(rc)));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    kCusadiExecute, CusadiExecute,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Attr<int64_t>("plan_id"));

NB_MODULE(_cusadi_jax_ffi, m) {
    m.def("install_plan", &install_plan,
          nb::arg("plan_id"), nb::arg("n_in"), nb::arg("n_out"),
          nb::arg("block_size"), nb::arg("shared_bytes"),
          nb::arg("const_slots"), nb::arg("const_vals"),
          nb::arg("input_slots"), nb::arg("input_args"),
          nb::arg("input_elems"), nb::arg("nnz_in"),
          nb::arg("level_offsets"), nb::arg("level_widths"),
          nb::arg("node_ops"), nb::arg("node_outs"),
          nb::arg("node_in_a"), nb::arg("node_in_b"),
          nb::arg("out_slots"), nb::arg("out_args"),
          nb::arg("out_elems"), nb::arg("nnz_out"));
    m.def("free_plan", &free_plan, nb::arg("plan_id"));
    m.def("handler", []() {
        nb::dict d;
        d["execute"] = nb::capsule(reinterpret_cast<void *>(kCusadiExecute));
        return d;
    });
}
