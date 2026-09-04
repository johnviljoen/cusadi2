// cusadi2: host-free C++/CUDA runtime for the level-parallel kernel.
// No Python at deployment: load a plan file written by cusadi2.plan.write_plan,
// upload it, evaluate. Compile this header's consumer together with the kernel
// source written by cusadi2.kernel_template.write_kernel_source.
//
//   #include "cusadi2.h"
//   cusadi2::Plan plan = cusadi2::load_plan("f.plan");
//   cusadi2::Program prog(plan);
//   prog.evaluate(d_in, d_out, batch, stream);   // pure launch; graph-capturable
//
// d_in / d_out are DEVICE-resident arrays of device buffer pointers, one per
// argument. Buffer a holds batch * plan.in[a].count * nnz elements, laid out
// (instance, stage_count, nnz); the horizon and per-arg stage counts are baked
// into the plan. Buffers are caller-owned.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Scalar type; must match the dtype the kernel .cu was written with.
#ifndef CUSADI2_SCALAR
#define CUSADI2_SCALAR double
#endif

#ifndef CUSADI2_CHECK
#define CUSADI2_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "cusadi2 CUDA error '%s' at %s:%d\n",              \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
            abort();                                                           \
        }                                                                      \
    } while (0)
#endif

// --- kernel entry points (defined in the emitted kernel .cu) -----------------
int cusadi2_launch(
    const CUSADI2_SCALAR* const* inputs, CUSADI2_SCALAR* const* outputs,
    int total_instances, int n_stages, int block_size, int shared_bytes,
    int n_consts, const int* const_slots, const CUSADI2_SCALAR* const_vals,
    int n_inputs, const int* input_slots, const int* input_args, const int* input_elems,
    const int* nnz_in, const int* in_stage_mode, const int* in_stage_offset,
    const int* in_stage_count,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    int n_outputs, const int* out_slots, const int* out_args, const int* out_elems,
    const int* nnz_out, const int* out_stage_mode, const int* out_stage_count,
    cudaStream_t stream);
int cusadi2_set_shared_memory(int shared_bytes);

namespace cusadi2 {

using Scalar = CUSADI2_SCALAR;

// CasADi compressed-column sparsity of one argument.
struct Sparsity {
    int rows = 0, cols = 0;
    std::vector<int> colptr;   // cols + 1
    std::vector<int> rowidx;   // nnz, column-major nonzero order
    int nnz() const { return (int)rowidx.size(); }
    // COO views in nonzero (column-major) order: the row of each nonzero is
    // rowidx itself; the column is expanded from colptr.
    const std::vector<int>& coo_row() const { return rowidx; }
    std::vector<int> coo_col() const {
        std::vector<int> c(rowidx.size());
        for (int j = 0; j < cols; ++j)
            for (int k = colptr[j]; k < colptr[j + 1]; ++k) c[k] = j;
        return c;
    }
};

// One argument: sparsity plus its stage-axis layout (mode 0=STAGE 1=SHARED
// 2=COUPLED; count = leading-axis length at the plan's horizon).
struct Arg {
    Sparsity sp;
    int mode = 0, offset = 0, span = 0, count = 1;
    int nnz() const { return sp.nnz(); }
    size_t len(int batch) const { return (size_t)batch * count * sp.nnz(); }  // elements
};

// A plan file, fully in host memory.
struct Plan {
    std::string name;
    int n_stages = 1, block_size = 32, peak_live_slots = 0;
    std::vector<Arg> in, out;
    std::vector<int> const_slots; std::vector<double> const_vals;
    std::vector<int> input_slots, input_args, input_elems;
    std::vector<int> level_offsets, level_widths;
    std::vector<int> node_ops, node_outs, node_in_a, node_in_b;
    std::vector<int> out_slots, out_args, out_elems;
    int n_in() const { return (int)in.size(); }
    int n_out() const { return (int)out.size(); }
    int shared_bytes() const { return peak_live_slots * (int)sizeof(Scalar); }
};

namespace detail {
struct Reader {
    std::ifstream f;
    std::string path;
    explicit Reader(const std::string& p) : f(p, std::ios::binary), path(p) {
        if (!f) throw std::runtime_error("cusadi2: cannot open plan file: " + p);
    }
    void raw(void* dst, size_t bytes) {
        f.read((char*)dst, (std::streamsize)bytes);
        if (!f) throw std::runtime_error("cusadi2: short read in " + path);
    }
    int32_t i() { int32_t v; raw(&v, 4); return v; }
    template <class T> std::vector<T> arr(size_t n) {
        std::vector<T> v(n); if (n) raw(v.data(), n * sizeof(T)); return v;
    }
    void ia(std::vector<int>& v, size_t n) { v = arr<int32_t>(n); }
    void sparsity(Sparsity& s, int rows, int cols, int nnz) {
        s.rows = rows; s.cols = cols;
        ia(s.colptr, (size_t)cols + 1); ia(s.rowidx, (size_t)nnz);
    }
};

template <class T> const T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    size_t n = h.empty() ? 1 : h.size();   // non-null even when empty
    CUSADI2_CHECK(cudaMalloc(&d, n * sizeof(T)));
    if (!h.empty()) CUSADI2_CHECK(cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
}  // namespace detail

inline Plan load_plan(const std::string& path) {
    detail::Reader r(path);
    char magic[4]; r.raw(magic, 4);
    if (std::string(magic, 4) != "CS2P")
        throw std::runtime_error("cusadi2: bad magic in " + path);
    int version = r.i();
    if (version != 1)
        throw std::runtime_error("cusadi2: unsupported plan version in " + path);
    Plan p;
    int name_len = r.i();
    p.name.resize(name_len); if (name_len) r.raw(&p.name[0], name_len);
    int n_in = r.i(), n_out = r.i();
    p.n_stages = r.i(); p.block_size = r.i(); p.peak_live_slots = r.i();
    p.in.resize(n_in);
    for (Arg& a : p.in) {
        int rows = r.i(), cols = r.i(), nnz = r.i();
        a.mode = r.i(); a.offset = r.i(); a.span = r.i(); a.count = r.i();
        r.sparsity(a.sp, rows, cols, nnz);
    }
    p.out.resize(n_out);
    for (Arg& a : p.out) {
        int rows = r.i(), cols = r.i(), nnz = r.i();
        a.mode = r.i(); a.count = r.i();
        r.sparsity(a.sp, rows, cols, nnz);
    }
    int n = r.i(); r.ia(p.const_slots, n); p.const_vals = r.arr<double>(n);
    n = r.i(); r.ia(p.input_slots, n); r.ia(p.input_args, n); r.ia(p.input_elems, n);
    n = r.i(); r.ia(p.level_offsets, n); r.ia(p.level_widths, n);
    n = r.i(); r.ia(p.node_ops, n); r.ia(p.node_outs, n); r.ia(p.node_in_a, n); r.ia(p.node_in_b, n);
    n = r.i(); r.ia(p.out_slots, n); r.ia(p.out_args, n); r.ia(p.out_elems, n);
    return p;
}

// A plan uploaded to the device. evaluate() is a single kernel launch on the
// given stream with no host-side work, so it can be captured into a CUDA graph.
class Program {
public:
    Program() = default;
    explicit Program(const Plan& p) { upload(p); }
    ~Program() { release(); }
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&& o) noexcept { *this = std::move(o); }
    Program& operator=(Program&& o) noexcept {
        if (this != &o) { release(); h_ = o.h_; d_ = o.d_; o.d_ = Dev{}; }
        return *this;
    }

    void upload(const Plan& p) {
        using detail::dev;
        release();
        h_ = Hdr{p.n_stages, p.block_size, p.shared_bytes(), (int)p.const_slots.size(),
                 (int)p.input_slots.size(), (int)p.level_offsets.size(),
                 (int)p.out_slots.size(), p.n_in(), p.n_out()};
        std::vector<int> nnz_in, nnz_out, in_mode, in_offset, in_count, out_mode, out_count;
        for (const Arg& a : p.in) {
            nnz_in.push_back(a.nnz()); in_mode.push_back(a.mode);
            in_offset.push_back(a.offset); in_count.push_back(a.count);
        }
        for (const Arg& a : p.out) {
            nnz_out.push_back(a.nnz()); out_mode.push_back(a.mode); out_count.push_back(a.count);
        }
        std::vector<Scalar> consts(p.const_vals.begin(), p.const_vals.end());
        d_.nnz_in = dev(nnz_in); d_.nnz_out = dev(nnz_out);
        d_.in_mode = dev(in_mode); d_.in_offset = dev(in_offset); d_.in_count = dev(in_count);
        d_.out_mode = dev(out_mode); d_.out_count = dev(out_count);
        d_.const_slots = dev(p.const_slots); d_.const_vals = dev(consts);
        d_.input_slots = dev(p.input_slots); d_.input_args = dev(p.input_args);
        d_.input_elems = dev(p.input_elems);
        d_.level_offsets = dev(p.level_offsets); d_.level_widths = dev(p.level_widths);
        d_.node_ops = dev(p.node_ops); d_.node_outs = dev(p.node_outs);
        d_.node_in_a = dev(p.node_in_a); d_.node_in_b = dev(p.node_in_b);
        d_.out_slots = dev(p.out_slots); d_.out_args = dev(p.out_args); d_.out_elems = dev(p.out_elems);
        CUSADI2_CHECK((cudaError_t)cusadi2_set_shared_memory(h_.shared_bytes));
    }

    // Launch on `stream`; returns cudaPeekAtLastError() (0 on success).
    int evaluate(const Scalar* const* d_in, Scalar* const* d_out, int batch,
                 cudaStream_t stream = 0) const {
        return cusadi2_launch(d_in, d_out, batch, h_.n_stages, h_.block_size, h_.shared_bytes,
            h_.n_consts, d_.const_slots, d_.const_vals,
            h_.n_input_maps, d_.input_slots, d_.input_args, d_.input_elems,
            d_.nnz_in, d_.in_mode, d_.in_offset, d_.in_count,
            h_.n_levels, d_.level_offsets, d_.level_widths,
            d_.node_ops, d_.node_outs, d_.node_in_a, d_.node_in_b,
            h_.n_output_maps, d_.out_slots, d_.out_args, d_.out_elems,
            d_.nnz_out, d_.out_mode, d_.out_count, stream);
    }

    int n_stages() const { return h_.n_stages; }
    int n_in() const { return h_.n_in; }
    int n_out() const { return h_.n_out; }

    void release() {
        const void* ptrs[] = {d_.nnz_in, d_.nnz_out, d_.in_mode, d_.in_offset, d_.in_count,
            d_.out_mode, d_.out_count, d_.const_slots, d_.const_vals, d_.input_slots,
            d_.input_args, d_.input_elems, d_.level_offsets, d_.level_widths, d_.node_ops,
            d_.node_outs, d_.node_in_a, d_.node_in_b, d_.out_slots, d_.out_args, d_.out_elems};
        for (const void* q : ptrs) if (q) cudaFree(const_cast<void*>(q));
        d_ = Dev{};
    }

private:
    struct Hdr {
        int n_stages = 1, block_size = 32, shared_bytes = 0, n_consts = 0, n_input_maps = 0,
            n_levels = 0, n_output_maps = 0, n_in = 0, n_out = 0;
    } h_;
    struct Dev {
        const int *nnz_in = nullptr, *nnz_out = nullptr, *in_mode = nullptr,
            *in_offset = nullptr, *in_count = nullptr, *out_mode = nullptr, *out_count = nullptr,
            *const_slots = nullptr, *input_slots = nullptr, *input_args = nullptr,
            *input_elems = nullptr, *level_offsets = nullptr, *level_widths = nullptr,
            *node_ops = nullptr, *node_outs = nullptr, *node_in_a = nullptr, *node_in_b = nullptr,
            *out_slots = nullptr, *out_args = nullptr, *out_elems = nullptr;
        const Scalar* const_vals = nullptr;
    } d_;
};

}  // namespace cusadi2
