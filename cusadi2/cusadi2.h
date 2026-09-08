// cusadi2: host-free C++/CUDA runtime for the level-parallel kernel.
// No Python at deployment: load a plan file written by cusadi2.plan.write_plan,
// upload it, evaluate. Compile this header's consumer together with the kernel
// source written by cusadi2.kernel_template.write_kernel_source.
//
//   #include "cusadi2.h"
//   cusadi2::Plan plan = cusadi2::load_plan("f.plan");
//   cusadi2::Program prog(plan);
//   prog.evaluate(d_in, d_out, batch, stream);   // pure launch; graph-capturable
//   prog.use_serial(scratch_bytes);              // optional: thread-per-instance mode
//                                                // (interprets CasADi's instruction stream)
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
    const int* nnz_in, const int* in_stage_mode, const int* in_stage_offset,
    const int* in_stage_count,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    const int* nnz_out, const int* out_stage_mode, const int* out_stage_count,
    cudaStream_t stream);
int cusadi2_launch_serial(
    const CUSADI2_SCALAR* const* inputs, CUSADI2_SCALAR* const* outputs,
    int total_instances, int n_stages, int block_size,
    const int* nnz_in, const int* in_stage_mode, const int* in_stage_offset,
    const int* in_stage_count, const int* nnz_out, const int* out_stage_count,
    int n_instr, const int* s_ops, const int* s_outs, const int* s_in_a, const int* s_in_b,
    const CUSADI2_SCALAR* s_const_vals, CUSADI2_SCALAR* scratch, int n_threads,
    int sz_w, int thread_major, cudaStream_t stream);
int cusadi2_set_shared_memory(int shared_bytes);
int cusadi2_max_active_blocks(int block_size, int shared_bytes);

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

// A plan file, fully in host memory: the level program (SSA-renamed node
// arrays with INPUT / OUTPUT nodes, constants, peak_live_slots) and the serial
// program (CasADi's instruction stream verbatim on its sz_w work slots, IO inline).
struct Plan {
    std::string name;
    int n_stages = 1, block_size = 0, peak_live_slots = 0, sz_w = 0;   // block_size 0: pick_block_size at upload
    std::vector<Arg> in, out;
    std::vector<int> const_slots; std::vector<double> const_vals;
    std::vector<int> level_offsets, level_widths;
    std::vector<int> node_ops, node_outs, node_in_a, node_in_b;
    std::vector<int> serial_ops, serial_outs, serial_in_a, serial_in_b;
    std::vector<double> serial_const_vals;
    int n_in() const { return (int)in.size(); }
    int n_out() const { return (int)out.size(); }
    int shared_bytes() const { return peak_live_slots * (int)sizeof(Scalar); }   // level mode, per block
    size_t serial_bytes() const { return (size_t)sz_w * sizeof(Scalar); }         // serial mode, per thread
};

// Threads per (instance, stage) block for `p`: the b in {32, ..., 1024} minimising
// the modelled per-SM time per instance T(b) / R(b), where
//   T(b) = sum over levels of ceil(width / b) + n_levels   interpreter passes, plus one
//                                                          barrier per level costed as a pass
//   R(b) = blocks of that size resident per SM at p's shared-memory footprint, from
//          the CUDA occupancy calculator (registers, shared memory, block and thread
//          limits of the current device and kernel build).
// Ties go to the smaller block (fewer idle warps at each barrier). Call after the
// kernel's shared-memory cap covers p (cusadi2_set_shared_memory).
inline int pick_block_size(const Plan& p) {
    int best = 0; double best_cost = 0;
    for (int b = 32; b <= 1024; b *= 2) {
        int resident = cusadi2_max_active_blocks(b, p.shared_bytes());
        if (resident <= 0) continue;
        double passes = (double)p.level_widths.size();
        for (int w : p.level_widths) passes += (w + b - 1) / b;
        double cost = passes / resident;
        if (!best || cost < best_cost) { best = b; best_cost = cost; }
    }
    if (!best) throw std::runtime_error("cusadi2: no block size fits plan " + p.name);
    return best;
}

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

inline int device() { int d = 0; cudaGetDevice(&d); return d; }

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
    if (version != 3)
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
    n = r.i(); r.ia(p.level_offsets, n); r.ia(p.level_widths, n);
    n = r.i(); r.ia(p.node_ops, n); r.ia(p.node_outs, n); r.ia(p.node_in_a, n); r.ia(p.node_in_b, n);
    p.sz_w = r.i();
    n = r.i(); r.ia(p.serial_ops, n); r.ia(p.serial_outs, n); r.ia(p.serial_in_a, n); r.ia(p.serial_in_b, n);
    n = r.i(); p.serial_const_vals = r.arr<double>(n);
    return p;
}

// A plan uploaded to the device. evaluate() is a single kernel launch on the
// given stream with no host-side work, so it can be captured into a CUDA graph.
// Default mode is level-parallel (one block per instance and stage, work vector
// in shared memory). use_serial() switches to one thread per instance and stage
// walking CasADi's instruction stream with its sz_w-slot work vector in a
// global scratch buffer; it suits functions whose levels are too narrow to fill
// a block, and it is the no-level-parallelism baseline. Scratch is allocated
// once, so evaluate() stays allocation-free.
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
        h_ = Hdr{p.n_stages, p.block_size, p.shared_bytes(), p.sz_w, (int)p.const_slots.size(),
                 (int)p.level_offsets.size(), p.n_in(), p.n_out(), (int)p.serial_ops.size()};
        std::vector<int> nnz_in, nnz_out, in_mode, in_offset, in_count, out_mode, out_count;
        for (const Arg& a : p.in) {
            nnz_in.push_back(a.nnz()); in_mode.push_back(a.mode);
            in_offset.push_back(a.offset); in_count.push_back(a.count);
        }
        for (const Arg& a : p.out) {
            nnz_out.push_back(a.nnz()); out_mode.push_back(a.mode); out_count.push_back(a.count);
        }
        std::vector<Scalar> consts(p.const_vals.begin(), p.const_vals.end());
        std::vector<Scalar> s_consts(p.serial_const_vals.begin(), p.serial_const_vals.end());
        d_.nnz_in = dev(nnz_in); d_.nnz_out = dev(nnz_out);
        d_.in_mode = dev(in_mode); d_.in_offset = dev(in_offset); d_.in_count = dev(in_count);
        d_.out_mode = dev(out_mode); d_.out_count = dev(out_count);
        d_.const_slots = dev(p.const_slots); d_.const_vals = dev(consts);
        d_.level_offsets = dev(p.level_offsets); d_.level_widths = dev(p.level_widths);
        d_.node_ops = dev(p.node_ops); d_.node_outs = dev(p.node_outs);
        d_.node_in_a = dev(p.node_in_a); d_.node_in_b = dev(p.node_in_b);
        d_.s_ops = dev(p.serial_ops); d_.s_outs = dev(p.serial_outs);
        d_.s_in_a = dev(p.serial_in_a); d_.s_in_b = dev(p.serial_in_b); d_.s_const_vals = dev(s_consts);
        // Raise the level kernel's shared-memory cap if the device can honour it;
        // a plan too large for level mode can still run in serial mode.
        int max_smem = 0;
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, detail::device());
        if (h_.shared_bytes <= max_smem) {
            CUSADI2_CHECK((cudaError_t)cusadi2_set_shared_memory(h_.shared_bytes));
            if (!h_.block_size) h_.block_size = pick_block_size(p);
        }
    }

    // Serial mode with sz_w scalars of scratch per thread: threads =
    // min(scratch_bytes / per-thread bytes, max_threads); when batch * n_stages
    // exceeds that, threads loop over items. Scratch is slot-major (coalesced)
    // unless thread_major, which reproduces the original cusadi's strided layout.
    void use_serial(size_t scratch_bytes = size_t(256) << 20, int block_size = 256,
                    int max_threads = 1 << 20, bool thread_major = false) {
        use_level();
        size_t per_thread = (size_t)h_.sz_w * sizeof(Scalar);
        size_t n = scratch_bytes / per_thread;
        if (n < 1) throw std::runtime_error("cusadi2: serial scratch budget below one thread");
        h_.serial_threads = (int)(n < (size_t)max_threads ? n : (size_t)max_threads);
        h_.serial_block = block_size;
        h_.thread_major = thread_major ? 1 : 0;
        CUSADI2_CHECK(cudaMalloc(&d_.scratch, per_thread * h_.serial_threads));
    }
    void use_level() { cudaFree(d_.scratch); d_.scratch = nullptr; h_.serial_threads = 0; }
    bool serial() const { return d_.scratch != nullptr; }

    // Launch on `stream`; returns cudaPeekAtLastError() (0 on success).
    int evaluate(const Scalar* const* d_in, Scalar* const* d_out, int batch,
                 cudaStream_t stream = 0) const {
        if (d_.scratch) {
            long long items = (long long)batch * h_.n_stages;
            int threads = items < h_.serial_threads ? (int)items : h_.serial_threads;
            return cusadi2_launch_serial(d_in, d_out, batch, h_.n_stages, h_.serial_block,
                d_.nnz_in, d_.in_mode, d_.in_offset, d_.in_count, d_.nnz_out, d_.out_count,
                h_.n_instr, d_.s_ops, d_.s_outs, d_.s_in_a, d_.s_in_b, d_.s_const_vals,
                d_.scratch, threads, h_.sz_w, h_.thread_major, stream);
        }
        return cusadi2_launch(d_in, d_out, batch, h_.n_stages, h_.block_size, h_.shared_bytes,
            h_.n_consts, d_.const_slots, d_.const_vals,
            d_.nnz_in, d_.in_mode, d_.in_offset, d_.in_count,
            h_.n_levels, d_.level_offsets, d_.level_widths,
            d_.node_ops, d_.node_outs, d_.node_in_a, d_.node_in_b,
            d_.nnz_out, d_.out_mode, d_.out_count, stream);
    }

    int n_stages() const { return h_.n_stages; }
    int n_in() const { return h_.n_in; }
    int n_out() const { return h_.n_out; }

    void release() {
        const void* ptrs[] = {d_.nnz_in, d_.nnz_out, d_.in_mode, d_.in_offset, d_.in_count,
            d_.out_mode, d_.out_count, d_.const_slots, d_.const_vals, d_.level_offsets,
            d_.level_widths, d_.node_ops, d_.node_outs, d_.node_in_a, d_.node_in_b,
            d_.s_ops, d_.s_outs, d_.s_in_a, d_.s_in_b, d_.s_const_vals};
        for (const void* q : ptrs) if (q) cudaFree(const_cast<void*>(q));
        cudaFree(d_.scratch);
        d_ = Dev{};
    }

private:
    struct Hdr {
        int n_stages = 1, block_size = 32, shared_bytes = 0, sz_w = 0, n_consts = 0,
            n_levels = 0, n_in = 0, n_out = 0, n_instr = 0;
        int serial_threads = 0, serial_block = 256, thread_major = 0;
    } h_;
    struct Dev {
        const int *nnz_in = nullptr, *nnz_out = nullptr, *in_mode = nullptr,
            *in_offset = nullptr, *in_count = nullptr, *out_mode = nullptr, *out_count = nullptr,
            *const_slots = nullptr, *level_offsets = nullptr, *level_widths = nullptr,
            *node_ops = nullptr, *node_outs = nullptr, *node_in_a = nullptr, *node_in_b = nullptr,
            *s_ops = nullptr, *s_outs = nullptr, *s_in_a = nullptr, *s_in_b = nullptr;
        const Scalar *const_vals = nullptr, *s_const_vals = nullptr;
        Scalar* scratch = nullptr;
    } d_;
};

}  // namespace cusadi2
