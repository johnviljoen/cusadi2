// cusadi2: host-free C++/CUDA runtime for the level-parallel kernel.
// No Python at deployment: load a plan file written by cusadi2.plan.write_plan,
// upload it, evaluate. Compile this header's consumer together with the kernel
// source written by cusadi2.kernel_template.write_kernel_source.
//
//   #include "cusadi2.h"
//   cusadi2::Plan plan = cusadi2::load_plan("f.plan");
//   cusadi2::Program prog(plan);
//   prog.evaluate(d_in, d_out, batch, stream);   // graph-capturable once reserve(batch) ran
//   prog.use_serial(scratch_bytes);              // optional: thread-per-instance mode
//                                                // (interprets CasADi's instruction stream)
//
// d_in / d_out are DEVICE-resident arrays of device buffer pointers, one per
// declared argument. Buffer a holds batch * plan.in[a].count * nnz elements,
// laid out (instance, stage_count, nnz); the horizon, the per-arg stage counts
// and the per-element tables that route each stage's inputs and outputs are
// baked into the plan (version 4; version 3 files load as the plain stage
// specs). A plan whose outputs sum several stage values (a composed Lagrangian
// Hessian, a cost summed over stages) streams them through a scratch buffer
// the Program owns and a second, deterministic segmented-sum launch; call
// reserve(batch) before capturing evaluate() into a CUDA graph so that no
// allocation happens inside the capture. Buffers are caller-owned.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
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

// Most buffer pointers a by-value launch (cusadi2_launch_v) carries per side;
// must match the emitted kernel's definition.
#ifndef CUSADI2_MAX_ARGS
#define CUSADI2_MAX_ARGS 16
#endif

// --- kernel entry points (defined in the emitted kernel .cu) -----------------
// IO routing arrays (per function input a / output k; see the kernel source):
//   in_src, in_nnz, in_off, in_stride, in_table; out_dst, out_nnz, out_off, out_stride, out_table; scratch
int cusadi2_launch(
    const CUSADI2_SCALAR* const* inputs, CUSADI2_SCALAR* const* outputs,
    int total_instances, int n_stages, int block_size, int shared_bytes,
    int n_consts, const int* const_slots, const CUSADI2_SCALAR* const_vals,
    const int* in_src, const int* in_nnz, const int* in_off, const int* in_stride, const int* in_table,
    const int* out_dst, const int* out_nnz, const int* out_off, const int* out_stride, const int* out_table,
    CUSADI2_SCALAR* scratch,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    cudaStream_t stream);
int cusadi2_launch_serial(
    const CUSADI2_SCALAR* const* inputs, CUSADI2_SCALAR* const* outputs,
    int total_instances, int n_stages, int block_size,
    const int* in_src, const int* in_nnz, const int* in_off, const int* in_stride, const int* in_table,
    const int* out_dst, const int* out_nnz, const int* out_off, const int* out_stride, const int* out_table,
    CUSADI2_SCALAR* scratch,
    int n_instr, const int* s_ops, const int* s_outs, const int* s_in_a, const int* s_in_b,
    const CUSADI2_SCALAR* s_const_vals, CUSADI2_SCALAR* work, int n_threads,
    int sz_w, int thread_major, cudaStream_t stream);
int cusadi2_launch_reduce(
    CUSADI2_SCALAR* const* outputs, int dst, const CUSADI2_SCALAR* scratch, int total_instances, int G,
    int out_stride, int scratch_stride, const int* segptr, const int* segidx, cudaStream_t stream);
int cusadi2_launch_v(
    const CUSADI2_SCALAR* const* h_inputs, CUSADI2_SCALAR* const* h_outputs, int n_in, int n_out,
    int total_instances, int n_stages, int block_size, int shared_bytes,
    int n_consts, const int* const_slots, const CUSADI2_SCALAR* const_vals,
    const int* in_src, const int* in_nnz, const int* in_off, const int* in_stride, const int* in_table,
    const int* out_dst, const int* out_nnz, const int* out_off, const int* out_stride, const int* out_table,
    CUSADI2_SCALAR* scratch,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    cudaStream_t stream);
int cusadi2_launch_serial_v(
    const CUSADI2_SCALAR* const* h_inputs, CUSADI2_SCALAR* const* h_outputs, int n_in, int n_out,
    int total_instances, int n_stages, int block_size,
    const int* in_src, const int* in_nnz, const int* in_off, const int* in_stride, const int* in_table,
    const int* out_dst, const int* out_nnz, const int* out_off, const int* out_stride, const int* out_table,
    CUSADI2_SCALAR* scratch,
    int n_instr, const int* s_ops, const int* s_outs, const int* s_in_a, const int* s_in_b,
    const CUSADI2_SCALAR* s_const_vals, CUSADI2_SCALAR* work, int n_threads,
    int sz_w, int thread_major, cudaStream_t stream);
int cusadi2_launch_reduce_v(
    CUSADI2_SCALAR* const* h_outputs, int n_out, int dst, const CUSADI2_SCALAR* scratch, int total_instances,
    int G, int out_stride, int scratch_stride, const int* segptr, const int* segidx, cudaStream_t stream);
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

// One declared argument: sparsity plus its stage-axis layout (mode 0=STAGE
// 1=SHARED 2=COUPLED; count = leading-axis length at the plan's horizon).
struct Arg {
    Sparsity sp;
    int mode = 0, offset = 0, span = 0, count = 1;
    int nnz() const { return sp.nnz(); }
    size_t len(int batch) const { return (size_t)batch * count * sp.nnz(); }  // elements
};

// Where one function argument reads from / writes to: declared argument `arg`
// (-1 on outputs: the scratch stream) and the (n_stages x nnz) table of
// indices inside that argument's instance block (-1: zero / discard).
struct StageMap {
    int arg = 0, nnz = 0;
    std::vector<int> table;
};

// Declared output `dst`: entry g is the ordered sum of the scratch-stream
// scalars segidx[segptr[g] .. segptr[g+1]) of its instance.
struct Reduce {
    int dst = 0;
    std::vector<int> segptr, segidx;
};

// A plan file, fully in host memory: the declared arguments, the IO routing
// (stage_in / stage_out / reduces), the level program (SSA-renamed node arrays
// with INPUT / OUTPUT nodes, constants, peak_live_slots) and the serial program
// (CasADi's instruction stream verbatim on its sz_w work slots, IO inline).
struct Plan {
    std::string name;
    int n_stages = 1, block_size = 0, peak_live_slots = 0, sz_w = 0;   // block_size 0: pick_block_size at upload
    std::vector<Arg> in, out;
    std::vector<StageMap> stage_in, stage_out;
    std::vector<Reduce> reduces;
    int scratch_per_instance = 0;
    std::vector<int> const_slots; std::vector<double> const_vals;
    std::vector<int> level_offsets, level_widths;
    std::vector<int> node_ops, node_outs, node_in_a, node_in_b;
    std::vector<int> serial_ops, serial_outs, serial_in_a, serial_in_b;
    std::vector<double> serial_const_vals;
    int n_in() const { return (int)in.size(); }
    int n_out() const { return (int)out.size(); }
    int shared_bytes() const { return peak_live_slots * (int)sizeof(Scalar); }   // level mode, per block
    size_t serial_bytes() const { return (size_t)sz_w * sizeof(Scalar); }         // serial mode, per thread
    size_t scratch_bytes(int batch) const { return (size_t)batch * scratch_per_instance * sizeof(Scalar); }
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
    std::string owned;  // file contents when constructed from a path
    const char* data;
    size_t size, pos = 0;
    std::string path;   // for messages
    Reader(const void* d, size_t n, const std::string& p) : data((const char*)d), size(n), path(p) {}
    explicit Reader(const std::string& p) : path(p) {
        std::ifstream f(p, std::ios::binary);
        if (!f) throw std::runtime_error("cusadi2: cannot open plan file: " + p);
        owned.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        data = owned.data(); size = owned.size();
    }
    void raw(void* dst, size_t bytes) {
        if (pos + bytes > size) throw std::runtime_error("cusadi2: short read in " + path);
        std::memcpy(dst, data + pos, bytes);
        pos += bytes;
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

// The plain stage specs as tables (a version-3 file, or a version-4 file written
// without a composition carries the same tables explicitly).
inline void legacy_tables(Plan& p) {
    p.stage_in.resize(p.n_in());
    for (int a = 0; a < p.n_in(); ++a) {
        const Arg& d = p.in[a];
        StageMap& m = p.stage_in[a];
        m.arg = a; m.nnz = d.nnz(); m.table.resize((size_t)p.n_stages * m.nnz);
        for (int s = 0; s < p.n_stages; ++s) {
            int knot = d.mode == 1 ? 0 : s + d.offset;
            for (int b = 0; b < m.nnz; ++b) m.table[(size_t)s * m.nnz + b] = knot * m.nnz + b;
        }
    }
    p.stage_out.resize(p.n_out());
    for (int k = 0; k < p.n_out(); ++k) {
        StageMap& m = p.stage_out[k];
        m.arg = k; m.nnz = p.out[k].nnz(); m.table.resize((size_t)p.n_stages * m.nnz);
        for (int s = 0; s < p.n_stages; ++s)
            for (int b = 0; b < m.nnz; ++b) m.table[(size_t)s * m.nnz + b] = s * m.nnz + b;
    }
    p.reduces.clear();
    p.scratch_per_instance = 0;
}
}  // namespace detail

// Parse a plan from memory (the bytes of a plan file), e.g. produced by
// cusadi2.plan_bytes() without touching the filesystem.
inline Plan load_plan_from_memory(const void* data, size_t size, const std::string& what = "<memory>") {
    detail::Reader r(data, size, what);
    char magic[4]; r.raw(magic, 4);
    if (std::string(magic, 4) != "CS2P")
        throw std::runtime_error("cusadi2: bad magic in " + what);
    int version = r.i();
    if (version != 3 && version != 4)
        throw std::runtime_error("cusadi2: unsupported plan version in " + what);
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
    if (version >= 4) {
        p.stage_in.resize(r.i());
        for (StageMap& m : p.stage_in) {
            m.arg = r.i(); m.nnz = r.i(); r.ia(m.table, (size_t)p.n_stages * m.nnz);
            if (m.arg < 0 || m.arg >= n_in) throw std::runtime_error("cusadi2: bad input routing in " + what);
        }
        p.stage_out.resize(r.i());
        for (StageMap& m : p.stage_out) {
            m.arg = r.i(); m.nnz = r.i(); r.ia(m.table, (size_t)p.n_stages * m.nnz);
            if (m.arg < -1 || m.arg >= n_out) throw std::runtime_error("cusadi2: bad output routing in " + what);
        }
        p.reduces.resize(r.i());
        for (Reduce& x : p.reduces) {
            x.dst = r.i(); int G = r.i(); r.ia(x.segptr, (size_t)G + 1); r.ia(x.segidx, (size_t)r.i());
            if (x.dst < 0 || x.dst >= n_out || G != (int)p.out[x.dst].len(1))
                throw std::runtime_error("cusadi2: bad reduce in " + what);
        }
        p.scratch_per_instance = r.i();
    } else {
        detail::legacy_tables(p);
    }
    int n = r.i(); r.ia(p.const_slots, n); p.const_vals = r.arr<double>(n);
    n = r.i(); r.ia(p.level_offsets, n); r.ia(p.level_widths, n);
    n = r.i(); r.ia(p.node_ops, n); r.ia(p.node_outs, n); r.ia(p.node_in_a, n); r.ia(p.node_in_b, n);
    p.sz_w = r.i();
    n = r.i(); r.ia(p.serial_ops, n); r.ia(p.serial_outs, n); r.ia(p.serial_in_a, n); r.ia(p.serial_in_b, n);
    n = r.i(); p.serial_const_vals = r.arr<double>(n);
    return p;
}

inline Plan load_plan(const std::string& path) {
    detail::Reader r(path);
    return load_plan_from_memory(r.data, r.size, path);
}

// A plan uploaded to the device. evaluate() is one kernel launch on the given
// stream (plus one segmented-sum launch per reduced output) with no host-side
// work, so it can be captured into a CUDA graph once the scratch stream is
// reserved for the batch. Default mode is level-parallel (one block per
// instance and stage, work vector in shared memory). use_serial() switches to
// one thread per instance and stage walking CasADi's instruction stream with
// its sz_w-slot work vector in a global buffer; it suits functions whose levels
// are too narrow to fill a block, and it is the no-level-parallelism baseline.
// The serial work buffer and the scratch stream are allocated once (use_serial,
// reserve), so evaluate() stays allocation-free after that.
class Program {
public:
    Program() = default;
    explicit Program(const Plan& p) { upload(p); }
    ~Program() { release(); }
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&& o) noexcept { *this = std::move(o); }
    Program& operator=(Program&& o) noexcept {
        if (this != &o) { release(); h_ = o.h_; d_ = o.d_; reduces_ = std::move(o.reduces_); o.d_ = Dev{}; o.reduces_.clear(); }
        return *this;
    }

    void upload(const Plan& p) {
        using detail::dev;
        release();
        h_ = Hdr{p.n_stages, p.block_size, p.shared_bytes(), p.sz_w, (int)p.const_slots.size(),
                 (int)p.level_offsets.size(), p.n_in(), p.n_out(), (int)p.serial_ops.size(),
                 p.scratch_per_instance};
        std::vector<int> in_src, in_nnz, in_off, in_stride, in_table, out_dst, out_nnz, out_off, out_stride, out_table;
        for (const StageMap& m : p.stage_in) {
            in_src.push_back(m.arg); in_nnz.push_back(m.nnz); in_off.push_back((int)in_table.size());
            in_stride.push_back((int)p.in[m.arg].len(1));
            in_table.insert(in_table.end(), m.table.begin(), m.table.end());
        }
        for (const StageMap& m : p.stage_out) {
            out_dst.push_back(m.arg); out_nnz.push_back(m.nnz); out_off.push_back((int)out_table.size());
            out_stride.push_back(m.arg < 0 ? p.scratch_per_instance : (int)p.out[m.arg].len(1));
            out_table.insert(out_table.end(), m.table.begin(), m.table.end());
        }
        std::vector<Scalar> consts(p.const_vals.begin(), p.const_vals.end());
        std::vector<Scalar> s_consts(p.serial_const_vals.begin(), p.serial_const_vals.end());
        d_.in_src = dev(in_src); d_.in_nnz = dev(in_nnz); d_.in_off = dev(in_off); d_.in_stride = dev(in_stride);
        d_.in_table = dev(in_table);
        d_.out_dst = dev(out_dst); d_.out_nnz = dev(out_nnz); d_.out_off = dev(out_off); d_.out_stride = dev(out_stride);
        d_.out_table = dev(out_table);
        for (const Reduce& x : p.reduces) {
            Red r; r.dst = x.dst; r.G = (int)x.segptr.size() - 1; r.out_stride = (int)p.out[x.dst].len(1);
            r.segptr = dev(x.segptr); r.segidx = dev(x.segidx);
            reduces_.push_back(r);
        }
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

    // Serial mode with sz_w scalars of work per thread: threads =
    // min(work_bytes / per-thread bytes, max_threads); when batch * n_stages
    // exceeds that, threads loop over items. Work is slot-major (coalesced)
    // unless thread_major, which reproduces the original cusadi's strided layout.
    void use_serial(size_t work_bytes = size_t(256) << 20, int block_size = 256,
                    int max_threads = 1 << 20, bool thread_major = false) {
        use_level();
        size_t per_thread = (size_t)h_.sz_w * sizeof(Scalar);
        size_t n = work_bytes / per_thread;
        if (n < 1) throw std::runtime_error("cusadi2: serial work budget below one thread");
        h_.serial_threads = (int)(n < (size_t)max_threads ? n : (size_t)max_threads);
        h_.serial_block = block_size;
        h_.thread_major = thread_major ? 1 : 0;
        CUSADI2_CHECK(cudaMalloc(&d_.work, per_thread * h_.serial_threads));
    }
    void use_level() { cudaFree(d_.work); d_.work = nullptr; h_.serial_threads = 0; }
    bool serial() const { return d_.work != nullptr; }

    // Make the scratch stream hold `batch` instances (no-op for plans without
    // reduces, or when it already does). evaluate() calls it when needed, which
    // allocates; call it yourself before a graph capture.
    void reserve(int batch) const {
        if (!h_.scratch_per_instance || batch <= scratch_cap_) return;
        cudaFree(scratch_); scratch_ = nullptr;
        CUSADI2_CHECK(cudaMalloc(&scratch_, (size_t)batch * h_.scratch_per_instance * sizeof(Scalar)));
        scratch_cap_ = batch;
    }
    bool needs_scratch() const { return h_.scratch_per_instance > 0; }
    size_t scratch_elements(int batch) const { return (size_t)batch * h_.scratch_per_instance; }

    // Launch on `stream`; returns cudaPeekAtLastError() (0 on success).
    int evaluate(const Scalar* const* d_in, Scalar* const* d_out, int batch,
                 cudaStream_t stream = 0) const {
        Scalar* scratch = nullptr;
        if (h_.scratch_per_instance) { reserve(batch); scratch = scratch_; }
        int e;
        if (d_.work) {
            long long items = (long long)batch * h_.n_stages;
            int threads = items < h_.serial_threads ? (int)items : h_.serial_threads;
            e = cusadi2_launch_serial(d_in, d_out, batch, h_.n_stages, h_.serial_block,
                d_.in_src, d_.in_nnz, d_.in_off, d_.in_stride, d_.in_table,
                d_.out_dst, d_.out_nnz, d_.out_off, d_.out_stride, d_.out_table, scratch,
                h_.n_instr, d_.s_ops, d_.s_outs, d_.s_in_a, d_.s_in_b, d_.s_const_vals,
                d_.work, threads, h_.sz_w, h_.thread_major, stream);
        } else {
            e = cusadi2_launch(d_in, d_out, batch, h_.n_stages, h_.block_size, h_.shared_bytes,
                h_.n_consts, d_.const_slots, d_.const_vals,
                d_.in_src, d_.in_nnz, d_.in_off, d_.in_stride, d_.in_table,
                d_.out_dst, d_.out_nnz, d_.out_off, d_.out_stride, d_.out_table, scratch,
                h_.n_levels, d_.level_offsets, d_.level_widths,
                d_.node_ops, d_.node_outs, d_.node_in_a, d_.node_in_b, stream);
        }
        for (const Red& r : reduces_) {
            if (e) break;
            e = cusadi2_launch_reduce(d_out, r.dst, scratch, batch, r.G, r.out_stride, h_.scratch_per_instance,
                                      r.segptr, r.segidx, stream);
        }
        return e;
    }

    // Same launch with the buffer pointers given as host arrays (n_in / n_out
    // device pointers, at most CUSADI2_MAX_ARGS each) passed by value in the
    // kernel parameters: nothing is staged through device memory, so callers
    // whose buffer addresses change every call (the JAX FFI handler) need no
    // per-call copy. `scratch` (scratch_elements(batch) scalars) replaces the
    // Program's own scratch stream when given. Graph-capturable like evaluate().
    int evaluate_ptrs(const Scalar* const* h_in, Scalar* const* h_out, int batch,
                      cudaStream_t stream = 0, Scalar* scratch = nullptr) const {
        if (h_.scratch_per_instance && !scratch) { reserve(batch); scratch = scratch_; }
        int e;
        if (d_.work) {
            long long items = (long long)batch * h_.n_stages;
            int threads = items < h_.serial_threads ? (int)items : h_.serial_threads;
            e = cusadi2_launch_serial_v(h_in, h_out, h_.n_in, h_.n_out, batch, h_.n_stages,
                h_.serial_block, d_.in_src, d_.in_nnz, d_.in_off, d_.in_stride, d_.in_table,
                d_.out_dst, d_.out_nnz, d_.out_off, d_.out_stride, d_.out_table, scratch,
                h_.n_instr, d_.s_ops, d_.s_outs, d_.s_in_a, d_.s_in_b, d_.s_const_vals,
                d_.work, threads, h_.sz_w, h_.thread_major, stream);
        } else {
            e = cusadi2_launch_v(h_in, h_out, h_.n_in, h_.n_out, batch, h_.n_stages, h_.block_size,
                h_.shared_bytes, h_.n_consts, d_.const_slots, d_.const_vals,
                d_.in_src, d_.in_nnz, d_.in_off, d_.in_stride, d_.in_table,
                d_.out_dst, d_.out_nnz, d_.out_off, d_.out_stride, d_.out_table, scratch,
                h_.n_levels, d_.level_offsets, d_.level_widths,
                d_.node_ops, d_.node_outs, d_.node_in_a, d_.node_in_b, stream);
        }
        for (const Red& r : reduces_) {
            if (e) break;
            e = cusadi2_launch_reduce_v(h_out, h_.n_out, r.dst, scratch, batch, r.G, r.out_stride,
                                        h_.scratch_per_instance, r.segptr, r.segidx, stream);
        }
        return e;
    }

    int n_stages() const { return h_.n_stages; }
    int n_in() const { return h_.n_in; }
    int n_out() const { return h_.n_out; }

    void release() {
        const void* ptrs[] = {d_.in_src, d_.in_nnz, d_.in_off, d_.in_stride, d_.in_table,
            d_.out_dst, d_.out_nnz, d_.out_off, d_.out_stride, d_.out_table,
            d_.const_slots, d_.const_vals, d_.level_offsets,
            d_.level_widths, d_.node_ops, d_.node_outs, d_.node_in_a, d_.node_in_b,
            d_.s_ops, d_.s_outs, d_.s_in_a, d_.s_in_b, d_.s_const_vals};
        for (const void* q : ptrs) if (q) cudaFree(const_cast<void*>(q));
        for (const Red& r : reduces_) { cudaFree(const_cast<int*>(r.segptr)); cudaFree(const_cast<int*>(r.segidx)); }
        reduces_.clear();
        cudaFree(d_.work);
        cudaFree(scratch_); scratch_ = nullptr; scratch_cap_ = 0;
        d_ = Dev{};
    }

private:
    struct Hdr {
        int n_stages = 1, block_size = 32, shared_bytes = 0, sz_w = 0, n_consts = 0,
            n_levels = 0, n_in = 0, n_out = 0, n_instr = 0, scratch_per_instance = 0;
        int serial_threads = 0, serial_block = 256, thread_major = 0;
    } h_;
    struct Dev {
        const int *in_src = nullptr, *in_nnz = nullptr, *in_off = nullptr, *in_stride = nullptr, *in_table = nullptr,
            *out_dst = nullptr, *out_nnz = nullptr, *out_off = nullptr, *out_stride = nullptr, *out_table = nullptr,
            *const_slots = nullptr, *level_offsets = nullptr, *level_widths = nullptr,
            *node_ops = nullptr, *node_outs = nullptr, *node_in_a = nullptr, *node_in_b = nullptr,
            *s_ops = nullptr, *s_outs = nullptr, *s_in_a = nullptr, *s_in_b = nullptr;
        const Scalar *const_vals = nullptr, *s_const_vals = nullptr;
        Scalar* work = nullptr;
    } d_;
    struct Red { int dst = 0, G = 0, out_stride = 0; const int *segptr = nullptr, *segidx = nullptr; };
    std::vector<Red> reduces_;
    mutable Scalar* scratch_ = nullptr;
    mutable int scratch_cap_ = 0;
};

}  // namespace cusadi2
