// XLA FFI custom call running the staged backup-CBF model (cusadi2 head plan + N step plans)
// inside a JAX program, plus a bare-kernel timer.
//
//   backup_init(head_plan, step_plan, N, max_batch) -> 0, or <0 on failure
//   BackupCusadi2  FFI handler: (z[B,14], z_des[B,18]) ->
//       u[B,7], P[B,nnzP], q[B,7], G0[B,nnzG0], h0[B,78], Gp[N,B,nnzGk], hp[N,B,64]
//   backup_out_nnz(a) / rowidx / colptr: output sparsity, a = 0..5 for (u, P, q, G0, h0) of the head
//       and a = 5, 6 for (G, h) of the step
//   backup_time(batch, n, &median_us, &min_us): head + N steps timed with CUDA events, no XLA
//
// The rollout state z_k and its 8 tangents T_k (14 x 8, column-major nnz = 112) ping-pong between
// two device scratch buffers sized for max_batch. Pointer tables for all N + 1 launches are staged
// in pinned memory (ring) and copied with one cudaMemcpyAsync per call on XLA's stream.
#include "cusadi2.h"
#include "xla/ffi/api/ffi.h"
#include <algorithm>
#include <string>
#include <vector>
namespace ffi = xla::ffi;
using namespace cusadi2;
using S = Scalar;

static Plan g_head, g_step;
static Program *g_ph = nullptr, *g_ps = nullptr;
static int g_N = 0, g_max_batch = 0;
static S *g_z[2] = {nullptr, nullptr}, *g_T[2] = {nullptr, nullptr};
// per call: head table (2 in + 6 out = 8) + N step tables (2 in + 4 out = 6)
static int g_tbl = 0;                      // pointers per call
static constexpr int RING = 2048;
static S** g_dtable = nullptr; static S** g_htable = nullptr; static int g_slot = 0;

extern "C" int backup_init(const char* head_plan, const char* step_plan, int N, int max_batch) {
    g_head = load_plan(head_plan); g_step = load_plan(step_plan);
    int smem = 0; cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if ((int)g_head.shared_bytes() > smem || (int)g_step.shared_bytes() > smem) return -1;
    if (g_head.n_in() != 2 || g_head.n_out() != 6 || g_step.n_in() != 2 || g_step.n_out() != 4) return -2;
    if (g_head.out[5].sp.nnz() != 112 || g_step.in[1].sp.nnz() != 112 || g_step.out[1].sp.nnz() != 112) return -3;
    g_ph = new Program(g_head); g_ph->upload(g_head);
    g_ps = new Program(g_step); g_ps->upload(g_step);
    g_N = N; g_max_batch = max_batch;
    for (int i = 0; i < 2; ++i) {
        CUSADI2_CHECK(cudaMalloc(&g_z[i], sizeof(S) * 14 * max_batch));
        CUSADI2_CHECK(cudaMalloc(&g_T[i], sizeof(S) * 112 * max_batch));
    }
    g_tbl = 8 + 6 * N;
    CUSADI2_CHECK(cudaMalloc(&g_dtable, sizeof(S*) * g_tbl * RING));
    CUSADI2_CHECK(cudaMallocHost(&g_htable, sizeof(S*) * g_tbl * RING));
    return 0;
}
static const Arg& out_arg(int a) { return a < 5 ? g_head.out[a] : g_step.out[a - 3]; }   // 5 -> step G (out 2), 6 -> step h (out 3)
extern "C" int backup_out_nnz(int a) { return out_arg(a).sp.nnz(); }
extern "C" int backup_out_rowidx(int a, int* dst) { const auto& sp = out_arg(a).sp; for (int k = 0; k < sp.nnz(); ++k) dst[k] = sp.rowidx[k]; return sp.nnz(); }
extern "C" int backup_out_colptr(int a, int* dst) { const auto& sp = out_arg(a).sp; for (int c = 0; c <= sp.cols; ++c) dst[c] = sp.colptr[c]; return sp.cols; }
extern "C" int backup_block_sizes(int* head, int* step) { *head = pick_block_size(g_head); *step = pick_block_size(g_step); return 0; }
extern "C" long long backup_shared_bytes(int which) { return (long long)(which ? g_step.shared_bytes() : g_head.shared_bytes()); }

// Enqueue head + N steps for `batch` instances with XLA/user buffers; ht is a pinned staging table of g_tbl pointers.
static cudaError_t run(cudaStream_t stream, int batch, S** ht, S** dt, const S* z, const S* zd,
                       S* u, S* P, S* q, S* G0, S* h0, S* Gp, S* hp) {
    const int nnzG = g_step.out[2].sp.nnz(), nnzh = g_step.out[3].sp.nnz();
    S** t = ht;
    t[0] = const_cast<S*>(z); t[1] = const_cast<S*>(zd); t[2] = u; t[3] = P; t[4] = q; t[5] = G0; t[6] = h0; t[7] = g_T[0];
    for (int k = 0; k < g_N; ++k) {
        S** s = ht + 8 + 6 * k;
        s[0] = k == 0 ? const_cast<S*>(z) : g_z[(k + 1) & 1]; s[1] = g_T[k & 1];
        s[2] = g_z[k & 1]; s[3] = g_T[(k + 1) & 1];
        s[4] = Gp + (size_t)k * batch * nnzG; s[5] = hp + (size_t)k * batch * nnzh;
    }
    cudaError_t e = cudaMemcpyAsync(dt, ht, sizeof(S*) * g_tbl, cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    e = (cudaError_t)g_ph->evaluate((const S* const*)dt, dt + 2, batch, stream);
    if (e != cudaSuccess) return e;
    for (int k = 0; k < g_N; ++k) {
        S** s = dt + 8 + 6 * k;
        e = (cudaError_t)g_ps->evaluate((const S* const*)s, s + 2, batch, stream);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

static ffi::Error Impl(cudaStream_t stream, ffi::Buffer<ffi::F64> z, ffi::Buffer<ffi::F64> zd,
                       ffi::ResultBuffer<ffi::F64> u, ffi::ResultBuffer<ffi::F64> P, ffi::ResultBuffer<ffi::F64> q,
                       ffi::ResultBuffer<ffi::F64> G0, ffi::ResultBuffer<ffi::F64> h0,
                       ffi::ResultBuffer<ffi::F64> Gp, ffi::ResultBuffer<ffi::F64> hp) {
    if (!g_ph) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "backup_init not called");
    const int batch = (int)z.dimensions()[0];
    if (batch > g_max_batch) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "batch exceeds max_batch of backup_init");
    const int slot = g_slot; g_slot = (g_slot + 1) % RING;
    cudaError_t e = run(stream, batch, g_htable + (size_t)slot * g_tbl, g_dtable + (size_t)slot * g_tbl,
                        z.typed_data(), zd.typed_data(), u->typed_data(), P->typed_data(), q->typed_data(),
                        G0->typed_data(), h0->typed_data(), Gp->typed_data(), hp->typed_data());
    if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(BackupCusadi2, Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>());

// ---- diagnostics: head + N steps by CUDA events, own buffers, default stream ----
extern "C" int backup_time(int batch, int n, double* median_us, double* min_us, double* head_us) {
    if (!g_ph || batch > g_max_batch) return -1;
    std::vector<S*> bufs;
    auto mk = [&](size_t count) { S* d; cudaMalloc(&d, count * sizeof(S)); cudaMemset(d, 0, count * sizeof(S)); bufs.push_back(d); return d; };
    S* z = mk(g_head.in[0].len(batch)); S* zd = mk(g_head.in[1].len(batch));
    S* o[5]; for (int a = 0; a < 5; ++a) o[a] = mk(g_head.out[a].len(batch));
    S* Gp = mk((size_t)g_N * g_step.out[2].len(batch)); S* hp = mk((size_t)g_N * g_step.out[3].len(batch));
    S** ht; cudaMallocHost(&ht, sizeof(S*) * g_tbl); S** dt; cudaMalloc(&dt, sizeof(S*) * g_tbl);
    cudaEvent_t e0, e1, eh; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&eh);
    for (int i = 0; i < 5; ++i) run(0, batch, ht, dt, z, zd, o[0], o[1], o[2], o[3], o[4], Gp, hp);
    cudaDeviceSynchronize();
    std::vector<float> ms(n), hms(n);
    for (int i = 0; i < n; ++i) {
        cudaEventRecord(e0, 0);
        // head alone, then the full sequence (head is re-run inside run())
        g_ph->evaluate((const S* const*)dt, dt + 2, batch, 0);
        cudaEventRecord(eh, 0);
        run(0, batch, ht, dt, z, zd, o[0], o[1], o[2], o[3], o[4], Gp, hp);
        cudaEventRecord(e1, 0); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&hms[i], e0, eh); cudaEventElapsedTime(&ms[i], eh, e1);
    }
    std::sort(ms.begin(), ms.end()); std::sort(hms.begin(), hms.end());
    *median_us = 1e3 * ms[n / 2]; *min_us = 1e3 * ms[0]; *head_us = 1e3 * hms[n / 2];
    for (S* p : bufs) cudaFree(p); cudaFree(dt); cudaFreeHost(ht);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(eh);
    return 0;
}
