// XLA FFI custom call that runs a cusadi2 level-plan kernel inside a JAX program.
//
//   oscbf_init(plan_path, serial) -> block size (>=0) or -1 if the plan does not fit shared memory
//                                   serial != 0 selects cusadi2's thread-per-instance serial mode
//   OscbfCusadi2           XLA FFI handler: (z[B,14], z_des[B,18]) -> 5 outputs [B, nnz_k]
//
// Buffers are XLA's; the handler only enqueues one H2D copy of the 7 device pointers
// (ring of pinned staging slots) and the cusadi2 kernel launch on XLA's stream.
#include "cusadi2.h"
#include "xla/ffi/api/ffi.h"
#include <string>
#include <vector>
#include <algorithm>
namespace ffi = xla::ffi;
using namespace cusadi2;
using S = Scalar;

static Plan g_plan;
static Program* g_prog = nullptr;
static int g_block = 0;
static constexpr int NPTR = 7, RING = 4096;
static S** g_dtable = nullptr;   // device: RING * NPTR pointers
static S** g_htable = nullptr;   // pinned host staging
static int g_slot = 0;

extern "C" int oscbf_init(const char* plan_path, int serial) {
    g_plan = load_plan(plan_path);
    int smem = 0; cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if ((int)g_plan.shared_bytes() > smem) return -1;
    if (g_plan.n_in() != 2 || g_plan.n_out() != 5) return -2;
    g_prog = new Program(g_plan);
    g_prog->upload(g_plan);
    g_block = g_plan.block_size ? g_plan.block_size : pick_block_size(g_plan);
    if (serial) { g_prog->use_serial(); g_block = 256; }
    CUSADI2_CHECK(cudaMalloc(&g_dtable, sizeof(S*) * NPTR * RING));
    CUSADI2_CHECK(cudaMallocHost(&g_htable, sizeof(S*) * NPTR * RING));
    return g_block;
}
extern "C" int oscbf_out_nnz(int a) { return g_plan.out[a].sp.nnz(); }
extern "C" int oscbf_out_rowidx(int a, int* dst) {
    const auto& sp = g_plan.out[a].sp;
    for (int k = 0; k < sp.nnz(); ++k) dst[k] = sp.rowidx[k];
    return sp.nnz();
}
extern "C" int oscbf_out_colptr(int a, int* dst) {
    const auto& sp = g_plan.out[a].sp;
    for (int c = 0; c <= sp.cols; ++c) dst[c] = sp.colptr[c];
    return sp.cols;
}
extern "C" long long oscbf_shared_bytes() { return (long long)g_plan.shared_bytes(); }

static ffi::Error Impl(cudaStream_t stream, ffi::Buffer<ffi::F64> z, ffi::Buffer<ffi::F64> zd,
                       ffi::ResultBuffer<ffi::F64> u, ffi::ResultBuffer<ffi::F64> P, ffi::ResultBuffer<ffi::F64> q,
                       ffi::ResultBuffer<ffi::F64> G, ffi::ResultBuffer<ffi::F64> h) {
    if (!g_prog) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "oscbf_init not called");
    const int batch = (int)z.dimensions()[0];
    const int slot = g_slot; g_slot = (g_slot + 1) % RING;
    S** ht = g_htable + slot * NPTR;
    ht[0] = const_cast<S*>(z.typed_data()); ht[1] = const_cast<S*>(zd.typed_data());
    ht[2] = u->typed_data(); ht[3] = P->typed_data(); ht[4] = q->typed_data(); ht[5] = G->typed_data(); ht[6] = h->typed_data();
    S** dt = g_dtable + slot * NPTR;
    cudaError_t e = cudaMemcpyAsync(dt, ht, sizeof(S*) * NPTR, cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    e = (cudaError_t)g_prog->evaluate((const S* const*)dt, dt + 2, batch, stream);
    if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(OscbfCusadi2, Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>());

// ---- diagnostics: bare kernel time by CUDA events, no XLA involved ----
// Allocates its own buffers for `batch`, runs `n` evaluations and returns the
// median and min kernel time in microseconds.
extern "C" int oscbf_time_kernel(int batch, int n, double* median_us, double* min_us) {
    if (!g_prog) return -1;
    std::vector<S*> bufs; std::vector<S*> h(NPTR);
    for (int a = 0; a < 2; ++a) { S* d; cudaMalloc(&d, g_plan.in[a].len(batch) * sizeof(S)); cudaMemset(d, 0, g_plan.in[a].len(batch) * sizeof(S)); bufs.push_back(d); h[a] = d; }
    for (int a = 0; a < 5; ++a) { S* d; cudaMalloc(&d, g_plan.out[a].len(batch) * sizeof(S)); bufs.push_back(d); h[2 + a] = d; }
    S** dt; cudaMalloc(&dt, sizeof(S*) * NPTR); cudaMemcpy(dt, h.data(), sizeof(S*) * NPTR, cudaMemcpyHostToDevice);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 5; ++i) g_prog->evaluate((const S* const*)dt, dt + 2, batch, 0);
    cudaDeviceSynchronize();
    std::vector<float> ms(n);
    for (int i = 0; i < n; ++i) {
        cudaEventRecord(e0, 0);
        g_prog->evaluate((const S* const*)dt, dt + 2, batch, 0);
        cudaEventRecord(e1, 0); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms[i], e0, e1);
    }
    std::sort(ms.begin(), ms.end());
    *median_us = 1e3 * ms[n / 2]; *min_us = 1e3 * ms[0];
    for (S* p : bufs) cudaFree(p); cudaFree(dt); cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
