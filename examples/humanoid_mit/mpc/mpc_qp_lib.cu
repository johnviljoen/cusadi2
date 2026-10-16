// The QP solver of mpc_cudss_bench.cu as a shared library with a C interface, so that an assembly written in
// another GPU framework (JAX / MJX, Warp, ...) can hand its device arrays to the same solver without a copy
// through the host: Ruiz equilibration (the cusadi2 plans mpc_ruiz_r1..r5), gather, cuDSS factorization and
// 25 x (cuDSS solve + mpc_admm_iter), captured once into a CUDA graph and replayed per solve.
//
// Inputs are device pointers in the layout `mpc_assemble` writes (see build_mpc_stage.py):
//   blk (B, N, NBLK)  [P_kk triu, P_{k-1,k}, A_kk, A_{k,k-1}] nonzeros      q (B, N, NS)      z_lb, z_ub (B, N, NG)
// They are copied into the solver's own buffers (the graph needs fixed addresses), so the caller's arrays may
// live anywhere on the device and must only be complete (synchronised) when mpcqp_solve is called.
//
// The cusadi2 assembly (the plan mpc_assemble) is exposed the same way (mpcasm_*), so that both assemblies can be
// timed through one harness with everything downstream identical.
//
// Build: nvcc -shared -Xcompiler -fPIC ... mpc_qp_lib.cu <build>/kernel_double.cu ... -o libmpc_qp_double.so
// Python: mpc_bridge.py.
#include "cusadi2.h"
#include <cudss.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
using namespace cusadi2;
using S = Scalar;

#define CK(x) do { cudaError_t e_ = (cudaError_t)(x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return FAIL; } } while (0)
#define DS(x) do { cudssStatus_t s_ = (x); if (s_ != CUDSS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuDSS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); return FAIL; } } while (0)

__global__ void k_gather(const S* __restrict__ src, const int* __restrict__ map, S* __restrict__ dst, int B, int nnz, long long src_stride) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < (long long)B * nnz) { long long b = i / nnz; int e = (int)(i - b * nnz); dst[i] = src[b * src_stride + map[e]]; }
}
__global__ void k_gamma(const S* __restrict__ m, S* __restrict__ gamma, int B, int N) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    S mx = m[(long long)b * N];
    for (int k = 1; k < N; ++k) mx = fmax(mx, m[(long long)b * N + k]);
    gamma[b] = S(1) / mx;
}
struct PoolCtx { cudaMemPool_t pool; };
static int pool_alloc(void* ctx, void** ptr, size_t size, cudaStream_t stream) {
    return cudaMallocFromPoolAsync(ptr, size, ((PoolCtx*)ctx)->pool, stream) == cudaSuccess ? 0 : 1;
}
static int pool_free(void*, void* ptr, size_t, cudaStream_t stream) { return cudaFreeAsync(ptr, stream) == cudaSuccess ? 0 : 1; }

struct Solver {
    int B = 0, n = 0, nnz = 0, N = 0, NB = 0, NKKT = 0, NBLK = 0, NS = 0, NG = 0;
    Program fn[6];                         // ruiz_r1..r5, admm_iter
    cudaStream_t st = nullptr; cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    cudssHandle_t hd = nullptr; cudssConfig_t cfg = nullptr; cudssData_t data = nullptr; cudssMatrix_t A = nullptr, Xm = nullptr, Bm = nullptr;
    PoolCtx pctx{}; cudssDeviceMemHandler_t handler{};
    S *blk = nullptr, *q = nullptr, *zlb = nullptr, *zub = nullptr, *dx = nullptr;
    std::vector<void*> owned;
    std::function<int()> launches;         // every launch of one solve, on st (eager, or under capture)
    template <class T> T* alloc(size_t count) {
        T* d = nullptr; if (cudaMalloc(&d, count * sizeof(T)) != cudaSuccess) return nullptr;
        cudaMemset(d, 0, count * sizeof(T)); owned.push_back(d); return d;
    }
    template <class T> T** table(const std::vector<T*>& h) {
        T** d = alloc<T*>(h.size()); cudaMemcpy(d, h.data(), h.size() * sizeof(T*), cudaMemcpyHostToDevice); return d;
    }
};

#define FAIL 1
static int setup(Solver& s, const std::string& dir, int B) {
    std::ifstream cf(dir + "/csr.bin", std::ios::binary); if (!cf) { fprintf(stderr, "no %s/csr.bin\n", dir.c_str()); return FAIL; }
    int hdr[5]; cf.read((char*)hdr, sizeof hdr);
    s.B = B; s.n = hdr[0]; s.nnz = hdr[1]; s.N = hdr[2]; s.NB = hdr[3]; s.NKKT = hdr[4];
    const int n = s.n, nnz = s.nnz, N = s.N, NB = s.NB, NKKT = s.NKKT;
    std::vector<int> Ap(n + 1), Ai(nnz), gmap(nnz);
    cf.read((char*)Ap.data(), Ap.size() * 4); cf.read((char*)Ai.data(), nnz * 4); cf.read((char*)gmap.data(), nnz * 4);
    std::ifstream sf(dir + "/inputs/scalars.f64", std::ios::binary); double sc[3]; sf.read((char*)sc, sizeof sc);
    if (!sf) { fprintf(stderr, "no inputs/scalars.f64\n"); return FAIL; }
    const char* names[6] = {"mpc_ruiz_r1", "mpc_ruiz_r2", "mpc_ruiz_r3", "mpc_ruiz_r4", "mpc_ruiz_r5", "mpc_admm_iter"};
    Plan plan[6];
    for (int i = 0; i < 6; ++i) { plan[i] = load_plan(dir + "/plans/" + names[i] + ".plan"); s.fn[i].upload(plan[i]); }
    s.NBLK = plan[0].in[0].nnz(); s.NS = plan[1].in[5].nnz(); s.NG = NB - s.NS;
    const int NBLK = s.NBLK, NS = s.NS, NG = s.NG, N_ITER = 25;
    if (plan[4].out[0].sp.nnz() != NKKT || plan[0].n_stages != N) { fprintf(stderr, "plan/csr mismatch\n"); return FAIL; }
    CK(cudaStreamCreateWithFlags(&s.st, cudaStreamNonBlocking));
    cudaStream_t st = s.st;
    auto buf = [&](size_t count) { return s.alloc<S>(count); };
    auto fill = [&](double v) { std::vector<S> h(B, (S)v); S* d = buf(B); cudaMemcpy(d, h.data(), B * sizeof(S), cudaMemcpyHostToDevice); return d; };
    s.blk = buf((size_t)B * N * NBLK); s.q = buf((size_t)B * N * NS); s.zlb = buf((size_t)B * N * NG); s.zub = buf((size_t)B * N * NG);
    S *blk = s.blk, *q = s.q, *zlb = s.zlb, *zub = s.zub;
    S *sigma = fill(sc[0]), *rho = fill(sc[1]), *alpha = fill(sc[2]);
    S *blk_pad = buf((size_t)B * (N + 1) * NBLK);
    S *d1 = buf((size_t)B * N * NB), *d1_pad = buf((size_t)B * (N + 2) * NB), *m1 = buf((size_t)B * N), *g1 = buf(B);
    S *d2 = buf((size_t)B * N * NB), *d2_pad = buf((size_t)B * (N + 2) * NB), *m2 = buf((size_t)B * N), *g2 = buf(B);
    S *kkt_blk = buf((size_t)B * N * NKKT), *vals = buf((size_t)B * nnz);
    S *q_s = buf((size_t)B * N * NS), *D = buf((size_t)B * N * NS), *zlb_s = buf((size_t)B * N * NG), *zub_s = buf((size_t)B * N * NG);
    S *b = buf((size_t)B * n), *sol = buf((size_t)B * n); s.dx = buf((size_t)B * N * NS); S* dx = s.dx;
    S *Xz = buf((size_t)B * N * NS), *Yz = buf((size_t)B * N * NG), *Zz = buf((size_t)B * N * NG);   // ADMM starts at x = y = z = 0
    S *X[2] = {buf((size_t)B * N * NS), buf((size_t)B * N * NS)}, *Y[2] = {buf((size_t)B * N * NG), buf((size_t)B * N * NG)},
      *Z[2] = {buf((size_t)B * N * NG), buf((size_t)B * N * NG)};
    if (!Z[1]) { fprintf(stderr, "out of device memory\n"); return FAIL; }
    int *d_Ap = s.alloc<int>(n + 1), *d_Ai = s.alloc<int>(nnz), *d_map = s.alloc<int>(nnz);
    CK(cudaMemcpy(d_Ap, Ap.data(), (n + 1) * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_Ai, Ai.data(), nnz * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_map, gmap.data(), nnz * 4, cudaMemcpyHostToDevice));
    const S** in_r1 = (const S**)s.table<S>({blk_pad, blk_pad});                                   S** out_r1 = s.table<S>({d1});
    const S** in_r2 = (const S**)s.table<S>({blk_pad, blk_pad, d1_pad, d1_pad, d1_pad, q});        S** out_r2 = s.table<S>({m1});
    const S** in_r3 = (const S**)s.table<S>({blk_pad, blk_pad, d1_pad, d1_pad, d1_pad, g1});       S** out_r3 = s.table<S>({d2});
    const S** in_r4 = (const S**)s.table<S>({blk_pad, blk_pad, d1_pad, d1_pad, d1_pad, d2_pad, d2_pad, d2_pad, g1, q}); S** out_r4 = s.table<S>({m2});
    const S** in_r5 = (const S**)s.table<S>({blk, d1_pad, d1_pad, d2_pad, d2_pad, g1, g2, q, zlb, zub, sigma, rho});
    S** out_r5 = s.table<S>({kkt_blk, q_s, D, zlb_s, zub_s, b});
    const S** in_it[3]; S** out_it[2];   // it 0 reads the zero state; then ping-pong
    in_it[0] = (const S**)s.table<S>({sol, Xz, Yz, Zz, q_s, zlb_s, zub_s, D, sigma, rho, alpha});
    for (int p = 0; p < 2; ++p) in_it[1 + p] = (const S**)s.table<S>({sol, X[p], Y[p], Z[p], q_s, zlb_s, zub_s, D, sigma, rho, alpha});
    for (int p = 0; p < 2; ++p) out_it[p] = s.table<S>({X[p], Y[p], Z[p], b, dx});

    DS(cudssCreate(&s.hd)); DS(cudssSetStream(s.hd, st));
    cudaMemPoolProps props{}; props.allocType = cudaMemAllocationTypePinned; props.location.type = cudaMemLocationTypeDevice; props.location.id = 0;
    CK(cudaMemPoolCreate(&s.pctx.pool, &props));
    unsigned long long thr = ~0ull; CK(cudaMemPoolSetAttribute(s.pctx.pool, cudaMemPoolAttrReleaseThreshold, &thr));
    s.handler.ctx = &s.pctx; s.handler.device_alloc = pool_alloc; s.handler.device_free = pool_free;
    strncpy(s.handler.name, "cusadi2 qp pool", CUDSS_ALLOCATOR_NAME_LEN); DS(cudssSetDeviceMemHandler(s.hd, &s.handler));
    DS(cudssConfigCreate(&s.cfg)); DS(cudssDataCreate(s.hd, &s.data));
    int ub = B; DS(cudssConfigSet(s.cfg, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof ub));
    const cudssDataType_t cdt = sizeof(S) == 4 ? CUDSS_R_32F : CUDSS_R_64F;
    DS(cudssMatrixCreateCsr(&s.A, n, n, nnz, d_Ap, nullptr, d_Ai, vals, CUDSS_R_32I, CUDSS_R_32I, cdt, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
    DS(cudssMatrixCreateDn(&s.Bm, n, 1, n, b, cdt, CUDSS_LAYOUT_COL_MAJOR));
    DS(cudssMatrixCreateDn(&s.Xm, n, 1, n, sol, cdt, CUDSS_LAYOUT_COL_MAJOR));

    const size_t szS = sizeof(S);
    Solver* sp = &s;
    auto pad = [=](S* dst, const S* src, int width, int before, int count_dst) {
        return cudaMemcpy2DAsync(dst + (size_t)before * width, (size_t)count_dst * width * szS, src, (size_t)N * width * szS,
                                 (size_t)N * width * szS, B, cudaMemcpyDeviceToDevice, st);
    };
    s.launches = [=]() -> int {
        Solver& s = *sp;
        CK(pad(blk_pad, blk, NBLK, 0, N + 1));
        CK(s.fn[0].evaluate(in_r1, out_r1, B, st));
        CK(pad(d1_pad, d1, NB, 1, N + 2));
        CK(s.fn[1].evaluate(in_r2, out_r2, B, st));
        k_gamma<<<(B + 127) / 128, 128, 0, st>>>(m1, g1, B, N);
        CK(s.fn[2].evaluate(in_r3, out_r3, B, st));
        CK(pad(d2_pad, d2, NB, 1, N + 2));
        CK(s.fn[3].evaluate(in_r4, out_r4, B, st));
        k_gamma<<<(B + 127) / 128, 128, 0, st>>>(m2, g2, B, N);
        CK(s.fn[4].evaluate(in_r5, out_r5, B, st));
        k_gather<<<(int)(((long long)B * nnz + 255) / 256), 256, 0, st>>>(kkt_blk, d_map, vals, B, nnz, (long long)N * NKKT);
        DS(cudssExecute(s.hd, CUDSS_PHASE_FACTORIZATION, s.cfg, s.data, s.A, s.Xm, s.Bm));
        for (int it = 0; it < N_ITER; ++it) {
            DS(cudssExecute(s.hd, CUDSS_PHASE_SOLVE, s.cfg, s.data, s.A, s.Xm, s.Bm));
            CK(s.fn[5].evaluate(in_it[it == 0 ? 0 : 1 + ((it - 1) & 1)], out_it[it & 1], B, st));
        }
        return 0;
    };
    DS(cudssExecute(s.hd, CUDSS_PHASE_ANALYSIS, s.cfg, s.data, s.A, s.Xm, s.Bm)); CK(cudaStreamSynchronize(st));
    return 0;
}
// The first solve runs eagerly twice on the caller's data (fills the cuDSS pool, as the benchmark's warm-up
// does), then the launches are captured into the graph every later solve replays.
static int capture(Solver& s) {
    for (int k = 0; k < 2; ++k) if (s.launches()) return FAIL;
    CK(cudaStreamSynchronize(s.st));
    CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal)); if (s.launches()) return FAIL; CK(cudaStreamEndCapture(s.st, &s.graph));
    CK(cudaGraphInstantiate(&s.exec, s.graph, 0));
    return 0;
}

// ---- the cusadi2 assembly: x_pad (B, N+2, NS), p_stage (B, N, NPS), p_shared (B, NPG) -> blk, q, z_lb, z_ub
struct Assembler {
    int B = 0, N = 0, NS = 0, NPS = 0, NPG = 0;
    Program fn; cudaStream_t st = nullptr;
    S *x_pad = nullptr, *p_stage = nullptr, *p_shared = nullptr, *out[4] = {};
    const S** d_in = nullptr; S** d_out = nullptr;
    std::vector<void*> owned;
};
static std::vector<double> read_f64(const std::string& path) {
    std::ifstream f(path, std::ios::binary); if (!f) return {};
    f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg() / 8; f.seekg(0);
    std::vector<double> v(n); f.read((char*)v.data(), n * 8); return v;
}
static int setup(Assembler& a, const std::string& dir, int B) {
    Plan plan = load_plan(dir + "/plans/mpc_assemble.plan"); a.fn.upload(plan);
    a.B = B; a.N = plan.n_stages; a.NS = plan.in[1].nnz(); a.NPS = plan.in[7].nnz(); a.NPG = plan.in[8].nnz();
    const int N = a.N;
    CK(cudaStreamCreateWithFlags(&a.st, cudaStreamNonBlocking));
    auto buf = [&](size_t count) { S* d = nullptr; if (cudaMalloc(&d, count * sizeof(S)) != cudaSuccess) return (S*)nullptr; cudaMemset(d, 0, count * sizeof(S)); a.owned.push_back(d); return d; };
    auto tile = [&](const std::vector<double>& h) {
        std::vector<S> t; for (int b = 0; b < B; ++b) t.insert(t.end(), h.begin(), h.end());
        S* d = buf(t.size()); if (d) cudaMemcpy(d, t.data(), t.size() * sizeof(S), cudaMemcpyHostToDevice); return d;
    };
    std::vector<double> dt = read_f64(dir + "/inputs/dt_pad.f64"), first = read_f64(dir + "/inputs/is_first.f64"), next = read_f64(dir + "/inputs/has_next.f64");
    if ((int)dt.size() != N + 1 || (int)first.size() != N || (int)next.size() != N) { fprintf(stderr, "bad inputs/ in %s\n", dir.c_str()); return FAIL; }
    a.x_pad = buf((size_t)B * (N + 2) * a.NS); a.p_stage = buf((size_t)B * N * a.NPS); a.p_shared = buf((size_t)B * a.NPG);
    S *dt_pad = tile(dt), *is_first = tile(first), *has_next = tile(next);
    for (int i = 0; i < 4; ++i) a.out[i] = buf((size_t)B * N * plan.out[i].sp.nnz());
    if (!a.out[3]) { fprintf(stderr, "out of device memory\n"); return FAIL; }
    const S* in[9] = {a.x_pad, a.x_pad, a.x_pad, dt_pad, dt_pad, is_first, has_next, a.p_stage, a.p_shared};
    void* t = nullptr; CK(cudaMalloc(&t, sizeof in)); a.owned.push_back(t); CK(cudaMemcpy(t, in, sizeof in, cudaMemcpyHostToDevice)); a.d_in = (const S**)t;
    CK(cudaMalloc(&t, sizeof a.out)); a.owned.push_back(t); CK(cudaMemcpy(t, a.out, sizeof a.out, cudaMemcpyHostToDevice)); a.d_out = (S**)t;
    return 0;
}

extern "C" {
void* mpcasm_create(const char* build_dir, int B) {
    Assembler* a = new Assembler();
    if (setup(*a, build_dir, B)) { delete a; return nullptr; }
    return a;
}
// Evaluate from device arrays; returns when blk, q, z_lb, z_ub (mpcasm_out 0..3) are ready on the device.
int mpcasm_evaluate(void* h, const S* x_pad, const S* p_stage, const S* p_shared) {
    Assembler& a = *(Assembler*)h; const size_t sz = sizeof(S);
    CK(cudaMemcpyAsync(a.x_pad, x_pad, (size_t)a.B * (a.N + 2) * a.NS * sz, cudaMemcpyDeviceToDevice, a.st));
    CK(cudaMemcpyAsync(a.p_stage, p_stage, (size_t)a.B * a.N * a.NPS * sz, cudaMemcpyDeviceToDevice, a.st));
    CK(cudaMemcpyAsync(a.p_shared, p_shared, (size_t)a.B * a.NPG * sz, cudaMemcpyDeviceToDevice, a.st));
    CK(a.fn.evaluate(a.d_in, a.d_out, a.B, a.st));
    CK(cudaStreamSynchronize(a.st));
    return 0;
}
const S* mpcasm_out(void* h, int i) { return ((Assembler*)h)->out[i]; }
void mpcasm_destroy(void* h) {
    Assembler* a = (Assembler*)h; if (!a) return;
    for (void* p : a->owned) cudaFree(p);
    if (a->st) cudaStreamDestroy(a->st);
    delete a;
}
// Returns a handle, or null on failure. build_dir holds plans/, csr.bin, inputs/scalars.f64.
void* mpcqp_create(const char* build_dir, int B) {
    Solver* s = new Solver();
    if (setup(*s, build_dir, B)) { delete s; return nullptr; }
    return s;
}
// out[0..5] = N, NS, NG, NBLK, B, sizeof(scalar)
void mpcqp_dims(void* h, int* out) { Solver* s = (Solver*)h; out[0] = s->N; out[1] = s->NS; out[2] = s->NG; out[3] = s->NBLK; out[4] = s->B; out[5] = (int)sizeof(S); }
// Solve from device arrays; returns when the solution is ready on the device (0 on success).
int mpcqp_solve(void* h, const S* blk, const S* q, const S* zlb, const S* zub) {
    Solver& s = *(Solver*)h; const size_t per = (size_t)s.B * s.N * sizeof(S);
    CK(cudaMemcpyAsync(s.blk, blk, per * s.NBLK, cudaMemcpyDeviceToDevice, s.st));
    CK(cudaMemcpyAsync(s.q, q, per * s.NS, cudaMemcpyDeviceToDevice, s.st));
    CK(cudaMemcpyAsync(s.zlb, zlb, per * s.NG, cudaMemcpyDeviceToDevice, s.st));
    CK(cudaMemcpyAsync(s.zub, zub, per * s.NG, cudaMemcpyDeviceToDevice, s.st));
    if (!s.exec && capture(s)) return FAIL;
    CK(cudaGraphLaunch(s.exec, s.st));
    CK(cudaStreamSynchronize(s.st));
    return 0;
}
// The step dx (B, N, NS): device pointer, or copied to the host.
const S* mpcqp_dx(void* h) { return ((Solver*)h)->dx; }
int mpcqp_get_dx(void* h, S* host) {
    Solver& s = *(Solver*)h;
    CK(cudaMemcpy(host, s.dx, (size_t)s.B * s.N * s.NS * sizeof(S), cudaMemcpyDeviceToHost));
    return 0;
}
#undef FAIL
void mpcqp_destroy(void* h) {
    Solver* s = (Solver*)h; if (!s) return;
    if (s->exec) cudaGraphExecDestroy(s->exec); if (s->graph) cudaGraphDestroy(s->graph);
    if (s->A) cudssMatrixDestroy(s->A); if (s->Xm) cudssMatrixDestroy(s->Xm); if (s->Bm) cudssMatrixDestroy(s->Bm);
    if (s->data) cudssDataDestroy(s->hd, s->data); if (s->cfg) cudssConfigDestroy(s->cfg); if (s->hd) cudssDestroy(s->hd);
    if (s->pctx.pool) cudaMemPoolDestroy(s->pctx.pool);
    for (void* p : s->owned) cudaFree(p);
    if (s->st) cudaStreamDestroy(s->st);
    delete s;
}
}
