// Stage-parallel MIT-humanoid MPC QP solve on the GPU: cusadi2 level kernels for
// the per-stage work, cuDSS (uniform batch, symmetric indefinite LDL^T) for the
// block-tridiagonal KKT solve. One instance = one QP (10 stages, 1340 KKT rows);
// B instances are solved in a batch. Per solve:
//
//   assemble        (B x N blocks)  KKT block row, gradient, bounds per stage
//   pad(blk); ruiz_r1; pad(delta1); ruiz_r2; reduce -> gamma1
//   ruiz_r3; pad(delta2); ruiz_r4; reduce -> gamma2; ruiz_r5 (scaled KKT, q, D, bounds, b0)
//   gather -> CSR values (B x nnz)
//   cuDSS factorization
//   25 x { cuDSS solve; admm_iter (ADMM step + next rhs + dx) }
//
// Eager mode times each phase with events; graph mode captures the whole solve
// into one cudaGraph (cuDSS device memory served from a pre-reserved pool so its
// phases capture) and replays it. Every instance is checked against the CPU
// reference dx (inputs/dx_ref.f64) and the CPU run of the staged pipeline.
//
//   mpc_cudss_bench <build_dir> <n_warm> <n_timed> <B...>
// build_dir holds plans/, csr.bin, inputs/ from build_mpc_stage.py.
// env CUDSS_ALLOC=pool|arena|default selects the cuDSS memory handler (default: pool).
#include "cusadi2.h"
#include <cudss.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
using namespace cusadi2;
using S = Scalar;

#define CK(x) do { cudaError_t e_ = (cudaError_t)(x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define DS(x) do { cudssStatus_t s_ = (x); if (s_ != CUDSS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuDSS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)

static std::vector<double> read_f64(const std::string& path) {
    std::ifstream f(path, std::ios::binary); if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg() / 8; f.seekg(0);
    std::vector<double> v(n); f.read((char*)v.data(), n * 8); return v;
}
static S* dalloc(size_t n) { S* d; CK(cudaMalloc(&d, n * sizeof(S))); CK(cudaMemset(d, 0, n * sizeof(S))); return d; }
template <class T> static T** dptrs(const std::vector<T*>& h) {
    T** d; CK(cudaMalloc(&d, h.size() * sizeof(T*))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T*), cudaMemcpyHostToDevice)); return d;
}
// Tile one instance's host data (float64) B times into a device buffer.
static S* tile(const std::vector<double>& h, int B) {
    std::vector<S> t((size_t)B * h.size());
    for (int b = 0; b < B; ++b) for (size_t i = 0; i < h.size(); ++i) t[(size_t)b * h.size() + i] = (S)h[i];
    S* d = dalloc(t.size()); CK(cudaMemcpy(d, t.data(), t.size() * sizeof(S), cudaMemcpyHostToDevice)); return d;
}

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

// --- cuDSS device memory handlers -------------------------------------------
struct PoolCtx { cudaMemPool_t pool; };
static int pool_alloc(void* ctx, void** ptr, size_t size, cudaStream_t stream) {
    return cudaMallocFromPoolAsync(ptr, size, ((PoolCtx*)ctx)->pool, stream) == cudaSuccess ? 0 : 1;
}
static int pool_free(void*, void* ptr, size_t, cudaStream_t stream) { return cudaFreeAsync(ptr, stream) == cudaSuccess ? 0 : 1; }
// Arena: one cudaMalloc up front, LIFO reuse; never calls the driver afterwards.
struct ArenaCtx { char* base = nullptr; size_t cap = 0, top = 0, peak = 0; std::vector<std::pair<char*, size_t>> live; };
static int arena_alloc(void* c, void** ptr, size_t size, cudaStream_t) {
    ArenaCtx* a = (ArenaCtx*)c; size_t sz = (size + 255) & ~size_t(255);
    if (a->top + sz > a->cap) { fprintf(stderr, "arena exhausted (%zu + %zu > %zu)\n", a->top, sz, a->cap); return 1; }
    *ptr = a->base + a->top; a->live.push_back({(char*)*ptr, sz}); a->top += sz; a->peak = std::max(a->peak, a->top); return 0;
}
static int arena_free(void* c, void* ptr, size_t, cudaStream_t) {
    ArenaCtx* a = (ArenaCtx*)c;
    for (auto it = a->live.begin(); it != a->live.end(); ++it) if (it->first == ptr) { it->second = 0; break; }
    while (!a->live.empty() && a->live.back().second == 0) { a->live.pop_back(); }
    a->top = a->live.empty() ? 0 : (size_t)(a->live.back().first - a->base) + a->live.back().second;
    return 0;
}

struct Fn {
    Plan plan; Program prog; std::string how;
    void setup(const std::string& path, int batch, int max_smem) {
        plan = load_plan(path); prog.upload(plan);
        bool level = plan.shared_bytes() <= max_smem;
        if (!level) prog.use_serial((size_t)plan.sz_w * sizeof(S) * (size_t)batch * plan.n_stages + 1, 256);
        how = level ? "level" : "serial";
    }
};

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <build_dir> <n_warm> <n_timed> <B...>\n", argv[0]); return 2; }
    const std::string dir = argv[1], inp = dir + "/inputs/";
    const int n_warm = atoi(argv[2]), n_timed = atoi(argv[3]);
    const char* alloc_mode = getenv("CUDSS_ALLOC") ? getenv("CUDSS_ALLOC") : "pool";
    const char* dt = sizeof(S) == 4 ? "f32" : "f64";
    int max_smem = 0; CK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    // csr.bin: i32 n, nnz, N, NB, NKKT; Ap[n+1]; Ai[nnz]; map[nnz]
    std::ifstream cf(dir + "/csr.bin", std::ios::binary); if (!cf) { fprintf(stderr, "no csr.bin\n"); return 2; }
    int hdr[5]; cf.read((char*)hdr, sizeof hdr);
    const int n = hdr[0], nnz = hdr[1], N = hdr[2], NB = hdr[3], NKKT = hdr[4];
    std::vector<int> Ap(n + 1), Ai(nnz), gmap(nnz);
    cf.read((char*)Ap.data(), Ap.size() * 4); cf.read((char*)Ai.data(), nnz * 4); cf.read((char*)gmap.data(), nnz * 4);
    std::vector<double> h_x = read_f64(inp + "x_pad.f64"), h_dt = read_f64(inp + "dt_pad.f64"), h_first = read_f64(inp + "is_first.f64"),
        h_next = read_f64(inp + "has_next.f64"), h_ps = read_f64(inp + "p_stage.f64"), h_pg = read_f64(inp + "p_shared.f64"),
        h_sc = read_f64(inp + "scalars.f64"), dx_ref = read_f64(inp + "dx_ref.f64"), dx_cpu = read_f64(inp + "dx_cpu_staged.f64");
    const int NS = (int)dx_ref.size() / N, NG = NB - NS, N_ITER = 25;
    const cudssDataType_t cdt = sizeof(S) == 4 ? CUDSS_R_32F : CUDSS_R_64F;

    printf("dtype,batch,mode,how,alloc,asm_ruiz_us,factor_us,iters_us,total_us,wall_us,analysis_ms,max_abs_err_ref,max_abs_err_cpu\n");
    for (int ai = 4; ai < argc; ++ai) {
        const int B = atoi(argv[ai]);
        const char* names[7] = {"mpc_assemble", "mpc_ruiz_r1", "mpc_ruiz_r2", "mpc_ruiz_r3", "mpc_ruiz_r4", "mpc_ruiz_r5", "mpc_admm_iter"};
        Fn fn[7];
        for (int i = 0; i < 7; ++i) fn[i].setup(dir + "/plans/" + names[i] + ".plan", B, max_smem);
        const int NBLK = fn[0].plan.out[0].sp.nnz();
        if (fn[5].plan.out[0].sp.nnz() != NKKT || fn[0].plan.n_stages != N) { fprintf(stderr, "plan/csr mismatch\n"); return 3; }
        std::string how = fn[0].how; for (int i = 1; i < 7; ++i) if (fn[i].how != how) how = "mixed";

        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        // ---- buffers: (instance, stage_count, nnz)
        S *x_pad = tile(h_x, B), *dt_pad = tile(h_dt, B), *is_first = tile(h_first, B), *has_next = tile(h_next, B);
        S *p_stage = tile(h_ps, B), *p_shared = tile(h_pg, B);
        S *sigma = tile({h_sc[0]}, B), *rho = tile({h_sc[1]}, B), *alpha = tile({h_sc[2]}, B);
        S *blk = dalloc((size_t)B * N * NBLK), *blk_pad = dalloc((size_t)B * (N + 1) * NBLK);
        S *q = dalloc((size_t)B * N * NS), *zlb = dalloc((size_t)B * N * NG), *zub = dalloc((size_t)B * N * NG);
        S *d1 = dalloc((size_t)B * N * NB), *d1_pad = dalloc((size_t)B * (N + 2) * NB), *m1 = dalloc((size_t)B * N), *g1 = dalloc(B);
        S *d2 = dalloc((size_t)B * N * NB), *d2_pad = dalloc((size_t)B * (N + 2) * NB), *m2 = dalloc((size_t)B * N), *g2 = dalloc(B);
        S *kkt_blk = dalloc((size_t)B * N * NKKT), *vals = dalloc((size_t)B * nnz);
        S *q_s = dalloc((size_t)B * N * NS), *D = dalloc((size_t)B * N * NS), *zlb_s = dalloc((size_t)B * N * NG), *zub_s = dalloc((size_t)B * N * NG);
        S *b = dalloc((size_t)B * n), *sol = dalloc((size_t)B * n), *dx = dalloc((size_t)B * N * NS);
        S *Xz = dalloc((size_t)B * N * NS), *Yz = dalloc((size_t)B * N * NG), *Zz = dalloc((size_t)B * N * NG);   // x_0 = y_0 = z_0 = 0
        S *X[2] = {dalloc((size_t)B * N * NS), dalloc((size_t)B * N * NS)}, *Y[2] = {dalloc((size_t)B * N * NG), dalloc((size_t)B * N * NG)},
          *Z[2] = {dalloc((size_t)B * N * NG), dalloc((size_t)B * N * NG)};
        int *d_Ap, *d_Ai, *d_map;
        CK(cudaMalloc(&d_Ap, (n + 1) * 4)); CK(cudaMemcpy(d_Ap, Ap.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_Ai, nnz * 4)); CK(cudaMemcpy(d_Ai, Ai.data(), nnz * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_map, nnz * 4)); CK(cudaMemcpy(d_map, gmap.data(), nnz * 4, cudaMemcpyHostToDevice));
        // ---- argument pointer tables
        const S** in_asm = (const S**)dptrs<S>({x_pad, x_pad, x_pad, dt_pad, dt_pad, is_first, has_next, p_stage, p_shared});
        S** out_asm = dptrs<S>({blk, q, zlb, zub});
        const S** in_r1 = (const S**)dptrs<S>({blk_pad, blk_pad});                                   S** out_r1 = dptrs<S>({d1});
        const S** in_r2 = (const S**)dptrs<S>({blk_pad, blk_pad, d1_pad, d1_pad, d1_pad, q});        S** out_r2 = dptrs<S>({m1});
        const S** in_r3 = (const S**)dptrs<S>({blk_pad, blk_pad, d1_pad, d1_pad, d1_pad, g1});       S** out_r3 = dptrs<S>({d2});
        const S** in_r4 = (const S**)dptrs<S>({blk_pad, blk_pad, d1_pad, d1_pad, d1_pad, d2_pad, d2_pad, d2_pad, g1, q}); S** out_r4 = dptrs<S>({m2});
        const S** in_r5 = (const S**)dptrs<S>({blk, d1_pad, d1_pad, d2_pad, d2_pad, g1, g2, q, zlb, zub, sigma, rho});
        S** out_r5 = dptrs<S>({kkt_blk, q_s, D, zlb_s, zub_s, b});
        const S** in_it[3]; S** out_it[3];   // it 0 reads the zero state; then ping-pong
        in_it[0] = (const S**)dptrs<S>({sol, Xz, Yz, Zz, q_s, zlb_s, zub_s, D, sigma, rho, alpha});
        for (int p = 0; p < 2; ++p) in_it[1 + p] = (const S**)dptrs<S>({sol, X[p], Y[p], Z[p], q_s, zlb_s, zub_s, D, sigma, rho, alpha});
        for (int p = 0; p < 2; ++p) out_it[p] = dptrs<S>({X[p], Y[p], Z[p], b, dx});
        out_it[2] = out_it[0];

        // ---- cuDSS
        cudssHandle_t hd; DS(cudssCreate(&hd)); DS(cudssSetStream(hd, st));
        PoolCtx pctx{}; ArenaCtx actx; cudssDeviceMemHandler_t handler{};
        if (!strcmp(alloc_mode, "pool")) {
            cudaMemPoolProps props{}; props.allocType = cudaMemAllocationTypePinned; props.location.type = cudaMemLocationTypeDevice; props.location.id = 0;
            CK(cudaMemPoolCreate(&pctx.pool, &props));
            unsigned long long thr = ~0ull; CK(cudaMemPoolSetAttribute(pctx.pool, cudaMemPoolAttrReleaseThreshold, &thr));
            handler.ctx = &pctx; handler.device_alloc = pool_alloc; handler.device_free = pool_free;
            strncpy(handler.name, "cusadi2 bench pool", CUDSS_ALLOCATOR_NAME_LEN); DS(cudssSetDeviceMemHandler(hd, &handler));
        } else if (!strcmp(alloc_mode, "arena")) {
            size_t freeb = 0, totb = 0; CK(cudaMemGetInfo(&freeb, &totb)); actx.cap = freeb / 2; CK(cudaMalloc(&actx.base, actx.cap));
            handler.ctx = &actx; handler.device_alloc = arena_alloc; handler.device_free = arena_free;
            strncpy(handler.name, "cusadi2 bench arena", CUDSS_ALLOCATOR_NAME_LEN); DS(cudssSetDeviceMemHandler(hd, &handler));
        }
        cudssConfig_t cfg; DS(cudssConfigCreate(&cfg)); cudssData_t data; DS(cudssDataCreate(hd, &data));
        int ub = B; DS(cudssConfigSet(cfg, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof ub));
        cudssMatrix_t A, Xm, Bm;
        DS(cudssMatrixCreateCsr(&A, n, n, nnz, d_Ap, nullptr, d_Ai, vals, CUDSS_R_32I, CUDSS_R_32I, cdt, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
        DS(cudssMatrixCreateDn(&Bm, n, 1, n, b, cdt, CUDSS_LAYOUT_COL_MAJOR));
        DS(cudssMatrixCreateDn(&Xm, n, 1, n, sol, cdt, CUDSS_LAYOUT_COL_MAJOR));

        cudaEvent_t ev[4]; for (auto& e : ev) CK(cudaEventCreate(&e));
        const size_t szS = sizeof(S);
        auto pad = [&](S* dst, const S* src, int width, int before, int count_dst) {
            CK(cudaMemcpy2DAsync(dst + (size_t)before * width, (size_t)count_dst * width * szS, src, (size_t)N * width * szS,
                                 (size_t)N * width * szS, B, cudaMemcpyDeviceToDevice, st));
        };
        auto solve = [&](bool timed) {
            if (timed) CK(cudaEventRecord(ev[0], st));
            CK(fn[0].prog.evaluate(in_asm, out_asm, B, st));
            pad(blk_pad, blk, NBLK, 0, N + 1);
            CK(fn[1].prog.evaluate(in_r1, out_r1, B, st));
            pad(d1_pad, d1, NB, 1, N + 2);
            CK(fn[2].prog.evaluate(in_r2, out_r2, B, st));
            k_gamma<<<(B + 127) / 128, 128, 0, st>>>(m1, g1, B, N);
            CK(fn[3].prog.evaluate(in_r3, out_r3, B, st));
            pad(d2_pad, d2, NB, 1, N + 2);
            CK(fn[4].prog.evaluate(in_r4, out_r4, B, st));
            k_gamma<<<(B + 127) / 128, 128, 0, st>>>(m2, g2, B, N);
            CK(fn[5].prog.evaluate(in_r5, out_r5, B, st));
            k_gather<<<(int)(((long long)B * nnz + 255) / 256), 256, 0, st>>>(kkt_blk, d_map, vals, B, nnz, (long long)N * NKKT);
            if (timed) CK(cudaEventRecord(ev[1], st));
            DS(cudssExecute(hd, CUDSS_PHASE_FACTORIZATION, cfg, data, A, Xm, Bm));
            if (timed) CK(cudaEventRecord(ev[2], st));
            for (int it = 0; it < N_ITER; ++it) {
                DS(cudssExecute(hd, CUDSS_PHASE_SOLVE, cfg, data, A, Xm, Bm));
                int src = it == 0 ? 0 : 1 + ((it - 1) & 1), dstp = it & 1;
                CK(fn[6].prog.evaluate(in_it[src], out_it[dstp], B, st));
            }
            if (timed) CK(cudaEventRecord(ev[3], st));
        };
        // analysis (reordering on the host) once; needs real values? cuDSS analysis is structural.
        auto t0 = std::chrono::steady_clock::now();
        DS(cudssExecute(hd, CUDSS_PHASE_ANALYSIS, cfg, data, A, Xm, Bm)); CK(cudaStreamSynchronize(st));
        double analysis_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        // warm-up (also fills the pool)
        for (int k = 0; k < n_warm; ++k) solve(false);
        CK(cudaStreamSynchronize(st));
        // correctness
        auto check = [&](double* err_ref, double* err_cpu) {
            std::vector<S> h((size_t)B * N * NS); CK(cudaMemcpy(h.data(), dx, h.size() * szS, cudaMemcpyDeviceToHost));
            *err_ref = 0; *err_cpu = 0;
            for (int bb = 0; bb < B; ++bb) for (int i = 0; i < N * NS; ++i) {
                double v = h[(size_t)bb * N * NS + i];
                *err_ref = std::max(*err_ref, std::fabs(v - dx_ref[i])); *err_cpu = std::max(*err_cpu, std::fabs(v - dx_cpu[i]));
                if (!std::isfinite(v)) *err_ref = *err_cpu = INFINITY;
            }
        };
        double e_ref, e_cpu; check(&e_ref, &e_cpu);
        // eager timing
        std::vector<double> t_ar, t_f, t_it, t_tot, t_wall;
        for (int k = 0; k < n_timed; ++k) {
            auto w0 = std::chrono::steady_clock::now();
            solve(true); CK(cudaEventSynchronize(ev[3]));
            t_wall.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count());
            float ms[3]; for (int i = 0; i < 3; ++i) CK(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]));
            t_ar.push_back(ms[0] * 1e3); t_f.push_back(ms[1] * 1e3); t_it.push_back(ms[2] * 1e3); t_tot.push_back((ms[0] + ms[1] + ms[2]) * 1e3);
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? NAN : v[v.size() / 2]; };
        printf("%s,%d,eager,%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.3e,%.3e\n", dt, B, how.c_str(), alloc_mode, med(t_ar), med(t_f), med(t_it), med(t_tot), med(t_wall),
               analysis_ms, e_ref, e_cpu); fflush(stdout);
        // graph
        {
            cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr; bool ok = true;
            cudaError_t ce = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
            if (ce == cudaSuccess) { solve(false); ce = cudaStreamEndCapture(st, &g); }
            if (ce != cudaSuccess) { fprintf(stderr, "graph capture failed: %s\n", cudaGetErrorString(ce)); cudaGetLastError(); ok = false; }
            if (ok && (ce = cudaGraphInstantiate(&ge, g, 0)) != cudaSuccess) { fprintf(stderr, "graph instantiate failed: %s\n", cudaGetErrorString(ce)); ok = false; }
            if (ok) {
                for (int k = 0; k < n_warm; ++k) CK(cudaGraphLaunch(ge, st));
                CK(cudaStreamSynchronize(st));
                double ge_ref, ge_cpu; check(&ge_ref, &ge_cpu);
                std::vector<double> tg, tw;
                for (int k = 0; k < n_timed; ++k) {
                    auto w0 = std::chrono::steady_clock::now();
                    CK(cudaEventRecord(ev[0], st)); CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(ev[3], st)); CK(cudaEventSynchronize(ev[3]));
                    tw.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count());
                    float ms; CK(cudaEventElapsedTime(&ms, ev[0], ev[3])); tg.push_back(ms * 1e3);
                }
                printf("%s,%d,graph,%s,%s,nan,nan,nan,%.1f,%.1f,%.2f,%.3e,%.3e\n", dt, B, how.c_str(), alloc_mode, med(tg), med(tw), analysis_ms, ge_ref, ge_cpu);
                CK(cudaGraphExecDestroy(ge));
            } else printf("%s,%d,graph,%s,%s,nan,nan,nan,nan,nan,%.2f,nan,nan\n", dt, B, how.c_str(), alloc_mode, analysis_ms);
            if (g) cudaGraphDestroy(g);
            fflush(stdout);
        }
        if (!strcmp(alloc_mode, "arena")) fprintf(stderr, "[B=%d] arena peak %.1f MB\n", B, actx.peak / 1048576.0);
        // teardown
        DS(cudssMatrixDestroy(A)); DS(cudssMatrixDestroy(Xm)); DS(cudssMatrixDestroy(Bm));
        DS(cudssDataDestroy(hd, data)); DS(cudssConfigDestroy(cfg)); DS(cudssDestroy(hd));
        if (pctx.pool) CK(cudaMemPoolDestroy(pctx.pool));
        if (actx.base) CK(cudaFree(actx.base));
        for (auto& e : ev) cudaEventDestroy(e);
        CK(cudaStreamDestroy(st));
        CK(cudaDeviceSynchronize());
        // buffers are freed with the context reset below is too coarse; free explicitly
        for (S* p : {x_pad, dt_pad, is_first, has_next, p_stage, p_shared, sigma, rho, alpha, blk, blk_pad, q, zlb, zub, d1, d1_pad, m1, g1,
                     d2, d2_pad, m2, g2, kkt_blk, vals, q_s, D, zlb_s, zub_s, b, sol, dx, Xz, Yz, Zz, X[0], X[1], Y[0], Y[1], Z[0], Z[1]}) cudaFree(p);
        cudaFree(d_Ap); cudaFree(d_Ai); cudaFree(d_map);
        for (auto p : {in_asm, in_r1, in_r2, in_r3, in_r4, in_r5, in_it[0], in_it[1], in_it[2]}) cudaFree((void*)p);
        for (auto p : {out_asm, out_r1, out_r2, out_r3, out_r4, out_r5, out_it[0], out_it[1]}) cudaFree(p);
    }
    return 0;
}
