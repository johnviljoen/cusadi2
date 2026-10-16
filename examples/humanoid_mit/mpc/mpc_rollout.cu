// Receding-horizon rollout of the MIT-humanoid MPC on the GPU: B instances in closed loop against the MPC's
// own model, one real-time iteration per control step, the whole step replayed as one CUDA graph.
//
// One control step (period dt_0, the first stage of the geometric grid):
//
//   solve           the staged QP of mpc_cudss_bench.cu (assemble, Ruiz, cuDSS factorization, 25 ADMM
//                   iterations), linearized at the current plan x instead of at rest
//   x+ = x + dx     one SQP step
//   advance         q_0 <- q_0 (+) dq_0+,  v_0 <- v_0+        (stage 0 of the plan is the state dt_0 ahead)
//   shift           x <- the plan x+ re-sampled dt_0 later and re-based onto the new q_0 (next linearization point)
//
// The stage parameters (references, contact schedule, swing phase) depend on time only; gen_batch12.py
// precomputes them for every step and they are uploaded once. Between graph launches the driver only copies
// that step's parameters in and the state and plan out, device to device.
//
//   mpc_rollout <build_dir> <B> [graph|eager]
// build_dir holds plans/, csr.bin, inputs/ (build_mpc_stage.py) and rollout/p_stage.f64 (gen_batch12.py).
// Writes rollout/state.f64 (n_steps+1, B, p_shared: q_0, v_0 first), rollout/plan.f64 (n_steps, B, N, 60:
// x+ of every step) and rollout/latency_us.f64 (n_steps).
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
static void write_f64(const std::string& path, const std::vector<double>& v) {
    std::ofstream f(path, std::ios::binary); f.write((const char*)v.data(), v.size() * 8);
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
}
static S* dalloc(size_t n) { S* d; CK(cudaMalloc(&d, n * sizeof(S))); CK(cudaMemset(d, 0, n * sizeof(S))); return d; }
template <class T> static T** dptrs(const std::vector<T*>& h) {
    T** d; CK(cudaMalloc(&d, h.size() * sizeof(T*))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T*), cudaMemcpyHostToDevice)); return d;
}
static S* upload(const std::vector<double>& h) {
    std::vector<S> t(h.begin(), h.end());
    S* d = dalloc(t.size()); CK(cudaMemcpy(d, t.data(), t.size() * sizeof(S), cudaMemcpyHostToDevice)); return d;
}
// Tile one instance's host data (float64) B times into a device buffer.
static S* tile(const std::vector<double>& h, int B) {
    std::vector<double> t; t.reserve((size_t)B * h.size());
    for (int b = 0; b < B; ++b) t.insert(t.end(), h.begin(), h.end());
    return upload(t);
}
static std::vector<double> download(const S* d, size_t n) {
    std::vector<S> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(S), cudaMemcpyDeviceToHost));
    return std::vector<double>(h.begin(), h.end());
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
// x+ = x + dx in the padded plan layout [pad, x_0 .. x_{N-1}, pad] (row = N * NS values per instance).
__global__ void k_sqp_step(const S* __restrict__ x_pad, const S* __restrict__ dx, S* __restrict__ xp_pad, int B, int row, int NS) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < (long long)B * row) { long long b = i / row; long long j = b * (row + 2 * NS) + NS + (i - b * row); xp_pad[j] = x_pad[j] + dx[i]; }
}

// cuDSS device memory from a pre-reserved pool, so its phases capture into a CUDA graph.
struct PoolCtx { cudaMemPool_t pool; };
static int pool_alloc(void* ctx, void** ptr, size_t size, cudaStream_t stream) {
    return cudaMallocFromPoolAsync(ptr, size, ((PoolCtx*)ctx)->pool, stream) == cudaSuccess ? 0 : 1;
}
static int pool_free(void*, void* ptr, size_t, cudaStream_t stream) { return cudaFreeAsync(ptr, stream) == cudaSuccess ? 0 : 1; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <build_dir> <B> [graph|eager]\n", argv[0]); return 2; }
    const std::string dir = argv[1], inp = dir + "/inputs/", roll = dir + "/rollout/";
    const int B = atoi(argv[2]); const bool use_graph = argc < 4 || strcmp(argv[3], "eager");
    if (sizeof(S) != 8) fprintf(stderr, "warning: fp32 is not accurate enough for this KKT (see README)\n");

    // csr.bin: i32 n, nnz, N, NB, NKKT; Ap[n+1]; Ai[nnz]; map[nnz]
    std::ifstream cf(dir + "/csr.bin", std::ios::binary); if (!cf) { fprintf(stderr, "no csr.bin\n"); return 2; }
    int hdr[5]; cf.read((char*)hdr, sizeof hdr);
    const int n = hdr[0], nnz = hdr[1], N = hdr[2], NB = hdr[3], NKKT = hdr[4];
    std::vector<int> Ap(n + 1), Ai(nnz), gmap(nnz);
    cf.read((char*)Ap.data(), Ap.size() * 4); cf.read((char*)Ai.data(), nnz * 4); cf.read((char*)gmap.data(), nnz * 4);
    std::vector<double> h_dt = read_f64(inp + "dt_pad.f64"), h_first = read_f64(inp + "is_first.f64"), h_next = read_f64(inp + "has_next.f64"),
        h_pg = read_f64(inp + "p_shared.f64"), h_sc = read_f64(inp + "scalars.f64"), h_w = read_f64(inp + "shift_w.f64"),
        h_ps = read_f64(roll + "p_stage.f64");
    const int N_ITER = 25;

    const char* names[9] = {"mpc_assemble", "mpc_ruiz_r1", "mpc_ruiz_r2", "mpc_ruiz_r3", "mpc_ruiz_r4", "mpc_ruiz_r5", "mpc_admm_iter",
                            "mpc_shift", "mpc_advance"};
    Plan plan[9]; Program fn[9];
    for (int i = 0; i < 9; ++i) { plan[i] = load_plan(dir + "/plans/" + names[i] + ".plan"); fn[i].upload(plan[i]); }
    const int NBLK = plan[0].out[0].sp.nnz(), NS = plan[0].in[1].nnz(), NG = NB - NS, NPS = plan[0].in[7].nnz(), NPG = plan[0].in[8].nnz();
    if (plan[5].out[0].sp.nnz() != NKKT || plan[0].n_stages != N || (int)h_pg.size() != NPG) { fprintf(stderr, "plan/csr mismatch\n"); return 3; }
    const size_t step_ps = (size_t)B * N * NPS, row = (size_t)N * NS;
    if (h_ps.size() % step_ps) { fprintf(stderr, "p_stage.f64 is not a multiple of B x N x %d\n", NPS); return 3; }
    const int n_steps = (int)(h_ps.size() / step_ps);

    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    // ---- buffers: (instance, stage_count, nnz)
    S *x_pad = dalloc((size_t)B * (N + 2) * NS), *xp_pad = dalloc((size_t)B * (N + 2) * NS), *x_shift = dalloc(B * row);   // plan: x_0 = 0
    S *dt_pad = tile(h_dt, B), *is_first = tile(h_first, B), *has_next = tile(h_next, B), *shift_w = tile(h_w, B), *dt_adv = tile({h_dt[0]}, B);
    S *p_all = upload(h_ps), *p_stage = dalloc(step_ps), *p_shared = tile(h_pg, B), *p_shared_next = dalloc((size_t)B * NPG);
    S *sigma = tile({h_sc[0]}, B), *rho = tile({h_sc[1]}, B), *alpha = tile({h_sc[2]}, B);
    S *blk = dalloc((size_t)B * N * NBLK), *blk_pad = dalloc((size_t)B * (N + 1) * NBLK);
    S *q = dalloc((size_t)B * N * NS), *zlb = dalloc((size_t)B * N * NG), *zub = dalloc((size_t)B * N * NG);
    S *d1 = dalloc((size_t)B * N * NB), *d1_pad = dalloc((size_t)B * (N + 2) * NB), *m1 = dalloc((size_t)B * N), *g1 = dalloc(B);
    S *d2 = dalloc((size_t)B * N * NB), *d2_pad = dalloc((size_t)B * (N + 2) * NB), *m2 = dalloc((size_t)B * N), *g2 = dalloc(B);
    S *kkt_blk = dalloc((size_t)B * N * NKKT), *vals = dalloc((size_t)B * nnz);
    S *q_s = dalloc((size_t)B * N * NS), *D = dalloc((size_t)B * N * NS), *zlb_s = dalloc((size_t)B * N * NG), *zub_s = dalloc((size_t)B * N * NG);
    S *b = dalloc((size_t)B * n), *sol = dalloc((size_t)B * n), *dx = dalloc(B * row);
    S *Xz = dalloc(B * row), *Yz = dalloc((size_t)B * N * NG), *Zz = dalloc((size_t)B * N * NG);   // ADMM starts at x = y = z = 0
    S *X[2] = {dalloc(B * row), dalloc(B * row)}, *Y[2] = {dalloc((size_t)B * N * NG), dalloc((size_t)B * N * NG)},
      *Z[2] = {dalloc((size_t)B * N * NG), dalloc((size_t)B * N * NG)};
    S *log_state = dalloc((size_t)(n_steps + 1) * B * NPG), *log_plan = dalloc((size_t)n_steps * B * row);
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
    const S** in_it[3]; S** out_it[2];   // it 0 reads the zero state; then ping-pong
    in_it[0] = (const S**)dptrs<S>({sol, Xz, Yz, Zz, q_s, zlb_s, zub_s, D, sigma, rho, alpha});
    for (int p = 0; p < 2; ++p) in_it[1 + p] = (const S**)dptrs<S>({sol, X[p], Y[p], Z[p], q_s, zlb_s, zub_s, D, sigma, rho, alpha});
    for (int p = 0; p < 2; ++p) out_it[p] = dptrs<S>({X[p], Y[p], Z[p], b, dx});
    const S** in_adv = (const S**)dptrs<S>({p_shared, xp_pad});                                  S** out_adv = dptrs<S>({p_shared_next});
    const S** in_shift = (const S**)dptrs<S>({xp_pad, xp_pad, has_next, shift_w, dt_adv, p_shared, p_shared_next});
    S** out_shift = dptrs<S>({x_shift});

    // ---- cuDSS
    cudssHandle_t hd; DS(cudssCreate(&hd)); DS(cudssSetStream(hd, st));
    PoolCtx pctx{}; cudssDeviceMemHandler_t handler{};
    cudaMemPoolProps props{}; props.allocType = cudaMemAllocationTypePinned; props.location.type = cudaMemLocationTypeDevice; props.location.id = 0;
    CK(cudaMemPoolCreate(&pctx.pool, &props));
    unsigned long long thr = ~0ull; CK(cudaMemPoolSetAttribute(pctx.pool, cudaMemPoolAttrReleaseThreshold, &thr));
    handler.ctx = &pctx; handler.device_alloc = pool_alloc; handler.device_free = pool_free;
    strncpy(handler.name, "cusadi2 rollout pool", CUDSS_ALLOCATOR_NAME_LEN); DS(cudssSetDeviceMemHandler(hd, &handler));
    cudssConfig_t cfg; DS(cudssConfigCreate(&cfg)); cudssData_t data; DS(cudssDataCreate(hd, &data));
    int ub = B; DS(cudssConfigSet(cfg, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof ub));
    const cudssDataType_t cdt = sizeof(S) == 4 ? CUDSS_R_32F : CUDSS_R_64F;
    cudssMatrix_t A, Xm, Bm;
    DS(cudssMatrixCreateCsr(&A, n, n, nnz, d_Ap, nullptr, d_Ai, vals, CUDSS_R_32I, CUDSS_R_32I, cdt, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
    DS(cudssMatrixCreateDn(&Bm, n, 1, n, b, cdt, CUDSS_LAYOUT_COL_MAJOR));
    DS(cudssMatrixCreateDn(&Xm, n, 1, n, sol, cdt, CUDSS_LAYOUT_COL_MAJOR));

    const size_t szS = sizeof(S);
    auto pad = [&](S* dst, const S* src, int width, int before, int count_dst) {
        CK(cudaMemcpy2DAsync(dst + (size_t)before * width, (size_t)count_dst * width * szS, src, (size_t)N * width * szS,
                             (size_t)N * width * szS, B, cudaMemcpyDeviceToDevice, st));
    };
    auto step = [&]() {
        // QP at the current plan x (x_pad) and state (p_shared), as in mpc_cudss_bench.cu
        CK(fn[0].evaluate(in_asm, out_asm, B, st));
        pad(blk_pad, blk, NBLK, 0, N + 1);
        CK(fn[1].evaluate(in_r1, out_r1, B, st));
        pad(d1_pad, d1, NB, 1, N + 2);
        CK(fn[2].evaluate(in_r2, out_r2, B, st));
        k_gamma<<<(B + 127) / 128, 128, 0, st>>>(m1, g1, B, N);
        CK(fn[3].evaluate(in_r3, out_r3, B, st));
        pad(d2_pad, d2, NB, 1, N + 2);
        CK(fn[4].evaluate(in_r4, out_r4, B, st));
        k_gamma<<<(B + 127) / 128, 128, 0, st>>>(m2, g2, B, N);
        CK(fn[5].evaluate(in_r5, out_r5, B, st));
        k_gather<<<(int)(((long long)B * nnz + 255) / 256), 256, 0, st>>>(kkt_blk, d_map, vals, B, nnz, (long long)N * NKKT);
        DS(cudssExecute(hd, CUDSS_PHASE_FACTORIZATION, cfg, data, A, Xm, Bm));
        for (int it = 0; it < N_ITER; ++it) {
            DS(cudssExecute(hd, CUDSS_PHASE_SOLVE, cfg, data, A, Xm, Bm));
            CK(fn[6].evaluate(in_it[it == 0 ? 0 : 1 + ((it - 1) & 1)], out_it[it & 1], B, st));
        }
        // SQP step, advance the state one stage, shift the plan
        k_sqp_step<<<(int)((B * row + 255) / 256), 256, 0, st>>>(x_pad, dx, xp_pad, B, (int)row, NS);
        CK(fn[8].evaluate(in_adv, out_adv, B, st));
        CK(fn[7].evaluate(in_shift, out_shift, B, st));
        pad(x_pad, x_shift, NS, 1, N + 2);
        CK(cudaMemcpyAsync(p_shared, p_shared_next, (size_t)B * NPG * szS, cudaMemcpyDeviceToDevice, st));
    };
    DS(cudssExecute(hd, CUDSS_PHASE_ANALYSIS, cfg, data, A, Xm, Bm)); CK(cudaStreamSynchronize(st));

    // Warm up (fills the cuDSS pool, compiles nothing), then restore the initial plan and state.
    auto reset = [&]() {
        CK(cudaMemsetAsync(x_pad, 0, (size_t)B * (N + 2) * NS * szS, st));
        std::vector<double> t; for (int i = 0; i < B; ++i) t.insert(t.end(), h_pg.begin(), h_pg.end());
        std::vector<S> ts(t.begin(), t.end()); CK(cudaMemcpy(p_shared, ts.data(), ts.size() * szS, cudaMemcpyHostToDevice));
        CK(cudaMemcpyAsync(p_stage, p_all, step_ps * szS, cudaMemcpyDeviceToDevice, st));
    };
    reset(); for (int k = 0; k < 3; ++k) step(); CK(cudaStreamSynchronize(st));
    cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
    if (use_graph) {
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal)); step(); CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int k = 0; k < 3; ++k) CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
    }
    reset(); CK(cudaStreamSynchronize(st));

    // ---- closed loop
    cudaEvent_t ev[2]; for (auto& e : ev) CK(cudaEventCreate(&e));
    std::vector<double> lat(n_steps);
    CK(cudaMemcpyAsync(log_state, p_shared, (size_t)B * NPG * szS, cudaMemcpyDeviceToDevice, st));
    auto w0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_steps; ++s) {
        CK(cudaMemcpyAsync(p_stage, p_all + (size_t)s * step_ps, step_ps * szS, cudaMemcpyDeviceToDevice, st));
        CK(cudaEventRecord(ev[0], st));
        if (use_graph) CK(cudaGraphLaunch(ge, st)); else step();
        CK(cudaEventRecord(ev[1], st));
        CK(cudaMemcpy2DAsync(log_plan + (size_t)s * B * row, row * szS, xp_pad + NS, (N + 2) * NS * szS, row * szS, B, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(log_state + (size_t)(s + 1) * B * NPG, p_shared, (size_t)B * NPG * szS, cudaMemcpyDeviceToDevice, st));
        CK(cudaEventSynchronize(ev[1]));
        float ms; CK(cudaEventElapsedTime(&ms, ev[0], ev[1])); lat[s] = ms * 1e3;
    }
    CK(cudaStreamSynchronize(st));
    double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();

    std::vector<double> h_state = download(log_state, (size_t)(n_steps + 1) * B * NPG), h_plan = download(log_plan, (size_t)n_steps * B * row);
    write_f64(roll + "state.f64", h_state); write_f64(roll + "plan.f64", h_plan); write_f64(roll + "latency_us.f64", lat);
    bool finite = true; for (double v : h_state) finite = finite && std::isfinite(v);
    std::vector<double> sl = lat; std::sort(sl.begin(), sl.end());
    printf("B=%d steps=%d (%.3f s simulated) mode=%s finite=%d | step latency us: median %.1f  p99 %.1f  max %.1f | wall %.1f ms (%.2fx real time)\n",
           B, n_steps, n_steps * h_dt[0], use_graph ? "graph" : "eager", (int)finite, sl[n_steps / 2], sl[(size_t)(n_steps * 0.99)], sl.back(),
           wall_ms, n_steps * h_dt[0] * 1e3 / wall_ms);

    if (ge) CK(cudaGraphExecDestroy(ge)); if (g) cudaGraphDestroy(g);
    DS(cudssMatrixDestroy(A)); DS(cudssMatrixDestroy(Xm)); DS(cudssMatrixDestroy(Bm));
    DS(cudssDataDestroy(hd, data)); DS(cudssConfigDestroy(cfg)); DS(cudssDestroy(hd));
    CK(cudaMemPoolDestroy(pctx.pool));
    for (auto& e : ev) cudaEventDestroy(e);
    CK(cudaStreamDestroy(st));
    return finite ? 0 : 1;
}
