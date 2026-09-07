// Full MIT-humanoid MPC QP solve (wip-branch "CusADi 2.0" ADMM formulation) on
// the GPU, batched over B instances, chaining the six CasADi functions exactly as
// admm_backend.solve_with_custom does:
//   kkt = KKT_matrix(x, p); (kkt_s, q, D, lb, ub) = ruiz(kkt, x, p, sigma, rho);
//   (Dl, L, perm) = ldl_fac(kkt_s);
//   25 x { b = KKT_vector(q, xk, yk, zk, sigma, rho); s = ldl_solve(b, Dl, L);
//          (xk, yk, zk) = ADMM_step(s, xk, yk, zk, lb, ub, alpha, rho) }
//   dx = D * xk
// Modes: c2 (cusadi2 level kernel where the plan fits shared memory, else its
// serial kernel), c2serial (cusadi2 serial kernel for every function), orig
// (original-cusadi unrolled kernels, linked in via unrolled_table.h), cudss
// (like c2, but the symbolic LDL pair is replaced by cuDSS uniform-batch
// factorization/solve on the Ruiz output, whose upper-triangle CCS is cuDSS's
// lower CSR; default cuDSS pivoting/algorithms, pool memory handler so the
// phases capture into the graph).
// Times one eager solve (sequential launches) and one CUDA-graph replay of the
// same 79 launches; checks instance 0 against the CPU reference dx.
//   mpc_solve_bench <plan_dir> <mode> <p_eval.f64> <dx_ref.f64> <n_warm> <n_timed> <B...>
#include "cusadi2.h"
#include <cudss.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
using namespace cusadi2;
using S = Scalar;

struct UnrolledFn {
    const char* name; int (*sz_w)(); int (*n_stages)();
    int (*launch)(const S* const*, S* const*, S*, int, int, cudaStream_t);
};
#ifdef NO_UNROLLED
static const UnrolledFn UNROLLED[] = {{nullptr, nullptr, nullptr, nullptr}};
#else
#include "unrolled_table.h"
#endif
static const UnrolledFn* find_unrolled(const std::string& n) {
    for (const UnrolledFn& u : UNROLLED) if (u.name && n == u.name) return &u;
    return nullptr;
}

#define DS(x) do { cudssStatus_t s_ = (x); if (s_ != CUDSS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuDSS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)
struct PoolCtx { cudaMemPool_t pool = nullptr; };
static int pool_alloc(void* ctx, void** ptr, size_t size, cudaStream_t stream) {
    return cudaMallocFromPoolAsync(ptr, size, ((PoolCtx*)ctx)->pool, stream) == cudaSuccess ? 0 : 1;
}
static int pool_free(void*, void* ptr, size_t, cudaStream_t stream) { return cudaFreeAsync(ptr, stream) == cudaSuccess ? 0 : 1; }
#define CK(x) do { cudaError_t e_ = (cudaError_t)(x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static std::vector<double> read_f64(const char* path) {
    std::ifstream f(path, std::ios::binary); if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg() / 8; f.seekg(0);
    std::vector<double> v(n); f.read((char*)v.data(), n * 8); return v;
}
static S* dalloc(size_t n) { S* d; CK(cudaMalloc(&d, n * sizeof(S))); CK(cudaMemset(d, 0, n * sizeof(S))); return d; }
template <class T> static T** dptrs(const std::vector<T*>& h) {
    T** d; CK(cudaMalloc(&d, h.size() * sizeof(T*))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T*), cudaMemcpyHostToDevice)); return d;
}

// One function: cusadi2 Program or the original unrolled kernel.
struct Fn {
    std::string name; Plan plan; Program prog; const UnrolledFn* unr = nullptr; S* work = nullptr;
    std::string how;
    void setup(const std::string& dir, const std::string& mode, int batch, int max_smem, size_t serial_budget) {
        plan = load_plan(dir + "/" + name + ".plan");
        if (mode == "orig") {
            unr = find_unrolled(name);
            if (!unr) { fprintf(stderr, "no unrolled kernel for %s\n", name.c_str()); exit(3); }
            CK(cudaMalloc(&work, (size_t)batch * unr->sz_w() * sizeof(S)));
            how = "orig";
        } else {
            prog.upload(plan);
            bool level = (mode == "c2" || mode == "cudss") && plan.shared_bytes() <= max_smem;
            if (!level) prog.use_serial((size_t)plan.sz_w * sizeof(S) * (size_t)batch + 1, 256);   // one thread per instance, no looping
            how = level ? "level" : "serial";
        }
    }
    int run(const S** d_in, S** d_out, int batch, cudaStream_t s) const {
        return unr ? unr->launch(d_in, d_out, work, batch, 256, s) : prog.evaluate(d_in, d_out, batch, s);
    }
};

int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s <plan_dir> <c2|c2serial|orig|cudss> <p_eval.f64> <dx_ref.f64> <n_warm> <n_timed> <B...>\n", argv[0]); return 2; }
    const std::string dir = argv[1], mode = argv[2];
    std::vector<double> p_ref = read_f64(argv[3]), dx_ref = read_f64(argv[4]);
    const int n_warm = atoi(argv[5]), n_timed = atoi(argv[6]);
    const int N_ITER = 25; const double SIGMA = 1e-6, RHO = 0.1, ALPHA = 1.6;
    int max_smem = 0; CK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    const char* names[6] = {"KKT_matrix_mit_humanoid_mpc", "ruiz_scaling_mit_humanoid_mpc", "ldl_factorize_mit_humanoid_mpc",
                            "KKT_vector_mit_humanoid_mpc", "ldl_solve_mit_humanoid_mpc", "ADMM_step_mit_humanoid_mpc"};
    printf("dtype,mode,batch,kkt_us,ruiz_us,fac_us,iters_us,eager_total_us,graph_total_us,max_abs_err,how\n");
    for (int ai = 7; ai < argc; ++ai) {
        const int B = atoi(argv[ai]);
        Fn fn[6];
        size_t freeb = 0, totb = 0; CK(cudaMemGetInfo(&freeb, &totb));
        // serial scratch: sz_w * B per function is needed for one-thread-per-instance without looping
        size_t need = 0; for (int i = 0; i < 6; ++i) { Plan p = load_plan(dir + "/" + std::string(names[i]) + ".plan"); need += (size_t)p.sz_w * sizeof(S) * B; }
        if (mode != "orig" && need > 0.6 * (double)freeb) { printf("%s,%s,%d,nan,nan,nan,nan,nan,nan,oom\n", sizeof(S) == 8 ? "f64" : "f32", mode.c_str(), B); continue; }
        const bool use_cudss = mode == "cudss";
        for (int i = 0; i < 6; ++i) {
            fn[i].name = names[i];
            if (use_cudss && (i == 2 || i == 4)) { fn[i].plan = load_plan(dir + "/" + std::string(names[i]) + ".plan"); fn[i].how = "cudss"; continue; }
            fn[i].setup(dir, mode, B, max_smem, (size_t)fn[i].plan.sz_w * sizeof(S) * B + 1);
        }
        // dims from the plans
        auto nnz_in = [&](int f, int a) { return (size_t)fn[f].plan.in[a].sp.nnz(); };
        auto nnz_out = [&](int f, int a) { return (size_t)fn[f].plan.out[a].sp.nnz(); };
        const size_t NX = nnz_in(0, 0), NP = nnz_in(0, 1), NKKT = nnz_out(0, 0), NKS = nnz_out(1, 0), NG = nnz_out(1, 3),
                     NSYS = nnz_out(3, 0), NL = nnz_out(2, 1);
        if (p_ref.size() != NP || dx_ref.size() != NX) { fprintf(stderr, "ref size mismatch\n"); return 4; }
        // buffers, (instance, nnz) layout
        S *x = dalloc(B * NX), *p = dalloc(B * NP), *sigma = dalloc(B), *rho = dalloc(B), *alpha = dalloc(B);
        S *kkt = dalloc(B * NKKT), *kkts = dalloc(B * NKS), *q = dalloc(B * NX), *D = dalloc(B * NX), *lb = dalloc(B * NG), *ub = dalloc(B * NG);
        S *Dl = dalloc(B * NSYS), *L = dalloc(B * NL), *perm = dalloc(B * NSYS), *b = dalloc(B * NSYS), *sol = dalloc(B * NSYS);
        S *X[2] = {dalloc(B * NX), dalloc(B * NX)}, *Y[2] = {dalloc(B * NG), dalloc(B * NG)}, *Z[2] = {dalloc(B * NG), dalloc(B * NG)};
        {   std::vector<S> hp(B * NP), h1(B);
            for (int i = 0; i < B; ++i) for (size_t j = 0; j < NP; ++j) hp[i * NP + j] = (S)p_ref[j];
            CK(cudaMemcpy(p, hp.data(), hp.size() * sizeof(S), cudaMemcpyHostToDevice));
            for (int i = 0; i < B; ++i) h1[i] = (S)SIGMA; CK(cudaMemcpy(sigma, h1.data(), B * sizeof(S), cudaMemcpyHostToDevice));
            for (int i = 0; i < B; ++i) h1[i] = (S)RHO;   CK(cudaMemcpy(rho, h1.data(), B * sizeof(S), cudaMemcpyHostToDevice));
            for (int i = 0; i < B; ++i) h1[i] = (S)ALPHA; CK(cudaMemcpy(alpha, h1.data(), B * sizeof(S), cudaMemcpyHostToDevice)); }
        const S** in_kkt = (const S**)dptrs<S>({x, p});                         S** out_kkt = dptrs<S>({kkt});
        const S** in_ruiz = (const S**)dptrs<S>({kkt, x, p, sigma, rho});       S** out_ruiz = dptrs<S>({kkts, q, D, lb, ub});
        const S** in_fac = (const S**)dptrs<S>({kkts});                         S** out_fac = dptrs<S>({Dl, L, perm});
        const S** in_vec[2], **in_step[2]; S** out_step[2]; S** out_vec = dptrs<S>({b});
        const S** in_sol = (const S**)dptrs<S>({b, Dl, L});                     S** out_sol = dptrs<S>({sol});
        for (int par = 0; par < 2; ++par) {
            in_vec[par] = (const S**)dptrs<S>({q, X[par], Y[par], Z[par], sigma, rho});
            in_step[par] = (const S**)dptrs<S>({sol, X[par], Y[par], Z[par], lb, ub, alpha, rho});
            out_step[par] = dptrs<S>({X[1 - par], Y[1 - par], Z[1 - par]});
        }
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        cudaEvent_t ev[5]; for (auto& e : ev) CK(cudaEventCreate(&e));
        // cuDSS on the Ruiz output: upper-triangle CCS (colptr, rowidx) == lower CSR of the symmetric KKT
        cudssHandle_t hd = nullptr; cudssConfig_t cfg = nullptr; cudssData_t data = nullptr; cudssMatrix_t A = nullptr, Xm = nullptr, Bm = nullptr;
        PoolCtx pctx; cudssDeviceMemHandler_t handler{}; int *d_Ap = nullptr, *d_Ai = nullptr; double analysis_ms = 0;
        if (use_cudss) {
            const Sparsity& sp = fn[1].plan.out[0].sp; const int n = (int)sp.colptr.size() - 1;
            if ((size_t)n != NSYS || (size_t)sp.nnz() != NKS) { fprintf(stderr, "kkt sparsity mismatch\n"); return 5; }
            CK(cudaMalloc(&d_Ap, (n + 1) * 4)); CK(cudaMemcpy(d_Ap, sp.colptr.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&d_Ai, sp.nnz() * 4)); CK(cudaMemcpy(d_Ai, sp.rowidx.data(), sp.nnz() * 4, cudaMemcpyHostToDevice));
            DS(cudssCreate(&hd)); DS(cudssSetStream(hd, st));
            cudaMemPoolProps props{}; props.allocType = cudaMemAllocationTypePinned; props.location.type = cudaMemLocationTypeDevice; props.location.id = 0;
            CK(cudaMemPoolCreate(&pctx.pool, &props)); unsigned long long thr = ~0ull; CK(cudaMemPoolSetAttribute(pctx.pool, cudaMemPoolAttrReleaseThreshold, &thr));
            handler.ctx = &pctx; handler.device_alloc = pool_alloc; handler.device_free = pool_free;
            strncpy(handler.name, "cusadi mono pool", CUDSS_ALLOCATOR_NAME_LEN); DS(cudssSetDeviceMemHandler(hd, &handler));
            DS(cudssConfigCreate(&cfg)); DS(cudssDataCreate(hd, &data));
            int ub = B; DS(cudssConfigSet(cfg, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof ub));
            const cudssDataType_t cdt = sizeof(S) == 4 ? CUDSS_R_32F : CUDSS_R_64F;
            DS(cudssMatrixCreateCsr(&A, n, n, sp.nnz(), d_Ap, nullptr, d_Ai, kkts, CUDSS_R_32I, CUDSS_R_32I, cdt, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
            DS(cudssMatrixCreateDn(&Bm, n, 1, n, b, cdt, CUDSS_LAYOUT_COL_MAJOR));
            DS(cudssMatrixCreateDn(&Xm, n, 1, n, sol, cdt, CUDSS_LAYOUT_COL_MAJOR));
            auto t0 = std::chrono::steady_clock::now();
            DS(cudssExecute(hd, CUDSS_PHASE_ANALYSIS, cfg, data, A, Xm, Bm)); CK(cudaStreamSynchronize(st));
            analysis_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        auto solve = [&](bool timed) {          // the 79 launches (+3 memsets); events only when timed
            CK(cudaMemsetAsync(X[0], 0, B * NX * sizeof(S), st)); CK(cudaMemsetAsync(Y[0], 0, B * NG * sizeof(S), st)); CK(cudaMemsetAsync(Z[0], 0, B * NG * sizeof(S), st));
            if (timed) CK(cudaEventRecord(ev[0], st));
            CK(fn[0].run(in_kkt, out_kkt, B, st));    if (timed) CK(cudaEventRecord(ev[1], st));
            CK(fn[1].run(in_ruiz, out_ruiz, B, st));  if (timed) CK(cudaEventRecord(ev[2], st));
            if (use_cudss) DS(cudssExecute(hd, CUDSS_PHASE_FACTORIZATION, cfg, data, A, Xm, Bm));
            else CK(fn[2].run(in_fac, out_fac, B, st));
            if (timed) CK(cudaEventRecord(ev[3], st));
            for (int it = 0; it < N_ITER; ++it) {
                int par = it & 1;
                CK(fn[3].run(in_vec[par], out_vec, B, st));
                if (use_cudss) DS(cudssExecute(hd, CUDSS_PHASE_SOLVE, cfg, data, A, Xm, Bm));
                else CK(fn[4].run(in_sol, out_sol, B, st));
                CK(fn[5].run(in_step[par], out_step[par], B, st));
            }
            if (timed) CK(cudaEventRecord(ev[4], st));
        };
        const int fin = N_ITER & 1;              // parity holding the final iterate
        // correctness on instance 0
        solve(false); CK(cudaStreamSynchronize(st));
        double max_err = 0;
        {   std::vector<S> hx(NX), hD(NX);
            CK(cudaMemcpy(hx.data(), X[fin], NX * sizeof(S), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hD.data(), D, NX * sizeof(S), cudaMemcpyDeviceToHost));
            for (size_t j = 0; j < NX; ++j) max_err = std::max(max_err, std::fabs((double)hx[j] * (double)hD[j] - dx_ref[j]));
            // instance B-1 is checked against the reference too (cuDSS is not bitwise deterministic across batch members)
            if (B > 1) { std::vector<S> hx2(NX), hD2(NX);
                CK(cudaMemcpy(hx2.data(), X[fin] + (size_t)(B - 1) * NX, NX * sizeof(S), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hD2.data(), D + (size_t)(B - 1) * NX, NX * sizeof(S), cudaMemcpyDeviceToHost));
                for (size_t j = 0; j < NX; ++j) max_err = std::max(max_err, std::fabs((double)hx2[j] * (double)hD2[j] - dx_ref[j])); }
        }
        // eager timing with phase breakdown
        for (int k = 0; k < n_warm; ++k) solve(false);
        CK(cudaStreamSynchronize(st));
        std::vector<double> t_kkt, t_ruiz, t_fac, t_it, t_tot;
        for (int k = 0; k < n_timed; ++k) {
            solve(true); CK(cudaEventSynchronize(ev[4]));
            float ms[4]; for (int i = 0; i < 4; ++i) CK(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]));
            t_kkt.push_back(ms[0] * 1e3); t_ruiz.push_back(ms[1] * 1e3); t_fac.push_back(ms[2] * 1e3); t_it.push_back(ms[3] * 1e3);
            t_tot.push_back((ms[0] + ms[1] + ms[2] + ms[3]) * 1e3);
        }
        // CUDA graph of the whole solve
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal)); solve(false); CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int k = 0; k < n_warm; ++k) CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        std::vector<double> t_g;
        for (int k = 0; k < n_timed; ++k) {
            CK(cudaEventRecord(ev[0], st)); CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(ev[4], st)); CK(cudaEventSynchronize(ev[4]));
            float ms; CK(cudaEventElapsedTime(&ms, ev[0], ev[4])); t_g.push_back(ms * 1e3);
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        std::string how; for (int i = 0; i < 6; ++i) how += (i ? "/" : "") + fn[i].how;
        printf("%s,%s,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.3e,%s\n", sizeof(S) == 8 ? "f64" : "f32", mode.c_str(), B,
               med(t_kkt), med(t_ruiz), med(t_fac), med(t_it), med(t_tot), med(t_g), max_err, how.c_str());
        fflush(stdout);
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
        if (use_cudss) {
            fprintf(stderr, "[B=%d] cuDSS analysis %.1f ms\n", B, analysis_ms);
            DS(cudssMatrixDestroy(A)); DS(cudssMatrixDestroy(Xm)); DS(cudssMatrixDestroy(Bm));
            DS(cudssDataDestroy(hd, data)); DS(cudssConfigDestroy(cfg)); DS(cudssDestroy(hd));
            CK(cudaMemPoolDestroy(pctx.pool)); cudaFree(d_Ap); cudaFree(d_Ai);
        }
        for (auto& e : ev) cudaEventDestroy(e); cudaStreamDestroy(st);
        for (S* d : {x, p, sigma, rho, alpha, kkt, kkts, q, D, lb, ub, Dl, L, perm, b, sol, X[0], X[1], Y[0], Y[1], Z[0], Z[1]}) cudaFree(d);
        for (int i = 0; i < 6; ++i) cudaFree(fn[i].work);
    }
    return 0;
}
