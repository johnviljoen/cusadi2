// Safe MPPI for one Franka Panda, one native CUDA driver with a pluggable model evaluator.
//
// One control tick (20 ms), K sampled rollouts of T steps:
//   sample     Y[k] = clip(Y0 + sigma_j eps, joint range)          N_KNOT knots of 7 joint targets; sample 0 is Y0
//   init       x[k] <- (q, v of the plant, tau = 0, cost = 0)
//   T times:   (u_nom, G, h) = model(x, Y, rp, par)                 THE EVALUATOR, call 1: nominal PD torque and
//                                                                   cbfpy's NCBF one-step CBF rows at the current state
//              u = project(u_nom, G, h, u_max)                      shared MPPI kernel: PROJ_ITERS greedy projections onto
//                                                                   the most violated half-space, torque box after each
//              x <- integrate(x, u, rp, par)                        THE EVALUATOR, call 2: semi-implicit Euler + running cost
//   update     w = softmax(-(cost - min) / (temp std)),  Y0 <- sum_k w Y[k]
//   plant      u_applied = project(model(x_plant, Y0, rp, par))     the torque the robot receives (same filter)
//              x_plant <- integrate(x_plant, u_applied)              only in model-plant mode
//   shift      Y0 re-sampled one control period later (last knot held)
//
// The evaluator is cusadi2 (two plans, pure kernel launches, the whole tick captured as one CUDA graph), a JAX/XLA
// program pair run through the PJRT C API on views of the driver's buffers (frax + cbfpy, or MJX + cbfpy), the original
// CusADi's unrolled one-thread-per-instance kernels of the same CasADi functions, or CasADi's C code generation on the
// CPU with the GPU <-> CPU transfers of every call. Everything else, the projection included, is the same code for all. R > 1 independent robots are supported for
// batch studies (instances = R * K); the demo uses R = 1.
//
// C API (driver.py wraps it with ctypes; host arrays are double):
//   void* sm_create(const char* cfg)     cfg = "key=value;..." : evaluator=cusadi2|pjrt|cusadi|cpu, R, K, T, graph, temp, seed,
//        dt, sigma=s0,s1,s2,s3, ylim=lo(7),hi(7), par=9 values, umax=7 values, proj_iters, model_plan, integrate_plan |
//        plugin, model_mlir, integrate_mlir, compile_options | model_lib, integrate_lib, block | cpu_lib, cpu_threads, cpu_transfer
//   int sm_tick(h, const double* x_plant[R*16] or NULL, const double* wp[R*3], double* u[R*7], double* stat[R*4],
//               double* gpu_ms, double* wall_ms)
//   int sm_time_rollout / sm_time_model / sm_time_integrate / sm_time_project (h, n_calls, double* median_ms, double* min_ms)
//   sm_get_plan / sm_set_plan / sm_get_plant / sm_set_plant / sm_get_samples / sm_get_filter / sm_eval_model / sm_eval_integrate /
//   sm_info / sm_error / sm_destroy
#include "cusadi2.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include <curand_kernel.h>
#include <dlfcn.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
using namespace cusadi2;
using S = Scalar;

namespace {
constexpr int NV = 7, NX = 16, NKNOT = 4, ND = NKNOT * NV, NPAR = 9, NRP = 3, I_TAU = 2 * NV, I_COST = 2 * NV + 1;
#ifndef NCBF_ROWS
#define NCBF_ROWS 64
#endif
constexpr int NCBF = NCBF_ROWS, RPL = (NCBF + 31) / 32;   // CBF rows, rows per lane in k_project (lane l owns rows l + 32 s)
static_assert(NCBF >= 32, "k_project: every lane owns at least one live row");
std::string g_err;
#define CK(x) do { cudaError_t e_ = (cudaError_t)(x); if (e_ != cudaSuccess) { \
    g_err = std::string("CUDA error ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__); \
    fprintf(stderr, "%s\n", g_err.c_str()); return -1; } } while (0)
#define CKV(x) do { cudaError_t e_ = (cudaError_t)(x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

std::vector<char> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary); if (!f) throw std::runtime_error("cannot open " + p);
    f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg(); f.seekg(0); std::vector<char> v(n); f.read(v.data(), n); return v;
}
S* dalloc(size_t n) { S* d; CKV(cudaMalloc(&d, n * sizeof(S))); CKV(cudaMemset(d, 0, n * sizeof(S))); return d; }
template <class T> T** dptrs(const std::vector<T*>& h) {
    T** d; CKV(cudaMalloc(&d, h.size() * sizeof(T*))); CKV(cudaMemcpy(d, h.data(), h.size() * sizeof(T*), cudaMemcpyHostToDevice)); return d;
}
std::vector<double> parse_list(const std::string& s) {
    std::vector<double> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(atof(t.c_str())); return v;
}

// ------------------------------------------------------------------------------------------------------ evaluators
// G is stored column-major per instance (cusadi2's nonzero order of a dense NCBF x 7 matrix): G[k][i + NCBF * j] = G_ij.
struct Evaluator {
    virtual ~Evaluator() {}
    virtual int model(int B, const S* x, const S* Y, const S* rp, const S* par, S* u_nom, S* G, S* h, cudaStream_t st) = 0;
    virtual int integrate(int B, const S* x, const S* u, const S* rp, const S* par, S* x1, cudaStream_t st) = 0;
    virtual bool capturable() const = 0;
    virtual std::string info() const = 0;
};

struct Cusadi2Evaluator : Evaluator {
    Plan plan_m, plan_i; Program fm, fi;
    std::map<std::vector<const void*>, const S**> tabs;
    Cusadi2Evaluator(const std::string& model_plan, const std::string& integrate_plan)
        : plan_m(load_plan(model_plan)), plan_i(load_plan(integrate_plan)), fm(plan_m), fi(plan_i) {
        if (plan_m.in[0].nnz() != NX || plan_m.in[1].nnz() != ND || plan_m.in[2].nnz() != NRP || plan_m.in[3].nnz() != NPAR ||
            plan_m.out[0].nnz() != NV || plan_m.out[1].nnz() != NCBF * NV || plan_m.out[2].nnz() != NCBF)
            throw std::runtime_error("model plan signature mismatch");
        if (plan_i.in[0].nnz() != NX || plan_i.in[1].nnz() != NV || plan_i.out[0].nnz() != NX) throw std::runtime_error("integrate plan signature mismatch");
    }
    const S** tab(std::vector<const void*> ptrs) {           // device pointer tables, built once per distinct tuple (before capture)
        auto it = tabs.find(ptrs);
        if (it == tabs.end()) { std::vector<S*> v; for (auto p : ptrs) v.push_back((S*)p); it = tabs.emplace(ptrs, (const S**)dptrs<S>(v)).first; }
        return it->second;
    }
    int model(int B, const S* x, const S* Y, const S* rp, const S* par, S* u_nom, S* G, S* h, cudaStream_t st) override {
        CK(fm.evaluate(tab({x, Y, rp, par}), (S* const*)tab({u_nom, G, h}), B, st)); return 0;
    }
    int integrate(int B, const S* x, const S* u, const S* rp, const S* par, S* x1, cudaStream_t st) override {
        CK(fi.evaluate(tab({x, u, rp, par}), (S* const*)tab({x1}), B, st)); return 0;
    }
    bool capturable() const override { return true; }
    std::string info() const override {
        return "cusadi2 plans " + plan_m.name + " (" + std::to_string(plan_m.node_ops.size()) + " nodes, " + std::to_string(plan_m.level_widths.size()) +
               " levels, " + std::to_string(plan_m.shared_bytes()) + " B shared, block " + std::to_string(pick_block_size(plan_m)) + ") + " + plan_i.name + " (" +
               std::to_string(plan_i.node_ops.size()) + " nodes, " + std::to_string(plan_i.level_widths.size()) + " levels)";
    }
};

// The original CusADi (se-hwan/cusadi) execution model: the same two CasADi functions as one unrolled kernel each,
// one thread per instance, one statement per CasADi instruction, thread-major global work vector (cusadi2.unrolled
// codegen, nvcc-compiled into build/unrolled/*.so and dlopen'ed here). Argument layout is cusadi2's (per-instance
// contiguous nonzeros), so no re-layout is needed. Block size 256 as in the paper's "CusADi" baseline.
struct CusadiUnrolledEvaluator : Evaluator {
    typedef int (*launch_t)(const S* const*, S* const*, S*, int, int, cudaStream_t);
    launch_t lm = nullptr, li = nullptr; int szw_m = 0, szw_i = 0, Bmax, block; S *work_m, *work_i; std::string desc;
    std::map<std::vector<const void*>, const S**> tabs;
    static void* sym(void* h, const char* n) { void* p = dlsym(h, n); if (!p) throw std::runtime_error(std::string("dlsym ") + n + ": " + dlerror()); return p; }
    CusadiUnrolledEvaluator(const std::string& model_lib, const std::string& integrate_lib, int B_max, int block_size) : Bmax(B_max), block(block_size) {
        void* hm = dlopen(model_lib.c_str(), RTLD_NOW | RTLD_LOCAL); if (!hm) throw std::runtime_error(std::string("dlopen ") + model_lib + ": " + dlerror());
        void* hi = dlopen(integrate_lib.c_str(), RTLD_NOW | RTLD_LOCAL); if (!hi) throw std::runtime_error(std::string("dlopen ") + integrate_lib + ": " + dlerror());
        lm = (launch_t)sym(hm, "safe_model_unrolled_launch"); li = (launch_t)sym(hi, "safe_integrate_unrolled_launch");
        szw_m = ((int (*)())sym(hm, "safe_model_unrolled_sz_w"))(); szw_i = ((int (*)())sym(hi, "safe_integrate_unrolled_sz_w"))();
        if (((int (*)())sym(hm, "safe_model_unrolled_n_stages"))() != 1 || ((int (*)())sym(hi, "safe_integrate_unrolled_n_stages"))() != 1)
            throw std::runtime_error("unrolled kernels must be single-stage");
        work_m = dalloc((size_t)Bmax * szw_m); work_i = dalloc((size_t)Bmax * szw_i);
        desc = "original CusADi unrolled kernels (one thread per instance, block " + std::to_string(block) + "): " + model_lib.substr(model_lib.rfind('/') + 1) +
               " (sz_w " + std::to_string(szw_m) + ") + " + integrate_lib.substr(integrate_lib.rfind('/') + 1) + " (sz_w " + std::to_string(szw_i) + ")";
    }
    const S** tab(std::vector<const void*> ptrs) {
        auto it = tabs.find(ptrs);
        if (it == tabs.end()) { std::vector<S*> v; for (auto p : ptrs) v.push_back((S*)p); it = tabs.emplace(ptrs, (const S**)dptrs<S>(v)).first; }
        return it->second;
    }
    int model(int B, const S* x, const S* Y, const S* rp, const S* par, S* u_nom, S* G, S* h, cudaStream_t st) override {
        if (B > Bmax) { g_err = "unrolled evaluator: B exceeds the work buffer"; return -1; }
        CK(lm(tab({x, Y, rp, par}), (S* const*)tab({u_nom, G, h}), work_m, B, block, st)); return 0;
    }
    int integrate(int B, const S* x, const S* u, const S* rp, const S* par, S* x1, cudaStream_t st) override {
        if (B > Bmax) { g_err = "unrolled evaluator: B exceeds the work buffer"; return -1; }
        CK(li(tab({x, u, rp, par}), (S* const*)tab({x1}), work_i, B, block, st)); return 0;
    }
    bool capturable() const override { return true; }
    std::string info() const override { return desc; }
};

// CasADi's own C code generation on the CPU (the paper's CPU baseline), 16 OpenMP threads over the instances, with
// the data living on the GPU: every call copies its inputs device -> pinned host, evaluates, and copies the outputs
// back on the driver's stream ("CPU + transfer"). With transfer=0 the copies are skipped (timing of the compute alone,
// "CPU idealized"; the results on the device are then stale, so that mode is only for the piece timings).
struct CpuEvaluator : Evaluator {
    typedef long long casadi_int;
    typedef int (*eval_t)(const S**, S**, casadi_int*, S*, int);
    typedef int (*work_t)(casadi_int*, casadi_int*, casadi_int*, casadi_int*);
    eval_t fm, fi; int threads, Bmax; bool transfer; std::string desc;
    S *hx, *hY, *hrp, *hpar, *hun, *hG, *hh, *hu, *hx1;
    std::vector<std::vector<S>> w_m, w_i; std::vector<std::vector<casadi_int>> iw_m, iw_i;
    static void* sym(void* h, const char* n) { void* p = dlsym(h, n); if (!p) throw std::runtime_error(std::string("dlsym ") + n + ": " + dlerror()); return p; }
    template <class T> static T* pinned(size_t n) { T* p; CKV(cudaMallocHost(&p, n * sizeof(T))); memset(p, 0, n * sizeof(T)); return p; }
    CpuEvaluator(const std::string& lib, int B_max, int n_threads, bool with_transfer) : threads(n_threads), Bmax(B_max), transfer(with_transfer) {
        void* h = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL); if (!h) throw std::runtime_error(std::string("dlopen ") + lib + ": " + dlerror());
        fm = (eval_t)sym(h, "safe_model"); fi = (eval_t)sym(h, "safe_integrate");
        casadi_int a, r, iwm, wm, iwi, wi;
        ((work_t)sym(h, "safe_model_work"))(&a, &r, &iwm, &wm); ((work_t)sym(h, "safe_integrate_work"))(&a, &r, &iwi, &wi);
        for (int t = 0; t < threads; ++t) { w_m.emplace_back(wm + 1); iw_m.emplace_back(iwm + 1); w_i.emplace_back(wi + 1); iw_i.emplace_back(iwi + 1); }
        hx = pinned<S>((size_t)Bmax * NX); hY = pinned<S>((size_t)Bmax * ND); hrp = pinned<S>((size_t)Bmax * NRP); hpar = pinned<S>((size_t)Bmax * NPAR);
        hun = pinned<S>((size_t)Bmax * NV); hG = pinned<S>((size_t)Bmax * NCBF * NV); hh = pinned<S>((size_t)Bmax * NCBF); hu = pinned<S>((size_t)Bmax * NV); hx1 = pinned<S>((size_t)Bmax * NX);
        desc = "CasADi C codegen on the CPU (" + std::to_string(threads) + " OpenMP threads, " + (transfer ? "GPU -> CPU -> GPU transfer in every call" : "NO transfer: compute-only timing") +
               "): " + lib.substr(lib.rfind('/') + 1) + " (sz_w " + std::to_string(wm) + " / " + std::to_string(wi) + ")";
    }
    int model(int B, const S* x, const S* Y, const S* rp, const S* par, S* u_nom, S* G, S* h, cudaStream_t st) override {
        if (B > Bmax) { g_err = "cpu evaluator: B exceeds the host buffers"; return -1; }
        if (transfer) {
            CK(cudaMemcpyAsync(hx, x, (size_t)B * NX * sizeof(S), cudaMemcpyDeviceToHost, st)); CK(cudaMemcpyAsync(hY, Y, (size_t)B * ND * sizeof(S), cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(hrp, rp, (size_t)B * NRP * sizeof(S), cudaMemcpyDeviceToHost, st)); CK(cudaMemcpyAsync(hpar, par, (size_t)B * NPAR * sizeof(S), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
        }
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int b = 0; b < B; ++b) {
            int t = omp_get_thread_num();
            const S* arg[4] = {hx + (size_t)b * NX, hY + (size_t)b * ND, hrp + (size_t)b * NRP, hpar + (size_t)b * NPAR};
            S* res[3] = {hun + (size_t)b * NV, hG + (size_t)b * NCBF * NV, hh + (size_t)b * NCBF};
            fm(arg, res, iw_m[t].data(), w_m[t].data(), 0);
        }
        if (transfer) {
            CK(cudaMemcpyAsync(u_nom, hun, (size_t)B * NV * sizeof(S), cudaMemcpyHostToDevice, st)); CK(cudaMemcpyAsync(G, hG, (size_t)B * NCBF * NV * sizeof(S), cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(h, hh, (size_t)B * NCBF * sizeof(S), cudaMemcpyHostToDevice, st));
        }
        return 0;
    }
    int integrate(int B, const S* x, const S* u, const S* rp, const S* par, S* x1, cudaStream_t st) override {
        if (B > Bmax) { g_err = "cpu evaluator: B exceeds the host buffers"; return -1; }
        if (transfer) {
            CK(cudaMemcpyAsync(hx, x, (size_t)B * NX * sizeof(S), cudaMemcpyDeviceToHost, st)); CK(cudaMemcpyAsync(hu, u, (size_t)B * NV * sizeof(S), cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(hrp, rp, (size_t)B * NRP * sizeof(S), cudaMemcpyDeviceToHost, st)); CK(cudaMemcpyAsync(hpar, par, (size_t)B * NPAR * sizeof(S), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
        }
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int b = 0; b < B; ++b) {
            int t = omp_get_thread_num();
            const S* arg[4] = {hx + (size_t)b * NX, hu + (size_t)b * NV, hrp + (size_t)b * NRP, hpar + (size_t)b * NPAR};
            S* res[1] = {hx1 + (size_t)b * NX};
            fi(arg, res, iw_i[t].data(), w_i[t].data(), 0);
        }
        if (transfer) CK(cudaMemcpyAsync(x1, hx1, (size_t)B * NX * sizeof(S), cudaMemcpyHostToDevice, st));
        return 0;
    }
    bool capturable() const override { return false; }
    std::string info() const override { return desc; }
};

// G row-major (B, NCBF, 7) -> column-major per instance (B, 7 cols of NCBF)
__global__ void k_transpose_G(const S* __restrict__ src, S* __restrict__ dst, int B) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (t >= (long long)B * NCBF * NV) return;
    int b = (int)(t / (NCBF * NV)), r = (int)(t % (NCBF * NV)); int i = r / NV, j = r % NV;
    dst[(size_t)b * NCBF * NV + i + NCBF * j] = src[t];
}

// frax + cbfpy through the PJRT C API: JAX's own CUDA plugin, the exported StableHLO programs compiled once, inputs as
// views of the driver's device buffers (no copies), outputs copied back on the driver's stream.
struct PjrtEvaluator : Evaluator {
    const PJRT_Api* api = nullptr; PJRT_Client* client = nullptr; PJRT_Device* device = nullptr; PJRT_Memory* memory = nullptr;
    PJRT_LoadedExecutable *exe_m = nullptr, *exe_i = nullptr;
    std::vector<char> copts; int B_m, B_i; std::string desc; double compile_s = 0, last_exec_ms = 0;

    std::string err(PJRT_Error* e) {
        PJRT_Error_Message_Args a{PJRT_Error_Message_Args_STRUCT_SIZE, nullptr, e, nullptr, 0}; api->PJRT_Error_Message(&a);
        std::string m(a.message, a.message_size); PJRT_Error_Destroy_Args d{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr, e}; api->PJRT_Error_Destroy(&d); return m;
    }
#define PJ(call) do { PJRT_Error* e_ = (call); if (e_) throw std::runtime_error(std::string("PJRT: ") + err(e_) + " [" #call "]"); } while (0)
#define PJR(call) do { PJRT_Error* e_ = (call); if (e_) { g_err = std::string("PJRT: ") + err(e_); fprintf(stderr, "%s\n", g_err.c_str()); return -1; } } while (0)

    PjrtEvaluator(const std::string& plugin, const std::string& model_mlir, const std::string& integrate_mlir, const std::string& copts_path, int B_model, int B_integrate)
        : B_m(B_model), B_i(B_integrate) {
        void* hdl = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!hdl) throw std::runtime_error(std::string("dlopen ") + plugin + ": " + dlerror());
        auto get = (const PJRT_Api* (*)())dlsym(hdl, "GetPjrtApi");
        if (!get) throw std::runtime_error("GetPjrtApi not found");
        api = get();
        if (api->PJRT_Plugin_Initialize) { PJRT_Plugin_Initialize_Args a{PJRT_Plugin_Initialize_Args_STRUCT_SIZE, nullptr}; PJ(api->PJRT_Plugin_Initialize(&a)); }
        PJRT_NamedValue opt{PJRT_NamedValue_STRUCT_SIZE, nullptr, "preallocate", strlen("preallocate"), PJRT_NamedValue_kBool, {}, 1};
        opt.bool_value = false;
        PJRT_Client_Create_Args c{}; c.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE; c.create_options = &opt; c.num_options = 1;
        PJ(api->PJRT_Client_Create(&c)); client = c.client;
        PJRT_Client_AddressableDevices_Args d{PJRT_Client_AddressableDevices_Args_STRUCT_SIZE, nullptr, client, nullptr, 0};
        PJ(api->PJRT_Client_AddressableDevices(&d)); if (!d.num_addressable_devices) throw std::runtime_error("no PJRT device"); device = d.addressable_devices[0];
        PJRT_Device_AddressableMemories_Args m{PJRT_Device_AddressableMemories_Args_STRUCT_SIZE, nullptr, device, nullptr, 0};
        PJ(api->PJRT_Device_AddressableMemories(&m)); memory = m.num_memories ? m.memories[0] : nullptr;
        copts = read_file(copts_path);
        auto t0 = std::chrono::steady_clock::now();
        size_t nm, ni; exe_m = compile(model_mlir, nm); exe_i = compile(integrate_mlir, ni);
        compile_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (nm != 3 || ni != 1) throw std::runtime_error("unexpected program outputs");
        desc = "JAX/XLA program via PJRT (" + plugin.substr(plugin.rfind('/') + 1) + ", api " + std::to_string(api->pjrt_api_version.major_version) + "." +
               std::to_string(api->pjrt_api_version.minor_version) + "): " + model_mlir.substr(model_mlir.rfind('/') + 1) + " + " +
               integrate_mlir.substr(integrate_mlir.rfind('/') + 1) + ", compile " + std::to_string(compile_s) + " s";
    }
    PJRT_LoadedExecutable* compile(const std::string& path, size_t& nout) {
        std::vector<char> code = read_file(path);
        PJRT_Program prog{PJRT_Program_STRUCT_SIZE, nullptr, code.data(), code.size(), "mlir", 4};
        PJRT_Client_Compile_Args a{PJRT_Client_Compile_Args_STRUCT_SIZE, nullptr, client, &prog, copts.data(), copts.size(), nullptr};
        PJ(api->PJRT_Client_Compile(&a));
        PJRT_LoadedExecutable_GetExecutable_Args g{PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE, nullptr, a.executable, nullptr};
        PJ(api->PJRT_LoadedExecutable_GetExecutable(&g));
        PJRT_Executable_NumOutputs_Args n{PJRT_Executable_NumOutputs_Args_STRUCT_SIZE, nullptr, g.executable, 0};
        PJ(api->PJRT_Executable_NumOutputs(&n)); nout = n.num_outputs;
        PJRT_Executable_Destroy_Args dd{PJRT_Executable_Destroy_Args_STRUCT_SIZE, nullptr, g.executable}; PJ(api->PJRT_Executable_Destroy(&dd));
        return a.executable;
    }
    PJRT_Buffer* view(const S* ptr, int B, int n) {
        int64_t dims[2] = {B, n};
        PJRT_Client_CreateViewOfDeviceBuffer_Args a{}; a.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
        a.client = client; a.device_buffer_ptr = (void*)ptr; a.dims = dims; a.num_dims = 2;
        a.element_type = sizeof(S) == 4 ? PJRT_Buffer_Type_F32 : PJRT_Buffer_Type_F64; a.layout = nullptr; a.device = device;
        a.on_delete_callback = [](void*, void*) {};     // a null callback makes the SE client throw bad_function_call on destroy
        a.on_delete_callback_arg = nullptr; a.stream = 0; a.memory = memory;   // PJRT accepts only its own DLPack streams here
        PJRT_Error* e = api->PJRT_Client_CreateViewOfDeviceBuffer(&a);
        if (e) { g_err = "PJRT view: " + err(e); return nullptr; }
        return a.buffer;
    }
    int destroy(PJRT_Buffer* b) { PJRT_Buffer_Destroy_Args a{PJRT_Buffer_Destroy_Args_STRUCT_SIZE, nullptr, b}; PJR(api->PJRT_Buffer_Destroy(&a)); return 0; }
    // run exe on 4 inputs (widths in_w) of batch B; copy its outputs to outs[] (device, our stream). JAX's programs produce
    // row-major (B, NCBF, 7) G; cusadi2's G is column-major per instance, so G is transposed on the way out (k_transpose_G).
    int run(PJRT_LoadedExecutable* exe, size_t nout, int B, const S* const* ins, const int* in_w, S* const* outs, const int* out_w, cudaStream_t st) {
        CK(cudaStreamSynchronize(st));                       // inputs ready (PJRT cannot wait on a foreign stream)
        PJRT_Buffer* in[4]; for (int i = 0; i < 4; ++i) { in[i] = view(ins[i], B, in_w[i]); if (!in[i]) { fprintf(stderr, "%s\n", g_err.c_str()); return -1; } }
        std::vector<PJRT_Buffer*> out(nout, nullptr); PJRT_Buffer** out_list = out.data(); PJRT_Buffer* const* in_list = in;
        PJRT_Event* done = nullptr;
        PJRT_ExecuteOptions opt{}; opt.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
        PJRT_LoadedExecutable_Execute_Args a{}; a.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE; a.executable = exe; a.options = &opt;
        a.argument_lists = &in_list; a.num_devices = 1; a.num_args = 4; a.output_lists = &out_list; a.device_complete_events = &done; a.execute_device = nullptr;
        auto t0 = std::chrono::steady_clock::now();
        PJR(api->PJRT_LoadedExecutable_Execute(&a));
        { PJRT_Event_Await_Args w{PJRT_Event_Await_Args_STRUCT_SIZE, nullptr, done}; PJR(api->PJRT_Event_Await(&w));
          PJRT_Event_Destroy_Args dd{PJRT_Event_Destroy_Args_STRUCT_SIZE, nullptr, done}; PJR(api->PJRT_Event_Destroy(&dd)); }
        last_exec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        for (size_t o = 0; o < nout; ++o) {
            PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args p{PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE, nullptr, out[o], nullptr};
            PJR(api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&p));
            if (out_w[o] == NCBF * NV) k_transpose_G<<<(int)(((long long)B * NCBF * NV + 255) / 256), 256, 0, st>>>((const S*)p.device_memory_ptr, outs[o], B);
            else CK(cudaMemcpyAsync(outs[o], p.device_memory_ptr, (size_t)B * out_w[o] * sizeof(S), cudaMemcpyDeviceToDevice, st));
        }
        CK(cudaStreamSynchronize(st));                        // the outputs are PJRT's buffers; free them after the copy landed
        for (size_t o = 0; o < nout; ++o) if (destroy(out[o])) return -1;
        for (auto b : in) if (destroy(b)) return -1;
        return 0;
    }
    int model(int B, const S* x, const S* Y, const S* rp, const S* par, S* u_nom, S* G, S* h, cudaStream_t st) override {
        if (B != B_m) { g_err = "PJRT model program was exported for batch " + std::to_string(B_m); fprintf(stderr, "%s\n", g_err.c_str()); return -1; }
        const S* ins[4] = {x, Y, rp, par}; const int iw[4] = {NX, ND, NRP, NPAR}; S* outs[3] = {u_nom, G, h}; const int ow[3] = {NV, NCBF * NV, NCBF};
        return run(exe_m, 3, B, ins, iw, outs, ow, st);
    }
    int integrate(int B, const S* x, const S* u, const S* rp, const S* par, S* x1, cudaStream_t st) override {
        if (B != B_i) { g_err = "PJRT integrate program was exported for batch " + std::to_string(B_i); fprintf(stderr, "%s\n", g_err.c_str()); return -1; }
        const S* ins[4] = {x, u, rp, par}; const int iw[4] = {NX, NV, NRP, NPAR}; S* outs[1] = {x1}; const int ow[1] = {NX};
        return run(exe_i, 1, B, ins, iw, outs, ow, st);
    }
    bool capturable() const override { return false; }
    std::string info() const override { return desc; }
};

// ------------------------------------------------------------------------------------------------------ MPPI kernels
// The CBF safety filter: one warp per rollout, lane l owns rows l and l + 32 (G entries in registers). `iters` times:
// every lane scores its rows by signed distance, a shuffle reduction finds the most violated row (lowest index on ties),
// the owner lane broadcasts that row, all lanes apply the projection and the torque box. projection_study2.py: 32 greedy
// iterations keep the barriers of 20-step rollouts above -0.01 as often as cbfpy's elastic QP does. G column-major.
__global__ void k_project(const S* __restrict__ u_nom, const S* __restrict__ G, const S* __restrict__ h, const S* __restrict__ u_max,
                          S* __restrict__ u_out, int B, int iters) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; if (wid >= B) return;
    const S* g = G + (size_t)wid * NCBF * NV; const S* hb = h + (size_t)wid * NCBF;
    // lane owns rows lane + 32 s, s < RPL; rows past NCBF are padding (zero row, never selected)
    S u[NV], um[NV], gr[RPL][NV], hr[RPL], ggr[RPL], inr[RPL];
    #pragma unroll
    for (int j = 0; j < NV; ++j) { u[j] = u_nom[(size_t)wid * NV + j]; um[j] = u_max[j]; }
    #pragma unroll
    for (int s = 0; s < RPL; ++s) {
        const int i = lane + 32 * s; const bool live = i < NCBF;
        hr[s] = live ? hb[i] : (S)0; S gg = (S)1e-12;
        #pragma unroll
        for (int j = 0; j < NV; ++j) { gr[s][j] = live ? g[i + NCBF * j] : (S)0; gg += gr[s][j] * gr[s][j]; }
        ggr[s] = gg; inr[s] = rsqrt(gg);
    }
    #pragma unroll
    for (int j = 0; j < NV; ++j) u[j] = fmin(fmax(u[j], -um[j]), um[j]);
    for (int it = 0; it < iters; ++it) {
        // lane-local argmax over the owned rows by signed distance (lowest index on ties), then a warp reduction
        S r = 0, viol = 0, gg = 1; int ri = 0;
        #pragma unroll
        for (int s = 0; s < RPL; ++s) {
            const int i = lane + 32 * s; S d = -hr[s];
            #pragma unroll
            for (int j = 0; j < NV; ++j) d += gr[s][j] * u[j];
            const S rs = d * inr[s];
            if (s == 0 || (i < NCBF && !(r >= rs))) { r = rs; viol = d; gg = ggr[s]; ri = i; }   // RPL = 2: exactly r0 >= r1 ? r0 : r1
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            S r2 = __shfl_down_sync(0xffffffffu, r, off), v2 = __shfl_down_sync(0xffffffffu, viol, off), gg2 = __shfl_down_sync(0xffffffffu, gg, off);
            int i2 = __shfl_down_sync(0xffffffffu, ri, off);
            if (r2 > r || (r2 == r && i2 < ri)) { r = r2; viol = v2; gg = gg2; ri = i2; }
        }
        r = __shfl_sync(0xffffffffu, r, 0); viol = __shfl_sync(0xffffffffu, viol, 0); gg = __shfl_sync(0xffffffffu, gg, 0); ri = __shfl_sync(0xffffffffu, ri, 0);
        if (!(r > 0)) break;                                   // every row satisfied (or NaN rows): done, warp-uniform
        const S step = viol / gg; const int owner = ri & 31, slot = ri >> 5;
        #pragma unroll
        for (int j = 0; j < NV; ++j) {
            S mine = gr[0][j];
            #pragma unroll
            for (int s = 1; s < RPL; ++s) if (s == slot) mine = gr[s][j];
            S gij = __shfl_sync(0xffffffffu, mine, owner);
            u[j] = fmin(fmax(u[j] - step * gij, -um[j]), um[j]);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < NV; ++j) u_out[(size_t)wid * NV + j] = u[j];
    }
}
__global__ void k_sample(const S* __restrict__ Y0, S* __restrict__ Y, const S* __restrict__ sigma, const S* __restrict__ ylim,
                         const unsigned* __restrict__ counter, int B, int K, unsigned long long seed) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    const S* y0 = Y0 + (size_t)(b / K) * ND; S* y = Y + (size_t)b * ND;
    if (b % K == 0) { for (int i = 0; i < ND; ++i) y[i] = y0[i]; return; }
    curandStatePhilox4_32_10_t st; curand_init(seed, (unsigned long long)b, (unsigned long long)counter[0] * 64ull, &st);
    for (int i = 0; i < ND; ++i) y[i] = fmin(fmax(y0[i] + sigma[i / NV] * (S)curand_normal(&st), ylim[i % NV]), ylim[NV + i % NV]);
}
__global__ void k_tick(unsigned* counter) { counter[0] += 1; }
// traj[step][k] = x[k] for the first n_rec samples (sample 0 is the mean plan), for the video
__global__ void k_record(const S* __restrict__ x, S* __restrict__ traj, int step, int n_rec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n_rec * NX) return; traj[(size_t)step * n_rec * NX + i] = x[i];
}
__global__ void k_init(const S* __restrict__ xp, const S* __restrict__ rp_r, S* __restrict__ x, S* __restrict__ rp_b, int B, int K) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= (long long)B * NX) return;
    int b = (int)(i / NX), e = (int)(i - (long long)b * NX);
    x[i] = (e == I_TAU || e == I_COST) ? S(0) : xp[(size_t)(b / K) * NX + e];
    if (e < NRP) rp_b[(size_t)b * NRP + e] = rp_r[(size_t)(b / K) * NRP + e];
}
__global__ void k_replicate(const S* __restrict__ src, S* __restrict__ dst, int n, int B) {   // dst[b] = src for b < B
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= (long long)B * n) return; dst[i] = src[i % n];
}
__global__ void k_upload_x(const double* __restrict__ h, S* __restrict__ x, int R) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= R * NX) return;
    int e = i % NX; x[i] = (e == I_TAU || e == I_COST) ? S(0) : (S)h[i];
}
__global__ void k_cast_in(const double* __restrict__ h, S* __restrict__ d, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) d[i] = (S)h[i]; }
__global__ void k_cast_out(const S* __restrict__ d, double* __restrict__ h, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) h[i] = (double)d[i]; }
__global__ void k_zero_tau(S* xp, int R) { int r = blockIdx.x * blockDim.x + threadIdx.x; if (r < R) { xp[(size_t)r * NX + I_TAU] = 0; xp[(size_t)r * NX + I_COST] = 0; } }
// One block per robot: softmax weights from the sample costs, weighted mean of the samples.
__global__ void k_update(const S* __restrict__ x, const S* __restrict__ Y, S* __restrict__ Y0, S* __restrict__ w, S* __restrict__ stat, int K, S temp) {
    const int r = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const S* c = x + (size_t)r * K * NX + I_COST; S* wr = w + (size_t)r * K;
    __shared__ double s_a[256], s_b[256], s_c[256], s_d[256];
    double sum = 0, sq = 0, mn = 1e300, cnt = 0;
    for (int k = tid; k < K; k += nt) { double v = c[(size_t)k * NX]; if (isfinite(v)) { sum += v; sq += v * v; mn = fmin(mn, v); cnt += 1; } }
    s_a[tid] = sum; s_b[tid] = sq; s_c[tid] = mn; s_d[tid] = cnt; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) { s_a[tid] += s_a[tid + s]; s_b[tid] += s_b[tid + s]; s_c[tid] = fmin(s_c[tid], s_c[tid + s]); s_d[tid] += s_d[tid + s]; }
        __syncthreads();
    }
    const double n = s_d[0], mean = s_a[0] / fmax(n, 1.0), cmin = s_c[0];
    const double sd = sqrt(fmax(s_b[0] / fmax(n, 1.0) - mean * mean, 1e-12));
    __syncthreads();
    double z = 0, z2 = 0;
    for (int k = tid; k < K; k += nt) {
        double v = c[(size_t)k * NX], wk = isfinite(v) ? exp(-(v - cmin) / ((double)temp * sd)) : 0.0;
        wr[k] = (S)wk; z += wk; z2 += wk * wk;
    }
    s_a[tid] = z; s_b[tid] = z2; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) { s_a[tid] += s_a[tid + s]; s_b[tid] += s_b[tid + s]; } __syncthreads(); }
    const double Z = s_a[0], Z2 = s_b[0];
    if (tid == 0) { S* st = stat + (size_t)r * 4; st[0] = (S)cmin; st[1] = (S)mean; st[2] = (S)sd; st[3] = (S)(Z > 0 ? Z * Z / Z2 : 0); }
    if (!(Z > 0)) return;
    for (int j = tid; j < ND; j += nt) {
        double acc = 0; const S* y = Y + (size_t)r * K * ND + j;
        for (int k = 0; k < K; ++k) acc += (double)wr[k] * y[(size_t)k * ND];
        Y0[(size_t)r * ND + j] = (S)(acc / Z);
    }
}
__global__ void k_shift(S* Y0, int R, S a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= R * NV) return;
    S* y = Y0 + (size_t)(i / NV) * ND + i % NV;
    for (int j = 0; j + 1 < NKNOT; ++j) y[j * NV] += a * (y[(j + 1) * NV] - y[j * NV]);
}

// ------------------------------------------------------------------------------------------------------ driver
struct Driver {
    int R, K, T, B, proj_iters, n_rec; bool use_graph; S temp; unsigned long long seed; S shift_a;
    std::unique_ptr<Evaluator> ev; cudaStream_t st; cudaGraphExec_t ge = nullptr; cudaGraph_t g = nullptr;
    S *xp, *xp1, *Y0, *rp_r, *par_r, *un_r, *G_r, *h_r, *u_r;                 // robot side (R)
    S *x[2], *Y, *rp_b, *par_b, *un_b, *G_b, *h_b, *u_b, *w, *stat;            // sample side (B = R K)
    S *sigma, *ylim, *umax, *traj; unsigned* counter;
    double *h_x_in, *h_rp_in, *h_u_out, *h_stat_out, *d_x_in, *d_rp_in, *d_u_out, *d_stat_out;
    cudaEvent_t ev0, ev1; bool graph_ready = false, captured_external = true; std::string info_str;

    Driver(std::map<std::string, std::string> c) {
        auto get = [&](const char* k, const char* def) { return c.count(k) ? c[k] : std::string(def); };
        R = atoi(get("R", "1").c_str()); K = atoi(get("K", "256").c_str()); T = atoi(get("T", "20").c_str()); B = R * K;
        proj_iters = atoi(get("proj_iters", "48").c_str()); n_rec = std::min(K, atoi(get("record", "32").c_str()));
        use_graph = atoi(get("graph", "1").c_str()) != 0; temp = (S)atof(get("temp", "0.3").c_str()); seed = (unsigned long long)atoll(get("seed", "1").c_str());
        double dt = atof(get("dt", "0.02").c_str()), kdt = T * dt / (NKNOT - 1); shift_a = (S)(dt / kdt);
        std::string kind = get("evaluator", "cusadi2");
        if (kind == "cusadi2") ev.reset(new Cusadi2Evaluator(c.at("model_plan"), c.at("integrate_plan")));
        else if (kind == "cusadi") ev.reset(new CusadiUnrolledEvaluator(c.at("model_lib"), c.at("integrate_lib"), B, atoi(get("block", "256").c_str())));
        else if (kind == "cpu") ev.reset(new CpuEvaluator(c.at("cpu_lib"), B, atoi(get("cpu_threads", "16").c_str()), atoi(get("cpu_transfer", "1").c_str()) != 0));
        else if (kind == "pjrt") {
            if (R != 1) throw std::runtime_error("the PJRT evaluator runs the plant step on the sample-batch program: use R = 1");
            ev.reset(new PjrtEvaluator(c.at("plugin"), c.at("model_mlir"), c.at("integrate_mlir"), c.at("compile_options"), B, B));
        } else throw std::runtime_error("unknown evaluator " + kind);
        use_graph = use_graph && ev->capturable();
        CKV(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        xp = dalloc((size_t)R * NX); xp1 = dalloc((size_t)R * NX); Y0 = dalloc((size_t)R * ND); rp_r = dalloc((size_t)R * NRP); par_r = dalloc((size_t)R * NPAR);
        un_r = dalloc((size_t)R * NV); G_r = dalloc((size_t)R * NCBF * NV); h_r = dalloc((size_t)R * NCBF); u_r = dalloc((size_t)R * NV);
        for (int p = 0; p < 2; ++p) x[p] = dalloc((size_t)B * NX);
        Y = dalloc((size_t)B * ND); rp_b = dalloc((size_t)B * NRP); par_b = dalloc((size_t)B * NPAR);
        un_b = dalloc((size_t)B * NV); G_b = dalloc((size_t)B * NCBF * NV); h_b = dalloc((size_t)B * NCBF); u_b = dalloc((size_t)B * NV);
        w = dalloc(B); stat = dalloc((size_t)R * 4); traj = dalloc((size_t)(T + 1) * n_rec * NX);
        std::vector<double> sig = parse_list(get("sigma", "0.1,0.1,0.1,0.1")), yl = parse_list(c.at("ylim")), par = parse_list(c.at("par")), um = parse_list(c.at("umax"));
        if ((int)sig.size() != NKNOT || (int)yl.size() != 2 * NV || (int)par.size() != NPAR || (int)um.size() != NV) throw std::runtime_error("sigma / ylim / par / umax length");
        std::vector<S> t;
        t.assign(sig.begin(), sig.end()); sigma = dalloc(NKNOT); CKV(cudaMemcpy(sigma, t.data(), NKNOT * sizeof(S), cudaMemcpyHostToDevice));
        t.assign(yl.begin(), yl.end()); ylim = dalloc(2 * NV); CKV(cudaMemcpy(ylim, t.data(), 2 * NV * sizeof(S), cudaMemcpyHostToDevice));
        t.assign(um.begin(), um.end()); umax = dalloc(NV); CKV(cudaMemcpy(umax, t.data(), NV * sizeof(S), cudaMemcpyHostToDevice));
        std::vector<S> pr((size_t)R * NPAR), pb((size_t)B * NPAR);
        for (int r = 0; r < R; ++r) for (int e = 0; e < NPAR; ++e) pr[(size_t)r * NPAR + e] = (S)par[e];
        for (int b = 0; b < B; ++b) for (int e = 0; e < NPAR; ++e) pb[(size_t)b * NPAR + e] = (S)par[e];
        CKV(cudaMemcpy(par_r, pr.data(), pr.size() * sizeof(S), cudaMemcpyHostToDevice)); CKV(cudaMemcpy(par_b, pb.data(), pb.size() * sizeof(S), cudaMemcpyHostToDevice));
        CKV(cudaMalloc(&counter, sizeof(unsigned))); CKV(cudaMemset(counter, 0, sizeof(unsigned)));
        CKV(cudaMallocHost(&h_x_in, (size_t)R * NX * 8)); CKV(cudaMallocHost(&h_rp_in, (size_t)R * NRP * 8));
        CKV(cudaMallocHost(&h_u_out, (size_t)R * NV * 8)); CKV(cudaMallocHost(&h_stat_out, (size_t)R * 4 * 8));
        CKV(cudaMalloc(&d_x_in, (size_t)R * NX * 8)); CKV(cudaMalloc(&d_rp_in, (size_t)R * NRP * 8)); CKV(cudaMalloc(&d_u_out, (size_t)R * NV * 8)); CKV(cudaMalloc(&d_stat_out, (size_t)R * 4 * 8));
        CKV(cudaEventCreate(&ev0)); CKV(cudaEventCreate(&ev1));
        info_str = ev->info() + " | R=" + std::to_string(R) + " K=" + std::to_string(K) + " T=" + std::to_string(T) + " proj_iters=" + std::to_string(proj_iters) + " graph=" + std::to_string(use_graph);
    }
    int project(int Bn, const S* un, const S* G, const S* h, S* u) {
        k_project<<<(Bn * 32 + 127) / 128, 128, 0, st>>>(un, G, h, umax, u, Bn, proj_iters); return 0;
    }
    // the T-step rollout of the B samples from x[0]; p_out = index of the buffer holding the final states
    int rollout(int& p_out) {
        int p = 0;
        k_record<<<(n_rec * NX + 255) / 256, 256, 0, st>>>(x[0], traj, 0, n_rec);
        for (int s = 0; s < T; ++s, p ^= 1) {
            if (ev->model(B, x[p], Y, rp_b, par_b, un_b, G_b, h_b, st)) return -1;
            if (project(B, un_b, G_b, h_b, u_b)) return -1;
            if (ev->integrate(B, x[p], u_b, rp_b, par_b, x[1 - p], st)) return -1;
            k_record<<<(n_rec * NX + 255) / 256, 256, 0, st>>>(x[1 - p], traj, s + 1, n_rec);
        }
        p_out = p; return 0;
    }
    // the plant-side filtered torque of the mean plan (and the model plant's next state when `advance`). The PJRT
    // programs are compiled for batch B, so for that evaluator the robot's state and plan are replicated over the
    // sample buffers and sample 0 is read back.
    int plant_step(bool advance) {
        if (ev->capturable()) {
            if (ev->model(R, xp, Y0, rp_r, par_r, un_r, G_r, h_r, st)) return -1;
            if (project(R, un_r, G_r, h_r, u_r)) return -1;
            if (advance && ev->integrate(R, xp, u_r, rp_r, par_r, xp1, st)) return -1;
        } else {
            k_init<<<(int)(((long long)B * NX + 255) / 256), 256, 0, st>>>(xp, rp_r, x[0], rp_b, B, K);
            k_replicate<<<(int)(((long long)B * ND + 255) / 256), 256, 0, st>>>(Y0, Y, ND, B);
            if (ev->model(B, x[0], Y, rp_b, par_b, un_b, G_b, h_b, st)) return -1;
            if (project(B, un_b, G_b, h_b, u_b)) return -1;
            CK(cudaMemcpyAsync(u_r, u_b, NV * sizeof(S), cudaMemcpyDeviceToDevice, st));
            if (advance) { if (ev->integrate(B, x[0], u_b, rp_b, par_b, x[1], st)) return -1; CK(cudaMemcpyAsync(xp1, x[1], NX * sizeof(S), cudaMemcpyDeviceToDevice, st)); }
        }
        return 0;
    }
    int body(bool external_plant) {
        if (external_plant) {
            CK(cudaMemcpyAsync(d_x_in, h_x_in, (size_t)R * NX * 8, cudaMemcpyHostToDevice, st));
            k_upload_x<<<(R * NX + 255) / 256, 256, 0, st>>>(d_x_in, xp, R);
        } else {
            k_zero_tau<<<(R + 255) / 256, 256, 0, st>>>(xp, R);
        }
        CK(cudaMemcpyAsync(d_rp_in, h_rp_in, (size_t)R * NRP * 8, cudaMemcpyHostToDevice, st));
        k_cast_in<<<(R * NRP + 255) / 256, 256, 0, st>>>(d_rp_in, rp_r, R * NRP);
        k_sample<<<(B + 255) / 256, 256, 0, st>>>(Y0, Y, sigma, ylim, counter, B, K, seed);
        k_tick<<<1, 1, 0, st>>>(counter);
        k_init<<<(int)(((long long)B * NX + 255) / 256), 256, 0, st>>>(xp, rp_r, x[0], rp_b, B, K);
        int p; if (rollout(p)) return -1;
        k_update<<<R, 256, 0, st>>>(x[p], Y, Y0, w, stat, K, temp);
        if (plant_step(!external_plant)) return -1;
        k_shift<<<(R * NV + 255) / 256, 256, 0, st>>>(Y0, R, shift_a);
        k_cast_out<<<(R * NV + 255) / 256, 256, 0, st>>>(u_r, d_u_out, R * NV);
        k_cast_out<<<(R * 4 + 255) / 256, 256, 0, st>>>(stat, d_stat_out, R * 4);
        CK(cudaMemcpyAsync(h_u_out, d_u_out, (size_t)R * NV * 8, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_stat_out, d_stat_out, (size_t)R * 4 * 8, cudaMemcpyDeviceToHost, st));
        if (!external_plant) CK(cudaMemcpyAsync(xp, xp1, (size_t)R * NX * sizeof(S), cudaMemcpyDeviceToDevice, st));
        return 0;
    }
    int capture(bool external_plant) {
        if (body(external_plant)) return -1; CK(cudaStreamSynchronize(st));         // warm-up builds the pointer tables
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal)); if (body(external_plant)) return -1; CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0)); graph_ready = true; return 0;
    }
    int tick(const double* x_plant, const double* wp, double* u_out, double* stat_out, double* gpu_ms, double* wall_ms) {
        bool external = x_plant != nullptr;
        if (external) memcpy(h_x_in, x_plant, (size_t)R * NX * 8);
        memcpy(h_rp_in, wp, (size_t)R * NRP * 8);
        auto t0 = std::chrono::steady_clock::now();
        CK(cudaEventRecord(ev0, st));
        if (use_graph) {
            if (!graph_ready || captured_external != external) {
                if (ge) { cudaGraphExecDestroy(ge); cudaGraphDestroy(g); ge = nullptr; }
                if (capture(external)) return -1; captured_external = external;
            }
            CK(cudaGraphLaunch(ge, st));
        } else if (body(external)) return -1;
        CK(cudaEventRecord(ev1, st)); CK(cudaEventSynchronize(ev1));
        float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1)); if (gpu_ms) *gpu_ms = ms;
        if (wall_ms) *wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        memcpy(u_out, h_u_out, (size_t)R * NV * 8); if (stat_out) memcpy(stat_out, h_stat_out, (size_t)R * 4 * 8);
        return 0;
    }
    template <class F> int timed(int n_calls, F fn, double* median_ms, double* min_ms) {
        if (fn()) return -1; CK(cudaStreamSynchronize(st));
        std::vector<float> ms(n_calls);
        for (int i = 0; i < n_calls; ++i) {
            CK(cudaEventRecord(ev0, st)); if (fn()) return -1; CK(cudaEventRecord(ev1, st)); CK(cudaEventSynchronize(ev1));
            CK(cudaEventElapsedTime(&ms[i], ev0, ev1));
        }
        std::sort(ms.begin(), ms.end()); *median_ms = ms[n_calls / 2]; *min_ms = ms[0]; return 0;
    }
    int time_rollout(int n, double* med, double* mn) { return timed(n, [&]() { int p; return rollout(p); }, med, mn); }
    int time_model(int n, double* med, double* mn) { return timed(n, [&]() { return ev->model(B, x[0], Y, rp_b, par_b, un_b, G_b, h_b, st); }, med, mn); }
    int time_integrate(int n, double* med, double* mn) { return timed(n, [&]() { return ev->integrate(B, x[0], u_b, rp_b, par_b, x[1], st); }, med, mn); }
    int time_project(int n, double* med, double* mn) { return timed(n, [&]() { return project(B, un_b, G_b, h_b, u_b); }, med, mn); }
    int get(const S* d, double* h, size_t n) {
        std::vector<S> t(n); CK(cudaMemcpy(t.data(), d, n * sizeof(S), cudaMemcpyDeviceToHost)); for (size_t i = 0; i < n; ++i) h[i] = (double)t[i]; return 0;
    }
    int set(S* d, const double* h, size_t n) {
        std::vector<S> t(h, h + n); CK(cudaMemcpy(d, t.data(), n * sizeof(S), cudaMemcpyHostToDevice)); return 0;
    }
};

std::map<std::string, std::string> parse_cfg(const char* cfg) {
    std::map<std::string, std::string> m; std::stringstream ss(cfg); std::string item;
    while (std::getline(ss, item, ';')) { auto eq = item.find('='); if (eq != std::string::npos) m[item.substr(0, eq)] = item.substr(eq + 1); }
    return m;
}
}  // namespace

extern "C" {
void* sm_create(const char* cfg) {
    try { return new Driver(parse_cfg(cfg)); }
    catch (const std::exception& e) { g_err = e.what(); fprintf(stderr, "sm_create: %s\n", e.what()); return nullptr; }
}
int sm_tick(void* h, const double* x_plant, const double* wp, double* u_out, double* stat_out, double* gpu_ms, double* wall_ms) {
    return ((Driver*)h)->tick(x_plant, wp, u_out, stat_out, gpu_ms, wall_ms);
}
int sm_time_rollout(void* h, int n, double* med, double* mn) { return ((Driver*)h)->time_rollout(n, med, mn); }
int sm_time_model(void* h, int n, double* med, double* mn) { return ((Driver*)h)->time_model(n, med, mn); }
int sm_time_integrate(void* h, int n, double* med, double* mn) { return ((Driver*)h)->time_integrate(n, med, mn); }
int sm_time_project(void* h, int n, double* med, double* mn) { return ((Driver*)h)->time_project(n, med, mn); }
int sm_get_plan(void* h, double* Y0) { Driver* d = (Driver*)h; return d->get(d->Y0, Y0, (size_t)d->R * ND); }
int sm_set_plan(void* h, const double* Y0) { Driver* d = (Driver*)h; return d->set(d->Y0, Y0, (size_t)d->R * ND); }
int sm_get_plant(void* h, double* x) { Driver* d = (Driver*)h; return d->get(d->xp, x, (size_t)d->R * NX); }
int sm_set_plant(void* h, const double* x) { Driver* d = (Driver*)h; return d->set(d->xp, x, (size_t)d->R * NX); }
int sm_get_samples(void* h, double* Y, double* xT) {       // the last tick's samples and the buffer x[T % 2] holding their final states
    Driver* d = (Driver*)h; if (d->get(d->Y, Y, (size_t)d->B * ND)) return -1; return d->get(d->x[d->T % 2], xT, (size_t)d->B * NX);
}
int sm_get_filter(void* h, double* u_nom, double* G, double* hh, double* u) {   // the last projection on the sample buffers
    Driver* d = (Driver*)h;
    if (d->get(d->un_b, u_nom, (size_t)d->B * NV) || d->get(d->G_b, G, (size_t)d->B * NCBF * NV) || d->get(d->h_b, hh, (size_t)d->B * NCBF)) return -1;
    return d->get(d->u_b, u, (size_t)d->B * NV);
}
int sm_get_traj(void* h, double* out) {         // (T+1, n_rec, 16) state trajectories of the first n_rec samples of the last tick
    Driver* d = (Driver*)h; return d->get(d->traj, out, (size_t)(d->T + 1) * d->n_rec * NX);
}
// Direct evaluator calls on the sample buffers (verification: the evaluator exactly as the rollout calls it, on host-given
// inputs of batch B). Overwrites x[0], Y / u_b, rp_b, par_b and the outputs.
int sm_eval_model(void* h, const double* x, const double* Y, const double* rp, const double* par, double* u_nom, double* G, double* hh) {
    Driver* d = (Driver*)h; const size_t B = d->B;
    if (d->set(d->x[0], x, B * NX) || d->set(d->Y, Y, B * ND) || d->set(d->rp_b, rp, B * NRP) || d->set(d->par_b, par, B * NPAR)) return -1;
    if (d->ev->model(d->B, d->x[0], d->Y, d->rp_b, d->par_b, d->un_b, d->G_b, d->h_b, d->st)) return -1;
    CK(cudaStreamSynchronize(d->st));
    if (d->get(d->un_b, u_nom, B * NV) || d->get(d->G_b, G, B * NCBF * NV)) return -1;
    return d->get(d->h_b, hh, B * NCBF);
}
int sm_eval_integrate(void* h, const double* x, const double* u, const double* rp, const double* par, double* x1) {
    Driver* d = (Driver*)h; const size_t B = d->B;
    if (d->set(d->x[0], x, B * NX) || d->set(d->u_b, u, B * NV) || d->set(d->rp_b, rp, B * NRP) || d->set(d->par_b, par, B * NPAR)) return -1;
    if (d->ev->integrate(d->B, d->x[0], d->u_b, d->rp_b, d->par_b, d->x[1], d->st)) return -1;
    CK(cudaStreamSynchronize(d->st));
    return d->get(d->x[1], x1, B * NX);
}
int sm_get_weights(void* h, double* out) { Driver* d = (Driver*)h; return d->get(d->w, out, (size_t)d->B); }
int sm_n_rec(void* h) { return ((Driver*)h)->n_rec; }
const char* sm_info(void* h) { return ((Driver*)h)->info_str.c_str(); }
const char* sm_error() { return g_err.c_str(); }
double sm_last_pjrt_exec_ms(void* h) { auto* p = dynamic_cast<PjrtEvaluator*>(((Driver*)h)->ev.get()); return p ? p->last_exec_ms : -1.0; }
void sm_destroy(void* h) { delete (Driver*)h; }
}
