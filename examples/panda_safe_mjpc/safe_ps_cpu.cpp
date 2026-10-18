// The safe predictive-sampling controller of safe_mppi.cu entirely on the CPU: the "CPU idealized" row.
//
// Same controller, same CasADi functions (CasADi's C code generation, build/cpu/libcpu_T{T}_{dtype}.so), same greedy
// projection, same best-sample selection, same cubic resample, same model plant step, no GPU and therefore no transfer
// anywhere. Per iteration the K rollouts are split statically over `threads` OpenMP threads; each thread runs its rollouts
// through all T steps (model -> projection -> integrate) with no synchronisation until the whole batch is done, which is
// the best a CPU implementation can do (one rollout per thread at a time, 16 threads on this machine). n_iter iterations
// per tick, the best sample of each becoming the nominal of the next (MJPC's OptimizePolicy, as the GPU driver).
//
// C API (cpu_driver.py wraps it with ctypes; host arrays are double):
//   void* spc_create(const char* cfg)   cfg = "key=value;..." : cpu_lib, K, T, n_iter, threads, seed, dt, proj_iters, keep_nominal, pin,
//                                       sigma=ND values (per knot x joint), ylim=lo(7),hi(7), par=6 values, umax=7 values
//   int spc_tick(h, const double* x_plant[16] or NULL, const double* wp[3], double* u[7], double* stat[4], double* ms)
//   int spc_set_samples(h, const double* Y[K*ND])   the next iteration uses these samples instead of drawing its own
//   spc_get_samples / spc_get_plan / spc_set_plan / spc_get_plant / spc_set_plant / spc_get_iter_stats (n_iter x 7)
//   int spc_time_iteration / spc_time_rollout / spc_time_model / spc_time_integrate / spc_time_project (h, n_calls, double* median_ms, double* min_ms)
//   spc_info / spc_error / spc_destroy
#include <dlfcn.h>
#include <omp.h>
#include <sched.h>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SCALAR
#define SCALAR double
#endif
using S = SCALAR;

namespace {
constexpr int NV = 7, NX = 16, NKNOT = 7, ND = NKNOT * NV, NPAR = 6, NRP = 3, I_TAU = 2 * NV, I_COST = 2 * NV + 1, NCBF = 64, NISTAT = 7;
typedef long long casadi_int;
typedef int (*eval_t)(const S**, S**, casadi_int*, S*, int);
typedef int (*work_t)(casadi_int*, casadi_int*, casadi_int*, casadi_int*);
std::string g_err;

std::map<std::string, std::string> parse_cfg(const char* s) {
    std::map<std::string, std::string> m; std::stringstream ss(s); std::string kv;
    while (std::getline(ss, kv, ';')) { auto p = kv.find('='); if (p != std::string::npos) m[kv.substr(0, p)] = kv.substr(p + 1); }
    return m;
}
std::vector<double> parse_list(const std::string& s) {
    std::vector<double> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(atof(t.c_str()));
    return v;
}
inline S clampS(S v, S lo, S hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Physical cores of this machine from sysfs: core index -> the logical CPUs (SMT siblings) of that core, in core_id order.
std::vector<std::vector<int>> physical_cores() {
    std::map<int, std::vector<int>> by_core;
    for (int cpu = 0; cpu < 4096; ++cpu) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/core_id");
        if (!f) break;
        int core = -1; f >> core; if (core >= 0) by_core[core].push_back(cpu);
    }
    std::vector<std::vector<int>> cores;
    for (auto& kv : by_core) cores.push_back(kv.second);
    return cores;
}
// Pin the calling OpenMP thread t of n to one physical core (both of its SMT siblings allowed), threads spread evenly
// over the cores in order (the binding OMP_PROC_BIND=close OMP_PLACES=cores would give). Unpinned threads on an SMT
// machine land two-per-core or migrate at the scheduler's whim, which makes the small-batch tick bimodal.
void pin_thread(int t, int n, const std::vector<std::vector<int>>& cores) {
    if (cores.empty()) return;
    const auto& cpus = cores[(size_t)((long long)t * (long long)cores.size() / n) % cores.size()];
    cpu_set_t s; CPU_ZERO(&s);
    for (int c : cpus) CPU_SET(c, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

// The greedy projection of k_project: `iters` times project onto the most violated row (signed distance, lowest index on
// ties), then clip to the torque box; exit when no row is violated. G column-major per instance: G(i, j) = G[i + NCBF j].
inline void project(const S* un, const S* G, const S* h, const S* um, S* u, int iters) {
    S gg[NCBF], inv[NCBF];
    for (int i = 0; i < NCBF; ++i) {
        S s = (S)1e-12;
        for (int j = 0; j < NV; ++j) s += G[i + NCBF * j] * G[i + NCBF * j];
        gg[i] = s; inv[i] = (S)1 / std::sqrt(s);
    }
    for (int j = 0; j < NV; ++j) u[j] = clampS(un[j], -um[j], um[j]);
    for (int it = 0; it < iters; ++it) {
        S best = -INFINITY, bd = 0; int bi = -1;
        for (int i = 0; i < NCBF; ++i) {
            S d = -h[i];
            for (int j = 0; j < NV; ++j) d += G[i + NCBF * j] * u[j];
            S r = d * inv[i];
            if (r > best) { best = r; bd = d; bi = i; }
        }
        if (!(best > 0)) break;
        S step = bd / gg[bi];
        for (int j = 0; j < NV; ++j) u[j] = clampS(u[j] - step * G[bi + NCBF * j], -um[j], um[j]);
    }
}

// MJPC's cubic spline (TimeSpline::Sample, kCubicSpline) as weights on the NKNOT knots at plan clock tau, knot j at j * kdt;
// the formula of safe_mppi.cu's cubic_weights / model.interp_weights.
inline void cubic_weights(S tau, S kdt, S* w) {
    S s = clampS(tau / kdt, (S)0, (S)(NKNOT - 1)); S segf = std::min(std::floor(s), (S)(NKNOT - 2)); int seg = (int)segf; S t = s - segf;
    S t2 = t * t, t3 = t2 * t;
    S h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
    for (int j = 0; j < NKNOT; ++j) w[j] = 0;
    w[seg] += h00; w[seg + 1] += h01;
    if (seg == 0) { w[1] += h10; w[0] -= h10; } else { w[seg + 1] += (S)0.5 * h10; w[seg - 1] -= (S)0.5 * h10; }
    if (seg + 1 == NKNOT - 1) { w[seg + 1] += h11; w[seg] -= h11; } else { w[seg + 2] += (S)0.5 * h11; w[seg] -= (S)0.5 * h11; }
}

struct Driver {
    int K, T, n_iter, threads, proj_iters, keep_nominal; double dt, kdt; unsigned long long seed; unsigned iter_count = 0;
    eval_t fm, fi; std::vector<std::vector<S>> w_m, w_i; std::vector<std::vector<casadi_int>> iw_m, iw_i;
    std::vector<S> sigma, ylim, umax, par, xp, xp1, Y0, rp, Y, xT, u0, x1, u_r, stat, istat;
    bool given_samples = false, pin = true; int n_cores = 0; std::string info_str;

    static void* sym(void* h, const char* n) { void* p = dlsym(h, n); if (!p) throw std::runtime_error(std::string("dlsym ") + n + ": " + dlerror()); return p; }
    explicit Driver(const char* cfg_s) {
        auto c = parse_cfg(cfg_s);
        auto get = [&](const char* k, const char* d) { auto it = c.find(k); return it == c.end() ? std::string(d) : it->second; };
        K = atoi(c.at("K").c_str()); T = atoi(c.at("T").c_str()); n_iter = atoi(get("n_iter", "1").c_str()); threads = atoi(get("threads", "16").c_str());
        if (n_iter < 1) throw std::runtime_error("n_iter >= 1");
        seed = (unsigned long long)atoll(get("seed", "1").c_str()); keep_nominal = atoi(get("keep_nominal", "1").c_str());
        dt = atof(get("dt", "0.02").c_str()); proj_iters = atoi(get("proj_iters", "48").c_str());
        kdt = (T - 1) * dt / (NKNOT - 1);                                                           // MJPC: knots spread over (T - 1) dt
        std::vector<double> sg = parse_list(c.at("sigma")), yl = parse_list(c.at("ylim")), pr = parse_list(c.at("par")), um = parse_list(c.at("umax"));
        if ((int)sg.size() != ND || (int)yl.size() != 2 * NV || (int)pr.size() != NPAR || (int)um.size() != NV) throw std::runtime_error("sigma / ylim / par / umax length");
        sigma.assign(sg.begin(), sg.end()); ylim.assign(yl.begin(), yl.end()); par.assign(pr.begin(), pr.end()); umax.assign(um.begin(), um.end());
        std::string lib = c.at("cpu_lib");
        void* h = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL); if (!h) throw std::runtime_error(std::string("dlopen ") + lib + ": " + dlerror());
        fm = (eval_t)sym(h, "safe_model"); fi = (eval_t)sym(h, "safe_integrate");
        casadi_int a, r, iwm, wm, iwi, wi;
        ((work_t)sym(h, "safe_model_work"))(&a, &r, &iwm, &wm); ((work_t)sym(h, "safe_integrate_work"))(&a, &r, &iwi, &wi);
        for (int t = 0; t < threads; ++t) { w_m.emplace_back(wm + 1); iw_m.emplace_back(iwm + 1); w_i.emplace_back(wi + 1); iw_i.emplace_back(iwi + 1); }
        xp.assign(NX, 0); xp1.assign(NX, 0); Y0.assign(ND, 0); rp.assign(NRP, 0); Y.assign((size_t)K * ND, 0); xT.assign((size_t)K * NX, 0);
        u0.assign((size_t)K * NV, 0); x1.assign((size_t)K * NX, 0); u_r.assign(NV, 0); stat.assign(4, 0); istat.assign((size_t)n_iter * NISTAT, 0);
        pin = atoi(get("pin", "1").c_str()) != 0;
        if (pin) {
            auto cores = physical_cores();
            #pragma omp parallel num_threads(threads)
            pin_thread(omp_get_thread_num(), threads, cores);          // libgomp keeps its thread pool, so the affinity sticks
            n_cores = (int)cores.size();
        }
        info_str = "CPU idealized: CasADi C codegen, " + std::to_string(threads) + " OpenMP threads" + (pin ? " pinned over " + std::to_string(n_cores) + " physical cores" : " (unpinned)") +
                   ", whole controller on the CPU, no transfer: " +
                   lib.substr(lib.rfind('/') + 1) + " (sz_w " + std::to_string(wm) + " / " + std::to_string(wi) + ") | K=" + std::to_string(K) + " T=" + std::to_string(T) +
                   " n_iter=" + std::to_string(n_iter) + " proj_iters=" + std::to_string(proj_iters) + " keep_nominal=" + std::to_string(keep_nominal) + " scalar=" + (sizeof(S) == 4 ? "float" : "double");
    }
    // Y[k] = clip(Y0 + sigma eps, joint range); sample 0 is Y0 (keep_nominal). One generator per rollout, keyed by (seed, k, iteration).
    void sample() {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int k = 0; k < K; ++k) {
            S* y = Y.data() + (size_t)k * ND;
            if (keep_nominal && k == 0) { std::copy(Y0.begin(), Y0.end(), y); continue; }
            std::mt19937_64 g(seed * 0x9E3779B97F4A7C15ULL ^ ((unsigned long long)k << 20) ^ ((unsigned long long)iter_count << 44));
            std::normal_distribution<double> n(0.0, 1.0);
            for (int i = 0; i < ND; ++i) y[i] = clampS((S)(Y0[i] + sigma[i] * n(g)), ylim[i % NV], ylim[NV + i % NV]);
        }
    }
    inline void model_one(int t, const S* x, const S* y, S* un, S* G, S* h) {
        const S* arg[4] = {x, y, rp.data(), par.data()}; S* res[3] = {un, G, h};
        fm(arg, res, iw_m[t].data(), w_m[t].data(), 0);
    }
    inline void integrate_one(int t, const S* x, const S* u, S* x1) {
        const S* arg[4] = {x, u, rp.data(), par.data()}; S* res[1] = {x1};
        fi(arg, res, iw_i[t].data(), w_i[t].data(), 0);
    }
    // the T-step rollout of the K samples from the plant state; each thread carries its rollouts through all T steps.
    // Step 0's filtered torque and next state are kept per rollout: the selected sample's become the applied torque and
    // (model plant) the next plant state, as the GPU driver's u0_b / x1_b.
    void rollout() {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int k = 0; k < K; ++k) {
            int t = omp_get_thread_num();
            S x[NX], xn[NX], un[NV], G[NCBF * NV], h[NCBF], u[NV];
            std::copy(xp.begin(), xp.end(), x); x[I_TAU] = 0; x[I_COST] = 0;
            const S* y = Y.data() + (size_t)k * ND;
            for (int s = 0; s < T; ++s) {
                model_one(t, x, y, un, G, h);
                project(un, G, h, umax.data(), u, proj_iters);
                integrate_one(t, x, u, xn);
                if (s == 0) { std::copy(u, u + NV, u0.data() + (size_t)k * NV); std::copy(xn, xn + NX, x1.data() + (size_t)k * NX); }
                std::copy(xn, xn + NX, x);
            }
            std::copy(x, x + NX, xT.data() + (size_t)k * NX);
        }
    }
    // k_select: cost statistics (double accumulation) and the best sample (lowest finite cost, lowest index on ties;
    // sample 0 if no cost is finite); its plan becomes the nominal, its step-0 torque the applied torque.
    void select(int it) {
        double sum = 0, sq = 0, mn = 1e300, cnt = 0; int ks = 0x7fffffff;
        for (int k = 0; k < K; ++k) {
            double v = xT[(size_t)k * NX + I_COST];
            if (std::isfinite(v)) { sum += v; sq += v * v; cnt += 1; if (v < mn || (v == mn && k < ks)) { mn = v; ks = k; } }
        }
        double n = cnt, mean = sum / std::max(n, 1.0), sd = std::sqrt(std::max(sq / std::max(n, 1.0) - mean * mean, 0.0));
        if (!(n > 0)) ks = 0;
        stat[0] = (S)(n > 0 ? xT[(size_t)ks * NX + I_COST] : NAN); stat[1] = (S)mean; stat[2] = (S)sd; stat[3] = (S)ks;
        S* si = istat.data() + (size_t)it * NISTAT; si[0] = stat[0]; si[1] = stat[1]; si[2] = stat[2]; si[3] = stat[3]; si[4] = xT[I_COST]; si[5] = 0; si[6] = 0;
        std::copy(Y.begin() + (size_t)ks * ND, Y.begin() + (size_t)(ks + 1) * ND, Y0.begin());
        std::copy(u0.begin() + (size_t)ks * NV, u0.begin() + (size_t)(ks + 1) * NV, u_r.begin());
        std::copy(x1.begin() + (size_t)ks * NX, x1.begin() + (size_t)(ks + 1) * NX, xp1.begin());
    }
    // one planner iteration (MJPC's OptimizePolicy): sample around the nominal, roll out, the best sample becomes the nominal
    void iteration(int it) {
        if (!given_samples) sample();
        given_samples = false; iter_count += 1;
        rollout();
        select(it);
    }
    // MJPC's UpdateNominalPolicy one control period later: knot j <- spline(dt + j * kdt) of the current plan
    void resample() {
        S old[NKNOT], w[NKNOT];
        for (int i = 0; i < NV; ++i) {
            for (int j = 0; j < NKNOT; ++j) old[j] = Y0[(size_t)j * NV + i];
            for (int k = 0; k < NKNOT; ++k) {
                cubic_weights((S)(dt + k * kdt), (S)kdt, w);
                S v = 0;
                for (int j = 0; j < NKNOT; ++j) v += w[j] * old[j];
                Y0[(size_t)k * NV + i] = v;
            }
        }
    }
    int tick(const double* x_plant, const double* wp, double* u_out, double* stat_out, double* ms) {
        auto t0 = std::chrono::steady_clock::now();
        if (x_plant) { for (int e = 0; e < NX; ++e) xp[e] = (S)x_plant[e]; }
        xp[I_TAU] = 0; xp[I_COST] = 0;
        for (int e = 0; e < NRP; ++e) rp[e] = (S)wp[e];
        for (int it = 0; it < n_iter; ++it) iteration(it);
        resample();
        if (!x_plant) std::copy(xp1.begin(), xp1.end(), xp.begin());         // model plant: the selected rollout's step-1 state
        for (int j = 0; j < NV; ++j) u_out[j] = (double)u_r[j];
        for (int e = 0; e < 4; ++e) stat_out[e] = (double)stat[e];
        *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return 0;
    }
    template <class F> int timed(int n_calls, F fn, double* median_ms, double* min_ms) {
        std::vector<double> t;
        fn();                                                               // warm-up
        for (int i = 0; i < n_calls; ++i) {
            auto t0 = std::chrono::steady_clock::now(); fn();
            t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end()); *median_ms = t[t.size() / 2]; *min_ms = t[0]; return 0;
    }
    // the pieces alone on the sample buffers (xT and Y of the last iteration), one parallel sweep over the K instances
    int time_model(int n, double* med, double* mn) {
        std::vector<S> un((size_t)K * NV), G((size_t)K * NCBF * NV), h((size_t)K * NCBF);
        return timed(n, [&] {
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (int k = 0; k < K; ++k) model_one(omp_get_thread_num(), xT.data() + (size_t)k * NX, Y.data() + (size_t)k * ND, un.data() + (size_t)k * NV, G.data() + (size_t)k * NCBF * NV, h.data() + (size_t)k * NCBF);
        }, med, mn);
    }
    int time_integrate(int n, double* med, double* mn) {
        std::vector<S> xn((size_t)K * NX);
        return timed(n, [&] {
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (int k = 0; k < K; ++k) integrate_one(omp_get_thread_num(), xT.data() + (size_t)k * NX, u_r.data(), xn.data() + (size_t)k * NX);
        }, med, mn);
    }
    int time_project(int n, double* med, double* mn) {
        std::vector<S> un((size_t)K * NV), G((size_t)K * NCBF * NV), h((size_t)K * NCBF), u((size_t)K * NV);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int k = 0; k < K; ++k) model_one(omp_get_thread_num(), xT.data() + (size_t)k * NX, Y.data() + (size_t)k * ND, un.data() + (size_t)k * NV, G.data() + (size_t)k * NCBF * NV, h.data() + (size_t)k * NCBF);
        return timed(n, [&] {
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (int k = 0; k < K; ++k) project(un.data() + (size_t)k * NV, G.data() + (size_t)k * NCBF * NV, h.data() + (size_t)k * NCBF, umax.data(), u.data() + (size_t)k * NV, proj_iters);
        }, med, mn);
    }
    int time_rollout(int n, double* med, double* mn) { return timed(n, [&] { rollout(); }, med, mn); }
    int time_iteration(int n, double* med, double* mn) { return timed(n, [&] { iteration(0); }, med, mn); }
    int get(const std::vector<S>& v, double* out) { for (size_t i = 0; i < v.size(); ++i) out[i] = (double)v[i]; return 0; }
    int set(std::vector<S>& v, const double* in) { for (size_t i = 0; i < v.size(); ++i) v[i] = (S)in[i]; return 0; }
};
}  // namespace

extern "C" {
void* spc_create(const char* cfg) { try { return new Driver(cfg); } catch (const std::exception& e) { g_err = e.what(); return nullptr; } }
int spc_tick(void* h, const double* x_plant, const double* wp, double* u, double* stat, double* ms) { return ((Driver*)h)->tick(x_plant, wp, u, stat, ms); }
int spc_set_samples(void* h, const double* Y) { Driver* d = (Driver*)h; d->set(d->Y, Y); d->given_samples = true; return 0; }
int spc_get_samples(void* h, double* Y, double* xT) { Driver* d = (Driver*)h; d->get(d->Y, Y); return d->get(d->xT, xT); }
int spc_get_iter_stats(void* h, double* out) { Driver* d = (Driver*)h; return d->get(d->istat, out); }
int spc_n_iter(void* h) { return ((Driver*)h)->n_iter; }
int spc_get_plan(void* h, double* Y0) { Driver* d = (Driver*)h; return d->get(d->Y0, Y0); }
int spc_set_plan(void* h, const double* Y0) { Driver* d = (Driver*)h; return d->set(d->Y0, Y0); }
int spc_get_plant(void* h, double* x) { Driver* d = (Driver*)h; return d->get(d->xp, x); }
int spc_set_plant(void* h, const double* x) { Driver* d = (Driver*)h; return d->set(d->xp, x); }
int spc_time_iteration(void* h, int n, double* med, double* mn) { return ((Driver*)h)->time_iteration(n, med, mn); }
int spc_time_rollout(void* h, int n, double* med, double* mn) { return ((Driver*)h)->time_rollout(n, med, mn); }
int spc_time_model(void* h, int n, double* med, double* mn) { return ((Driver*)h)->time_model(n, med, mn); }
int spc_time_integrate(void* h, int n, double* med, double* mn) { return ((Driver*)h)->time_integrate(n, med, mn); }
int spc_time_project(void* h, int n, double* med, double* mn) { return ((Driver*)h)->time_project(n, med, mn); }
const char* spc_info(void* h) { return ((Driver*)h)->info_str.c_str(); }
const char* spc_error() { return g_err.c_str(); }
void spc_destroy(void* h) { delete (Driver*)h; }
}
