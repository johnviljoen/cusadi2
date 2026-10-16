// Closed-loop MPPI for R Unitree Go2 robots on the parkour course, everything on the GPU: K sampled
// rollouts per robot through the cusadi2 step function, annealed MPPI updates, and the plant (the same
// function), one control step replayed as one CUDA graph.
//
// One control step (period substeps * dt):
//
//   for i in 0 .. iters-1                      annealed sampling iterations (DIAL-MPC style)
//     sample    Y[r,k] = Y0[r] + sigma_i,j * eps      knots j of 12 joint-target offsets; sample 0 is Y0 itself
//     rollout   x[r,k] <- step^(horizon)(x_plant[r], Y[r,k])    cusadi2 launches over the R*K batch, cost in x
//     update    w = softmax(-(cost - min) / (temp * std)),  Y0[r] <- sum_k w Y[r,k]
//   plant       x_plant[r] <- step^(one control period)(x_plant[r], Y0[r])
//   shift       Y0[r] re-sampled one control period later (last knot held)
//
//   mppi_rollout <build_dir> <run_dir> [key=value ...]
// build_dir: plans/, meta.txt (build.py). run_dir: robots.f64 (R x [x0 y0 yaw0 goal_x goal_y goal_yaw]),
// x0.f64 (R x nx initial states), par.f64 (R x model parameters). Keys: K iters steps temp sigma anneal hfac
// seed fused graph. Writes state.f64 (steps+1, R, nx), plan.f64 (steps, R, knots; after the shift), stat.f64 (steps, R, 4:
// min / mean / std of the sample costs of the last iteration, effective sample size), latency_us.f64.
#include "cusadi2.h"
#include <curand_kernel.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
using namespace cusadi2;
using S = Scalar;

#define CK(x) do { cudaError_t e_ = (cudaError_t)(x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

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
static std::vector<double> download(const S* d, size_t n) {
    std::vector<S> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(S), cudaMemcpyDeviceToHost));
    return std::vector<double>(h.begin(), h.end());
}

// Y[r,k] = clip(Y0[r] + sigma[knot] * N(0,1), joint range); sample 0 of every robot is the unperturbed plan. One Philox
// stream per sample, offset by the device-side draw counter so replays of the graph draw fresh noise.
__global__ void k_sample(const S* __restrict__ Y0, S* __restrict__ Y, const S* __restrict__ sigma, const S* __restrict__ ylim,
                         const unsigned* __restrict__ counter, int B, int K, int n_knot, int nj, unsigned long long seed) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    const int nd = n_knot * nj; const S* y0 = Y0 + (size_t)(b / K) * nd; S* y = Y + (size_t)b * nd;
    if (b % K == 0) { for (int i = 0; i < nd; ++i) y[i] = y0[i]; return; }
    curandStatePhilox4_32_10_t st; curand_init(seed, (unsigned long long)b, (unsigned long long)counter[0] * 64ull, &st);
    for (int i = 0; i < nd; ++i) y[i] = fmin(fmax(y0[i] + sigma[i / nj] * (S)curand_normal(&st), ylim[i % nj]), ylim[nj + i % nj]);
}
__global__ void k_tick(unsigned* counter) { counter[0] += 1; }
// Every sample starts at its robot's plant state with the plan clock and the cost at zero.
__global__ void k_init(const S* __restrict__ xp, S* __restrict__ x, int B, int K, int nx, int i_tau, int i_cost) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= (long long)B * nx) return;
    int b = (int)(i / nx), e = (int)(i - (long long)b * nx);
    x[i] = (e == i_tau || e == i_cost) ? S(0) : xp[(size_t)(b / K) * nx + e];
}
__global__ void k_zero_tau(S* xp, int R, int nx, int i_tau) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r < R) xp[(size_t)r * nx + i_tau] = 0;
}
// One block per robot: weights from the sample costs, then the weighted mean of the samples.
__global__ void k_update(const S* __restrict__ x, const S* __restrict__ Y, S* __restrict__ Y0, S* __restrict__ w, S* __restrict__ stat,
                         int K, int nx, int i_cost, int nd, S temp) {
    const int r = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const S* c = x + (size_t)r * K * nx + i_cost; S* wr = w + (size_t)r * K;
    __shared__ double s_a[256], s_b[256], s_c[256], s_d[256];
    double sum = 0, sq = 0, mn = 1e300, cnt = 0;
    for (int k = tid; k < K; k += nt) { double v = c[(size_t)k * nx]; if (isfinite(v)) { sum += v; sq += v * v; mn = fmin(mn, v); cnt += 1; } }
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
        double v = c[(size_t)k * nx], wk = isfinite(v) ? exp(-(v - cmin) / ((double)temp * sd)) : 0.0;
        wr[k] = (S)wk; z += wk; z2 += wk * wk;
    }
    s_a[tid] = z; s_b[tid] = z2; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) { s_a[tid] += s_a[tid + s]; s_b[tid] += s_b[tid + s]; } __syncthreads(); }
    const double Z = s_a[0], Z2 = s_b[0];
    if (tid == 0) { S* st = stat + (size_t)r * 4; st[0] = (S)cmin; st[1] = (S)mean; st[2] = (S)sd; st[3] = (S)(Z > 0 ? Z * Z / Z2 : 0); }
    if (!(Z > 0)) return;                       // every rollout diverged: keep the plan
    for (int j = tid; j < nd; j += nt) {
        double acc = 0; const S* y = Y + (size_t)r * K * nd + j;
        for (int k = 0; k < K; ++k) acc += (double)wr[k] * y[(size_t)k * nd];
        Y0[(size_t)r * nd + j] = (S)(acc / Z);
    }
}
// Re-sample the plan one control period later: knot j <- lerp(knot j, knot j+1, a); the last knot holds.
__global__ void k_shift(S* Y0, int R, int n_knot, int nj, S a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= R * nj) return;
    S* y = Y0 + (size_t)(i / nj) * n_knot * nj + i % nj;
    for (int j = 0; j + 1 < n_knot; ++j) y[j * nj] += a * (y[(j + 1) * nj] - y[j * nj]);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <build_dir> <run_dir> [key=value ...]\n", argv[0]); return 2; }
    const std::string dir = argv[1], run = argv[2];
    std::map<std::string, double> cfg = {{"K", 1024}, {"iters", 4}, {"steps", 250}, {"temp", 0.1}, {"sigma", 0.25}, {"anneal", 0.5},
                                         {"hfac", 0.8}, {"seed", 1}, {"fused", 1}, {"graph", 1}};
    std::ifstream mf(dir + "/meta.txt"); if (!mf) { fprintf(stderr, "no meta.txt in %s\n", dir.c_str()); return 2; }
    std::map<std::string, double> meta; { std::string k; double v; while (mf >> k >> v) meta[k] = v; }
    for (int i = 3; i < argc; ++i) {
        const char* eq = strchr(argv[i], '='); if (!eq) { fprintf(stderr, "bad argument %s\n", argv[i]); return 2; }
        std::string k(argv[i], eq - argv[i]); if (!cfg.count(k)) { fprintf(stderr, "unknown key %s\n", k.c_str()); return 2; }
        cfg[k] = atof(eq + 1);
    }
    const int K = (int)cfg["K"], iters = (int)cfg["iters"], n_steps = (int)cfg["steps"];
    const bool fused = cfg["fused"] != 0, use_graph = cfg["graph"] != 0;
    const int NX = (int)meta["nx"], NKNOT = (int)meta["n_knot"], NJ = 12, ND = NKNOT * NJ, SUB = (int)meta["substeps"], H = (int)meta["horizon"];
    const int I_TAU = NX - 2, I_COST = NX - 1;
    const double dt_ctrl = SUB * meta["dt"];
    std::vector<double> h_rob = read_f64(run + "/robots.f64"), h_x0 = read_f64(run + "/x0.f64"), h_par = read_f64(run + "/par.f64");
    const int R = (int)(h_x0.size() / NX), NP = (int)meta["npar"], B = R * K;
    if ((int)h_rob.size() != R * 6 || (int)h_par.size() != R * NP) { fprintf(stderr, "run_dir does not match the build\n"); return 3; }

    const std::string tag = sizeof(S) == 4 ? "float" : "double";
    const int n_sub = fused ? SUB : 1, per_ctrl = SUB / n_sub;        // physics steps per launch, launches per control step
    Plan plan = load_plan(dir + "/plans/go2_step" + std::to_string(n_sub) + "_" + tag + ".plan");
    Program fn(plan);
    if (plan.in[0].nnz() != NX || plan.in[1].nnz() != ND || plan.in[2].nnz() != 4 || plan.in[3].nnz() != NP) { fprintf(stderr, "plan mismatch\n"); return 3; }

    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    // ---- buffers
    std::vector<double> h_rp_r((size_t)R * 4, 0.0), h_rp_b((size_t)B * 4), h_par_r, h_par_b;
    for (int r = 0; r < R; ++r) for (int e = 0; e < 3; ++e) h_rp_r[r * 4 + e] = h_rob[r * 6 + 3 + e];
    for (int b = 0; b < B; ++b) for (int e = 0; e < 4; ++e) h_rp_b[(size_t)b * 4 + e] = h_rp_r[(b / K) * 4 + e];
    h_par_r = h_par;
    for (int b = 0; b < B; ++b) h_par_b.insert(h_par_b.end(), h_par.begin() + (size_t)(b / K) * NP, h_par.begin() + (size_t)(b / K + 1) * NP);
    S *xp[2] = {upload(h_x0), dalloc((size_t)R * NX)}, *Y0 = dalloc((size_t)R * ND), *rp_r = upload(h_rp_r), *par_r = upload(h_par_r);
    S *x[2] = {dalloc((size_t)B * NX), dalloc((size_t)B * NX)}, *Y = dalloc((size_t)B * ND), *rp_b = upload(h_rp_b), *par_b = upload(h_par_b);
    S *w = dalloc(B), *stat = dalloc((size_t)R * 4);
    std::vector<double> h_sig((size_t)iters * NKNOT);
    for (int i = 0; i < iters; ++i) for (int j = 0; j < NKNOT; ++j)
        h_sig[(size_t)i * NKNOT + j] = cfg["sigma"] * pow(cfg["anneal"], i) * pow(cfg["hfac"], NKNOT - 1 - j);
    S *sigma = upload(h_sig), *ylim = upload(read_f64(dir + "/ylim.f64"));
    unsigned* counter; CK(cudaMalloc(&counter, sizeof(unsigned))); CK(cudaMemset(counter, 0, sizeof(unsigned)));
    S *log_state = dalloc((size_t)(n_steps + 1) * R * NX), *log_plan = dalloc((size_t)n_steps * R * ND), *log_stat = dalloc((size_t)n_steps * R * 4);
    const S** in_roll[2]; S** out_roll[2]; const S** in_plant[2]; S** out_plant[2];
    for (int p = 0; p < 2; ++p) {
        in_roll[p] = (const S**)dptrs<S>({x[p], Y, rp_b, par_b});     out_roll[p] = dptrs<S>({x[1 - p]});
        in_plant[p] = (const S**)dptrs<S>({xp[p], Y0, rp_r, par_r});  out_plant[p] = dptrs<S>({xp[1 - p]});
    }

    const size_t szS = sizeof(S);
    const unsigned long long seed = (unsigned long long)cfg["seed"];
    const S shift_a = (S)(dt_ctrl / meta["knot_dt"]);
    auto step = [&]() {
        for (int i = 0; i < iters; ++i) {
            k_sample<<<(B + 255) / 256, 256, 0, st>>>(Y0, Y, sigma + (size_t)i * NKNOT, ylim, counter, B, K, NKNOT, NJ, seed);
            k_tick<<<1, 1, 0, st>>>(counter);
            k_init<<<(int)(((long long)B * NX + 255) / 256), 256, 0, st>>>(xp[0], x[0], B, K, NX, I_TAU, I_COST);
            int p = 0;
            for (int s = 0; s < H * per_ctrl; ++s, p ^= 1) CK(fn.evaluate(in_roll[p], out_roll[p], B, st));
            k_update<<<R, 256, 0, st>>>(x[p], Y, Y0, w, stat, K, NX, I_COST, ND, (S)cfg["temp"]);
        }
        k_zero_tau<<<(R + 255) / 256, 256, 0, st>>>(xp[0], R, NX, I_TAU);
        int p = 0;
        for (int s = 0; s < per_ctrl; ++s, p ^= 1) CK(fn.evaluate(in_plant[p], out_plant[p], R, st));
        if (p) CK(cudaMemcpyAsync(xp[0], xp[1], (size_t)R * NX * szS, cudaMemcpyDeviceToDevice, st));
        k_shift<<<(R * NJ + 255) / 256, 256, 0, st>>>(Y0, R, NKNOT, NJ, shift_a);
    };
    auto reset = [&]() {
        std::vector<S> t(h_x0.begin(), h_x0.end()); CK(cudaMemcpy(xp[0], t.data(), t.size() * szS, cudaMemcpyHostToDevice));
        CK(cudaMemset(Y0, 0, (size_t)R * ND * szS)); CK(cudaMemset(counter, 0, sizeof(unsigned)));
    };
    step(); CK(cudaStreamSynchronize(st));
    cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
    if (use_graph) {
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal)); step(); CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
    }
    reset();

    // ---- closed loop
    cudaEvent_t ev[2]; for (auto& e : ev) CK(cudaEventCreate(&e));
    std::vector<double> lat(n_steps);
    CK(cudaMemcpyAsync(log_state, xp[0], (size_t)R * NX * szS, cudaMemcpyDeviceToDevice, st));
    auto w0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_steps; ++s) {
        CK(cudaEventRecord(ev[0], st));
        if (use_graph) CK(cudaGraphLaunch(ge, st)); else step();
        CK(cudaEventRecord(ev[1], st));
        CK(cudaMemcpyAsync(log_plan + (size_t)s * R * ND, Y0, (size_t)R * ND * szS, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(log_stat + (size_t)s * R * 4, stat, (size_t)R * 4 * szS, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(log_state + (size_t)(s + 1) * R * NX, xp[0], (size_t)R * NX * szS, cudaMemcpyDeviceToDevice, st));
        CK(cudaEventSynchronize(ev[1]));
        float ms; CK(cudaEventElapsedTime(&ms, ev[0], ev[1])); lat[s] = ms * 1e3;
    }
    CK(cudaStreamSynchronize(st));
    double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();

    std::vector<double> h_state = download(log_state, (size_t)(n_steps + 1) * R * NX);
    write_f64(run + "/state.f64", h_state); write_f64(run + "/plan.f64", download(log_plan, (size_t)n_steps * R * ND));
    write_f64(run + "/stat.f64", download(log_stat, (size_t)n_steps * R * 4)); write_f64(run + "/latency_us.f64", lat);
    bool finite = true; for (double v : h_state) finite = finite && std::isfinite(v);
    std::vector<double> sl = lat; std::sort(sl.begin(), sl.end());
    printf("R=%d K=%d iters=%d horizon=%d x %d steps (%s, %s) | %d control steps (%.2f s) finite=%d | step latency ms: median %.2f  p99 %.2f | "
           "wall %.1f s (%.3fx real time)\n", R, K, iters, H, SUB, tag.c_str(), fused ? "fused" : "per-step", n_steps, n_steps * dt_ctrl,
           (int)finite, sl[n_steps / 2] / 1e3, sl[(size_t)(n_steps * 0.99)] / 1e3, wall_ms / 1e3, n_steps * dt_ctrl * 1e3 / wall_ms);
    for (int r = 0; r < R; ++r) {
        const double* xe = &h_state[((size_t)n_steps * R + r) * NX];
        printf("  robot %d: end (%.2f, %.2f, %.2f)  goal (%.2f, %.2f)  dist %.3f\n", r, xe[0], xe[1], xe[2], h_rob[r * 6 + 3], h_rob[r * 6 + 4],
               hypot(xe[0] - h_rob[r * 6 + 3], xe[1] - h_rob[r * 6 + 4]));
    }
    if (ge) CK(cudaGraphExecDestroy(ge)); if (g) cudaGraphDestroy(g);
    for (auto& e : ev) cudaEventDestroy(e);
    CK(cudaStreamDestroy(st));
    return finite ? 0 : 1;
}
