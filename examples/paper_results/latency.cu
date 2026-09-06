// Latency-vs-batch saturation benchmark for cusadi2's three parallelism axes.
//
// For each function (A1/H1 forward kinematics, frame Jacobians, frame Hessians) it times one
// kernel launch, sweeping batch size B at fixed horizon N, for four configs:
//
//   full    staged + level-parallel + batch   per-stage plan, level kernel, grid=(B,N)
//   staged  staged + batch (no LP)            per-stage plan, serial kernel, B*N threads
//   lp      level-parallel + batch (no stage) unrolled plan,  level kernel, grid=(B,1)
//   batch   batch only (= original cusadi)    unrolled plan,  serial kernel, B threads
//
// full/lp run the level-parallel kernel (one block per item, work vector in shared
// memory); staged/batch run the serial kernel (one thread per item walking CasADi's
// instruction stream, sz_w-slot work vector in global scratch, Program::use_serial),
// the original cusadi execution model interpreted rather than unrolled. The
// unrolled plans stamp the per-stage function N times into one DAG (~N x the work
// vector). All four do the same total work (B*N evaluations).
//
// Plans are read from <plan_dir>/<func>.plan (per-stage, horizon baked in) and
// <plan_dir>/<func>_unr.plan (unrolled, n_stages == 1), written by bench.py.
// One CSV row per (func, config, B):
//
//   dtype,func,config,batch,n_stages,grid_blocks,latency_us,wall_us
//
// latency_us is the cudaEvent kernel-only time; wall_us is host wall-clock around
// launch + kernel + sync, symmetric with the MJX / mujoco-warp timing. 'nan' means
// the case was skipped: a level-kernel plan exceeds the device per-block shared-
// memory limit (unrolled plans without tiling) or buffers + scratch exceed ~90% of
// free GPU memory.
//
//   ./latency <plan_dir> <n_warmup> <n_timed> <B...>
//
// Compiled with -DCUSADI2_BENCH_LIB (-shared -Xcompiler -fPIC) this file instead
// builds a ctypes-callable library (main() dropped) so bench.py can time the same
// evaluate() from Python. Cases are used one at a time:
//   c2_setup(plan_path, serial, batch) -> handle (>= 0) or -1 if infeasible
//   c2_eval(handle)                   -> one evaluate() incl. sync; kernel seconds
//   c2_teardown(handle)
#include "cusadi2.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace cusadi2;
using S = Scalar;
static const size_t SERIAL_SCRATCH = size_t(2) << 30;   // serial-mode scratch budget (bytes)

// Deterministic fill in [-1, 1]: keeps trig ops off NaN/denormal paths.
static void fill_rand(std::vector<S>& v, unsigned seed) {
    unsigned x = seed * 2654435761u + 1u;
    for (S& s : v) { x = x * 1664525u + 1013904223u; s = (S)((x >> 8) / (double)(1u << 24)) * 2 - 1; }
}

// One benchmark case: uploaded program, filled device buffers, timing events.
struct Case {
    Program prog;
    const S** d_in = nullptr; S** d_out = nullptr;
    std::vector<S*> bufs;
    int batch = 0, n_stages = 1;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    ~Case() {
        for (S* p : bufs) cudaFree(p);
        cudaFree(d_in); cudaFree(d_out);
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
    }
    double eval() {                       // kernel-only seconds; synchronizes
        cudaEventRecord(e0);
        CUSADI2_CHECK((cudaError_t)prog.evaluate(d_in, d_out, batch));
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        return ms * 1e-3;
    }
};

template <class T> static T** upload_ptrs(const std::vector<T*>& h) {
    T** d; cudaMalloc(&d, h.size() * sizeof(T*));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T*), cudaMemcpyHostToDevice);
    return d;
}

// nullptr if the plan overflows shared memory (level mode) or memory would not fit.
static Case* setup_case(const Plan& p, bool serial, int batch) {
    int smem = 0; cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (!serial && p.shared_bytes() > smem) return nullptr;
    double bytes = serial ? (double)std::min(SERIAL_SCRATCH, p.serial_bytes() << 20) : 0;
    for (const Arg& a : p.in) bytes += (double)a.len(batch) * sizeof(S);
    for (const Arg& a : p.out) bytes += (double)a.len(batch) * sizeof(S);
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    if (bytes > 0.9 * (double)freeb) return nullptr;

    Case* c = new Case;
    c->batch = batch; c->n_stages = p.n_stages;
    c->prog.upload(p);
    if (serial) c->prog.use_serial(SERIAL_SCRATCH);
    std::vector<S*> in, out;
    for (const Arg& a : p.in) {
        std::vector<S> h(a.len(batch)); fill_rand(h, (unsigned)(in.size() * 7919 + batch));
        S* d; cudaMalloc(&d, h.size() * sizeof(S));
        cudaMemcpy(d, h.data(), h.size() * sizeof(S), cudaMemcpyHostToDevice);
        in.push_back(d); c->bufs.push_back(d);
    }
    for (const Arg& a : p.out) {
        S* d; cudaMalloc(&d, a.len(batch) * sizeof(S)); out.push_back(d); c->bufs.push_back(d);
    }
    c->d_in = (const S**)upload_ptrs(in); c->d_out = upload_ptrs(out);
    cudaEventCreate(&c->e0); cudaEventCreate(&c->e1);
    return c;
}

// --- ctypes API (bench.py --pylib) ------------------------------------------
static std::vector<Case*> g_cases;
extern "C" int c2_setup(const char* plan_path, int serial, int batch) {
    Case* c = setup_case(load_plan(plan_path), serial != 0, batch);
    if (!c) return -1;
    g_cases.push_back(c);
    return (int)g_cases.size() - 1;
}
extern "C" double c2_eval(int h) {
    return (h >= 0 && h < (int)g_cases.size() && g_cases[h]) ? g_cases[h]->eval() : -1.0;
}
extern "C" void c2_teardown(int h) {
    if (h >= 0 && h < (int)g_cases.size()) { delete g_cases[h]; g_cases[h] = nullptr; }
}

#ifndef CUSADI2_BENCH_LIB
static const char* FUNCS[] = {"a1_fk", "a1_fkjac", "a1_fkhess", "h1_fk", "h1_fkjac", "h1_fkhess"};
static const char* CONFIGS[] = {"full", "staged", "lp", "batch"};

// config -> plan file and whether it runs the serial kernel.
static std::string config_plan(const std::string& dir, const char* func, const char* cfg, bool* serial) {
    bool unrolled = !strcmp(cfg, "lp") || !strcmp(cfg, "batch");
    *serial = !strcmp(cfg, "staged") || !strcmp(cfg, "batch");
    return dir + "/" + func + (unrolled ? "_unr" : "") + ".plan";
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <plan_dir> <n_warmup> <n_timed> <B...>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const int n_warmup = atoi(argv[2]), n_timed = atoi(argv[3]);
    const char* dt = sizeof(S) == 4 ? "f32" : "f64";
    printf("dtype,func,config,batch,n_stages,grid_blocks,latency_us,wall_us\n");
    for (const char* fn : FUNCS)
        for (int i = 4; i < argc; ++i) {
            const int B = atoi(argv[i]);
            for (const char* cfg : CONFIGS) {
                bool serial = false;
                Plan p = load_plan(config_plan(dir, fn, cfg, &serial));
                Case* c = setup_case(p, serial, B);
                if (!c) {
                    printf("%s,%s,%s,%d,%d,%lld,nan,nan\n", dt, fn, cfg, B, p.n_stages,
                           (long long)B * p.n_stages);
                    continue;
                }
                for (int k = 0; k < n_warmup; ++k) c->eval();
                std::vector<double> kus, wus;
                for (int k = 0; k < n_timed; ++k) {
                    auto t0 = std::chrono::steady_clock::now();
                    kus.push_back(c->eval() * 1e6);
                    wus.push_back(std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count());
                }
                std::sort(kus.begin(), kus.end()); std::sort(wus.begin(), wus.end());
                printf("%s,%s,%s,%d,%d,%lld,%.3f,%.3f\n", dt, fn, cfg, B, c->n_stages,
                       (long long)B * c->n_stages, kus[n_timed / 2], wus[n_timed / 2]);
                fflush(stdout);
                delete c;
            }
        }
    return 0;
}
#endif
