// Latency-vs-batch saturation benchmark for cusadi2's three parallelism axes —
// per-robot Menagerie variant of examples/paper_results/latency.cu.
//
// Identical semantics to the paper_results harness, but for exactly TWO functions
// (this robot's forward kinematics and frame Jacobians) with generic plan prefixes
// (ROBOT_FK / ROBOT_FKJAC); each robot gets its own build dir + binary, so the
// robot identity lives in the directory, not the symbol names. It times one kernel
// launch, sweeping batch size B at fixed horizon N, for four configs:
//
//   full    staged + level-parallel + batch   per-stage plan, grid=(B,N), block=pick
//   staged  staged + batch (no LP)            per-stage plan, grid=(B,N), block=1
//   lp      level-parallel + batch (no stage) UNROLLED plan,  grid=(B,1), block=pick
//   batch   batch only (~ original cusadi)    UNROLLED plan,  grid=(B,1), block=1
//
// full/staged spread the horizon over the grid's stage axis (B*N small one-stage
// blocks); lp/batch run the per-stage function stamped N times into one DAG, one
// big block per batch instance (B blocks, ~N x the per-block shared memory). Both
// do the same total work (B*N evaluations) on identically-sized total buffers; the
// stage axis just changes how that work is packed into the grid and shared memory.
// The level-parallel axis is switched off by forcing block_size=1 (one thread runs
// every level serially). Emits one CSV row per (func,config,B):
//
//   dtype,func,config,batch,n_stages,grid_blocks,latency_us,wall_us
//
// latency_us is the cudaEvent kernel-only time (excludes the CPU-side launch cost);
// wall_us is the host wall-clock around the whole evaluate() call (count upload +
// launch dispatch + kernel + sync) — the end-to-end per-call latency, symmetric with
// the wall-clock timing used for the MJX / mujoco-warp baselines.
//
// A 'nan' latency means that case was skipped: either its unrolled plan exceeds the
// device per-block shared-memory limit (infeasible without tiling) or its buffers
// exceed ~90% of free GPU memory. Built and driven by run.py.
// Usage: ./latency <n_stages> <n_warmup> <n_timed> <B...>
//
// Compiled with -DCUSADI2_BENCH_LIB (as -shared -Xcompiler -fPIC) this file
// instead builds a ctypes-callable library (main() is dropped) so run.py can time
// the SAME evaluate() call from Python — the host-language-symmetric counterpart
// to the MJX/warp wall-clock numbers, including the Python + FFI dispatch a real
// Python deployment would pay. One case at a time (setup -> eval* -> teardown;
// sequential use only — upload() reconfigures the kernel's shared-memory carveout
// globally, so live handles of different plans must not interleave):
//
//   c2_setup(func, config, B, N) -> handle (>=0), -1 infeasible, -2 bad name
//   c2_eval(handle)              -> one evaluate() incl. device sync; returns
//                                   the kernel-only cudaEvent seconds
//   c2_teardown(handle)          -> frees the case's input/output buffers
#include "cusadi2.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace cusadi2;
using S = CUSADI2_SCALAR;   // scalar precision (double, or float via -DCUSADI2_SCALAR=float)

#include "fk_plan.h"         // this robot's forward kinematics, per-stage
#include "fk_unr_plan.h"     // forward kinematics, unrolled over N stages
#include "fkjac_plan.h"      // frame Jacobians, per-stage
#include "fkjac_unr_plan.h"  // frame Jacobians, unrolled over N stages

struct Func {
    const char* name;
    const StagePlanHost* per_stage;   // for full / staged  (grid B x N)
    const StagePlanHost* unrolled;    // for lp   / batch   (grid B x 1)
};

static const Func FUNCS[] = {
    {"fk",    &ROBOT_FK_plan,    &ROBOT_FK_UNR_plan},
    {"fkjac", &ROBOT_FKJAC_plan, &ROBOT_FKJAC_UNR_plan},
};

// Deterministic host fill in [-1, 1] — avoids NaN/denormal paths in trig ops
// while keeping the benchmark reproducible (values are irrelevant to latency).
static void fill_rand(std::vector<S>& v, unsigned seed) {
    unsigned x = seed * 2654435761u + 1u;
    for (size_t i = 0; i < v.size(); ++i) {
        x = x * 1664525u + 1013904223u;
        v[i] = (S)((x >> 8) / (double)(1u << 24)) * (S)2 - (S)1;
    }
}

// One fully-staged benchmark case: uploaded program + filled device buffers,
// ready for repeated evaluate() calls. Shared by the C++ timing loop (run_plan)
// and the ctypes API (c2_setup / c2_eval / c2_teardown).
struct Case {
    StageProgram prog;
    const S** d_in = nullptr;
    S** d_out = nullptr;
    std::vector<S*> in_bufs, out_bufs;
    int batch = 0, n_stages = 0;
    bool live = false;
};

// Upload `plan`, allocate + fill buffers for (batch, n_stages) with block_size
// override `block`. Returns false (case untouched apart from the upload) if the
// plan overflows the device shared-memory limit or the buffers won't fit in
// ~90% of free GPU memory.
static bool setup_case(const StagePlanHost& plan, int batch, int n_stages, int block,
                       int smem_limit, Case& c) {
    const LPPlanHost& b = plan.base;
    const int ni = b.n_args_in, no = b.n_args_out;

    if (b.shared_bytes > smem_limit) return false;   // infeasible without tiling

    c.prog.upload(plan);
    c.prog.h.base.block_size = block;
    c.batch = batch; c.n_stages = n_stages;

    long long bytes = 0;
    for (int a = 0; a < ni; ++a)
        bytes += (long long)batch * c.prog.in_stage_count(a, n_stages) * b.nnz_in[a] * sizeof(S);
    for (int a = 0; a < no; ++a)
        bytes += (long long)batch * c.prog.out_stage_count(a, n_stages) * b.nnz_out[a] * sizeof(S);
    size_t freeb = 0, totb = 0;
    cudaMemGetInfo(&freeb, &totb);
    if ((double)bytes > 0.90 * (double)freeb) return false;   // would OOM

    c.in_bufs.resize(ni);
    for (int a = 0; a < ni; ++a) {
        long long len = (long long)batch * c.prog.in_stage_count(a, n_stages) * b.nnz_in[a];
        std::vector<S> h(len);
        fill_rand(h, (unsigned)(a * 7919 + batch + n_stages));
        cudaMalloc(&c.in_bufs[a], len * sizeof(S));
        cudaMemcpy(c.in_bufs[a], h.data(), len * sizeof(S), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&c.d_in, ni * sizeof(S*));
    cudaMemcpy(c.d_in, c.in_bufs.data(), ni * sizeof(S*), cudaMemcpyHostToDevice);

    c.out_bufs.resize(no);
    for (int a = 0; a < no; ++a)
        cudaMalloc(&c.out_bufs[a], (long long)batch * c.prog.out_stage_count(a, n_stages)
                                   * b.nnz_out[a] * sizeof(S));
    cudaMalloc(&c.d_out, no * sizeof(S*));
    cudaMemcpy(c.d_out, c.out_bufs.data(), no * sizeof(S*), cudaMemcpyHostToDevice);

    c.live = true;
    return true;
}

// Frees the case's input/output buffers. The uploaded plan metadata is small and
// intentionally leaked (cases run sequentially; StageProgram has no free()).
static void free_case(Case& c) {
    for (S* p : c.in_bufs) cudaFree(p);
    for (S* p : c.out_bufs) cudaFree(p);
    cudaFree(c.d_in); cudaFree(c.d_out);
    c.live = false;
}

// config name -> (plan, effective n_stages, block override) for horizon N.
// Returns nullptr for an unknown config.
static const StagePlanHost* map_config(const Func& fn, const char* config, int N,
                                       int* nst, int* block) {
    if (!std::strcmp(config, "full"))   { *nst = N; *block = fn.per_stage->base.block_size; return fn.per_stage; }
    if (!std::strcmp(config, "staged")) { *nst = N; *block = 1;                             return fn.per_stage; }
    if (!std::strcmp(config, "lp"))     { *nst = 1; *block = fn.unrolled->base.block_size;  return fn.unrolled; }
    if (!std::strcmp(config, "batch"))  { *nst = 1; *block = 1;                             return fn.unrolled; }
    return nullptr;
}

// --- ctypes API (used by run.py --pylib; also linked into the binary, unused) ---

static std::vector<Case*> g_cases;

extern "C" int c2_setup(const char* func, const char* config, int batch, int n_stages) {
    const Func* fn = nullptr;
    for (const Func& f : FUNCS)
        if (!std::strcmp(f.name, func)) { fn = &f; break; }
    if (!fn) return -2;
    int nst = 0, block = 0;
    const StagePlanHost* plan = map_config(*fn, config, n_stages, &nst, &block);
    if (!plan) return -2;

    int smem_limit = 0;
    cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    Case* c = new Case;
    if (!setup_case(*plan, batch, nst, block, smem_limit, *c)) { delete c; return -1; }
    g_cases.push_back(c);
    return (int)g_cases.size() - 1;
}

extern "C" double c2_eval(int handle) {
    if (handle < 0 || handle >= (int)g_cases.size() || !g_cases[handle]->live) return -1.0;
    Case* c = g_cases[handle];
    return c->prog.evaluate(c->d_in, c->d_out, c->batch, c->n_stages);
}

extern "C" void c2_teardown(int handle) {
    if (handle >= 0 && handle < (int)g_cases.size() && g_cases[handle]->live)
        free_case(*g_cases[handle]);
}

#ifndef CUSADI2_BENCH_LIB

struct Timing { double kernel_us; double wall_us; };   // cudaEvent-only, host wall-clock

// Set up `plan` at (batch, n_stages) with block_size override `block`, time
// `n_timed` launches, free, and return the median kernel-only (cudaEvent) and
// end-to-end wall-clock µs. Returns {NaN,NaN} if setup_case deems it infeasible.
// prog.evaluate() returns the cudaEvent-timed kernel duration in seconds and
// internally synchronizes, so a host timer around the call captures launch
// dispatch + kernel + sync.
static Timing run_plan(const StagePlanHost& plan, int batch, int n_stages, int block,
                       int smem_limit, int n_warmup, int n_timed) {
    Case c;
    if (!setup_case(plan, batch, n_stages, block, smem_limit, c)) return {NAN, NAN};

    for (int i = 0; i < n_warmup; ++i) c.prog.evaluate(c.d_in, c.d_out, batch, n_stages);
    std::vector<double> us, wall;
    us.reserve(n_timed); wall.reserve(n_timed);
    for (int i = 0; i < n_timed; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double k = c.prog.evaluate(c.d_in, c.d_out, batch, n_stages) * 1e6;   // kernel-only, syncs
        auto t1 = std::chrono::high_resolution_clock::now();
        us.push_back(k);
        wall.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    std::sort(wall.begin(), wall.end());

    free_case(c);
    return {us[us.size() / 2], wall[wall.size() / 2]};
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <n_stages> <n_warmup> <n_timed> <B...>\n", argv[0]);
        return 2;
    }
    const int N        = atoi(argv[1]);
    const int n_warmup = atoi(argv[2]);
    const int n_timed  = atoi(argv[3]);
    std::vector<int> batches;
    for (int i = 4; i < argc; ++i) batches.push_back(atoi(argv[i]));

    int smem_limit = 0;
    cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    fprintf(stderr, "device max dynamic shared memory/block = %.1f KB\n", smem_limit / 1024.0);

    const char* dt = sizeof(S) == 4 ? "f32" : "f64";

    printf("dtype,func,config,batch,n_stages,grid_blocks,latency_us,wall_us\n");
    fflush(stdout);

    for (const Func& fn : FUNCS) {
        for (int B : batches) {
            static const char* CONFIGS[] = {"full", "staged", "lp", "batch"};
            for (const char* cfg : CONFIGS) {
                int nst = 0, block = 0;
                const StagePlanHost* plan = map_config(fn, cfg, N, &nst, &block);
                Timing t = run_plan(*plan, B, nst, block, smem_limit, n_warmup, n_timed);
                long long blocks = (long long)B * nst;
                if (std::isnan(t.kernel_us))
                    printf("%s,%s,%s,%d,%d,%lld,nan,nan\n", dt, fn.name, cfg, B, nst, blocks);
                else
                    printf("%s,%s,%s,%d,%d,%lld,%.3f,%.3f\n", dt, fn.name, cfg, B, nst,
                           blocks, t.kernel_us, t.wall_us);
            }
            fflush(stdout);
        }
    }
    return 0;
}

#endif  // CUSADI2_BENCH_LIB
