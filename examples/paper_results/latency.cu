// Latency-vs-batch saturation benchmark for cusadi2's three parallelism axes.
//
// For each robot function (A1/H1 forward kinematics and frame Jacobians) it times
// one kernel launch, sweeping batch size B at fixed horizon N, for four configs:
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
#include "cusadi2.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace cusadi2;
using S = CUSADI2_SCALAR;   // scalar precision (double, or float via -DCUSADI2_SCALAR=float)

#include "a1_fk_plan.h"         // A1 forward kinematics, per-stage
#include "a1_fk_unr_plan.h"     // A1 forward kinematics, unrolled over N stages
#include "a1_fkjac_plan.h"      // A1 frame Jacobians, per-stage
#include "a1_fkjac_unr_plan.h"
#include "h1_fk_plan.h"
#include "h1_fk_unr_plan.h"
#include "h1_fkjac_plan.h"
#include "h1_fkjac_unr_plan.h"

struct Func {
    const char* name;
    const StagePlanHost* per_stage;   // for full / staged  (grid B x N)
    const StagePlanHost* unrolled;    // for lp   / batch   (grid B x 1)
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

struct Timing { double kernel_us; double wall_us; };   // cudaEvent-only, host wall-clock

// Allocate buffers for `plan` at (batch, n_stages), time `n_timed` launches with
// block_size override `block`, free, and return the median kernel-only (cudaEvent)
// and end-to-end wall-clock µs. Returns {NaN,NaN} if the plan overflows the device
// shared-memory limit or the buffers won't fit in free GPU memory. prog.evaluate()
// returns the cudaEvent-timed kernel duration in seconds and internally synchronizes,
// so a host timer around the call captures launch dispatch + kernel + sync.
static Timing run_plan(const StagePlanHost& plan, int batch, int n_stages, int block,
                       int smem_limit, int n_warmup, int n_timed) {
    const LPPlanHost& b = plan.base;
    const int ni = b.n_args_in, no = b.n_args_out;

    if (b.shared_bytes > smem_limit) return {NAN, NAN};   // infeasible without tiling

    StageProgram prog; prog.upload(plan);
    prog.h.base.block_size = block;

    long long bytes = 0;
    for (int a = 0; a < ni; ++a)
        bytes += (long long)batch * prog.in_stage_count(a, n_stages) * b.nnz_in[a] * sizeof(S);
    for (int a = 0; a < no; ++a)
        bytes += (long long)batch * prog.out_stage_count(a, n_stages) * b.nnz_out[a] * sizeof(S);
    size_t freeb = 0, totb = 0;
    cudaMemGetInfo(&freeb, &totb);
    if ((double)bytes > 0.90 * (double)freeb) return {NAN, NAN};   // would OOM

    std::vector<S*> hin(ni);
    for (int a = 0; a < ni; ++a) {
        long long len = (long long)batch * prog.in_stage_count(a, n_stages) * b.nnz_in[a];
        std::vector<S> h(len);
        fill_rand(h, (unsigned)(a * 7919 + batch + n_stages));
        S* d; cudaMalloc(&d, len * sizeof(S));
        cudaMemcpy(d, h.data(), len * sizeof(S), cudaMemcpyHostToDevice);
        hin[a] = d;
    }
    const S** d_in; cudaMalloc(&d_in, ni * sizeof(S*));
    cudaMemcpy(d_in, hin.data(), ni * sizeof(S*), cudaMemcpyHostToDevice);

    std::vector<S*> hout(no);
    for (int a = 0; a < no; ++a)
        cudaMalloc(&hout[a], (long long)batch * prog.out_stage_count(a, n_stages)
                             * b.nnz_out[a] * sizeof(S));
    S** d_out; cudaMalloc(&d_out, no * sizeof(S*));
    cudaMemcpy(d_out, hout.data(), no * sizeof(S*), cudaMemcpyHostToDevice);

    for (int i = 0; i < n_warmup; ++i) prog.evaluate(d_in, d_out, batch, n_stages);
    std::vector<double> us, wall;
    us.reserve(n_timed); wall.reserve(n_timed);
    for (int i = 0; i < n_timed; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double k = prog.evaluate(d_in, d_out, batch, n_stages) * 1e6;   // kernel-only, syncs
        auto t1 = std::chrono::high_resolution_clock::now();
        us.push_back(k);
        wall.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    std::sort(wall.begin(), wall.end());

    for (int a = 0; a < ni; ++a) cudaFree(hin[a]);
    for (int a = 0; a < no; ++a) cudaFree(hout[a]);
    cudaFree(d_in); cudaFree(d_out);
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

    Func funcs[] = {
        {"a1_fk",    &A1_FK_plan,    &A1_FK_UNR_plan},
        {"a1_fkjac", &A1_FKJAC_plan, &A1_FKJAC_UNR_plan},
        {"h1_fk",    &H1_FK_plan,    &H1_FK_UNR_plan},
        {"h1_fkjac", &H1_FKJAC_plan, &H1_FKJAC_UNR_plan},
    };
    const char* dt = sizeof(S) == 4 ? "f32" : "f64";

    printf("dtype,func,config,batch,n_stages,grid_blocks,latency_us,wall_us\n");
    fflush(stdout);

    for (Func& fn : funcs) {
        const int ps_block  = fn.per_stage->base.block_size;   // pick_block_size(per-stage)
        const int unr_block = fn.unrolled->base.block_size;    // pick_block_size(unrolled)
        for (int B : batches) {
            // config, plan, batch_eff, nstages_eff, block, grid_blocks
            Timing full   = run_plan(*fn.per_stage, B, N, ps_block,  smem_limit, n_warmup, n_timed);
            Timing staged = run_plan(*fn.per_stage, B, N, 1,         smem_limit, n_warmup, n_timed);
            Timing lp     = run_plan(*fn.unrolled,  B, 1, unr_block, smem_limit, n_warmup, n_timed);
            Timing batch  = run_plan(*fn.unrolled,  B, 1, 1,         smem_limit, n_warmup, n_timed);

            struct Row { const char* name; Timing t; int nst; long long blocks; } rows[] = {
                {"full",   full,   N, (long long)B * N},
                {"staged", staged, N, (long long)B * N},
                {"lp",     lp,     1, (long long)B},
                {"batch",  batch,  1, (long long)B},
            };
            for (Row& r : rows) {
                if (std::isnan(r.t.kernel_us))
                    printf("%s,%s,%s,%d,%d,%lld,nan,nan\n", dt, fn.name, r.name, B, r.nst, r.blocks);
                else
                    printf("%s,%s,%s,%d,%d,%lld,%.3f,%.3f\n", dt, fn.name, r.name, B, r.nst,
                           r.blocks, r.t.kernel_us, r.t.wall_us);
            }
            fflush(stdout);
        }
    }
    return 0;
}
