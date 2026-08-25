// End-to-end self-check for the cusadi2 native backend.
//
// For each generated plan it uploads the program, runs evaluate() on the GPU
// over a single instance, and compares the outputs to the generate_reference
// header (random inputs evaluated through CasADi at build time). Exits
// non-zero on any mismatch.
//
// Built and driven by run.py, which emits kernel.cu + the *_plan.h / *_ref.h
// headers.
#include "cusadi2.h"
#include <cstdio>
#include <cmath>
#include <vector>
using namespace cusadi2;
using S = CUSADI2_SCALAR;   // scalar precision (double, or float via -DCUSADI2_SCALAR=float)

// Run one plan's embedded reference check on the GPU. Returns 0 on pass.
static int run(const char* name, const StagePlanHost& plan, int n_stages,
               const S* const* ref_in, const S* ref_out, int ref_nout) {
    const LPPlanHost& b = plan.base;
    const int ni = b.n_args_in, no = b.n_args_out, batch = 1;
    StageProgram prog; prog.upload(plan);

    // upload reference inputs (already laid out as the kernel indexes them)
    std::vector<const S*> hin(ni);
    for (int a = 0; a < ni; ++a) {
        int len = batch * prog.in_stage_count(a, n_stages) * b.nnz_in[a];
        S* d; cudaMalloc(&d, len * sizeof(S));
        cudaMemcpy(d, ref_in[a], len * sizeof(S), cudaMemcpyHostToDevice);
        hin[a] = d;
    }
    const S** d_in; cudaMalloc(&d_in, ni * sizeof(S*));
    cudaMemcpy(d_in, hin.data(), ni * sizeof(S*), cudaMemcpyHostToDevice);

    // output buffers
    std::vector<S*> hout(no); std::vector<int> olen(no);
    for (int a = 0; a < no; ++a) {
        olen[a] = batch * prog.out_stage_count(a, n_stages) * b.nnz_out[a];
        cudaMalloc(&hout[a], olen[a] * sizeof(S));
    }
    S** d_out; cudaMalloc(&d_out, no * sizeof(S*));
    cudaMemcpy(d_out, hout.data(), no * sizeof(S*), cudaMemcpyHostToDevice);

    prog.evaluate(d_in, d_out, batch, n_stages);
    cudaDeviceSynchronize();

    std::vector<std::vector<S>> ho(no);
    for (int a = 0; a < no; ++a) {
        ho[a].resize(olen[a]);
        cudaMemcpy(ho[a].data(), hout[a], olen[a] * sizeof(S), cudaMemcpyDeviceToHost);
    }
    // reference order: for each stage, for each output arg, its nonzeros
    std::vector<S> got;
    for (int k = 0; k < n_stages; ++k)
        for (int a = 0; a < no; ++a)
            for (int e = 0; e < b.nnz_out[a]; ++e)
                got.push_back(ho[a][k * b.nnz_out[a] + e]);

    // relative error: magnitude-robust across functions, and float accumulates
    // more error than double over deep DAGs (e.g. h1_jac at 78k ops).
    double maxrel = 0;
    for (int i = 0; i < ref_nout; ++i)
        maxrel = fmax(maxrel, fabs((double)got[i] - (double)ref_out[i]) / (1.0 + fabs((double)ref_out[i])));
    double tol = sizeof(S) == 4 ? 1e-3 : 1e-9;
    bool ok = maxrel < tol;
    printf("[%-6s] %s n_stages=%d nout=%d maxrel=%.3e -> %s\n",
           name, sizeof(S) == 4 ? "f32" : "f64", n_stages, ref_nout, maxrel, ok ? "PASS" : "FAIL");

    for (int a = 0; a < ni; ++a) cudaFree((void*)hin[a]);
    for (int a = 0; a < no; ++a) cudaFree(hout[a]);
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}

#include "demo_plan.h"   // batch-only LP  (every arg "stage", n_stages=1)
#include "dyn_plan.h"    // staged         (coupled + shared inputs, 4 stages)
#include "a1_plan.h"     // Pinocchio A1 quadruped ABA  (batch LP)
#include "h1_plan.h"     // Pinocchio H1 humanoid ABA   (batch LP)
#include "a1_jac_plan.h" // Pinocchio A1 ABA Jacobians  (batch LP, ~41k instrs)
#include "h1_jac_plan.h" // Pinocchio H1 ABA Jacobians  (batch LP, ~78k instrs)
#include "demo_ref.h"    // CasADi-evaluated self-check references (generate_reference)
#include "dyn_ref.h"
#include "a1_ref.h"
#include "h1_ref.h"
#include "a1_jac_ref.h"
#include "h1_jac_ref.h"

int main() {
    int rc = 0;
    rc |= run("demo",   DEMO_plan,   DEMO_ref_nstages,   DEMO_ref_in,   DEMO_ref_out,   DEMO_ref_nout);
    rc |= run("dyn",    DYN_plan,    DYN_ref_nstages,    DYN_ref_in,    DYN_ref_out,    DYN_ref_nout);
    rc |= run("a1",     A1_plan,     A1_ref_nstages,     A1_ref_in,     A1_ref_out,     A1_ref_nout);
    rc |= run("h1",     H1_plan,     H1_ref_nstages,     H1_ref_in,     H1_ref_out,     H1_ref_nout);
    rc |= run("a1_jac", A1_JAC_plan, A1_JAC_ref_nstages, A1_JAC_ref_in, A1_JAC_ref_out, A1_JAC_ref_nout);
    rc |= run("h1_jac", H1_JAC_plan, H1_JAC_ref_nstages, H1_JAC_ref_in, H1_JAC_ref_out, H1_JAC_ref_nout);
    printf(rc ? "FAIL\n" : "ALL PASS\n");
    return rc;
}
