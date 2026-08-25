// Self-contained native-backend example: drives the 1-axis LP program and the
// 2-axis stage program host-free (no torch/jax/python) and self-checks both
// against the references embedded in the generated headers.

#include <cstdio>
#include <cmath>
#include <vector>

#include "cusadi_native.h"
#include "lp_example_plan.h"
#include "stage_example_plan.h"

static const double** dInPtrs(std::vector<double*>& v) {
    const double** d; cudaMalloc(&d, v.size() * sizeof(double*));
    cudaMemcpy(d, v.data(), v.size() * sizeof(double*), cudaMemcpyHostToDevice);
    return d;
}
static double** dOutPtrs(std::vector<double*>& v) {
    double** d; cudaMalloc(&d, v.size() * sizeof(double*));
    cudaMemcpy(d, v.data(), v.size() * sizeof(double*), cudaMemcpyHostToDevice);
    return d;
}

static int check_lp() {
    const auto& P = LP_EXAMPLE_plan;
    const int B = 8;
    std::vector<double*> in(P.n_args_in), out(P.n_args_out);
    for (int a = 0; a < P.n_args_in; ++a) {
        int nnz = P.nnz_in[a];
        std::vector<double> h(size_t(B) * nnz);
        for (int b = 0; b < B; ++b)
            for (int e = 0; e < nnz; ++e) h[b * nnz + e] = LP_EXAMPLE_ref_in[a][e];
        cudaMalloc(&in[a], h.size() * sizeof(double));
        cudaMemcpy(in[a], h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice);
    }
    for (int a = 0; a < P.n_args_out; ++a)
        cudaMalloc(&out[a], size_t(B) * P.nnz_out[a] * sizeof(double));

    cusadi::LPProgram prog;
    prog.upload(P);
    prog.evaluate(dInPtrs(in), dOutPtrs(out), B);
    cudaDeviceSynchronize();

    double err = 0; int idx = 0;
    for (int a = 0; a < P.n_args_out; ++a) {
        std::vector<double> h(size_t(B) * P.nnz_out[a]);
        cudaMemcpy(h.data(), out[a], h.size() * sizeof(double), cudaMemcpyDeviceToHost);
        for (int e = 0; e < P.nnz_out[a]; ++e)
            err = fmax(err, fabs(h[e] - LP_EXAMPLE_ref_out[idx++]));   // instance 0
    }
    printf("  LP    program: max|err| vs reference = %.3e  (%s)\n", err, err < 1e-9 ? "PASS" : "FAIL");
    return err < 1e-9 ? 0 : 1;
}

static int check_stage() {
    const auto& P = STAGE_EXAMPLE_plan;
    const int B = 8, H = STAGE_EXAMPLE_ref_nstages;
    cusadi::StageProgram prog;
    prog.upload(P);

    std::vector<double*> in(P.base.n_args_in), out(P.base.n_args_out);
    for (int a = 0; a < P.base.n_args_in; ++a) {
        int n = prog.default_in_spacing(a, H);  // dense: stage_count * nnz doubles per instance
        std::vector<double> h(size_t(B) * n);
        for (int b = 0; b < B; ++b)
            for (int s = 0; s < n; ++s) h[size_t(b) * n + s] = STAGE_EXAMPLE_ref_in[a][s];
        cudaMalloc(&in[a], h.size() * sizeof(double));
        cudaMemcpy(in[a], h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice);
    }
    for (int a = 0; a < P.base.n_args_out; ++a)
        cudaMalloc(&out[a], size_t(B) * H * P.base.nnz_out[a] * sizeof(double));

    prog.evaluate(dInPtrs(in), dOutPtrs(out), B, H);
    cudaDeviceSynchronize();

    // ref_out is knot-major: for k, for each output arg, its nonzeros.
    std::vector<std::vector<double>> ho(P.base.n_args_out);
    for (int a = 0; a < P.base.n_args_out; ++a) {
        ho[a].resize(size_t(B) * H * P.base.nnz_out[a]);
        cudaMemcpy(ho[a].data(), out[a], ho[a].size() * sizeof(double), cudaMemcpyDeviceToHost);
    }
    double err = 0; int idx = 0;
    for (int k = 0; k < H; ++k)
        for (int a = 0; a < P.base.n_args_out; ++a) {
            int nnz = P.base.nnz_out[a];
            for (int e = 0; e < nnz; ++e)                          // instance 0, knot k
                err = fmax(err, fabs(ho[a][size_t(k) * nnz + e] - STAGE_EXAMPLE_ref_out[idx++]));
        }
    printf("  STAGE program: max|err| vs reference = %.3e  (%s)\n", err, err < 1e-9 ? "PASS" : "FAIL");
    return err < 1e-9 ? 0 : 1;
}

int main() {
    printf("cusadi native backend example (host-free: no torch/jax/python)\n");
    int rc = check_lp() | check_stage();
    printf("%s\n", rc == 0 ? "all checks PASSED" : "CHECK FAILED");
    return rc;
}
