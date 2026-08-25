// Head-to-head: cusadi2 data-driven interpreter vs resurrected fully-unrolled
// level-parallel codegen (cusadi 2b352bd design), same LevelPlan, same device
// buffers. Compiled per function with:
//   -DPLAN_HEADER='"<stem>_plan.h"' -DPLAN=<PREFIX>_plan -DFUNC_NAME='"<stem>"'
// Usage: bench <n_warmup> <n_timed> <batch>...
// CSV on stdout: func,impl,batch,kernel_us,max_abs_diff
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cusadi2.h"
#include PLAN_HEADER

extern "C" float unr_evaluate(const CUSADI2_SCALAR* const inputs[],
                              CUSADI2_SCALAR* outputs[], int total_instances);
extern "C" int unr_configure();

using S = CUSADI2_SCALAR;

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <n_warmup> <n_timed> <batch>...\n", argv[0]);
        return 1;
    }
    const int n_warmup = atoi(argv[1]), n_timed = atoi(argv[2]);
    std::vector<int> batches;
    for (int i = 3; i < argc; ++i) batches.push_back(atoi(argv[i]));

    cusadi2::StageProgram prog;
    prog.upload(PLAN);
    unr_configure();
    const cusadi2::LPPlanHost& b = PLAN.base;

    std::mt19937_64 rng(0);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    printf("func,impl,batch,kernel_us,max_abs_diff\n");
    for (int B : batches) {
        // per-arg input buffers, batch-major (inst * nnz + elem)
        std::vector<const S*> hin(b.n_args_in);
        for (int a = 0; a < b.n_args_in; ++a) {
            size_t n = (size_t)B * b.nnz_in[a];
            std::vector<S> h(n);
            for (size_t i = 0; i < n; ++i) h[i] = (S)U(rng);
            S* d;
            CUSADI2_CHECK(cudaMalloc(&d, n * sizeof(S)));
            CUSADI2_CHECK(cudaMemcpy(d, h.data(), n * sizeof(S), cudaMemcpyHostToDevice));
            hin[a] = d;
        }
        const S** d_in;
        CUSADI2_CHECK(cudaMalloc(&d_in, b.n_args_in * sizeof(S*)));
        CUSADI2_CHECK(cudaMemcpy(d_in, hin.data(), b.n_args_in * sizeof(S*),
                                 cudaMemcpyHostToDevice));

        // two independent output buffer sets (interpreter / unrolled)
        std::vector<S*> hout_i(b.n_args_out), hout_u(b.n_args_out);
        for (int a = 0; a < b.n_args_out; ++a) {
            size_t n = (size_t)B * b.nnz_out[a];
            CUSADI2_CHECK(cudaMalloc(&hout_i[a], n * sizeof(S)));
            CUSADI2_CHECK(cudaMalloc(&hout_u[a], n * sizeof(S)));
        }
        S **d_out_i, **d_out_u;
        CUSADI2_CHECK(cudaMalloc(&d_out_i, b.n_args_out * sizeof(S*)));
        CUSADI2_CHECK(cudaMalloc(&d_out_u, b.n_args_out * sizeof(S*)));
        CUSADI2_CHECK(cudaMemcpy(d_out_i, hout_i.data(), b.n_args_out * sizeof(S*),
                                 cudaMemcpyHostToDevice));
        CUSADI2_CHECK(cudaMemcpy(d_out_u, hout_u.data(), b.n_args_out * sizeof(S*),
                                 cudaMemcpyHostToDevice));

        // correctness: both backends on the same inputs
        prog.evaluate(d_in, d_out_i, B, 1);
        unr_evaluate(d_in, d_out_u, B);
        double md = 0.0;
        for (int a = 0; a < b.n_args_out; ++a) {
            size_t n = (size_t)B * b.nnz_out[a];
            std::vector<S> oi(n), ou(n);
            CUSADI2_CHECK(cudaMemcpy(oi.data(), hout_i[a], n * sizeof(S), cudaMemcpyDeviceToHost));
            CUSADI2_CHECK(cudaMemcpy(ou.data(), hout_u[a], n * sizeof(S), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i)
                md = std::max(md, (double)std::abs((double)oi[i] - (double)ou[i]));
        }

        for (int impl = 0; impl < 2; ++impl) {
            for (int i = 0; i < n_warmup; ++i)
                impl == 0 ? prog.evaluate(d_in, d_out_i, B, 1)
                          : unr_evaluate(d_in, d_out_u, B);
            std::vector<double> us(n_timed);
            for (int i = 0; i < n_timed; ++i)
                us[i] = 1e6 * (impl == 0 ? prog.evaluate(d_in, d_out_i, B, 1)
                                         : unr_evaluate(d_in, d_out_u, B));
            printf("%s,%s,%d,%.3f,%.3e\n", FUNC_NAME,
                   impl == 0 ? "interpreter" : "unrolled", B, median(us), md);
            fflush(stdout);
        }

        for (int a = 0; a < b.n_args_in; ++a) cudaFree((void*)hin[a]);
        for (int a = 0; a < b.n_args_out; ++a) { cudaFree(hout_i[a]); cudaFree(hout_u[a]); }
        cudaFree(d_in); cudaFree(d_out_i); cudaFree(d_out_u);
    }
    return 0;
}
