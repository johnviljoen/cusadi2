// Head-to-head: standard cusadi2 interpreter (nodes-across-lanes) vs the
// warp-batched interpreter (instances-across-lanes), same plan, same device
// buffers, same uploaded plan arrays. Compiled per function with:
//   -DPLAN_HEADER='"<stem>_plan.h"' -DPLAN=<PREFIX>_plan -DFUNC_NAME='"<stem>"'
// Usage: bench_wb <n_warmup> <n_timed> <block_warps> <batch>...
// CSV on stdout: func,impl,batch,kernel_us,max_abs_diff
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cusadi2.h"
#include PLAN_HEADER

using S = CUSADI2_SCALAR;

extern "C" float wb_evaluate(
    const S* const* inputs, S** outputs, int total_instances,
    int block_warps, int shared_bytes,
    int n_consts, const int* const_slots, const S* const_vals,
    int n_inputs, const int* input_slots, const int* input_args,
    const int* input_elems, const int* nnz_in,
    int n_levels, const int* level_offsets, const int* level_widths,
    const int* node_ops, const int* node_outs, const int* node_in_a, const int* node_in_b,
    int n_outputs, const int* out_slots, const int* out_args,
    const int* out_elems, const int* nnz_out);
extern "C" int wb_configure(int shared_bytes);

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <n_warmup> <n_timed> <block_warps> <batch>...\n", argv[0]);
        return 1;
    }
    const int n_warmup = atoi(argv[1]), n_timed = atoi(argv[2]);
    const int block_warps = atoi(argv[3]);
    std::vector<int> batches;
    for (int i = 4; i < argc; ++i) batches.push_back(atoi(argv[i]));

    cusadi2::StageProgram prog;
    prog.upload(PLAN);                       // device plan arrays, shared by both
    const cusadi2::LPPlanHost& b = PLAN.base;
    const int wb_smem = b.shared_bytes * 32; // transposed [slot][lane32] vector
    wb_configure(wb_smem);

    auto wb_run = [&](const S** d_in, S** d_out, int B) {
        return wb_evaluate(d_in, d_out, B, block_warps, wb_smem,
            b.n_consts, prog.d_const_slots, prog.d_const_vals,
            b.n_input_maps, prog.d_input_slots, prog.d_input_args,
            prog.d_input_elems, prog.d_nnz_in,
            b.n_levels, prog.d_level_offsets, prog.d_level_widths,
            prog.d_node_ops, prog.d_node_outs, prog.d_node_in_a, prog.d_node_in_b,
            b.n_output_maps, prog.d_out_slots, prog.d_out_args,
            prog.d_out_elems, prog.d_nnz_out);
    };

    std::mt19937_64 rng(0);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    printf("func,impl,batch,kernel_us,max_abs_diff\n");
    for (int B : batches) {
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

        std::vector<S*> hout_i(b.n_args_out), hout_w(b.n_args_out);
        for (int a = 0; a < b.n_args_out; ++a) {
            size_t n = (size_t)B * b.nnz_out[a];
            CUSADI2_CHECK(cudaMalloc(&hout_i[a], n * sizeof(S)));
            CUSADI2_CHECK(cudaMalloc(&hout_w[a], n * sizeof(S)));
        }
        S **d_out_i, **d_out_w;
        CUSADI2_CHECK(cudaMalloc(&d_out_i, b.n_args_out * sizeof(S*)));
        CUSADI2_CHECK(cudaMalloc(&d_out_w, b.n_args_out * sizeof(S*)));
        CUSADI2_CHECK(cudaMemcpy(d_out_i, hout_i.data(), b.n_args_out * sizeof(S*),
                                 cudaMemcpyHostToDevice));
        CUSADI2_CHECK(cudaMemcpy(d_out_w, hout_w.data(), b.n_args_out * sizeof(S*),
                                 cudaMemcpyHostToDevice));

        prog.evaluate(d_in, d_out_i, B, 1);
        wb_run(d_in, d_out_w, B);
        double md = 0.0;
        for (int a = 0; a < b.n_args_out; ++a) {
            size_t n = (size_t)B * b.nnz_out[a];
            std::vector<S> oi(n), ow(n);
            CUSADI2_CHECK(cudaMemcpy(oi.data(), hout_i[a], n * sizeof(S), cudaMemcpyDeviceToHost));
            CUSADI2_CHECK(cudaMemcpy(ow.data(), hout_w[a], n * sizeof(S), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i)
                md = std::max(md, (double)std::abs((double)oi[i] - (double)ow[i]));
        }

        for (int impl = 0; impl < 2; ++impl) {
            for (int i = 0; i < n_warmup; ++i)
                impl == 0 ? prog.evaluate(d_in, d_out_i, B, 1) : wb_run(d_in, d_out_w, B);
            std::vector<double> us(n_timed);
            for (int i = 0; i < n_timed; ++i)
                us[i] = 1e6 * (impl == 0 ? prog.evaluate(d_in, d_out_i, B, 1)
                                         : wb_run(d_in, d_out_w, B));
            if (impl == 0)
                printf("%s,interpreter,%d,%.3f,%.3e\n", FUNC_NAME, B, median(us), md);
            else
                printf("%s,wb_w%d,%d,%.3f,%.3e\n", FUNC_NAME, block_warps, B, median(us), md);
            fflush(stdout);
        }

        for (int a = 0; a < b.n_args_in; ++a) cudaFree((void*)hin[a]);
        for (int a = 0; a < b.n_args_out; ++a) { cudaFree(hout_i[a]); cudaFree(hout_w[a]); }
        cudaFree(d_in); cudaFree(d_out_i); cudaFree(d_out_w);
    }
    return 0;
}
