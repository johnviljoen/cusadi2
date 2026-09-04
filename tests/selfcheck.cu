// GPU self-check: for each (plan, reference) pair on the command line, load the
// plan, run it over a batch of two identical instances in both the level-parallel
// and the serial mode, and compare against the CasADi reference. Exit code is the
// number of failures.
//
//   selfcheck f.plan f.ref [g.plan g.ref ...]
#include "cusadi2.h"
#include <cmath>
#include <cstdio>
#include <vector>
using cusadi2::Scalar;

static std::vector<std::vector<double>> read_ref(const char* path, int& n_stages) {
    cusadi2::detail::Reader r(path);
    char magic[4]; r.raw(magic, 4);
    if (std::string(magic, 4) != "CS2R") throw std::runtime_error("bad reference magic");
    r.i();                                       // version
    int n_in = r.i(), n_out = r.i(); n_stages = r.i();
    std::vector<std::vector<double>> bufs;
    for (int a = 0; a < n_in + n_out; ++a) bufs.push_back(r.arr<double>(r.i()));
    return bufs;
}

static int run(const char* plan_path, const char* ref_path) {
    const int batch = 2;
    cusadi2::Plan plan = cusadi2::load_plan(plan_path);
    int ref_stages;
    auto ref = read_ref(ref_path, ref_stages);
    if (ref_stages != plan.n_stages) { printf("[%s] horizon mismatch\n", plan.name.c_str()); return 1; }
    cusadi2::Program prog(plan);

    // inputs: the reference layout replicated for each instance
    std::vector<const Scalar*> hin(plan.n_in());
    for (int a = 0; a < plan.n_in(); ++a) {
        std::vector<Scalar> h;
        for (int b = 0; b < batch; ++b) h.insert(h.end(), ref[a].begin(), ref[a].end());
        Scalar* d; cudaMalloc(&d, h.size() * sizeof(Scalar));
        cudaMemcpy(d, h.data(), h.size() * sizeof(Scalar), cudaMemcpyHostToDevice);
        hin[a] = d;
    }
    const Scalar** d_in; cudaMalloc(&d_in, hin.size() * sizeof(Scalar*));
    cudaMemcpy(d_in, hin.data(), hin.size() * sizeof(Scalar*), cudaMemcpyHostToDevice);

    std::vector<Scalar*> hout(plan.n_out());
    for (int a = 0; a < plan.n_out(); ++a) cudaMalloc(&hout[a], plan.out[a].len(batch) * sizeof(Scalar));
    Scalar** d_out; cudaMalloc(&d_out, hout.size() * sizeof(Scalar*));
    cudaMemcpy(d_out, hout.data(), hout.size() * sizeof(Scalar*), cudaMemcpyHostToDevice);

    int fails = 0;
    for (const char* mode : {"level", "serial"}) {
        if (mode[0] == 's')                            // room for 3 threads: staged cases loop
            prog.use_serial(3 * plan.peak_live_slots * sizeof(Scalar), 32);
        for (int a = 0; a < plan.n_out(); ++a) cudaMemset(hout[a], 0, plan.out[a].len(batch) * sizeof(Scalar));
        int rc = prog.evaluate(d_in, d_out, batch);
        cudaError_t err = cudaDeviceSynchronize();
        if (rc || err) { printf("[%s] %s launch failed: %s\n", plan.name.c_str(), mode, cudaGetErrorString(err ? err : (cudaError_t)rc)); return 1; }

        // relative error, so deep DAGs and float builds are judged fairly
        double maxrel = 0;
        for (int a = 0; a < plan.n_out(); ++a) {
            std::vector<Scalar> got(plan.out[a].len(batch));
            cudaMemcpy(got.data(), hout[a], got.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
            const auto& want = ref[plan.n_in() + a];
            for (int b = 0; b < batch; ++b)
                for (size_t i = 0; i < want.size(); ++i) {
                    double g = got[b * want.size() + i], w = want[i];
                    maxrel = fmax(maxrel, fabs(g - w) / (1.0 + fabs(w)));
                }
        }
        double tol = sizeof(Scalar) == 4 ? 1e-4 : 1e-9;
        bool ok = maxrel < tol;
        printf("[%-8s] %s %-6s n_stages=%d batch=%d maxrel=%.3e -> %s\n", plan.name.c_str(),
               sizeof(Scalar) == 4 ? "f32" : "f64", mode, plan.n_stages, batch, maxrel, ok ? "PASS" : "FAIL");
        fails += !ok;
    }

    for (auto p : hin) cudaFree((void*)p);
    for (auto p : hout) cudaFree(p);
    cudaFree(d_in); cudaFree(d_out);
    return fails;
}

int main(int argc, char** argv) {
    int fails = 0;
    for (int i = 1; i + 1 < argc; i += 2) fails += run(argv[i], argv[i + 1]);
    printf(fails ? "FAIL\n" : "ALL PASS\n");
    return fails;
}
