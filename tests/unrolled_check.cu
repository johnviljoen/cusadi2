// GPU check of one unrolled kernel (cusadi2.unrolled.write_unrolled_source)
// against its CasADi reference file. Compiled per function together with the
// emitted .cu:  nvcc -DFN=<name> unrolled_check.cu <name>_unrolled.cu
//   unrolled_check f.ref
#include "cusadi2.h"
#include <cmath>
#include <cstdio>
#include <vector>
using cusadi2::Scalar;
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define STR_(a) #a
#define STR(a) STR_(a)
extern "C" int CAT(FN, _unrolled_sz_w)();
extern "C" int CAT(FN, _unrolled_n_stages)();
extern "C" int CAT(FN, _unrolled_launch)(const Scalar* const*, Scalar* const*, Scalar*, int, int,
                                          cudaStream_t);

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s f.ref\n", argv[0]); return 2; }
    const int batch = 2;
    cusadi2::detail::Reader r(argv[1]);
    char magic[4]; r.raw(magic, 4);
    if (std::string(magic, 4) != "CS2R") { printf("bad reference magic\n"); return 1; }
    r.i();
    int n_in = r.i(), n_out = r.i(), n_stages = r.i();
    std::vector<std::vector<double>> ref;
    for (int a = 0; a < n_in + n_out; ++a) ref.push_back(r.arr<double>(r.i()));
    if (n_stages != CAT(FN, _unrolled_n_stages)()) { printf("horizon mismatch\n"); return 1; }

    std::vector<const Scalar*> hin(n_in);
    for (int a = 0; a < n_in; ++a) {
        std::vector<Scalar> h;
        for (int b = 0; b < batch; ++b) h.insert(h.end(), ref[a].begin(), ref[a].end());
        Scalar* d; cudaMalloc(&d, h.size() * sizeof(Scalar));
        cudaMemcpy(d, h.data(), h.size() * sizeof(Scalar), cudaMemcpyHostToDevice);
        hin[a] = d;
    }
    const Scalar** d_in; cudaMalloc(&d_in, n_in * sizeof(Scalar*));
    cudaMemcpy(d_in, hin.data(), n_in * sizeof(Scalar*), cudaMemcpyHostToDevice);
    std::vector<Scalar*> hout(n_out);
    for (int a = 0; a < n_out; ++a) {
        cudaMalloc(&hout[a], batch * ref[n_in + a].size() * sizeof(Scalar));
        cudaMemset(hout[a], 0, batch * ref[n_in + a].size() * sizeof(Scalar));
    }
    Scalar** d_out; cudaMalloc(&d_out, n_out * sizeof(Scalar*));
    cudaMemcpy(d_out, hout.data(), n_out * sizeof(Scalar*), cudaMemcpyHostToDevice);
    Scalar* work; cudaMalloc(&work, (size_t)batch * n_stages * CAT(FN, _unrolled_sz_w)() * sizeof(Scalar));

    int rc = CAT(FN, _unrolled_launch)(d_in, d_out, work, batch, 32, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (rc || err) { printf("launch failed: %s\n", cudaGetErrorString(err ? err : (cudaError_t)rc)); return 1; }

    double maxrel = 0;
    for (int a = 0; a < n_out; ++a) {
        const auto& want = ref[n_in + a];
        std::vector<Scalar> got(batch * want.size());
        cudaMemcpy(got.data(), hout[a], got.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
        for (int b = 0; b < batch; ++b)
            for (size_t i = 0; i < want.size(); ++i)
                maxrel = fmax(maxrel, fabs(got[b * want.size() + i] - want[i]) / (1.0 + fabs(want[i])));
    }
    double tol = sizeof(Scalar) == 4 ? 1e-4 : 1e-9;
    bool ok = maxrel < tol;
    printf("[%-8s] %s unrolled n_stages=%d batch=%d maxrel=%.3e -> %s\n", STR(FN),
           sizeof(Scalar) == 4 ? "f32" : "f64", n_stages, batch, maxrel, ok ? "PASS" : "FAIL");
    return !ok;
}
