// Latency of one MPPI rollout sweep: B samples carried through horizon * substeps chained evaluations
// of the cusadi2 step function (the part of a control step that scales with the sample count), as one
// CUDA graph. Level-parallel (the default) against the serial mode (one thread per sample walking
// CasADi's instruction stream, the original cusadi execution model) on the same plan.
//
//   mppi_bench <build_dir> <level|serial> <n_timed> B [B ...]
// Reads bench/x0.f64 (one state), bench/par.f64 and ylim.f64 from build_dir; every sample gets its own
// random knots (uniform within +-0.25 rad, clipped to the joint range). Prints one CSV row per B:
//   mode,dtype,B,launches,median_ms,min_ms,checksum
// checksum is the mean accumulated cost over the samples (identical inputs across modes and on the CPU).
#include "cusadi2.h"
#include <algorithm>
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
static S* upload(const std::vector<double>& h) {
    std::vector<S> t(h.begin(), h.end()); S* d; CK(cudaMalloc(&d, t.size() * sizeof(S)));
    CK(cudaMemcpy(d, t.data(), t.size() * sizeof(S), cudaMemcpyHostToDevice)); return d;
}
template <class T> static T** dptrs(const std::vector<T*>& h) {
    T** d; CK(cudaMalloc(&d, h.size() * sizeof(T*))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T*), cudaMemcpyHostToDevice)); return d;
}
// The knots of sample b, knot element i: a fixed hash, so the CPU baseline draws the same inputs.
static double knot_value(long long b, int i, double lo, double hi) {
    unsigned long long z = (unsigned long long)b * 0x9E3779B97F4A7C15ull + (unsigned long long)i * 0xBF58476D1CE4E5B9ull + 1;
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull; z ^= z >> 27; z *= 0x94D049BB133111EBull; z ^= z >> 31;
    double u = (double)(z >> 11) / 9007199254740992.0;
    return std::min(std::max(0.5 * u - 0.25, lo), hi);
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <build_dir> <level|serial> <n_timed> B [B ...]\n", argv[0]); return 2; }
    const std::string dir = argv[1]; const bool serial = !strcmp(argv[2], "serial"); const int n_timed = atoi(argv[3]);
    std::ifstream mf(dir + "/meta.txt"); std::map<std::string, double> meta; { std::string k; double v; while (mf >> k >> v) meta[k] = v; }
    const int NX = (int)meta["nx"], ND = (int)meta["n_knot"] * 12, NP = (int)meta["npar"], L = (int)meta["horizon"] * (int)meta["substeps"];
    const std::string tag = sizeof(S) == 4 ? "float" : "double";
    Plan plan = load_plan(dir + "/plans/go2_step1_" + tag + ".plan");
    std::vector<double> x0 = read_f64(dir + "/bench/x0.f64"), par = read_f64(dir + "/bench/par.f64"), ylim = read_f64(dir + "/ylim.f64");
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    for (int a = 4; a < argc; ++a) {
        const int B = atoi(argv[a]);
        Program fn(plan);
        if (serial) fn.use_serial((size_t)B * plan.serial_bytes());           // one thread per sample
        std::vector<double> hx((size_t)B * NX), hy((size_t)B * ND), hr((size_t)B * 4, 0.0), hp((size_t)B * NP);
        for (int b = 0; b < B; ++b) {
            std::copy(x0.begin(), x0.end(), hx.begin() + (size_t)b * NX); std::copy(par.begin(), par.end(), hp.begin() + (size_t)b * NP);
            for (int i = 0; i < ND; ++i) hy[(size_t)b * ND + i] = knot_value(b, i, ylim[i % 12], ylim[12 + i % 12]);
            hr[(size_t)b * 4] = x0[0] + 3.0; hr[(size_t)b * 4 + 1] = x0[1];
        }
        S *x[2] = {upload(hx), upload(hx)}, *xs = upload(hx), *Y = upload(hy), *rp = upload(hr), *pp = upload(hp);
        const S** in[2] = {(const S**)dptrs<S>({x[0], Y, rp, pp}), (const S**)dptrs<S>({x[1], Y, rp, pp})};
        S** out[2] = {dptrs<S>({x[1]}), dptrs<S>({x[0]})};
        auto sweep = [&]() {
            CK(cudaMemcpyAsync(x[0], xs, (size_t)B * NX * sizeof(S), cudaMemcpyDeviceToDevice, st));
            for (int s = 0; s < L; ++s) CK(fn.evaluate(in[s & 1], out[s & 1], B, st));
        };
        sweep(); CK(cudaStreamSynchronize(st));
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal)); sweep(); CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        std::vector<double> t(n_timed);
        for (int k = 0; k < n_timed; ++k) {
            CK(cudaEventRecord(e0, st)); CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t[k] = ms;
        }
        std::vector<S> res((size_t)B * NX); CK(cudaMemcpy(res.data(), x[L & 1], res.size() * sizeof(S), cudaMemcpyDeviceToHost));
        double sum = 0; for (int b = 0; b < B; ++b) sum += res[(size_t)b * NX + NX - 1];
        std::sort(t.begin(), t.end());
        printf("%s,%s,%d,%d,%.4f,%.4f,%.9g\n", serial ? "serial" : "level", tag.c_str(), B, L, t[n_timed / 2], t[0], sum / B);
        fflush(stdout);
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); cudaEventDestroy(e0); cudaEventDestroy(e1);
        for (S* p : {x[0], x[1], xs, Y, rp, pp}) cudaFree(p);
        for (int p = 0; p < 2; ++p) { cudaFree((void*)in[p]); cudaFree(out[p]); }
    }
    CK(cudaStreamDestroy(st));
    return 0;
}
