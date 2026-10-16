// ctypes-callable cusadi2 driver for the frax head-to-head (bench.py).
//
// One case = one plan uploaded at one batch size, with device-resident input and
// output buffers. bench.py times two calls from Python:
//
//   fb_eval(h)                  launch + stream sync; data stays on the device
//   fb_eval_io(h, h_in, h_out)  cudaMemcpy of every input host->device, launch,
//                               cudaMemcpy of every output device->host
//                               (host buffers are the caller's, pageable)
//
//   fb_setup(plan_path, batch) -> handle (>= 0), or -1 if the plan does not fit
//                                 the device's per-block shared memory
//   fb_n_in / fb_n_out / fb_in_len / fb_out_len   argument counts, elements per buffer
//   fb_block_size(h)            the runtime-picked threads per block
//   fb_upload / fb_download     one buffer host<->device (setup, output checks)
//   fb_teardown(h)
#include "cusadi2.h"
#include <vector>
using namespace cusadi2;
using S = Scalar;

struct Case {
    Plan plan;
    Program prog;
    std::vector<S*> in, out;
    const S** d_in = nullptr; S** d_out = nullptr;
    int batch = 0, block_size = 0;
    ~Case() {
        for (S* p : in) cudaFree(p);
        for (S* p : out) cudaFree(p);
        cudaFree(d_in); cudaFree(d_out);
    }
};

template <class T> static T** upload_ptrs(const std::vector<T*>& h) {
    T** d; CUSADI2_CHECK(cudaMalloc(&d, h.size() * sizeof(T*)));
    CUSADI2_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T*), cudaMemcpyHostToDevice));
    return d;
}

static std::vector<Case*> g_cases;
static Case* get(int h) { return g_cases[h]; }

extern "C" int fb_setup(const char* plan_path, int batch) {
    Case* c = new Case;
    c->plan = load_plan(plan_path);
    int smem = 0; cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (c->plan.shared_bytes() > smem) { delete c; return -1; }
    c->batch = batch;
    c->prog.upload(c->plan);
    c->block_size = c->plan.block_size ? c->plan.block_size : pick_block_size(c->plan);
    for (const Arg& a : c->plan.in) {
        S* d; CUSADI2_CHECK(cudaMalloc(&d, a.len(batch) * sizeof(S))); c->in.push_back(d);
    }
    for (const Arg& a : c->plan.out) {
        S* d; CUSADI2_CHECK(cudaMalloc(&d, a.len(batch) * sizeof(S))); c->out.push_back(d);
    }
    c->d_in = (const S**)upload_ptrs(c->in); c->d_out = upload_ptrs(c->out);
    g_cases.push_back(c);
    return (int)g_cases.size() - 1;
}
extern "C" int fb_n_in(int h) { return get(h)->plan.n_in(); }
extern "C" int fb_n_out(int h) { return get(h)->plan.n_out(); }
extern "C" long long fb_in_len(int h, int a) { return (long long)get(h)->plan.in[a].len(get(h)->batch); }
extern "C" long long fb_out_len(int h, int a) { return (long long)get(h)->plan.out[a].len(get(h)->batch); }
extern "C" int fb_block_size(int h) { return get(h)->block_size; }
extern "C" void fb_upload(int h, int a, const S* host) {
    Case* c = get(h);
    CUSADI2_CHECK(cudaMemcpy(c->in[a], host, c->plan.in[a].len(c->batch) * sizeof(S), cudaMemcpyHostToDevice));
}
extern "C" void fb_download(int h, int a, S* host) {
    Case* c = get(h);
    CUSADI2_CHECK(cudaMemcpy(host, c->out[a], c->plan.out[a].len(c->batch) * sizeof(S), cudaMemcpyDeviceToHost));
}
extern "C" void fb_eval(int h) {
    Case* c = get(h);
    CUSADI2_CHECK((cudaError_t)c->prog.evaluate(c->d_in, c->d_out, c->batch));
    CUSADI2_CHECK(cudaStreamSynchronize(0));
}
extern "C" void fb_eval_io(int h, const S* const* host_in, S* const* host_out) {
    Case* c = get(h);
    for (size_t a = 0; a < c->in.size(); ++a)
        CUSADI2_CHECK(cudaMemcpy(c->in[a], host_in[a], c->plan.in[a].len(c->batch) * sizeof(S),
                                 cudaMemcpyHostToDevice));
    CUSADI2_CHECK((cudaError_t)c->prog.evaluate(c->d_in, c->d_out, c->batch));
    for (size_t a = 0; a < c->out.size(); ++a)
        CUSADI2_CHECK(cudaMemcpy(host_out[a], c->out[a], c->plan.out[a].len(c->batch) * sizeof(S),
                                 cudaMemcpyDeviceToHost));
}
extern "C" void fb_teardown(int h) { delete g_cases[h]; g_cases[h] = nullptr; }
