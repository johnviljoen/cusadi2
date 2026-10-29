// The G1 step on cusadi2 as a shared library: B instances advanced n_steps times inside one CUDA graph
// (a reset to the initial state, then per step: smooth -> contact kernel -> integrate).
//
//   h = g1_create(build_dir, variant, B, n_steps, qpos0, qvel0, ctrl)
//        variant 0: contacts off (smooth_nogeom plan, qfrc_constraint = 0: the phase-1 premise workload)
//        variant 1: smooth plan + the fused contact kernel (cusadi2_contact.h, one block per instance)
//   g1_launch(h)                     one graph launch + stream synchronize (host wall-clock timing)
//   g1_time_events(h, n, ms)         n launches timed with CUDA events; ms[n] sorted; returns the median
//   g1_state(h, qpos, qvel)          the state after the last launch, host arrays B x nq, B x nv (double)
//   g1_solver(h, qacc, qfc, stats)   the last step's solver outputs: qacc, qfrc_constraint (B x nv) and
//                                    per-instance ints {nefc, ncon, niter, overflow} (B x 4)
//   g1_destroy(h)
//
// G1_NJMAX (compile-time, default 128) caps the constraint rows per instance, as mujoco-warp's njmax.
#include "cusadi2.h"
#include "cusadi2_contact.h"
#include "g1_model.h"
#ifndef G1_NJMAX
#define G1_NJMAX 128
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <string>
#include <vector>
using namespace cusadi2;
using S = Scalar;

#define CK(x) do { cudaError_t e_ = (cudaError_t)(x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

namespace {

template <class T> T** dptrs(const std::vector<T*>& h) {
    T** d; CK(cudaMalloc(&d, h.size() * sizeof(T*))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T*), cudaMemcpyHostToDevice)); return d;
}
S* upload(const double* h, size_t n) {
    std::vector<S> t(h, h + n); S* d; CK(cudaMalloc(&d, n * sizeof(S)));
    CK(cudaMemcpy(d, t.data(), n * sizeof(S), cudaMemcpyHostToDevice)); return d;
}
S* zeros(size_t n) { S* d; CK(cudaMalloc(&d, n * sizeof(S))); CK(cudaMemset(d, 0, n * sizeof(S))); return d; }

using Model = cusadi2_contact::model;
using Contact = cusadi2_contact::FusedStep<S, Model, G1_NJMAX, 128>;

struct Handle {
    int B, n_steps, nq, nv, nu, variant;
    cudaStream_t st;
    Plan plan_s, plan_i;
    Program* smooth; Program* integ;
    S *q[2], *v[2], *q0, *v0, *ctrl, *M, *qfs, *qfc, *qacc, *gpos, *grot, *cdof, *warm, *qacc_s;
    cusadi2_contact::Stats* stats;
    cusadi2_contact::RowDump<S> dump = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    const S **in_s[2], **in_i[2];
    S **out_s, **out_i[2];
    cudaGraph_t g; cudaGraphExec_t ge;

    void step_sequence() {
        CK(cudaMemcpyAsync(q[0], q0, (size_t)B * nq * sizeof(S), cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(v[0], v0, (size_t)B * nv * sizeof(S), cudaMemcpyDeviceToDevice, st));
        if (variant) CK(cudaMemsetAsync(warm, 0, (size_t)B * nv * sizeof(S), st));      // MuJoCo's reset: qacc_warmstart = 0
        for (int s = 0; s < n_steps; ++s) {
            CK(smooth->evaluate(in_s[s & 1], out_s, B, st));
            if (variant) CK(Contact::launch(B, q[s & 1], v[s & 1], M, qfs, gpos, grot, cdof, warm, qfc, qacc_s, stats, st, 0, dump));
            CK(integ->evaluate(in_i[s & 1], out_i[s & 1], B, st));
        }
    }
};

}  // namespace

extern "C" {

void* g1_create(const char* build_dir, int variant, int B, int n_steps, const double* qpos, const double* qvel, const double* ctrl) {
    Handle* h = new Handle();
    h->B = B; h->n_steps = n_steps; h->variant = variant;
    const std::string dir = build_dir, tag = sizeof(S) == 4 ? "float" : "double";
    h->plan_s = load_plan(dir + "/plans/" + (variant ? "smooth_" : "smooth_nogeom_") + tag + ".plan");
    h->plan_i = load_plan(dir + "/plans/integrate_" + tag + ".plan");
    h->nq = h->plan_s.in[0].nnz(); h->nv = h->plan_s.in[1].nnz(); h->nu = h->plan_s.in[2].nnz();
    const int nq = h->nq, nv = h->nv, nu = h->nu;
    CK(cudaStreamCreateWithFlags(&h->st, cudaStreamNonBlocking));
    h->smooth = new Program(h->plan_s); h->integ = new Program(h->plan_i);
    h->q0 = upload(qpos, (size_t)B * nq); h->v0 = upload(qvel, (size_t)B * nv); h->ctrl = upload(ctrl, (size_t)B * nu);
    for (int p = 0; p < 2; ++p) { h->q[p] = zeros((size_t)B * nq); h->v[p] = zeros((size_t)B * nv); }
    h->M = zeros((size_t)B * nv * nv); h->qfs = zeros((size_t)B * nv); h->qfc = zeros((size_t)B * nv); h->qacc = zeros((size_t)B * nv);
    std::vector<S*> outs = {h->M, h->qfs};
    h->warm = h->qacc_s = nullptr; h->stats = nullptr;
    if (variant) {
        if (h->plan_s.out[0].nnz() != Model::M_NNZ || nv != Model::NV) { fprintf(stderr, "plan / model header mismatch\n"); exit(3); }
        h->gpos = zeros(h->plan_s.out[2].len(B)); h->grot = zeros(h->plan_s.out[3].len(B)); h->cdof = zeros(h->plan_s.out[4].len(B));
        outs.push_back(h->gpos); outs.push_back(h->grot); outs.push_back(h->cdof);
        h->warm = zeros((size_t)B * nv); h->qacc_s = zeros((size_t)B * nv);
        CK(cudaMalloc(&h->stats, (size_t)B * sizeof(cusadi2_contact::Stats)));
        CK(Contact::setup());
        if (getenv("G1_DUMP")) {
            h->dump.J = zeros((size_t)B * G1_NJMAX * nv); h->dump.D = zeros((size_t)B * G1_NJMAX); h->dump.aref = zeros((size_t)B * G1_NJMAX);
            h->dump.jaref = zeros((size_t)B * G1_NJMAX); CK(cudaMalloc(&h->dump.type, (size_t)B * G1_NJMAX * sizeof(int)));
            h->dump.iters = zeros((size_t)B * Model::ITERATIONS * 8);
            CK(cudaMalloc(&h->dump.prof, (size_t)B * 32 * sizeof(long long))); CK(cudaMemset(h->dump.prof, 0, (size_t)B * 32 * sizeof(long long)));
        }
    } else {
        h->gpos = h->grot = h->cdof = nullptr;
    }
    h->out_s = dptrs<S>(outs);
    for (int p = 0; p < 2; ++p) {
        h->in_s[p] = (const S**)dptrs<S>({h->q[p], h->v[p], h->ctrl});
        h->in_i[p] = (const S**)dptrs<S>({h->q[p], h->v[p], h->M, h->qfs, h->qfc});
        h->out_i[p] = dptrs<S>({h->q[1 - p], h->v[1 - p], h->qacc});
    }
    h->step_sequence(); CK(cudaStreamSynchronize(h->st));          // warm (module load, shared-memory setup) outside the capture
    CK(cudaStreamBeginCapture(h->st, cudaStreamCaptureModeThreadLocal)); h->step_sequence(); CK(cudaStreamEndCapture(h->st, &h->g));
    CK(cudaGraphInstantiate(&h->ge, h->g, 0));
    CK(cudaGraphLaunch(h->ge, h->st)); CK(cudaStreamSynchronize(h->st));
    return h;
}

void g1_launch(void* hp) {
    Handle* h = (Handle*)hp;
    CK(cudaGraphLaunch(h->ge, h->st)); CK(cudaStreamSynchronize(h->st));
}

double g1_time_events(void* hp, int n, double* ms) {
    Handle* h = (Handle*)hp;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int k = 0; k < n; ++k) {
        CK(cudaEventRecord(e0, h->st)); CK(cudaGraphLaunch(h->ge, h->st)); CK(cudaEventRecord(e1, h->st)); CK(cudaEventSynchronize(e1));
        float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms[k] = t;
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    std::sort(ms, ms + n);
    return ms[n / 2];
}

void g1_state(void* hp, double* qpos, double* qvel) {
    Handle* h = (Handle*)hp;
    const int p = h->n_steps & 1;
    std::vector<S> tq((size_t)h->B * h->nq), tv((size_t)h->B * h->nv);
    CK(cudaMemcpy(tq.data(), h->q[p], tq.size() * sizeof(S), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tv.data(), h->v[p], tv.size() * sizeof(S), cudaMemcpyDeviceToHost));
    std::copy(tq.begin(), tq.end(), qpos); std::copy(tv.begin(), tv.end(), qvel);
}

void g1_solver(void* hp, double* qacc, double* qfc, int* stats) {
    Handle* h = (Handle*)hp;
    if (!h->variant) return;
    std::vector<S> ta((size_t)h->B * h->nv), tf((size_t)h->B * h->nv);
    CK(cudaMemcpy(ta.data(), h->qacc_s, ta.size() * sizeof(S), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tf.data(), h->qfc, tf.size() * sizeof(S), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(stats, h->stats, (size_t)h->B * sizeof(cusadi2_contact::Stats), cudaMemcpyDeviceToHost));
    std::copy(ta.begin(), ta.end(), qacc); std::copy(tf.begin(), tf.end(), qfc);
}

// the last step's assembled rows (needs G1_DUMP set when the handle was created): J (B x njmax x nv), D, aref, jaref (B x njmax), type (ints)
int g1_rows(void* hp, double* J, double* D, double* aref, double* jaref, int* type) {
    Handle* h = (Handle*)hp;
    if (!h->dump.J) return 0;
    const size_t nr = (size_t)h->B * G1_NJMAX;
    std::vector<S> tJ(nr * h->nv), tD(nr), tA(nr), tR(nr);
    CK(cudaMemcpy(tJ.data(), h->dump.J, tJ.size() * sizeof(S), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tD.data(), h->dump.D, nr * sizeof(S), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tA.data(), h->dump.aref, nr * sizeof(S), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tR.data(), h->dump.jaref, nr * sizeof(S), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(type, h->dump.type, nr * sizeof(int), cudaMemcpyDeviceToHost));
    std::copy(tJ.begin(), tJ.end(), J); std::copy(tD.begin(), tD.end(), D); std::copy(tA.begin(), tA.end(), aref); std::copy(tR.begin(), tR.end(), jaref);
    return G1_NJMAX;
}

// per-iteration solver statistics of the last step (needs G1_DUMP): B x ITERATIONS x 8 doubles; returns ITERATIONS
int g1_iters(void* hp, double* out) {
    Handle* h = (Handle*)hp;
    if (!h->dump.iters) return 0;
    std::vector<S> tt((size_t)h->B * Model::ITERATIONS * 8);
    CK(cudaMemcpy(tt.data(), h->dump.iters, tt.size() * sizeof(S), cudaMemcpyDeviceToHost));
    std::copy(tt.begin(), tt.end(), out);
    return Model::ITERATIONS;
}

// the clock64() phase stamps of the last step (needs G1_DUMP): B x 32 int64; returns 32
int g1_prof(void* hp, long long* out) {
    Handle* h = (Handle*)hp;
    if (!h->dump.prof) return 0;
    CK(cudaMemcpy(out, h->dump.prof, (size_t)h->B * 32 * sizeof(long long), cudaMemcpyDeviceToHost));
    return 32;
}

void g1_destroy(void* hp) {
    Handle* h = (Handle*)hp;
    CK(cudaGraphExecDestroy(h->ge)); CK(cudaGraphDestroy(h->g));
    delete h->smooth; delete h->integ;
    for (S* p : {h->q[0], h->q[1], h->v[0], h->v[1], h->q0, h->v0, h->ctrl, h->M, h->qfs, h->qfc, h->qacc, h->gpos, h->grot, h->cdof, h->warm, h->qacc_s})
        if (p) cudaFree(p);
    if (h->stats) cudaFree(h->stats);
    for (void* p : {(void*)h->dump.J, (void*)h->dump.D, (void*)h->dump.aref, (void*)h->dump.jaref, (void*)h->dump.type, (void*)h->dump.iters, (void*)h->dump.prof}) if (p) cudaFree(p);
    for (int p = 0; p < 2; ++p) { cudaFree((void*)h->in_s[p]); cudaFree((void*)h->in_i[p]); cudaFree(h->out_i[p]); }
    cudaFree(h->out_s);
    CK(cudaStreamDestroy(h->st));
    delete h;
}

}  // extern "C"
