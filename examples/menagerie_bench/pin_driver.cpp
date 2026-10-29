// Pure-Pinocchio CPU baseline for the Menagerie campaign: Pinocchio's own C++
// algorithms (no CasADi, no code generation) evaluated over n_items problems
// (distinct configurations) split statically over n_threads OpenMP threads,
// each with its own pinocchio::Data and output buffer: median of n_timed
// repetitions, us -- the protocol of paper_results' CPU_DRIVER.
//
//   pin_driver <model.pin> <frames.txt> <fk|fkjac|fkhess> <n_warm> <n_timed> <n_items> <n_threads>
//   pin_driver <model.pin> <frames.txt> <fk|fkjac|fkhess> check <q.bin> <out.bin>
//
// model.pin is the gated Pinocchio model (model.load_pinocchio) written by
// Model.saveToBinary; frames.txt lists the BODY frame ids of the campaign's
// frame set in order. Outputs, per frame in that order:
//   fk      translation (3) + column-major rotation (9)         -- as model.build_fk
//   fkjac   the 6 x nv LOCAL_WORLD_ALIGNED frame Jacobian, column-major (dense)
//   fkhess  Pinocchio's 6 x nv x nv LOCAL_WORLD_ALIGNED frame kinematic Hessian
//           (getFrameKinematicHessian), column-major
// The outputs are what the library writes: fk copies each frame's placement
// (12 scalars) out of data.oMf; fkjac / fkhess are written by getFrameJacobian /
// getFrameKinematicHessian themselves into per-frame buffers that ARE the
// output (the thread's own in the idealized build; for fkjac in the transfer
// build an Eigen::Map onto the item's pinned slot). Those routines write only
// the frame's supporting columns, so every buffer is zeroed once at allocation
// and never again (Pinocchio's contract: "initialized with zero when calling
// this function for the first time"); nothing is copied twice. The one
// exception is the transfer build's Hessian: getFrameKinematicHessian takes an
// owning Eigen::Tensor, so each frame's tensor is memcpy'd into the pinned
// slot (the cost of packing Pinocchio's dense Hessian for the GPU).
// -DSCALAR=float builds the single-precision model (Model::cast<float>()).
// -DTRANSFER: the data lives on the GPU. The timed region also copies every
// input device->host (pinned) before and every output host->device after the
// evaluation; each thread writes its items' outputs into the full host buffer.
// check mode: one configuration from q.bin (nq doubles) -> out.bin (doubles).
#include <pinocchio/serialization.hpp>
#include <pinocchio/multibody.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <time.h>
#ifndef SCALAR
#define SCALAR double
#endif
#ifdef TRANSFER
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define HOST_ALLOC(p, bytes) CK(cudaMallocHost((void**)&(p), (bytes)))
#else
#define HOST_ALLOC(p, bytes) (p) = (SCALAR*)malloc(bytes)
#endif

typedef pinocchio::ModelTpl<SCALAR> Model;
typedef pinocchio::DataTpl<SCALAR> Data;
typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> Vec;
typedef Eigen::Matrix<SCALAR, 6, Eigen::Dynamic> Mat6x;
typedef pinocchio::Tensor<SCALAR, 3> Tensor3;
enum Func { FK, FKJAC, FKHESS };

struct Worker {                        // one per thread: Data + the per-frame outputs
    Data data;
    std::vector<Mat6x> J;              // fkjac: per frame (the output, idealized build)
    std::vector<Tensor3> H;            // fkhess: per frame (the output; staged to the slot under TRANSFER)
    explicit Worker(const Model& m) : data(m) {}
};

// Evaluate one configuration. `out` is the item's output slot: fk's 12 scalars
// per frame always; the Jacobian's 6 x nv per frame when `slot` (TRANSFER /
// check: written in place through a Map), else the worker's own J; the
// Hessian is computed into the worker's tensors and copied to `out` when `slot`.
static inline void eval(Func f, const Model& model, Worker& w, const std::vector<pinocchio::FrameIndex>& frames,
                        const SCALAR* q_in, SCALAR* out, bool slot) {
    Eigen::Map<const Vec> q(q_in, model.nq);
    const int nv = model.nv;
    switch (f) {
    case FK:
        pinocchio::framesForwardKinematics(model, w.data, q);
        for (size_t i = 0; i < frames.size(); ++i) {
            const auto& M = w.data.oMf[frames[i]];
            Eigen::Map<Eigen::Matrix<SCALAR, 3, 1>>(out + 12 * i) = M.translation();
            Eigen::Map<Eigen::Matrix<SCALAR, 3, 3>>(out + 12 * i + 3) = M.rotation();
        }
        break;
    case FKJAC:
        pinocchio::computeJointJacobians(model, w.data, q);
        pinocchio::updateFramePlacements(model, w.data);
        for (size_t i = 0; i < frames.size(); ++i) {
            if (slot) {
                Eigen::Map<Mat6x> Jm(out + (size_t)6 * nv * i, 6, nv);
                pinocchio::getFrameJacobian(model, w.data, frames[i], pinocchio::LOCAL_WORLD_ALIGNED, Jm);
            } else {
                pinocchio::getFrameJacobian(model, w.data, frames[i], pinocchio::LOCAL_WORLD_ALIGNED, w.J[i]);
            }
        }
        break;
    case FKHESS:
        pinocchio::computeJointKinematicHessians(model, w.data, q);
        for (size_t i = 0; i < frames.size(); ++i) {
            const auto& fr = model.frames[frames[i]];
            pinocchio::getFrameKinematicHessian(model, w.data, fr.parentJoint, fr.placement,
                                                pinocchio::LOCAL_WORLD_ALIGNED, w.H[i]);
            if (slot)
                std::memcpy(out + (size_t)6 * nv * nv * i, w.H[i].data(), sizeof(SCALAR) * 6 * nv * nv);
        }
        break;
    }
}

static int cmp(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y);
}

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: see header\n"); return 2; }
    pinocchio::Model md;
    md.loadFromBinary(argv[1]);
    const Model model = md.cast<SCALAR>();
    std::vector<pinocchio::FrameIndex> frames;
    { std::ifstream f(argv[2]); long id; while (f >> id) frames.push_back((pinocchio::FrameIndex)id); }
    const std::string fname = argv[3];
    const Func func = fname == "fk" ? FK : fname == "fkjac" ? FKJAC : FKHESS;
    const int nq = model.nq, nv = model.nv;
    const size_t nf = frames.size();
    const size_t out_per = func == FK ? 12 * nf : func == FKJAC ? (size_t)6 * nv * nf : (size_t)6 * nv * nv * nf;

    auto make_worker = [&](Worker& w) {
        if (func == FKJAC) { w.J.resize(nf); for (auto& J : w.J) { J.resize(6, nv); J.setZero(); } }
        if (func == FKHESS) { w.H.resize(nf); for (auto& H : w.H) { H.resize(6, nv, nv); H.setZero(); } }
    };

    if (std::string(argv[4]) == "check") {        // one configuration, outputs in double
        std::vector<double> qd(nq);
        FILE* fq = fopen(argv[5], "rb");
        if (!fq || fread(qd.data(), sizeof(double), nq, fq) != (size_t)nq) { fprintf(stderr, "bad q.bin\n"); return 1; }
        fclose(fq);
        std::vector<SCALAR> q(qd.begin(), qd.end()), out(out_per, SCALAR(0));
        Worker w(model); make_worker(w);
        eval(func, model, w, frames, q.data(), out.data(), true);
        std::vector<double> od(out.begin(), out.end());
        FILE* fo = fopen(argv[6], "wb");
        fwrite(od.data(), sizeof(double), od.size(), fo); fclose(fo);
        printf("%zu\n", out_per);
        return 0;
    }

    const int n_warm = atoi(argv[4]), n_timed = atoi(argv[5]);
    const long n_items = atol(argv[6]);
    const int n_threads = atoi(argv[7]);
    omp_set_num_threads(n_threads);

    // Inputs: n_items valid configurations, neutral + a random tangent step
    // (quaternions stay unit), as uniform [-1, 1] in the tangent.
    srand(0);
    SCALAR* in;
    HOST_ALLOC(in, (size_t)n_items * nq * sizeof(SCALAR));
    {
        const Vec q0 = pinocchio::neutral(model);
        Vec v(nv), q(nq);
        for (long j = 0; j < n_items; ++j) {
            for (int i = 0; i < nv; ++i) v[i] = (SCALAR)(2.0 * rand() / RAND_MAX - 1.0);
            q = pinocchio::integrate(model, q0, v);
            std::memcpy(in + (size_t)j * nq, q.data(), sizeof(SCALAR) * nq);
        }
    }
    std::vector<Worker> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) { workers.emplace_back(model); make_worker(workers.back()); }
#ifdef TRANSFER
    // Device copy of the inputs (filled once) and of the outputs; every item's
    // outputs in one pinned host buffer.
    SCALAR *d_in, *d_out, *out;
    const size_t in_bytes = (size_t)n_items * nq * sizeof(SCALAR), out_bytes = (size_t)n_items * out_per * sizeof(SCALAR);
    CK(cudaMalloc((void**)&d_in, in_bytes));
    CK(cudaMemcpy(d_in, in, in_bytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**)&d_out, out_bytes));
    HOST_ALLOC(out, out_bytes);
    std::memset(out, 0, out_bytes);            // zero once (see header)
#define OUT_OF(t, j) (out + (size_t)(j) * out_per)
#define SLOT true
#define TRANSFER_IN  CK(cudaMemcpy(in, d_in, in_bytes, cudaMemcpyDeviceToHost));
#define TRANSFER_OUT CK(cudaMemcpy(d_out, out, out_bytes, cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
#else
    // per thread: fk's placements (the Jacobian / Hessian outputs are the worker's own buffers)
    std::vector<std::vector<SCALAR>> outs(n_threads, std::vector<SCALAR>(func == FK ? out_per : 0, SCALAR(0)));
#define OUT_OF(t, j) (outs[t].data())
#define SLOT false
#define TRANSFER_IN
#define TRANSFER_OUT
#endif
#define EVAL_ALL                                                                            \
    if (n_threads == 1) {                     /* no fork/join on the 1-thread path */       \
        for (long j = 0; j < n_items; ++j) eval(func, model, workers[0], frames, in + (size_t)j * nq, OUT_OF(0, j), SLOT); \
    } else {                                                                                \
        _Pragma("omp parallel for schedule(static)")                                        \
        for (long j = 0; j < n_items; ++j) {                                                \
            const int t = omp_get_thread_num();                                             \
            eval(func, model, workers[t], frames, in + (size_t)j * nq, OUT_OF(t, j), SLOT); \
        }                                                                                   \
    }
    for (int k = 0; k < n_warm; ++k) { TRANSFER_IN EVAL_ALL TRANSFER_OUT }
    std::vector<double> ts(n_timed);
    struct timespec t0, t1;
    for (int k = 0; k < n_timed; ++k) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        TRANSFER_IN EVAL_ALL TRANSFER_OUT
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ts[k] = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) * 1e-3;
    }
    qsort(ts.data(), n_timed, sizeof(double), cmp);
    printf("%.4f\n", ts[n_timed / 2]);
    return 0;
}
