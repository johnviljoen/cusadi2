// MuJoCo CPU baseline for the Menagerie campaign: libmujoco's own C
// algorithms over n_items problems split statically over n_threads OpenMP
// threads, each with its own mjData and output buffer: median of n_timed
// repetitions, us -- the protocol of paper_results' CPU_DRIVER, the series of
// the MJX baseline (run.py run_mjx), double precision only (mjtNum).
//
//   mj_driver <model.mjb> <bodies.txt> <fk|jac|fk_rollout|jac_rollout> <n_warm> <n_timed> <n_items> <n_threads> <n_steps>
//   mj_driver <model.mjb> <bodies.txt> <fk|jac|fk_rollout|jac_rollout> check <q.bin> <out.bin> <n_steps>
//
// model.mjb is the campaign's MJCF compiled by MuJoCo with contact detection
// and the constraint solver disabled (_disable_contacts_and_constraints) and
// saved by mj_saveModel; bodies.txt lists the named body ids (world excluded)
// in MuJoCo body order. Per item:
//   fk           mj_kinematics + mj_comPos; per body xpos (3) + xmat (9, row-major)
//   jac          + mj_jacBody per body (MuJoCo's analytical body Jacobian);
//                per body jacp (3 x nv, row-major) then jacr (3 x nv)
//   *_rollout    one instance: qpos from the input, qvel = act = 0, n_steps x
//                mj_step, the fk / jac block extracted after every step
//                (n_steps blocks per item; the placements of the state the step
//                started from, as mjx.step's output carries them)
// -DTRANSFER: the data lives on the GPU. The timed region also copies every
// input device->host (pinned) before and every output host->device after the
// evaluation; each thread writes its items' outputs into the full host buffer.
// check mode: one configuration from q.bin (nq doubles) -> out.bin (doubles).
#include <mujoco/mujoco.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef TRANSFER
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define HOST_ALLOC(p, bytes) CK(cudaMallocHost((void**)&(p), (bytes)))
#else
#define HOST_ALLOC(p, bytes) (p) = malloc(bytes)
#endif

enum Series { FK, JAC, FK_ROLLOUT, JAC_ROLLOUT };

static void extract(const mjModel* m, mjData* d, const int* bodies, int nb, int with_jac, mjtNum* out) {
    const int nv = m->nv;
    if (!with_jac) {
        for (int i = 0; i < nb; ++i) {
            memcpy(out + 12 * i, d->xpos + 3 * bodies[i], 3 * sizeof(mjtNum));
            memcpy(out + 12 * i + 3, d->xmat + 9 * bodies[i], 9 * sizeof(mjtNum));
        }
    } else {
        for (int i = 0; i < nb; ++i)
            mj_jacBody(m, d, out + (size_t)6 * nv * i, out + (size_t)6 * nv * i + 3 * nv, bodies[i]);
    }
}

// One item into out: a batched series evaluates the kinematics of one
// configuration; a rollout steps one instance n_steps times.
static inline void eval(enum Series s, const mjModel* m, mjData* d, const int* bodies, int nb,
                        const mjtNum* q, int n_steps, size_t block, mjtNum* out) {
    memcpy(d->qpos, q, m->nq * sizeof(mjtNum));
    if (s == FK || s == JAC) {
        mj_kinematics(m, d);
        mj_comPos(m, d);
        extract(m, d, bodies, nb, s == JAC, out);
        return;
    }
    memset(d->qvel, 0, m->nv * sizeof(mjtNum));
    memset(d->act, 0, m->na * sizeof(mjtNum));
    memset(d->ctrl, 0, m->nu * sizeof(mjtNum));
    d->time = 0;
    for (int k = 0; k < n_steps; ++k) {
        mj_step(m, d);
        extract(m, d, bodies, nb, s == JAC_ROLLOUT, out + block * k);
    }
}

static int cmp(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y);
}

int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "usage: see header\n"); return 2; }
    mjModel* m = mj_loadModel(argv[1], NULL);
    if (!m) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    int* bodies = malloc(m->nbody * sizeof(int));
    int nb = 0;
    { FILE* f = fopen(argv[2], "r"); int id; while (f && fscanf(f, "%d", &id) == 1) bodies[nb++] = id; if (f) fclose(f); }
    const char* sname = argv[3];
    const enum Series s = !strcmp(sname, "fk") ? FK : !strcmp(sname, "jac") ? JAC
                        : !strcmp(sname, "fk_rollout") ? FK_ROLLOUT : JAC_ROLLOUT;
    const int nq = m->nq, nv = m->nv;
    const int with_jac = (s == JAC || s == JAC_ROLLOUT), rollout = (s == FK_ROLLOUT || s == JAC_ROLLOUT);
    const size_t block = with_jac ? (size_t)6 * nv * nb : (size_t)12 * nb;     // one fk / jac extraction

    if (!strcmp(argv[4], "check")) {
        const int n_steps = atoi(argv[7]);
        const size_t out_per = block * (rollout ? n_steps : 1);
        mjtNum* q = malloc(nq * sizeof(mjtNum));
        FILE* fq = fopen(argv[5], "rb");
        if (!fq || fread(q, sizeof(mjtNum), nq, fq) != (size_t)nq) { fprintf(stderr, "bad q.bin\n"); return 1; }
        fclose(fq);
        mjtNum* out = calloc(out_per, sizeof(mjtNum));
        mjData* d = mj_makeData(m);
        eval(s, m, d, bodies, nb, q, n_steps, block, out);
        FILE* fo = fopen(argv[6], "wb");
        fwrite(out, sizeof(mjtNum), out_per, fo); fclose(fo);
        printf("%zu\n", out_per);
        return 0;
    }
    const int n_warm = atoi(argv[4]), n_timed = atoi(argv[5]);
    const long n_items = atol(argv[6]);
    const int n_threads = atoi(argv[7]);
    const int n_steps = atoi(argv[8]);
    const size_t out_per = block * (rollout ? n_steps : 1);
    omp_set_num_threads(n_threads);

    // Inputs: n_items valid configurations = qpos0 integrated along a random
    // tangent step, uniform [-1, 1] (quaternions stay unit).
    srand(0);
    mjtNum* in;
    HOST_ALLOC(in, (size_t)n_items * nq * sizeof(mjtNum));
    {
        mjtNum* v = malloc(nv * sizeof(mjtNum));
        for (long j = 0; j < n_items; ++j) {
            mjtNum* q = in + (size_t)j * nq;
            memcpy(q, m->qpos0, nq * sizeof(mjtNum));
            for (int i = 0; i < nv; ++i) v[i] = 2.0 * rand() / RAND_MAX - 1.0;
            mj_integratePos(m, q, v, 1.0);
        }
        free(v);
    }
    mjData** datas = malloc(n_threads * sizeof(*datas));
    for (int t = 0; t < n_threads; ++t) datas[t] = mj_makeData(m);
#ifdef TRANSFER
    mjtNum *d_in, *d_out, *out;
    const size_t in_bytes = (size_t)n_items * nq * sizeof(mjtNum), out_bytes = (size_t)n_items * out_per * sizeof(mjtNum);
    CK(cudaMalloc((void**)&d_in, in_bytes));
    CK(cudaMemcpy(d_in, in, in_bytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**)&d_out, out_bytes));
    HOST_ALLOC(out, out_bytes);
    memset(out, 0, out_bytes);
#define OUT_OF(t, j) (out + (size_t)(j) * out_per)
#define TRANSFER_IN  CK(cudaMemcpy(in, d_in, in_bytes, cudaMemcpyDeviceToHost));
#define TRANSFER_OUT CK(cudaMemcpy(d_out, out, out_bytes, cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
#else
    mjtNum** outs = malloc(n_threads * sizeof(*outs));                 // per thread
    for (int t = 0; t < n_threads; ++t) outs[t] = calloc(out_per, sizeof(mjtNum));
#define OUT_OF(t, j) (outs[t])
#define TRANSFER_IN
#define TRANSFER_OUT
#endif
#define EVAL_ALL                                                                            \
    if (n_threads == 1) {                     /* no fork/join on the 1-thread path */       \
        for (long j = 0; j < n_items; ++j)                                                  \
            eval(s, m, datas[0], bodies, nb, in + (size_t)j * nq, n_steps, block, OUT_OF(0, j)); \
    } else {                                                                                \
        _Pragma("omp parallel for schedule(static)")                                        \
        for (long j = 0; j < n_items; ++j) {                                                \
            const int t = omp_get_thread_num();                                             \
            eval(s, m, datas[t], bodies, nb, in + (size_t)j * nq, n_steps, block, OUT_OF(t, j)); \
        }                                                                                   \
    }
    for (int k = 0; k < n_warm; ++k) { TRANSFER_IN EVAL_ALL TRANSFER_OUT }
    double* ts = malloc(n_timed * sizeof(double));
    struct timespec t0, t1;
    for (int k = 0; k < n_timed; ++k) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        TRANSFER_IN EVAL_ALL TRANSFER_OUT
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ts[k] = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) * 1e-3;
    }
    qsort(ts, n_timed, sizeof(double), cmp);
    printf("%.4f\n", ts[n_timed / 2]);
    return 0;
}
