// Batched dense primal-dual interior-point solver for the elastic CBF-QP, one thread block per instance.
//
// Solves exactly qpax.solve_qp_elastic (explicit backend) for n = 7 variables and m rows:
//
//     min  1/2 x'Qx + q'x + penalty't     s.t.  G x - t <= h,  t >= 0
//
// with qpax's slack/dual split (s1, z1 for t >= 0; s2, z2 for the inequality), its initialisation,
// its Mehrotra predictor-corrector step (same residuals r1..r6, same centering sigma = (..)^3,
// same 0.99 line search), its convergence test (inf-norm of all residuals < tol) and its iteration
// cap, so a converged solution is qpax's to tolerance.  Each block stops at its own convergence.
//
// Data layout: thread tid owns rows tid, tid + THREADS, ..., R = ceil(m / THREADS) rows, and keeps
// those rows of G, their slacks, duals and step directions in registers for the whole solve (R is
// a template parameter picked at launch).  Per iteration the KKT system reduces to the 7 x 7
// normal equations H = Q + G' diag(1/a3) G whose 28 + 7 entries are block reductions over the rows
// (warp shuffles, then one shared-memory pass); the Cholesky and triangular solves are done by
// thread 0 in shared memory.  Only the inputs and the result touch global memory.
//
//   ipm_init(max_batch, m)                                kept for the API (no scratch needed)
//   IpmSolve (XLA FFI): (Q[B,7,7], q[B,7], G[B,m,7], h[B,m], pen[m]) -> x[B,7], iters[B] i32, conv[B] i32
//   ipm_solve_ptr(...)                                    same on raw device pointers (tests / timing)
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
namespace ffi = xla::ffi;

#define NX 7
#define NH 28                     // upper triangle of H
#define THREADS 512               // 16 warps: 1358 rows -> R = 3; register limit 128 / thread
#define WARPS (THREADS / 32)
#define TOL 1e-5
#ifndef MAX_ITER
#define MAX_ITER 30
#endif
#define MAXR 8                    // rows per thread supported -> m <= THREADS * MAXR
#define MAXM (THREADS * MAXR)
#define NBOX (2 * NX)             // the last 2 NX rows are the torque bounds u <= u_max, -u <= u_max (cbfpy layout)
static int g_screen = 1;              // drop CBF rows that cannot be active inside the torque box (exact for the box-constrained QP)
#define NRED (1 + 2 * NX + NH)    // widest reduction: inf-norm, G'z2 (7), G'(p1/a3) (7), H (28)

__device__ __forceinline__ double warp_sum(double v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o); return v; }
__device__ __forceinline__ double warp_min(double v) { for (int o = 16; o; o >>= 1) v = fmin(v, __shfl_xor_sync(0xffffffff, v, o)); return v; }
__device__ __forceinline__ double warp_max(double v) { for (int o = 16; o; o >>= 1) v = fmax(v, __shfl_xor_sync(0xffffffff, v, o)); return v; }

struct Shared {
    double wsm[NRED][WARPS];
    double red[NRED];
    double Qs[NX * NX], qs[NX], x[NX], dx[NX], L[NX * NX], Ld[NX], rhs[NX], r1[NX];
    int done, count;
    int idx[MAXM];                 // compacted list of the rows the solve uses (screening)
};

// lane 0 of each warp deposits its warp-reduced value k
__device__ __forceinline__ void put(Shared& sh, int k, double r) { if ((threadIdx.x & 31) == 0) sh.wsm[k][threadIdx.x >> 5] = r; }
// combine the warps: entries [0, KMAX) are max-reduced, [KMAX, K) summed; red[] valid for all threads afterwards
template <int K, int KMAX>
__device__ __forceinline__ void finish(Shared& sh) {
    __syncthreads();
    if (threadIdx.x < K) {
        double r = sh.wsm[threadIdx.x][0];
        for (int w = 1; w < WARPS; ++w) r = threadIdx.x < KMAX ? fmax(r, sh.wsm[threadIdx.x][w]) : r + sh.wsm[threadIdx.x][w];
        sh.red[threadIdx.x] = r;
    }
    __syncthreads();
}
template <int K>
__device__ __forceinline__ void finish_min(Shared& sh) {
    __syncthreads();
    if (threadIdx.x < K) {
        double r = sh.wsm[threadIdx.x][0];
        for (int w = 1; w < WARPS; ++w) r = fmin(r, sh.wsm[threadIdx.x][w]);
        sh.red[threadIdx.x] = r;
    }
    __syncthreads();
}

// Cholesky H = L L' with the reciprocal diagonal Ld[i] = 1 / L[i][i] kept, so the solves need no divisions.
__device__ void chol7(const double* u, double* L, double* Ld) {   // u: upper triangle row-major (i <= j)
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = u[j * NX - j * (j - 1) / 2 + (i - j)];
            for (int k = 0; k < j; ++k) s -= L[i * NX + k] * L[j * NX + k];
            if (i == j) { Ld[i] = rsqrt(s); L[i * NX + i] = s * Ld[i]; } else L[i * NX + j] = s * Ld[j];
        }
}
__device__ void chol7_solve(const double* L, const double* Ld, const double* b, double* x) {
    double y[NX];
    for (int i = 0; i < NX; ++i) { double s = b[i]; for (int k = 0; k < i; ++k) s -= L[i * NX + k] * y[k]; y[i] = s * Ld[i]; }
    for (int i = NX - 1; i >= 0; --i) { double s = y[i]; for (int k = i + 1; k < NX; ++k) s -= L[k * NX + i] * x[k]; x[i] = s * Ld[i]; }
}
// thread 0: H = Q + red[off .. off + 28) (upper triangle), factor into sh.L / sh.Ld
__device__ __forceinline__ void factor_from(Shared& sh, int off) {
    double Hu[NH]; int k = 0;
    for (int r = 0; r < NX; ++r) for (int c = r; c < NX; ++c) { Hu[k] = sh.Qs[r * NX + c] + sh.red[off + k]; ++k; }
    chol7(Hu, sh.L, sh.Ld);
}

template <int R>
__global__ void __launch_bounds__(THREADS) ipm_kernel(const double* __restrict__ Q, const double* __restrict__ q, const double* __restrict__ G,
                                                       const double* __restrict__ h, const double* __restrict__ pen, int m, int screen,
                                                       double* __restrict__ x_out, int* __restrict__ iters_out, int* __restrict__ conv_out) {
    __shared__ Shared sh;
    const int b = blockIdx.x, tid = threadIdx.x;
    const double* Gb = G + (size_t)b * m * NX; const double* hb = h + (size_t)b * m;
    if (tid < NX * NX) sh.Qs[tid] = 0.5 * (Q[(size_t)b * NX * NX + tid] + Q[(size_t)b * NX * NX + (tid % NX) * NX + tid / NX]);
    if (tid < NX) sh.qs[tid] = q[(size_t)b * NX + tid];
    if (tid == 0) { sh.done = 0; sh.count = 0; }
    __syncthreads();
    // ---- row screening: a CBF row with h_i > sum_j |G_ij| u_max_j cannot be violated by any torque inside the box, so it
    //      cannot be active; keep the others (and always the box rows) in a compacted list (order is irrelevant) ----
    if (screen) {
        double um[NX];
        for (int c = 0; c < NX; ++c) um[c] = hb[m - NBOX + c];
        for (int i = tid; i < m; i += THREADS) {
            double bound = 0.0;
            for (int c = 0; c < NX; ++c) bound += fabs(Gb[(size_t)i * NX + c]) * um[c];
            if (i >= m - NBOX || !(hb[i] > bound)) sh.idx[atomicAdd(&sh.count, 1)] = i;
        }
    } else {
        for (int i = tid; i < m; i += THREADS) sh.idx[i] = i;
        if (tid == 0) sh.count = m;
    }
    __syncthreads();
    const int mm = sh.count;                                        // rows in the solve

    // ---- this thread's rows in registers (slots past mm: g = 0, h = 0, pen = 1 -> inert, excluded from reductions) ----
    double g[R][NX], hr[R], pr[R];
    bool ok[R];
#pragma unroll
    for (int k = 0; k < R; ++k) {
        const int slot = tid + k * THREADS; ok[k] = slot < mm;
        const int i = ok[k] ? sh.idx[slot] : 0;
#pragma unroll
        for (int c = 0; c < NX; ++c) g[k][c] = ok[k] ? Gb[(size_t)i * NX + c] : 0.0;
        hr[k] = ok[k] ? hb[i] : 0.0; pr[k] = ok[k] ? pen[i] : 1.0;
    }

    // ---- initialisation (qpax solve_init_elastic_ls): H0 = Q + 1/2 G'G, x = H0^-1 (-q - 1/2 G'(pen - h)) ----
    {
        int kk = 1 + 2 * NX;
#pragma unroll
        for (int r = 0; r < NX; ++r)
#pragma unroll
            for (int c = r; c < NX; ++c) {
                double v = 0.0;
#pragma unroll
                for (int k = 0; k < R; ++k) if (ok[k]) v += 0.5 * g[k][r] * g[k][c];
                put(sh, kk++, warp_sum(v));
            }
#pragma unroll
        for (int c = 0; c < NX; ++c) {
            double v = 0.0;
#pragma unroll
            for (int k = 0; k < R; ++k) if (ok[k]) v += 0.5 * g[k][c] * (pr[k] - hr[k]);
            put(sh, 1 + c, warp_sum(v)); put(sh, 1 + NX + c, 0.0);
        }
        put(sh, 0, 0.0);
        finish<NRED, 1>(sh);
        if (tid == 0) {
            factor_from(sh, 1 + 2 * NX);
            for (int c = 0; c < NX; ++c) sh.rhs[c] = -sh.qs[c] - sh.red[1 + c];
            chol7_solve(sh.L, sh.Ld, sh.rhs, sh.x);
        }
        __syncthreads();
    }
    double s1[R], s2[R], z1[R], z2[R], tt[R];
    {
        double zmax = -1e300, zmin = 1e300;
#pragma unroll
        for (int k = 0; k < R; ++k) {
            z1[k] = z2[k] = 1.0; tt[k] = 0.0;
            if (!ok[k]) continue;
            double gx = 0.0;
#pragma unroll
            for (int c = 0; c < NX; ++c) gx += g[k][c] * sh.x[c];
            z2[k] = 0.5 * (gx + pr[k] - hr[k]); z1[k] = pr[k] - z2[k]; tt[k] = -z1[k];
            zmax = fmax(zmax, fmax(z1[k], z2[k])); zmin = fmin(zmin, fmin(z1[k], z2[k]));
        }
        put(sh, 0, warp_max(zmax)); put(sh, 1, -warp_min(zmin));
        finish<2, 2>(sh);                                         // red[0] = max z = alpha_p, red[1] = -min z = alpha_d
        const double alpha_p = sh.red[0], alpha_d = sh.red[1];
#pragma unroll
        for (int k = 0; k < R; ++k) {
            s1[k] = s2[k] = 1.0;
            if (!ok[k]) continue;
            s1[k] = alpha_p < 0 ? -z1[k] : -z1[k] + (1.0 + alpha_p);
            s2[k] = alpha_p < 0 ? -z2[k] : -z2[k] + (1.0 + alpha_p);
            if (alpha_d >= 0) { z1[k] += 1.0 + alpha_d; z2[k] += 1.0 + alpha_d; }
        }
    }

    // ---- PDIP iterations ----
    double ia[R], p1[R], ds1[R], ds2[R], dz1[R], dz2[R], rz1[R], rz2[R];
    int it = 0;
    for (; it < MAX_ITER; ++it) {
        // (1) residual inf-norm, G'z2, and H = Q + G' diag(1/a3) G in one reduction; a3, p1 per row (affine rhs, r' = -r)
        double rmax = 0.0;
#pragma unroll
        for (int k = 0; k < R; ++k) {
            ia[k] = 0.0; p1[k] = 0.0; rz1[k] = rz2[k] = 1.0;
            if (!ok[k]) continue;
            double gx = 0.0;
#pragma unroll
            for (int c = 0; c < NX; ++c) gx += g[k][c] * sh.x[c];
            const double r2 = pr[k] - z1[k] - z2[k], r3 = s1[k] * z1[k], r4 = s2[k] * z2[k], r5 = s1[k] - tt[k], r6 = gx - tt[k] + s2[k] - hr[k];
            rmax = fmax(rmax, fmax(fmax(fabs(r2), fabs(r3)), fmax(fmax(fabs(r4), fabs(r5)), fabs(r6))));
#ifndef EXP_NODIV
            rz1[k] = 1.0 / z1[k]; rz2[k] = 1.0 / z2[k];
            const double a1 = s1[k] * rz1[k], a2 = s2[k] * rz2[k];
            ia[k] = 1.0 / (a1 + a2);
#else
            rz1[k] = 1.0 * z1[k]; rz2[k] = 1.0 * z2[k];
            const double a1 = s1[k] * rz1[k], a2 = s2[k] * rz2[k];
            ia[k] = 1.0 * (a1 + a2);
#endif
            p1[k] = (-r5 + r6 - s2[k] + s1[k] + a1 * r2) * ia[k];     // p1 / a3
        }
        put(sh, 0, warp_max(rmax));
#pragma unroll
        for (int c = 0; c < NX; ++c) {
            double v = 0.0, w = 0.0;
#pragma unroll
            for (int k = 0; k < R; ++k) if (ok[k]) { v += g[k][c] * z2[k]; w += g[k][c] * p1[k]; }
            put(sh, 1 + c, warp_sum(v)); put(sh, 1 + NX + c, warp_sum(w));
        }
        {
            int kk = 1 + 2 * NX;
#pragma unroll
            for (int r = 0; r < NX; ++r)
#pragma unroll
                for (int c = r; c < NX; ++c) {
                    double v = 0.0;
#pragma unroll
#ifndef EXP_NOH
                    for (int k = 0; k < R; ++k) if (ok[k]) v += g[k][r] * g[k][c] * ia[k];
#else
                    v = (r == c) ? 1.0 : 0.0;
#endif
                    put(sh, kk++, warp_sum(v));
                }
        }
        finish<NRED, 1>(sh);
        if (tid == 0) {
            double rm = sh.red[0];
            for (int r = 0; r < NX; ++r) {
                double s = sh.qs[r] + sh.red[1 + r];
                for (int c = 0; c < NX; ++c) s += sh.Qs[r * NX + c] * sh.x[c];
                sh.r1[r] = s; rm = fmax(rm, fabs(s));
            }
            sh.done = rm < TOL;
#ifdef EXP_NOCHOL
            for (int c = 0; c < NX; ++c) sh.dx[c] = 1e-3 * sh.red[1 + NX + c];
#else
            if (!sh.done) {
                factor_from(sh, 1 + 2 * NX);
                for (int c = 0; c < NX; ++c) sh.rhs[c] = -sh.r1[c] - sh.red[1 + NX + c];
                chol7_solve(sh.L, sh.Ld, sh.rhs, sh.dx);
            }
#endif
        }
        __syncthreads();
        if (sh.done) break;

        // (3) affine directions; centering data (sum s.z, affine step length)
        double ssz = 0.0, amin = 1.0;
#pragma unroll
        for (int k = 0; k < R; ++k) {
            ds1[k] = ds2[k] = dz1[k] = dz2[k] = 0.0;
            if (!ok[k]) continue;
            double gdx = 0.0;
#pragma unroll
            for (int c = 0; c < NX; ++c) gdx += g[k][c] * sh.dx[c];
            const double r2 = pr[k] - z1[k] - z2[k];
            dz2[k] = p1[k] + gdx * ia[k]; dz1[k] = r2 - dz2[k];
            ds1[k] = -s1[k] * (1.0 + dz1[k] * rz1[k]); ds2[k] = -s2[k] * (1.0 + dz2[k] * rz2[k]);
            {
                ssz += s1[k] * z1[k] + s2[k] * z2[k];
                // ratio -v/dv < amin  <=>  -v > amin * dv  (dv < 0): one multiply per test, a division only when it improves
                if (ds1[k] < 0 && -s1[k] > amin * ds1[k]) amin = -s1[k] / ds1[k];
                if (ds2[k] < 0 && -s2[k] > amin * ds2[k]) amin = -s2[k] / ds2[k];
                if (dz1[k] < 0 && -z1[k] > amin * dz1[k]) amin = -z1[k] / dz1[k];
                if (dz2[k] < 0 && -z2[k] > amin * dz2[k]) amin = -z2[k] / dz2[k];
            }
        }
        put(sh, 0, -warp_min(amin)); put(sh, 1, warp_sum(ssz));
        finish<2, 1>(sh);                                         // red[0] = -min alpha, red[1] = sum s.z
        const double a_aff = -sh.red[0], sum_sz = sh.red[1];
        double snew = 0.0;
#pragma unroll
        for (int k = 0; k < R; ++k)
            if (ok[k]) snew += (s1[k] + a_aff * ds1[k]) * (z1[k] + a_aff * dz1[k]) + (s2[k] + a_aff * ds2[k]) * (z2[k] + a_aff * dz2[k]);
        put(sh, 0, warp_sum(snew));
        finish<1, 0>(sh);
        const double mu = sum_sz / (2.0 * mm), ratio = sh.red[0] / sum_sz, sm = ratio * ratio * ratio * mu;

        // (4) corrector rhs: r3c = r3 - (sm - ds1a dz1a), r4c likewise; p1 = -r5 + r6 + w2 - w1 + a1 r2, w1 = -r3c/z1, w2 = -r4c/z2
#pragma unroll
        for (int k = 0; k < R; ++k) {
            if (!ok[k]) continue;
            double gx = 0.0;
#pragma unroll
            for (int c = 0; c < NX; ++c) gx += g[k][c] * sh.x[c];
            const double r2 = pr[k] - z1[k] - z2[k], r5 = s1[k] - tt[k], r6 = gx - tt[k] + s2[k] - hr[k];
            const double r3c = s1[k] * z1[k] - (sm - ds1[k] * dz1[k]), r4c = s2[k] * z2[k] - (sm - ds2[k] * dz2[k]);
            p1[k] = (-r5 + r6 - r4c * rz2[k] + r3c * rz1[k] + (s1[k] * rz1[k]) * r2) * ia[k];   // p1 / a3
        }
#pragma unroll
        for (int c = 0; c < NX; ++c) {
            double v = 0.0;
#pragma unroll
            for (int k = 0; k < R; ++k) if (ok[k]) v += g[k][c] * p1[k];
            put(sh, c, warp_sum(v));
        }
        finish<NX, 0>(sh);
#ifdef EXP_NOCHOL
        if (tid == 0) for (int c = 0; c < NX; ++c) sh.dx[c] = 1e-3 * sh.red[c];
#else
        if (tid == 0) { for (int c = 0; c < NX; ++c) sh.rhs[c] = -sh.r1[c] - sh.red[c]; chol7_solve(sh.L, sh.Ld, sh.rhs, sh.dx); }
#endif
        __syncthreads();

        // (5) corrector directions and step length
        amin = 1.0;
#pragma unroll
        for (int k = 0; k < R; ++k) {
            if (!ok[k]) continue;
            double gdx = 0.0;
#pragma unroll
            for (int c = 0; c < NX; ++c) gdx += g[k][c] * sh.dx[c];
            const double r2 = pr[k] - z1[k] - z2[k];
            const double r3c = s1[k] * z1[k] - (sm - ds1[k] * dz1[k]), r4c = s2[k] * z2[k] - (sm - ds2[k] * dz2[k]);   // from the affine directions
            dz2[k] = p1[k] + gdx * ia[k]; dz1[k] = r2 - dz2[k];
            ds1[k] = (-r3c - s1[k] * dz1[k]) * rz1[k]; ds2[k] = (-r4c - s2[k] * dz2[k]) * rz2[k];
            {
                if (ds1[k] < 0 && -s1[k] > amin * ds1[k]) amin = -s1[k] / ds1[k];
                if (ds2[k] < 0 && -s2[k] > amin * ds2[k]) amin = -s2[k] / ds2[k];
                if (dz1[k] < 0 && -z1[k] > amin * dz1[k]) amin = -z1[k] / dz1[k];
                if (dz2[k] < 0 && -z2[k] > amin * dz2[k]) amin = -z2[k] / dz2[k];
            }
        }
        put(sh, 0, warp_min(amin));
        finish_min<1>(sh);
        const double alpha = 0.99 * fmin(1.0, 0.99 * sh.red[0]);
#pragma unroll
        for (int k = 0; k < R; ++k) {
            if (!ok[k]) continue;
            const double r5 = s1[k] - tt[k];
            s1[k] += alpha * ds1[k]; s2[k] += alpha * ds2[k]; z1[k] += alpha * dz1[k]; z2[k] += alpha * dz2[k];
            tt[k] += alpha * (ds1[k] + r5);                         // dt = ds1 - r5' = ds1 + r5
        }
        if (tid < NX) sh.x[tid] += alpha * sh.dx[tid];
        __syncthreads();
    }
    if (tid < NX) x_out[(size_t)b * NX + tid] = sh.x[tid];
    if (tid == 0) { iters_out[b] = it; conv_out[b] = sh.done; }
}

extern "C" int ipm_init(int, int) { return 0; }
extern "C" void ipm_set_screen(int on) { g_screen = on; }

extern "C" int ipm_solve_ptr(const double* Q, const double* q, const double* G, const double* h, const double* pen, int batch, int m,
                             double* x, int* iters, int* conv, cudaStream_t stream) {
    const int R = (m + THREADS - 1) / THREADS;
    if (R < 1 || R > MAXR) return -1;
#define LAUNCH(RR) ipm_kernel<RR><<<batch, THREADS, 0, stream>>>(Q, q, G, h, pen, m, g_screen, x, iters, conv)
    switch (R) { case 1: LAUNCH(1); break; case 2: LAUNCH(2); break; case 3: LAUNCH(3); break; case 4: LAUNCH(4); break;
                 case 5: LAUNCH(5); break; case 6: LAUNCH(6); break; case 7: LAUNCH(7); break; default: LAUNCH(8); break; }
#undef LAUNCH
    return (int)cudaPeekAtLastError();
}

static ffi::Error Impl(cudaStream_t stream, ffi::Buffer<ffi::F64> Q, ffi::Buffer<ffi::F64> q, ffi::Buffer<ffi::F64> G, ffi::Buffer<ffi::F64> h,
                       ffi::Buffer<ffi::F64> pen, ffi::ResultBuffer<ffi::F64> x, ffi::ResultBuffer<ffi::S32> iters, ffi::ResultBuffer<ffi::S32> conv) {
    const int batch = (int)G.dimensions()[0], m = (int)G.dimensions()[1];
    int rc = ipm_solve_ptr(Q.typed_data(), q.typed_data(), G.typed_data(), h.typed_data(), pen.typed_data(), batch, m,
                           x->typed_data(), iters->typed_data(), conv->typed_data(), stream);
    if (rc) return ffi::Error(ffi::ErrorCode::kInternal, rc == -1 ? "too many rows for the kernel (m > THREADS * MAXR)" : cudaGetErrorString((cudaError_t)rc));
    return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(IpmSolve, Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::S32>>());
