// cusadi2_contact.h: MuJoCo's constraint step (plane contacts, joint limits, dof frictionloss; Newton
// solver with the pyramidal cone) as one fused CUDA kernel, one thread block per instance, between the
// cusadi2 `smooth` and `integrate` plans built by cusadi2.contact:
//
//   smooth  -> M (sparse nnz), qfrc_smooth, geom_pos, geom_rot, cdof
//   FusedStep<S, Model, NJMAX>::launch(...)   (this file)  -> qfrc_constraint, qacc, qacc_warmstart (updated), stats
//   integrate(qpos, qvel, M, qfrc_smooth, qfrc_constraint) -> qpos', qvel'
//
// The algorithm is MuJoCo's. Rows follow mujoco-warp's assembly (constraint.py: _efc_row, _friction_dof,
// _limit_slide_hinge, _efc_contact_*; collision_primitive.py: plane_sphere / plane_capsule), with the same
// row order (frictionloss, limits, contacts), impedance, reference acceleration and pyramid rows; the
// capsule tangent follows MuJoCo C (mjc_PlaneCapsule), where warp deviates. The Newton solver follows
// MuJoCo C's engine_solver.c (PrimalSolve / PrimalSearch, 3.12.0): the zero-iteration certificate exits,
// the one-sided then bracketed line search with its evaluation budget, and the termination test; warp's
// line search is a different bracketing scheme and its iterates differ from C's whenever the iteration
// cap is reached. `Model` is the struct cusadi2.contact.Model.write_model_header
// emits (constexpr sizes and options, accessors over __constant__ arrays); S is float or double; NJMAX caps
// the constraint rows per instance (a row set that does not fit is dropped and flagged in stats, as
// mujoco-warp's njmax does).
//
// Per instance, in order inside the block: narrowphase on every pair geom (thread per contact point),
// row assembly (thread per row), then the Newton loop with per-instance early exit: Ma, Jaref, constraint
// state and force, gradient, H = M + J^T D J (dense in shared memory), Cholesky (warp 0), the bracketing
// line search over the rows (block reductions), until the tolerance or Model::ITERATIONS.
#pragma once
#include <cuda_runtime.h>
#include <math.h>

namespace cusadi2_contact {

constexpr double MJ_MINVAL = 1e-15, MJ_MINIMP = 0.0001, MJ_MAXIMP = 0.9999;
enum { ROW_FRICTION = 0, ROW_LIMIT = 1, ROW_CONTACT = 2 };
enum { STATE_SATISFIED = 0, STATE_QUADRATIC = 1, STATE_LINEARNEG = 2, STATE_LINEARPOS = 3 };
enum { OVERFLOW_ROWS = 1, OVERFLOW_ITERATIONS = 2, OVERFLOW_LS = 4 };

struct Stats { int nefc, ncon, niter, overflow; };
// optional per-instance dump of the assembled rows (debugging / gates): J (NJMAX x NV), D, aref, type, jaref (final)
template <class S> struct RowDump { S* J; S* D; S* aref; int* type; S* jaref; S* iters; long long* prof; };
// iters: per instance 8 x ITERATIONS: alpha, improvement, gradient, decrement, ls_iter, nactive, cost, 0
// prof: per instance 32 clock64() stamps: 0 start, 1 narrowphase, 2 rows, 3 M factor + warm start, 4 certificate + first H,
//       then per iteration i: 5+4i linesearch, 6+4i constraint+grad, 7+4i hessian, 8+4i solve; 31 end

template <class S> __device__ __forceinline__ S safe_div(S x, S y) { return x / (y != S(0) ? y : S(MJ_MINVAL)); }

template <class S> struct V3 {
    S x, y, z;
    __device__ __forceinline__ V3 operator+(V3 o) const { return {x + o.x, y + o.y, z + o.z}; }
    __device__ __forceinline__ V3 operator-(V3 o) const { return {x - o.x, y - o.y, z - o.z}; }
    __device__ __forceinline__ V3 operator*(S a) const { return {x * a, y * a, z * a}; }
};
template <class S> __device__ __forceinline__ S dot(V3<S> a, V3<S> b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
template <class S> __device__ __forceinline__ V3<S> cross(V3<S> a, V3<S> b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
template <class S> __device__ __forceinline__ S norm(V3<S> a) { return sqrt(dot(a, a)); }

// mujoco-warp math.orthogonals: the two tangents of a unit normal
template <class S> __device__ __forceinline__ void orthogonals(V3<S> a, V3<S>& b, V3<S>& c) {
    b = (S(-0.5) < a.y && a.y < S(0.5)) ? V3<S>{0, 1, 0} : V3<S>{0, 0, 1};
    b = b - a * dot(a, b);
    S n = norm(b);
    b = b * (S(1) / n);
    if (norm(a) == S(0)) b = {0, 0, 0};
    c = cross(a, b);
}

// block-wide sum of K values held per thread; result in sh_out[0..K) (shared), valid for all threads on return
template <class S, int K, int NT>
__device__ __forceinline__ void block_sum(S* v, S* sh_scratch /* (NT/32) * K */, S* sh_out /* K */) {
    for (int k = 0; k < K; ++k)
        for (int o = 16; o > 0; o >>= 1) v[k] += __shfl_xor_sync(0xffffffffu, v[k], o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) for (int k = 0; k < K; ++k) sh_scratch[warp * K + k] = v[k];
    __syncthreads();
    if (threadIdx.x < K) {
        S s = 0;
        for (int w = 0; w < NT / 32; ++w) s += sh_scratch[w * K + threadIdx.x];
        sh_out[threadIdx.x] = s;
    }
    __syncthreads();
}

template <class S, class Model, int NJMAX>
struct Shared {
    static constexpr int NV = Model::NV, NP = Model::NPOINT, MC = Model::MAX_CHAIN;
    static_assert(NV * NV >= NP * 13, "the narrowphase scratch aliases H");
    S M[NV * NV];                 // dense mass matrix (row-major)
    S H[NV * NV];                 // Newton Hessian, lower triangle used; Cholesky factor in place; narrowphase scratch before that
    S J[NJMAX * MC];              // constraint Jacobian rows, sparse: value a of row i at J[i*MC+a] is for the a-th set bit of mask[i]
    S D[NJMAX], aref[NJMAX], jaref[NJMAX], jv[NJMAX], force[NJMAX], floss[NJMAX];
    int type[NJMAX], state[NJMAX], nnz[NJMAX];
    unsigned long long mask[NJMAX];   // bit d set when dof d is in the row (ascending, so the value index of dof d is a popcount)
    S dinv[NV];                   // 1 / sqrt(diag) during the factorization
    S qacc[NV], Ma[NV], grad[NV], mgrad[NV], search[NV], mv[NV], qfs[NV], qfc[NV], qvel[NV], cdof[6 * NV];
    S qg1, qg2;                   // the gauss quadratic of the line search
    S qacc_smooth[NV];
    S p0[3];                      // base position: the cdof linear reference
    int point_row[NP], limit_row[Model::NLIMIT];
    __device__ __forceinline__ S* cpos() { return H; }
    __device__ __forceinline__ S* cframe() { return H + NP * 3; }
    __device__ __forceinline__ S* cdist() { return H + NP * 12; }
    S red_scratch[32 * 9], red[9];
    int nefc, ncon, nl, overflow;
    S search_dot, newton_dec, grad_dot;
};

// MuJoCo's constraint impedance and reference acceleration (mujoco-warp constraint._efc_row)
template <class S, class Model>
__device__ void efc_row(S pos_aref, S pos_imp, S invweight, const double* solref, const double* solimp, S vel, S& D, S& aref) {
    S timeconst = (S)solref[0], dampratio = (S)solref[1];
    S dmin = (S)solimp[0], dmax = (S)solimp[1], width = (S)solimp[2], mid = (S)solimp[3], power = (S)solimp[4];
    if (Model::REFSAFE) timeconst = fmax(timeconst, S(2.0 * Model::DT));
    dmin = fmin(fmax(dmin, S(MJ_MINIMP)), S(MJ_MAXIMP));
    dmax = fmin(fmax(dmax, S(MJ_MINIMP)), S(MJ_MAXIMP));
    width = fmax(S(MJ_MINVAL), width);
    mid = fmin(fmax(mid, S(MJ_MINIMP)), S(MJ_MAXIMP));
    power = fmax(S(1), power);
    S dmax_sq = dmax * dmax;
    S k = S(1) / (dmax_sq * timeconst * timeconst * dampratio * dampratio);
    S b = S(2) / (dmax * timeconst);
    if (solref[0] <= 0) k = -(S)solref[0] / dmax_sq;
    if (solref[1] <= 0) b = -(S)solref[1] / dmax;
    S imp_x = fabs(pos_imp) / width;
    S imp_a = (S(1) / pow(mid, power - S(1))) * pow(imp_x, power);
    S imp_b = S(1) - (S(1) / pow(S(1) - mid, power - S(1))) * pow(S(1) - imp_x, power);
    S imp_y = imp_x < mid ? imp_a : imp_b;
    S imp = dmin + imp_y * (dmax - dmin);
    imp = fmin(fmax(imp, dmin), dmax);
    if (imp_x > S(1)) imp = dmax;
    D = S(1) / fmax(invweight * (S(1) - imp) / imp, S(MJ_MINVAL));
    aref = -k * imp * pos_aref - b * vel;
}

// (cost - cost(0), grad, hessian) of one row at alpha along the search direction
// (mujoco-warp solver._compute_efc_eval_pt_pyramidal and _compute_efc_eval_pt_alpha_zero_pyramidal)
template <class S>
__device__ __forceinline__ void eval_row(int type, S jaref, S jv, S D, S f, S alpha, S& c, S& g, S& h) {
    if (type != ROW_FRICTION) {
        S x = jaref + alpha * jv;
        S quad0 = S(0.5) * D * jaref * jaref;
        S cost0 = jaref < S(0) ? quad0 : S(0);
        S offset = quad0 - cost0;
        if (x < S(0)) {
            S jvD = jv * D, hh = jv * jvD, ah = alpha * hh;
            c = alpha * (jvD * jaref + S(0.5) * ah) + offset; g = jvD * jaref + ah; h = hh;
        } else { c = -cost0; g = 0; h = 0; }
    } else {
        S rf = safe_div(f, D);
        S x = jaref + alpha * jv;
        S cost0 = (-rf < jaref && jaref < rf) ? S(0.5) * D * jaref * jaref : (jaref <= -rf ? f * (S(-0.5) * rf - jaref) : f * (S(-0.5) * rf + jaref));
        if (-rf < x && x < rf) { S jvD = jv * D; c = S(0.5) * D * x * x - cost0; g = jvD * x; h = jv * jvD; }
        else if (x <= -rf) { c = f * (S(-0.5) * rf - x) - cost0; g = -f * jv; h = 0; }
        else { c = f * (S(-0.5) * rf + x) - cost0; g = f * jv; h = 0; }
    }
}
template <class S>
__device__ __forceinline__ void eval_row_alpha_zero(int type, S jaref, S jv, S D, S f, S& c, S& g, S& h) {
    if (type != ROW_FRICTION) {
        if (jaref < S(0)) { S jvD = jv * D; c = S(0.5) * D * jaref * jaref; g = jvD * jaref; h = jv * jvD; }
        else { c = g = h = 0; }
    } else {
        S rf = safe_div(f, D);
        if (-rf < jaref && jaref < rf) { S jvD = jv * D; c = S(0.5) * D * jaref * jaref; g = jvD * jaref; h = jv * jvD; }
        else if (jaref <= -rf) { c = f * (S(-0.5) * rf - jaref); g = -f * jv; h = 0; }
        else { c = f * (S(-0.5) * rf + jaref); g = f * jv; h = 0; }
    }
}

template <class S, class Model, int NJMAX, int NT>
struct Fused {
    using Sh = Shared<S, Model, NJMAX>;
    static constexpr int NV = Model::NV, NP = Model::NPOINT, MC = Model::MAX_CHAIN;
    static_assert(NV <= 64, "row dof masks are 64-bit");
    static_assert(NT % 32 == 0 && NT >= 32, "whole warps");

    // ---- constraint state, force and qfrc_constraint = J^T force
    __device__ static void update_constraint(Sh& s) {
        const int nefc = s.nefc;
        for (int i = threadIdx.x; i < nefc; i += NT) {
            S ja = s.jaref[i], D = s.D[i];
            if (s.type[i] == ROW_FRICTION) {
                S f = s.floss[i], rf = safe_div(f, D);
                if (ja <= -rf) { s.force[i] = f; s.state[i] = STATE_LINEARNEG; }
                else if (ja >= rf) { s.force[i] = -f; s.state[i] = STATE_LINEARPOS; }
                else { s.force[i] = -D * ja; s.state[i] = STATE_QUADRATIC; }
            } else if (ja >= S(0)) { s.force[i] = 0; s.state[i] = STATE_SATISFIED; }
            else { s.force[i] = -D * ja; s.state[i] = STATE_QUADRATIC; }
        }
        __syncthreads();
        for (int k = threadIdx.x; k < NV; k += NT) {
            const unsigned long long bit = 1ull << k, below = bit - 1;
            S acc = 0;
            for (int i = 0; i < nefc; ++i) {
                const unsigned long long mk = s.mask[i];
                if (mk & bit) acc += s.J[i * MC + __popcll(mk & below)] * s.force[i];
            }
            s.qfc[k] = acc;
        }
        __syncthreads();
    }

    // ---- grad = Ma - qfrc_smooth - qfrc_constraint and |grad|^2
    __device__ static void update_grad(Sh& s) {
        S v[1] = {0};
        for (int k = threadIdx.x; k < NV; k += NT) {
            S g = s.Ma[k] - s.qfs[k] - s.qfc[k];
            s.grad[k] = g; v[0] += g * g;
        }
        block_sum<S, 1, NT>(v, s.red_scratch, s.red);
        if (threadIdx.x == 0) s.grad_dot = s.red[0];
        __syncthreads();
    }

    static constexpr int NPAIR = NV * (NV + 1) / 2, PPT = (NPAIR + NT - 1) / NT;   // lower-triangle entries, per thread
    // the lower-triangle entry (i, k), i >= k, of index p (row-major over the triangle)
    __device__ static __forceinline__ void pair(int p, int& i, int& k) {
        int r = (int)((sqrtf(8.0f * (float)p + 1.0f) - 1.0f) * 0.5f);
        while (r * (r + 1) / 2 > p) --r;
        while ((r + 1) * (r + 2) / 2 <= p) ++r;
        i = r; k = p - r * (r + 1) / 2;
    }

    // ---- Cholesky of the lower triangle of s.H in place, all threads. Column j's entries are final once
    // columns < j have been applied, so the rank-1 update of the statically owned entries (NPAIR / NT each)
    // uses the unscaled column divided by its diagonal: one barrier per column, and one scaling pass at
    // the end (L[i][j] = H[i][j] / sqrt(H[j][j])). Call from all threads; syncs inside and at the end.
    __device__ static void factor_H(Sh& s) {
        int pi[PPT], pk[PPT];
        for (int m = 0; m < PPT; ++m) { const int p = threadIdx.x + m * NT; if (p < NPAIR) pair(p, pi[m], pk[m]); else pi[m] = pk[m] = -1; }
        S* H = s.H;
        for (int j = 0; j < NV - 1; ++j) {
            __syncthreads();
            const S inv = S(1) / H[j * NV + j];
            for (int m = 0; m < PPT; ++m) {
                const int i = pi[m], k = pk[m];
                if (k > j) H[i * NV + k] -= H[i * NV + j] * H[k * NV + j] * inv;
            }
        }
        __syncthreads();
        for (int j = threadIdx.x; j < NV; j += NT) s.dinv[j] = S(1) / sqrt(H[j * NV + j]);
        __syncthreads();
        for (int m = 0; m < PPT; ++m) {
            const int i = pi[m], k = pk[m];
            if (i < 0) continue;
            if (i == k) H[i * NV + i] = S(1) / s.dinv[i]; else H[i * NV + k] *= s.dinv[k];
        }
        __syncthreads();
    }

    // ---- out = (L L^T)^-1 rhs with the factor in s.H, by warp 0 (column-oriented substitution: no reductions);
    // call from all threads, syncs at the end
    __device__ static void chol_solve(Sh& s, const S* rhs, S* out) {
        if (threadIdx.x < 32) {
            const int lane = threadIdx.x;
            const S* H = s.H;
            for (int k = lane; k < NV; k += 32) out[k] = rhs[k];
            __syncwarp();
            for (int i = 0; i < NV; ++i) {                          // L y = rhs
                const S x = out[i] / H[i * NV + i];
                __syncwarp();
                if (lane == 0) out[i] = x;
                for (int k = i + 1 + lane; k < NV; k += 32) out[k] -= H[k * NV + i] * x;
                __syncwarp();
            }
            for (int i = NV - 1; i >= 0; --i) {                     // L^T x = y
                const S x = out[i] / H[i * NV + i];
                __syncwarp();
                if (lane == 0) out[i] = x;
                for (int k = lane; k < i; k += 32) out[k] -= H[i * NV + k] * x;
                __syncwarp();
            }
        }
        __syncthreads();
    }

    // ---- mgrad = (factor)^-1 grad, search = -mgrad, search_dot, newton_dec = grad . mgrad
    __device__ static void solve_direction(Sh& s) {
        chol_solve(s, s.grad, s.mgrad);
        S v[2] = {0, 0};
        for (int k = threadIdx.x; k < NV; k += NT) { S x = s.mgrad[k]; s.search[k] = -x; v[0] += x * x; v[1] += s.grad[k] * x; }
        block_sum<S, 2, NT>(v, s.red_scratch, s.red);
        if (threadIdx.x == 0) { s.search_dot = s.red[0]; s.newton_dec = s.red[1]; }
        __syncthreads();
    }

    // ---- the constraint cost at jar = J qacc - aref (mj_constraintUpdate_impl's cost), summed over the block
    __device__ static S constraint_cost(Sh& s, const S* jar) {
        const int nefc = s.nefc;
        S v[1] = {0};
        for (int i = threadIdx.x; i < nefc; i += NT) {
            const S ja = jar[i], D = s.D[i];
            if (s.type[i] == ROW_FRICTION) {
                const S f = s.floss[i], Rf = safe_div(f, D), R = S(1) / D;
                if (ja <= -Rf) v[0] += S(-0.5) * R * f * f - f * ja;
                else if (ja >= Rf) v[0] += S(-0.5) * R * f * f + f * ja;
                else v[0] += S(0.5) * D * ja * ja;
            } else if (ja < S(0)) v[0] += S(0.5) * D * ja * ja;
        }
        block_sum<S, 1, NT>(v, s.red_scratch, s.red);
        return s.red[0];
    }

    // ---- H = M + sum over quadratic rows of D J^T J (lower triangle), factored (MuJoCo's MakeHessian + FactorizeHessian).
    // Thread per lower-triangle entry (a, b): the rows holding both dofs contribute D J_a J_b (value index by popcount)
    __device__ static void make_hessian(Sh& s) {
        const int nefc = s.nefc;
        for (int p = threadIdx.x; p < NPAIR; p += NT) {
            int a, b; pair(p, a, b);
            const unsigned long long ba = 1ull << a, bb = 1ull << b, la = ba - 1, lb = bb - 1;
            S acc = s.M[a * NV + b];
            for (int i = 0; i < nefc; ++i) {
                const unsigned long long mk = s.mask[i];
                if ((mk & ba) && (mk & bb) && s.state[i] == STATE_QUADRATIC)
                    acc += s.D[i] * s.J[i * MC + __popcll(mk & la)] * s.J[i * MC + __popcll(mk & lb)];
            }
            s.H[a * NV + b] = acc;
        }
        __syncthreads();
        factor_H(s);
    }

    // ---- one line-search point (MuJoCo C PrimalEval): cost(alpha) - cost(0), its first and second derivative
    __device__ static void eval_point(Sh& s, S alpha, S& cost, S& d0, S& d1) {
        const int nefc = s.nefc;
        // partial sums: quadTotal[0..2] (inequality rows), friction cost, friction d0, friction d1
        S v[6] = {0, 0, 0, 0, 0, 0};
        for (int i = threadIdx.x; i < nefc; i += NT) {
            const S start = s.jaref[i], jv = s.jv[i], D = s.D[i];
            const S x = start + alpha * jv;
            if (s.type[i] == ROW_FRICTION) {
                const S f = s.floss[i], Rf = safe_div(f, D);
                // frictionCostDif(start, x)
                const int st0 = (-Rf < start && start < Rf) ? 0 : (start <= -Rf ? -1 : 1);
                const int stx = (-Rf < x && x < Rf) ? 0 : (x <= -Rf ? -1 : 1);
                S c;
                if (st0 == 0 && stx == 0) c = S(0.5) * D * (x - start) * (x + start);
                else if (st0 == -1 && stx == -1) c = f * (start - x);
                else if (st0 == 1 && stx == 1) c = f * (x - start);
                else {
                    auto fc = [&](S y) { return (-Rf < y && y < Rf) ? S(0.5) * D * y * y : (y <= -Rf ? f * (S(-0.5) * Rf - y) : f * (S(-0.5) * Rf + y)); };
                    c = fc(x) - fc(start);
                }
                v[3] += c;
                if (-Rf < x && x < Rf) { v[4] += D * x * jv; v[5] += D * jv * jv; }
                else if (x <= -Rf) v[4] += -f * jv;
                else v[4] += f * jv;
            } else {
                const S DJ0 = D * start;
                const S q0 = S(0.5) * start * DJ0, q1 = jv * DJ0, q2 = S(0.5) * jv * D * jv;
                const S cost0 = start < S(0) ? q0 : S(0);
                if (x < S(0)) { v[0] += q0 - cost0; v[1] += q1; v[2] += q2; }
                else v[3] -= cost0;
            }
        }
        block_sum<S, 6, NT>(v, s.red_scratch, s.red);
        const S qt0 = s.red[0], qt1 = s.qg1 + s.red[1], qt2 = s.qg2 + s.red[2];
        cost = s.red[3] + alpha * alpha * qt2 + alpha * qt1 + qt0;
        d0 = s.red[4] + S(2) * alpha * qt2 + qt1;
        d1 = s.red[5] + S(2) * qt2;
        if (d1 <= S(0)) d1 = S(MJ_MINVAL);
    }

    struct Pnt { S alpha, cost, d0, d1; };

    // ---- MuJoCo C's PrimalSearch along s.search: returns alpha (0 = no improvement), sets `improvement`
    __device__ static S primal_search(Sh& s, S gtol_scale, S& improvement, int& ls_iter) {
        const int nefc = s.nefc;
        improvement = 0; ls_iter = 0;
        const S snorm = sqrt(s.search_dot);
        if (snorm < S(MJ_MINVAL)) return 0;
        const S gtol = gtol_scale * snorm;                   // tolerance * ls_tolerance * snorm / scale
        // Mv = M search, Jv = J search, the gauss quadratic
        S v[2] = {0, 0};
        for (int k = threadIdx.x; k < NV; k += NT) {
            S acc = 0;
            for (int c = 0; c < NV; ++c) acc += s.M[k * NV + c] * s.search[c];
            s.mv[k] = acc;
            v[0] += s.search[k] * s.Ma[k] - s.qfs[k] * s.search[k];
            v[1] += S(0.5) * s.search[k] * acc;
        }
        for (int i = threadIdx.x; i < nefc; i += NT) {
            S acc = 0; const S* Ji = s.J + i * MC;
            unsigned long long mk = s.mask[i];
            for (int a = 0; mk; ++a) { const int d = __ffsll(mk) - 1; mk &= mk - 1; acc += Ji[a] * s.search[d]; }
            s.jv[i] = acc;
        }
        block_sum<S, 2, NT>(v, s.red_scratch, s.red);
        if (threadIdx.x == 0) { s.qg1 = s.red[0]; s.qg2 = s.red[1]; }
        __syncthreads();
        auto eval = [&](Pnt& p) { eval_point(s, p.alpha, p.cost, p.d0, p.d1); ++ls_iter; };
        Pnt p0, p1, p2, pmid, p1next, p2next;
        p0.alpha = 0; eval(p0);
        p1.alpha = p0.alpha - p0.d0 / p0.d1; eval(p1);
        if (fabs(p1.d0) < gtol && (p1.alpha == S(0) || p1.cost < S(0))) { improvement = -p1.cost; return p1.alpha; }
        const S dir = p1.d0 < S(0) ? S(1) : S(-1);
        p2 = p0;
        while (p1.d0 * dir <= -gtol && ls_iter < Model::LS_ITERATIONS) {
            p2 = p1;
            p1.alpha -= p1.d0 / p1.d1; eval(p1);
            if (fabs(p1.d0) < gtol && p1.cost < S(0)) { improvement = -p1.cost; return p1.alpha; }
        }
        if (ls_iter >= Model::LS_ITERATIONS) { improvement = -p1.cost; return p1.alpha; }
        p2next = p1;
        p1next.alpha = p1.alpha - p1.d0 / p1.d1; eval(p1next);
        auto update_bracket = [&](Pnt& p, const Pnt* cand, Pnt& pnext) {
            int flag = 0;
            for (int i = 0; i < 3; ++i) {
                if (p.d0 < S(0) && cand[i].d0 < S(0) && p.d0 < cand[i].d0) { p = cand[i]; flag = 1; }
                else if (p.d0 > S(0) && cand[i].d0 > S(0) && p.d0 > cand[i].d0) { p = cand[i]; flag = 2; }
            }
            if (flag) { pnext.alpha = p.alpha - p.d0 / p.d1; eval(pnext); }
            return flag;
        };
        while (ls_iter < Model::LS_ITERATIONS) {
            pmid.alpha = S(0.5) * (p1.alpha + p2.alpha); eval(pmid);
            const Pnt cand[3] = {p1next, p2next, pmid};
            S bestcost = 0; int bestind = -1;
            for (int i = 0; i < 3; ++i)
                if (fabs(cand[i].d0) < gtol && (bestind == -1 || cand[i].cost < bestcost)) { bestcost = cand[i].cost; bestind = i; }
            if (bestind >= 0) { improvement = -cand[bestind].cost; return cand[bestind].alpha; }
            const int b1 = update_bracket(p1, cand, p1next);
            const int b2 = update_bracket(p2, cand, p2next);
            if (!b1 && !b2) { improvement = -pmid.cost; return pmid.alpha; }
        }
        if (p1.cost <= p2.cost && p1.cost < S(0)) { improvement = -p1.cost; return p1.alpha; }
        if (p2.cost <= p1.cost && p2.cost < S(0)) { improvement = -p2.cost; return p2.alpha; }
        return 0;
    }

    // ---- narrowphase + rows + Newton for one instance
    __device__ static void run(Sh& s, const S* __restrict__ qpos, const S* __restrict__ qvel, const S* __restrict__ Mnnz,
                               const S* __restrict__ qfs, const S* __restrict__ gpos, const S* __restrict__ grot,
                               const S* __restrict__ cdof, S* __restrict__ warmstart, S* __restrict__ qfrc_constraint,
                               S* __restrict__ qacc_out, Stats* __restrict__ stats, RowDump<S> dump, int b) {
        const int t = threadIdx.x;
        long long* prof = dump.prof ? dump.prof + (size_t)b * 32 : nullptr;
        auto stamp = [&](int k) { if (prof && t == 0) prof[k] = clock64(); };
        stamp(0);
        // ---- load
        for (int e = t; e < NV * NV; e += NT) s.M[e] = 0;
        for (int k = t; k < NV; k += NT) { s.qvel[k] = qvel[k]; s.qfs[k] = qfs[k]; s.qacc[k] = warmstart[k]; }
        for (int e = t; e < 6 * NV; e += NT) s.cdof[e] = cdof[e];
        if (t < 3) s.p0[t] = qpos[t];
        if (t == 0) { s.overflow = 0; s.nl = 0; }
        __syncthreads();
        for (int e = t; e < Model::M_NNZ; e += NT) s.M[Model::m_row(e) * NV + Model::m_col(e)] = Mnnz[e];
        // ---- narrowphase: thread per contact point (sphere: 1, capsule: 2 ends)
        for (int p = t; p < NP; p += NT) {
            int g = 0, end = 0, acc = 0;
            for (int gg = 0; gg < Model::NGEOM; ++gg) {
                int np = Model::geom_type(gg) == 3 ? 2 : 1;
                if (p < acc + np) { g = gg; end = p - acc; break; }
                acc += np;
            }
            const V3<S> n = {(S)Model::plane_normal(0), (S)Model::plane_normal(1), (S)Model::plane_normal(2)};
            const V3<S> pp = {(S)Model::plane_pos(0), (S)Model::plane_pos(1), (S)Model::plane_pos(2)};
            V3<S> c = {gpos[3 * g], gpos[3 * g + 1], gpos[3 * g + 2]};
            const S r = (S)Model::geom_radius(g);
            V3<S> b, cc;
            if (Model::geom_type(g) == 3) {
                const V3<S> axis = {grot[9 * g + 6], grot[9 * g + 7], grot[9 * g + 8]};       // column 2 of the rotation
                // tangent b along the capsule axis projected onto the plane, as MuJoCo C (mjc_PlaneCapsule:
                // mju_normalize3 and the makeFrame fallback only for a degenerate projection). mujoco-warp
                // instead falls back to a fixed tangent whenever the projected norm is below 0.5.
                V3<S> bb = axis - n * dot(n, axis);
                const S bn = norm(bb);
                if (bn < S(MJ_MINVAL)) orthogonals(n, b, cc);
                else { b = bb * (S(1) / bn); cc = cross(n, b); }
                c = c + axis * ((end == 0 ? S(1) : S(-1)) * (S)Model::geom_halflen(g));
            } else {
                orthogonals(n, b, cc);
            }
            const S dist = dot(c - pp, n) - r;
            const V3<S> pos = c - n * (r + S(0.5) * dist);
            s.cdist()[p] = dist;
            s.cpos()[3 * p] = pos.x; s.cpos()[3 * p + 1] = pos.y; s.cpos()[3 * p + 2] = pos.z;
            S* f = s.cframe() + 9 * p;
            f[0] = n.x; f[1] = n.y; f[2] = n.z; f[3] = b.x; f[4] = b.y; f[5] = b.z; f[6] = cc.x; f[7] = cc.y; f[8] = cc.z;
        }
        __syncthreads();
        stamp(1);
        // ---- row bookkeeping (thread 0): frictionloss rows, then active limits, then active contacts.
        // A row set that does not fit under NJMAX is dropped whole and flagged (mujoco-warp's njmax).
        if (t == 0) {
            int row = Model::NFRICTION, nl = 0, ncon = 0;
            for (int l = 0; l < Model::NLIMIT; ++l) {
                const S q = qpos[Model::limit_qpos(l)];
                const S pos = fmin(q - (S)Model::limit_range(l)[0], (S)Model::limit_range(l)[1] - q) - (S)Model::limit_margin(l);
                s.limit_row[l] = -1;
                if (pos < S(0)) {
                    ++nl;
                    if (row < NJMAX) s.limit_row[l] = row++; else s.overflow |= OVERFLOW_ROWS;
                }
            }
            int acc = 0;
            for (int g = 0; g < Model::NGEOM; ++g) {
                const int np = Model::geom_type(g) == 3 ? 2 : 1;
                const int ndim = Model::geom_condim(g) == 1 ? 1 : 2 * (Model::geom_condim(g) - 1);
                for (int e = 0; e < np; ++e) {
                    const int p = acc + e;
                    s.point_row[p] = -1;
                    // active: dist < margin (includemargin); detected: dist < margin + gap
                    if (s.cdist()[p] < (S)Model::geom_margin(g) + (S)Model::geom_gap(g)) {
                        ++ncon;
                        if (s.cdist()[p] < (S)Model::geom_margin(g)) {
                            if (row + ndim <= NJMAX) { s.point_row[p] = row; row += ndim; } else s.overflow |= OVERFLOW_ROWS;
                        }
                    }
                }
                acc += np;
            }
            s.nl = nl; s.ncon = ncon; s.nefc = row;
        }
        __syncthreads();
        const int nefc = s.nefc;
        // ---- rows: frictionloss
        for (int i = t; i < Model::NFRICTION; i += NT) {
            const int d = Model::friction_dof(i);
            s.type[i] = ROW_FRICTION; s.nnz[i] = 1; s.J[i * MC] = 1; s.mask[i] = 1ull << d;
            efc_row<S, Model>(S(0), S(0), (S)Model::friction_invweight(i), Model::friction_solref(i), Model::friction_solimp(i), s.qvel[d], s.D[i], s.aref[i]);
            s.floss[i] = (S)Model::friction_loss(i);
        }
        // ---- rows: limits
        for (int l = t; l < Model::NLIMIT; l += NT) {
            const int i = s.limit_row[l];
            if (i < 0) continue;
            const int d = Model::limit_dof(l);
            const S q = qpos[Model::limit_qpos(l)];
            const S dmin = q - (S)Model::limit_range(l)[0], dmax = (S)Model::limit_range(l)[1] - q;
            const S pos = fmin(dmin, dmax) - (S)Model::limit_margin(l);
            const S Jv = dmin < dmax ? S(1) : S(-1);
            s.type[i] = ROW_LIMIT; s.nnz[i] = 1; s.J[i * MC] = Jv; s.floss[i] = 0; s.mask[i] = 1ull << d;
            efc_row<S, Model>(pos, pos, (S)Model::limit_invweight(l), Model::limit_solref(l), Model::limit_solimp(l), Jv * s.qvel[d], s.D[i], s.aref[i]);
        }
        // ---- rows: contacts (thread per row)
        for (int rr = t; rr < NP * 4; rr += NT) {
            const int p = rr >> 2, dim = rr & 3;
            const int base = s.point_row[p];
            if (base < 0) continue;
            int g = 0, acc = 0;
            for (int gg = 0; gg < Model::NGEOM; ++gg) { int np = Model::geom_type(gg) == 3 ? 2 : 1; if (p < acc + np) { g = gg; break; } acc += np; }
            const int condim = Model::geom_condim(g);
            const int ndim = condim == 1 ? 1 : 2 * (condim - 1);
            if (dim >= ndim) continue;
            const int i = base + dim;
            const S* f = s.cframe() + 9 * p;
            const V3<S> n = {f[0], f[1], f[2]};
            V3<S> tdir = {0, 0, 0};
            S fri = 0;
            if (condim > 1) {
                const int dimid2 = dim / 2 + 1;                       // 1: tangent b, 2: tangent c
                const S* tt = f + 3 * dimid2;
                tdir = {tt[0], tt[1], tt[2]};
                fri = (S)Model::geom_friction(g)[dimid2 - 1] * (S(1) - S(2) * (S)(dim & 1));
            }
            const V3<S> rel = {s.cpos()[3 * p] - s.p0[0], s.cpos()[3 * p + 1] - s.p0[1], s.cpos()[3 * p + 2] - s.p0[2]};
            const int nz = Model::geom_nchain(g);
            S Jqvel = 0;
            unsigned long long mk = 0;
            for (int a = 0; a < nz; ++a) {
                const int d = Model::geom_chain(g)[a];
                mk |= 1ull << d;
                const S* cd = s.cdof + 6 * d;
                const V3<S> lin = {cd[0], cd[1], cd[2]}, ang = {cd[3], cd[4], cd[5]};
                const V3<S> jacp = lin + cross(ang, rel);
                const S Jd = dot(jacp, n) + fri * dot(jacp, tdir);
                s.J[i * MC + a] = Jd;
                Jqvel += Jd * s.qvel[d];
            }
            s.nnz[i] = nz; s.type[i] = ROW_CONTACT; s.floss[i] = 0; s.mask[i] = mk;
            const S pos = s.cdist()[p] - (S)Model::geom_margin(g);
            S invweight = (S)Model::geom_invweight(g);
            if (condim > 1) {
                const S fri0 = (S)Model::geom_friction(g)[0];
                invweight = invweight + fri0 * fri0 * invweight;
                invweight = invweight * S(2) * fri0 * fri0 / (S)Model::IMPRATIO;
            }
            efc_row<S, Model>(pos, pos, invweight, Model::geom_solref(g), Model::geom_solimp(g), Jqvel, s.D[i], s.aref[i]);
        }
        __syncthreads();
        stamp(2);
        // ---- factor M (into H), qacc_smooth = M^-1 qfrc_smooth
        for (int e = t; e < NV * NV; e += NT) s.H[e] = s.M[e];
        __syncthreads();
        factor_H(s);
        chol_solve(s, s.qfs, s.qacc_smooth);
        // ---- warm start: the better of qacc_warmstart and qacc_smooth (MuJoCo C warmstart()):
        // cost = constraint cost(jar) + 0.5 (M qacc - qfrc_smooth) . (qacc - qacc_smooth), the latter zero at qacc_smooth
        for (int k = t; k < NV; k += NT) { S acc = 0; for (int c = 0; c < NV; ++c) acc += s.M[k * NV + c] * s.qacc[c]; s.Ma[k] = acc; }
        for (int i = t; i < nefc; i += NT) {
            S a1 = 0, a2 = 0; const S* Ji = s.J + i * MC;
            unsigned long long mk = s.mask[i];
            for (int a = 0; mk; ++a) { const int d = __ffsll(mk) - 1; mk &= mk - 1; a1 += Ji[a] * s.qacc[d]; a2 += Ji[a] * s.qacc_smooth[d]; }
            s.jaref[i] = a1 - s.aref[i]; s.jv[i] = a2 - s.aref[i];               // jv holds jar(qacc_smooth) until the line search
        }
        __syncthreads();
        {
            S v[1] = {0};
            for (int k = t; k < NV; k += NT) v[0] += S(0.5) * (s.Ma[k] - s.qfs[k]) * (s.qacc[k] - s.qacc_smooth[k]);
            block_sum<S, 1, NT>(v, s.red_scratch, s.red);
            const S gauss_ws = s.red[0];
            const S cost_ws = constraint_cost(s, s.jaref) + gauss_ws;
            const S cost_sm = constraint_cost(s, s.jv);
            if (cost_ws > cost_sm) {
                for (int k = t; k < NV; k += NT) { s.qacc[k] = s.qacc_smooth[k]; s.Ma[k] = s.qfs[k]; }
                for (int i = t; i < nefc; i += NT) s.jaref[i] = s.jv[i];
                __syncthreads();
            }
        }
        stamp(3);
        update_constraint(s);
        update_grad(s);
        const S tol = (S)Model::TOLERANCE;
        // MuJoCo's tolerance scale: 1 / (meaninertia nv), or with the island solver (the default since
        // MuJoCo 3.3; this model is one tree, so one island of all dofs) 1 / trace(M)
        S scale;
        if (Model::ISLANDS) {
            S v[1] = {0};
            for (int k = t; k < NV; k += NT) v[0] += s.M[k * NV + k];
            block_sum<S, 1, NT>(v, s.red_scratch, s.red);
            scale = S(1) / s.red[0];
        } else {
            scale = S(1) / ((S)Model::MEANINERTIA * (S)(NV > 1 ? NV : 1));
        }
        const S gtol_scale = tol * (S)Model::LS_TOLERANCE / scale;
        // certificate: Mgrad = M^-1 grad (the factor of M is still in H), gap = 0.5 scale grad . Mgrad;
        // Newton also needs the gradient test
        solve_direction(s);
        const bool flg_gradient = scale * sqrt(s.grad_dot) < tol;
        bool done = fmax(S(0), S(0.5) * scale * s.newton_dec) < tol && flg_gradient;
        if (!done) {
            make_hessian(s);
            solve_direction(s);
            done = flg_gradient && fmax(S(0), S(0.5) * scale * s.newton_dec) < tol;
        }
        int niter = 0;
        stamp(4);
        while (!done && niter < Model::ITERATIONS) {
            S ls_improvement; int ls_iter;
            const S alpha = primal_search(s, gtol_scale, ls_improvement, ls_iter);
            stamp(5 + 4 * niter);
            if (alpha == S(0)) break;
            for (int k = t; k < NV; k += NT) { s.qacc[k] += alpha * s.search[k]; s.Ma[k] += alpha * s.mv[k]; }
            for (int i = t; i < nefc; i += NT) s.jaref[i] += alpha * s.jv[i];
            __syncthreads();
            update_constraint(s);
            update_grad(s);
            stamp(6 + 4 * niter);
            make_hessian(s);
            stamp(7 + 4 * niter);
            solve_direction(s);
            stamp(8 + 4 * niter);
            const S improvement = scale * ls_improvement, gradient = scale * sqrt(s.grad_dot);
            const S decrement = fmax(S(0), S(0.5) * scale * s.newton_dec);
            if (dump.iters && t == 0) {
                int nactive = 0; for (int i = 0; i < nefc; ++i) nactive += s.state[i] != STATE_SATISFIED;
                S* it = dump.iters + ((size_t)b * Model::ITERATIONS + niter) * 8;
                it[0] = alpha; it[1] = improvement; it[2] = gradient; it[3] = decrement; it[4] = (S)ls_iter; it[5] = (S)nactive; it[6] = 0; it[7] = 0;
            }
            ++niter;
            if ((improvement > S(0) && improvement < tol) || gradient < tol || decrement < tol) break;
            if (niter == Model::ITERATIONS && t == 0) s.overflow |= OVERFLOW_ITERATIONS;
        }
        __syncthreads();
        stamp(31);
        for (int k = t; k < NV; k += NT) { qfrc_constraint[k] = s.qfc[k]; qacc_out[k] = s.qacc[k]; warmstart[k] = s.qacc[k]; }
        if (t == 0) { stats->nefc = s.nefc; stats->ncon = s.ncon; stats->niter = niter; stats->overflow = s.overflow; }
        if (dump.J) {
            for (int e = t; e < nefc * NV; e += NT) dump.J[(size_t)b * NJMAX * NV + e] = 0;
            __syncthreads();
            for (int i = t; i < nefc; i += NT) {
                unsigned long long mk = s.mask[i];
                for (int a = 0; mk; ++a) { const int d = __ffsll(mk) - 1; mk &= mk - 1; dump.J[(size_t)b * NJMAX * NV + (size_t)i * NV + d] = s.J[i * MC + a]; }
                dump.D[(size_t)b * NJMAX + i] = s.D[i]; dump.aref[(size_t)b * NJMAX + i] = s.aref[i];
                dump.type[(size_t)b * NJMAX + i] = s.type[i]; dump.jaref[(size_t)b * NJMAX + i] = s.jaref[i];
            }
        }
    }
};

template <class S, class Model, int NJMAX, int NT>
__global__ void __launch_bounds__(NT) fused_kernel(int B, const S* qpos, const S* qvel, const S* Mnnz, const S* qfs, const S* gpos,
                                                   const S* grot, const S* cdof, S* warmstart, S* qfrc_constraint, S* qacc, Stats* stats,
                                                   RowDump<S> dump) {
    extern __shared__ unsigned char smem[];
    using Sh = Shared<S, Model, NJMAX>;
    Sh& s = *reinterpret_cast<Sh*>(smem);
    constexpr int NV = Model::NV, NQ = Model::NQ, NG = Model::NGEOM;
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        Fused<S, Model, NJMAX, NT>::run(s, qpos + (size_t)b * NQ, qvel + (size_t)b * NV, Mnnz + (size_t)b * Model::M_NNZ, qfs + (size_t)b * NV,
                                        gpos + (size_t)b * 3 * NG, grot + (size_t)b * 9 * NG, cdof + (size_t)b * 6 * NV, warmstart + (size_t)b * NV,
                                        qfrc_constraint + (size_t)b * NV, qacc + (size_t)b * NV, stats + b, dump, b);
        __syncthreads();
    }
}

// Launcher: one block per instance (grid capped at `max_blocks`, 0 = one block per instance). Call
// `setup` once per process before capturing a graph (the shared-memory opt-in attribute).
template <class S, class Model, int NJMAX = 128, int NT = 128>
struct FusedStep {
    static constexpr size_t shared_bytes = sizeof(Shared<S, Model, NJMAX>);
    static cudaError_t setup() {
        return cudaFuncSetAttribute(fused_kernel<S, Model, NJMAX, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);
    }
    static cudaError_t launch(int B, const S* qpos, const S* qvel, const S* Mnnz, const S* qfs, const S* gpos, const S* grot, const S* cdof,
                              S* warmstart, S* qfrc_constraint, S* qacc, Stats* stats, cudaStream_t st, int max_blocks = 0,
                              RowDump<S> dump = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}) {
        int grid = max_blocks > 0 && B > max_blocks ? max_blocks : B;
        fused_kernel<S, Model, NJMAX, NT><<<grid, NT, shared_bytes, st>>>(B, qpos, qvel, Mnnz, qfs, gpos, grot, cdof, warmstart, qfrc_constraint, qacc, stats, dump);
        return cudaGetLastError();
    }
};

}  // namespace cusadi2_contact
