"""Stage-wise MIT Humanoid MPC QP for cusadi2 + cuDSS.

Re-expresses one generic stage of the wip-branch (CusADi 2.0) MIT Humanoid MPC
formulation as small CasADi functions with a stage axis, so that cusadi2's
level-parallel kernel evaluates all (instance, stage) pairs in one launch and
the KKT linear solve is handed to cuDSS as a uniform batch of block-tridiagonal
systems. The ADMM (OSQP-style) loop and Ruiz equilibration mirror the wip
backend's admm_backend.py exactly, including operation order, so the staged
pipeline is checked bit-for-bit (or to rounding) against the wip's monolithic
whole-horizon functions.

Stage k (N=10 stages, 60 primal x_k, 74 constraints g_k):
  assemble   (x_{k-1}, x_k, x_{k+1}, dt_k, dt_{k+1}, is_first, has_next, p_stage_k, p_shared)
             -> blk_k = [P_kk^triu, P_{k-1,k}, A_kk, A_{k,k-1}] nonzeros, q_k, z_lb_k, z_ub_k
  ruiz_r1    column inf-norms of the symmetric KKT window          -> delta1_k
  ruiz_r2    per-stage max of scaled P column norms and |q|        -> m1_k   (reduce -> gamma1)
  ruiz_r3    second Ruiz pass column norms                         -> delta2_k
  ruiz_r4    per-stage max for gamma2                              -> m2_k   (reduce -> gamma2)
  ruiz_r5    scaled KKT block row (+sigma I, -1/rho), q_s, D, bounds, b0
  admm_iter  fused ADMM step + next RHS (+ dx = D x)

KKT rows are stage-interleaved ([x_0; g_0; x_1; g_1; ...]); csr.bin holds the
lower-triangular CSR pattern and the gather map from ruiz_r5's per-stage output
to the cuDSS values array.

usage: python build_mpc_stage.py <build_dir> [double,float]
Environment: CUSADI_WIP points at the directory holding wip_pkg/, stubs/,
mit_humanoid_urdf/ and humanoid_mpc_fns/ (reference functions + p_eval/dx).
"""
import json, os, struct, sys, time
import numpy as np, casadi as ca

SCRATCH = os.environ.get("CUSADI_WIP", "/tmp/claude-1000/-home-jovi-code-cusecmpc-cusadi2/71234236-054e-4990-b81c-ea03e2ae3482/scratchpad")
sys.path[:0] = [os.path.join(SCRATCH, "wip_pkg"), os.path.join(SCRATCH, "stubs")]
import cusadi2
from cusadi.controllers import MITHumanoidModelPredictiveController

OUT = sys.argv[1]; DTYPES = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["double", "float"])
REF = os.path.join(SCRATCH, "humanoid_mpc_fns")
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)

# ---------------------------------------------------------------- wip problem
c = MITHumanoidModelPredictiveController(os.path.join(SCRATCH, "mit_humanoid_urdf/humanoid_full_sf_toeheel.urdf"))
c.build()
c.setup_solver(qp_solver='osqp', qp_cfg={'max_iter': 25}, sqp_cfg={'max_iter': 1})
form, m, qp = c.formulation, c.model, c.solver.qp_backend
N, NV, NQ, NF = c.N_HORIZON, c.N_V, c.N_Q, c.N_F
NS, NG = c.N_STAGE, form.n_g // c.N_HORIZON
NB = NS + NG
assert form.n_x == N * NS and form.n_g == N * NG
cfg = qp.osqp_cfg
SIGMA, RHO_BAR, ALPHA, MAX_ITER, N_SCALING = cfg['sigma'], cfg['rho'], cfg['alpha'], cfg['max_iter'], cfg['scaling']
assert N_SCALING == 2, "staged Ruiz is written for scaling=2"
rho_norm = np.asarray(qp.rho_norm).reshape(-1)
assert all(np.array_equal(rho_norm[k*NG:(k+1)*NG], rho_norm[:NG]) for k in range(N))
rho_norm_s = rho_norm[:NG]
fn_opts = dict(c.fn_opts)                              # {'cse': True, 'post_expand': True}
p_eval = np.load(os.path.join(REF, "p_eval.npy")).reshape(-1)
par = {k: v.idx for k, v in form.parameters.items()}
lbg_f = ca.Function('lbg', [form.opti.x, form.opti.p], [form.opti.lbg, form.opti.ubg])
lbg, ubg = [np.asarray(v).reshape(-1) for v in lbg_f(np.zeros(form.n_x), p_eval)]
assert all(np.array_equal(lbg[k*NG:(k+1)*NG], lbg[:NG]) and np.array_equal(ubg[k*NG:(k+1)*NG], ubg[:NG]) for k in range(N))
lbg_s, ubg_s = lbg[:NG], ubg[:NG]
eq_s = np.zeros(NG, bool); eq_s[[i % NG for i in form.eq_idx if i < NG]] = True
assert np.array_equal(np.sort([i for i in form.eq_idx if i < NG]), np.flatnonzero(eq_s))
log(f"problem: N={N} NS={NS} NG={NG} NV={NV} NQ={NQ}  n_x={form.n_x} n_g={form.n_g} n_p={form.n_par}")

# ---------------------------------------------------------------- one stage, symbolically
x_prev, x_cur, x_next = (ca.MX.sym(n, NS) for n in ("x_prev", "x_cur", "x_next"))
dt_cur, dt_next, is_first, has_next = (ca.MX.sym(n) for n in ("dt_cur", "dt_next", "is_first", "has_next"))
P_STAGE = [("q_des", NQ), ("v_des", NV), ("F_des", NF), ("phi_traj", 4), ("contact_traj", 4)]
P_SHARED = [("q_0", NQ), ("v_0", NV), ("Q_fb", 6), ("Q_leg", 5), ("Q_arm", 4), ("Qd_fb", 6), ("Qd_leg", 5),
            ("Qd_arm", 4), ("Qa_fb", 6), ("Qa_leg", 5), ("Qa_arm", 4), ("R_F", NF)]
n_ps, n_pg = sum(n for _, n in P_STAGE), sum(n for _, n in P_SHARED)
p_stage, p_shared = ca.MX.sym("p_stage", n_ps), ca.MX.sym("p_shared", n_pg)
def split(sym, spec):
    out, o = {}, 0
    for name, n in spec: out[name] = sym[o:o+n]; o += n
    return out
S, G = split(p_stage, P_STAGE), split(p_shared, P_SHARED)
q_des_k, v_des_k, F_des_k, phi_k, contact_k = S["q_des"], S["v_des"], S["F_des"], S["phi_traj"], S["contact_traj"]
q_0, v_0 = G["q_0"], G["v_0"]
Q_q = ca.diag(ca.vertcat(G["Q_fb"], G["Q_leg"], G["Q_leg"], G["Q_arm"], G["Q_arm"]))
Q_v = ca.diag(ca.vertcat(G["Qd_fb"], G["Qd_leg"], G["Qd_leg"], G["Qd_arm"], G["Qd_arm"]))
Q_a = ca.diag(ca.vertcat(G["Qa_fb"], G["Qa_leg"], G["Qa_leg"], G["Qa_arm"], G["Qa_arm"]))
R_F = G["R_F"]

# Bezier swing height (wip build_bezier_swing_fn, un-mapped)
import math
_t, _vTO, _vTD, _h = (ca.MX.sym(n) for n in ("t_swing", "v_TO", "v_TD", "h_swing"))
_P = ca.MX(5, 1); _P[2] = _vTO/4; _P[3] = 8*(_h - _vTO/16 + _vTD/16)/3; _P[4] = -_vTD/4
_B = sum(math.comb(4, i) * (1 - _t) ** (4 - i) * _t ** i * _P[i] for i in range(5))
fn_swing = ca.Function('fn_swing_traj', [_t, _vTO, _vTD, _h], [_B, ca.jacobian(_B, _t), ca.jacobian(ca.jacobian(_B, _t), _t)],
                       ['t_swing', 'v_TO', 'v_TD', 'h_swing'], ['B_t', 'B_dot_t', 'B_ddot_t'], fn_opts)

dq_p, v_p, F_p = x_prev[c.dq_idx], x_prev[c.v_idx], x_prev[c.F_idx]
dq_k, v_k, F_k = x_cur[c.dq_idx], x_cur[c.v_idx], x_cur[c.F_idx]
v_n = x_next[c.v_idx]
q_k = m.get_integrated_states(q_0, dq_k)
v_prev = ca.if_else(is_first, v_0, v_p)
a_k = (v_k - v_prev) / dt_cur
fwd_kin_k = m.get_forward_kinematics(q=q_k, v=v_k, frames=c.end_eff_frames)
dot_quat = (q_k[3]*q_des_k[3] + q_k[4]*q_des_k[4] + q_k[5]*q_des_k[5] + q_k[6]*q_des_k[6])
sign = ca.if_else(dot_quat >= 0, 1.0, -1.0)
q_des_k_aligned = ca.vertcat(q_des_k[:3], sign * q_des_k[3:7], q_des_k[7:])
q_err_k = m.get_state_error(q_k, q_des_k_aligned)
v_err_k = v_des_k - v_k
F_err_k = F_des_k - F_k
cost = 0
cost += q_err_k.T @ Q_q @ q_err_k * dt_cur
cost += v_err_k.T @ Q_v @ v_err_k * dt_cur
cost += a_k.T @ Q_a @ a_k * dt_cur
cost += F_err_k.T @ ca.diag(R_F) @ F_err_k * dt_cur
a_n = (v_n - v_k) / dt_next                     # stage k+1's acceleration term depends on v_k
cost += has_next * (a_n.T @ Q_a @ a_n * dt_next)

q_prev = ca.if_else(is_first, q_0, m.get_integrated_states(q_0, dq_p))
v_int = v_prev + a_k * dt_cur
q_integrated = m.get_integrated_states(q_prev, v_int * dt_cur)
g_pos = m.get_state_error(q_k, q_integrated)
tau_inertial_k = m.get_inverse_dynamics(q_k, v_k, a_k)[:6]
tau_external_k = m.get_external_wrench(q_k, F_k*m.bodyweight, c.end_eff_frames)[:6]
g_dyn = tau_inertial_k - tau_external_k
p_contact_k = ca.vertcat(fwd_kin_k['p_right_toe'], fwd_kin_k['p_left_toe'], fwd_kin_k['p_right_heel'], fwd_kin_k['p_left_heel'])
h_des_k = ca.vertcat(*[fn_swing(phi_k[i], 0.02, -0.02, 0.05)[0] for i in range(4)])
g_swing = p_contact_k[c.contact_z_idx] - h_des_k
F_x, F_y, F_z = F_k[c.contact_x_idx], F_k[c.contact_y_idx], F_k[c.contact_z_idx]
v_contact_k = ca.vertcat(fwd_kin_k['v_right_toe'], fwd_kin_k['v_left_toe'], fwd_kin_k['v_right_heel'], fwd_kin_k['v_left_heel'])
mu = 0.7
lb_x, ub_x = -mu*F_z/np.sqrt(2), mu*F_z/np.sqrt(2)
g_fric_x = ca.vertcat(lb_x - F_x, F_x - ub_x)       # Opti canonical form of lb <= F_x <= ub
g_fric_y = ca.vertcat(lb_x - F_y, F_y - ub_x)
g_cvx = v_contact_k[c.contact_x_idx] * contact_k
g_cvy = v_contact_k[c.contact_y_idx] * contact_k
g_cF = ca.vertcat(F_k[c.right_toe_idx] * (1 - contact_k[0]), F_k[c.left_toe_idx] * (1 - contact_k[1]),
                  F_k[c.right_heel_idx] * (1 - contact_k[2]), F_k[c.left_heel_idx] * (1 - contact_k[3]))
q_fb_k = q_k[:7]
qx, qy, qz, qw = q_fb_k[3], q_fb_k[4], q_fb_k[5], q_fb_k[6]
siny_cosp = 2 * (qw * qz + qx * qy); cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
body_yaw = ca.atan2(siny_cosp, cosy_cosp); c_yaw = ca.cos(body_yaw); s_yaw = ca.sin(body_yaw)
b_R_w = ca.vertcat(ca.horzcat(c_yaw, s_yaw, 0), ca.horzcat(-s_yaw, c_yaw, 0), ca.horzcat(0, 0, 1))
p_rt = b_R_w @ p_contact_k[c.right_toe_idx]; p_lt = b_R_w @ p_contact_k[c.left_toe_idx]
p_rh = b_R_w @ p_contact_k[c.right_heel_idx]; p_lh = b_R_w @ p_contact_k[c.left_heel_idx]
dist_toe = (p_lt[1] - p_rt[1] - c.collision_radius**2 - c.offset_lateral_stance)
dist_heel = (p_lh[1] - p_rh[1] - c.collision_radius**2 - c.offset_lateral_stance)
fwd_kin_hip = m.get_forward_kinematics(q=q_k, frames=['right_hip_yaw', 'left_hip_yaw'])
d_right = 0.5 * (p_contact_k[c.right_toe_idx] + p_contact_k[c.right_heel_idx]) - fwd_kin_hip['p_right_hip_yaw']
d_left = 0.5 * (p_contact_k[c.left_toe_idx] + p_contact_k[c.left_heel_idx]) - - fwd_kin_hip['p_left_hip_yaw']
l_leg_max = 0.58
g_len = ca.vertcat(d_right.T @ d_right - l_leg_max ** 2, d_left.T @ d_left - l_leg_max ** 2)
g_k = ca.vertcat(g_pos, g_dyn, g_swing, g_fric_x, g_fric_y, g_cvx, g_cvy, g_cF, dist_toe, dist_heel, g_len)
assert g_k.shape == (NG, 1)

# Derivatives: P_kk (upper), P_{k-1,k} (rows x_{k-1}, cols x_k), A_kk, A_{k,k-1}
P_kk, c_k = ca.hessian(cost, x_cur)
P_triu = ca.triu(P_kk)
P_up = ca.jacobian(ca.gradient(cost, x_prev), x_cur)
A_kk = ca.jacobian(g_k, x_cur)
A_prev = ca.jacobian(g_k, x_prev)
SP = dict(Ptriu=P_triu.sparsity(), Pup=P_up.sparsity(), A=A_kk.sparsity(), Aprev=A_prev.sparsity())
z_ub = ca.vertcat(*[(-g_k[i]) if eq_s[i] else (ubg_s[i] - g_k[i]) for i in range(NG)])
z_lb = ca.vertcat(*[(-g_k[i]) if eq_s[i] else (lbg_s[i] - g_k[i]) for i in range(NG)])
flat = lambda M: ca.sparsity_cast(M, ca.Sparsity.dense(M.nnz(), 1))
blk = ca.vertcat(flat(P_triu), flat(P_up), flat(A_kk), flat(A_prev))
NBLK = blk.numel()
fn_asm = ca.Function("mpc_assemble", [x_prev, x_cur, x_next, dt_cur, dt_next, is_first, has_next, p_stage, p_shared],
                     [blk, c_k, z_lb, z_ub], ["x_prev", "x_cur", "x_next", "dt_cur", "dt_next", "is_first", "has_next", "p_stage", "p_shared"],
                     ["blk", "q", "z_lb", "z_ub"], fn_opts)
log(f"assemble: {fn_asm.n_instructions()} instr, sz_w {fn_asm.sz_w()}, blk {NBLK} = " +
    " + ".join(f"{k} {v.nnz()}" for k, v in SP.items()))

# ---------------------------------------------------------------- Ruiz (osqp scaling=2), staged
def unflat(b):
    out, o = {}, 0
    for k, sp in SP.items(): out[k] = ca.sparsity_cast(b[o:o+sp.nnz()], sp); o += sp.nnz()
    return out
Z = lambda r, c_: ca.MX(r, c_)
def window(bk, bn):
    """Symmetric-KKT window: rows [x_{k-1}, x_k, x_{k+1}, g_k, g_{k+1}], cols [x_k, g_k] (no diagonal -1/rho)."""
    Bk, Bn = unflat(bk), unflat(bn)
    Pxx = ca.triu2symm(Bk["Ptriu"])
    P_top = Bk["Pup"]                     # rows x_{k-1}, cols x_k
    P_bot = Bn["Pup"].T                   # rows x_{k+1}, cols x_k  (P_{k+1,k} = P_{k,k+1}^T)
    return dict(P=ca.vertcat(P_top, Pxx, P_bot),                          # 3*NS x NS
                A=ca.vertcat(Bk["A"], Bn["Aprev"]),                       # 2*NG x NS  (rows g_k, g_{k+1})
                AT=ca.vertcat(Bk["Aprev"].T, Bk["A"].T))                  # 2*NS x NG  (rows x_{k-1}, x_k)
def col_norms(Pw, Aw, ATw):
    """Column inf-norms exactly as the wip: mmax(fabs(M_bar[:, j]))."""
    Mx = ca.vertcat(Pw, Aw)               # x-columns: P rows then A rows
    Mg = ATw                              # g-columns
    return ca.vertcat(*[ca.mmax(ca.fabs(Mx[:, j])) for j in range(NS)],
                      *[ca.mmax(ca.fabs(Mg[:, j])) for j in range(NG)])
def delta_of(norms):
    return ca.vertcat(*[ca.if_else(ca.sqrt(norms[j]) <= 0, 1, 1/ca.sqrt(norms[j])) for j in range(NB)])
def scaleP(Pw, r, s):  return ca.diag(r) @ Pw @ ca.diag(s)      # (r_i P_ij) s_j
def scaleA(Aw, e, d):  return ca.diag(e) @ Aw @ ca.diag(d)      # (e_i A_ij) d_j

blk_k, blk_n = ca.MX.sym("blk_k", NBLK), ca.MX.sym("blk_n", NBLK)
d1p, d1k, d1n, d2p, d2k, d2n = (ca.MX.sym(n, NB) for n in ("d1_prev", "d1_k", "d1_next", "d2_prev", "d2_k", "d2_next"))
gamma1, gamma2, sigma, rho_bar, alpha = (ca.MX.sym(n) for n in ("gamma1", "gamma2", "sigma", "rho_bar", "alpha"))
q_k_s, zlb_s, zub_s = ca.MX.sym("q_k", NS), ca.MX.sym("z_lb", NG), ca.MX.sym("z_ub", NG)
W = window(blk_k, blk_n)
# r1: delta1 from the unscaled window
fn_r1 = ca.Function("mpc_ruiz_r1", [blk_k, blk_n], [delta_of(col_norms(W["P"], W["A"], W["AT"]))],
                    ["blk_k", "blk_n"], ["delta1"], fn_opts)
# scales for the window rows/cols from per-stage deltas: x rows use d[:NS], g rows use d[NS:]
def rows_P(dp, dk, dn): return ca.vertcat(dp[:NS], dk[:NS], dn[:NS])
def rows_A(dk, dn):     return ca.vertcat(dk[NS:], dn[NS:])
def rows_AT(dp, dk):    return ca.vertcat(dp[:NS], dk[:NS])
P1 = scaleP(W["P"], rows_P(d1p, d1k, d1n), d1k[:NS])
A1 = scaleA(W["A"], rows_A(d1k, d1n), d1k[:NS])
AT1 = scaleA(W["AT"], rows_AT(d1p, d1k), d1k[NS:])
q1 = d1k[:NS] * q_k_s                                        # D @ q
# r2: partial max for gamma1: max over x-columns of P1 col norms and |q1|
P1_inf = ca.vertcat(*[ca.mmax(ca.fabs(P1[:, j])) for j in range(NS)])
fn_r2 = ca.Function("mpc_ruiz_r2", [blk_k, blk_n, d1p, d1k, d1n, q_k_s],
                    [ca.fmax(ca.mmax(P1_inf), ca.mmax(ca.fabs(q1)))],
                    ["blk_k", "blk_n", "d1_prev", "d1_k", "d1_next", "q_k"], ["m1"], fn_opts)
# r3: delta2 from the iteration-1 window (P part scaled by gamma1)
fn_r3 = ca.Function("mpc_ruiz_r3", [blk_k, blk_n, d1p, d1k, d1n, gamma1],
                    [delta_of(col_norms(gamma1 * P1, A1, AT1))],
                    ["blk_k", "blk_n", "d1_prev", "d1_k", "d1_next", "gamma1"], ["delta2"], fn_opts)
# r4: partial max for gamma2
P2 = scaleP(gamma1 * P1, rows_P(d2p, d2k, d2n), d2k[:NS])
q2 = d2k[:NS] * (gamma1 * q1)
P2_inf = ca.vertcat(*[ca.mmax(ca.fabs(P2[:, j])) for j in range(NS)])
fn_r4 = ca.Function("mpc_ruiz_r4", [blk_k, blk_n, d1p, d1k, d1n, d2p, d2k, d2n, gamma1, q_k_s],
                    [ca.fmax(ca.mmax(P2_inf), ca.mmax(ca.fabs(q2)))],
                    ["blk_k", "blk_n", "d1_prev", "d1_k", "d1_next", "d2_prev", "d2_k", "d2_next", "gamma1", "q_k"], ["m2"], fn_opts)
# r5: final scaling of block row k (needs stage k-1 scales for the coupling blocks)
Bk = unflat(blk_k)
S_k, S_p = d2k * d1k, d2p * d1p                              # diag(delta2) @ diag(delta1)
c_scale = gamma2 * gamma1
d_k, e_k, d_p = S_k[:NS], S_k[NS:], S_p[:NS]
cD_k = c_scale * d_k
P_s = ca.diag(cD_k) @ Bk["Ptriu"] @ ca.diag(d_k) + sigma * ca.MX.eye(NS)          # c*D P D + sigma I (upper)
Pup_s = ca.diag(c_scale * d_p) @ Bk["Pup"] @ ca.diag(d_k)                          # rows x_{k-1}, cols x_k
A_s = ca.diag(e_k) @ Bk["A"] @ ca.diag(d_k)
Aprev_s = ca.diag(e_k) @ Bk["Aprev"] @ ca.diag(d_p)
rho = rho_norm_s * rho_bar; rho_inv = 1 / rho
diag_g = -rho_inv
q_s = cD_k * q_k_s                                           # (c_scale @ D_scale) @ c
zlb_sc, zub_sc = e_k * zlb_s, e_k * zub_s
b0 = ca.vertcat(sigma * ca.MX.zeros(NS) - q_s, ca.MX.zeros(NG) - rho_inv * ca.MX.zeros(NG))
kkt_blk = ca.vertcat(flat(P_s), flat(Pup_s), flat(A_s), flat(Aprev_s), diag_g)
SP5 = dict(Ptriu=P_s.sparsity(), Pup=Pup_s.sparsity(), A=A_s.sparsity(), Aprev=Aprev_s.sparsity())
NKKT = kkt_blk.numel()
fn_r5 = ca.Function("mpc_ruiz_r5", [blk_k, d1p, d1k, d2p, d2k, gamma1, gamma2, q_k_s, zlb_s, zub_s, sigma, rho_bar],
                    [kkt_blk, q_s, d_k, zlb_sc, zub_sc, b0],
                    ["blk_k", "d1_prev", "d1_k", "d2_prev", "d2_k", "gamma1", "gamma2", "q_k", "z_lb", "z_ub", "sigma", "rho_bar"],
                    ["kkt_blk", "q_s", "D", "z_lb_s", "z_ub_s", "b0"], fn_opts)
# admm_iter: ADMM step + next RHS + dx
x_solve = ca.MX.sym("x_solve", NB); xk, yk, zk = ca.MX.sym("x_k", NS), ca.MX.sym("y_k", NG), ca.MX.sym("z_k", NG)
D_in = ca.MX.sym("D", NS)
x_sub, nu_sub = x_solve[:NS], x_solve[NS:]
w = alpha * nu_sub + (1 - alpha) * yk
x_n = alpha * x_sub + (1 - alpha) * xk
z_n = ca.fmax(ca.fmin(rho_inv * w + zk, zub_s), zlb_s)
y_n = w + rho * (zk - z_n)
b_n = ca.vertcat(sigma * x_n - q_k_s, z_n - rho_inv * y_n)
fn_it = ca.Function("mpc_admm_iter", [x_solve, xk, yk, zk, q_k_s, zlb_s, zub_s, D_in, sigma, rho_bar, alpha],
                    [x_n, y_n, z_n, b_n, D_in * x_n],
                    ["x_solve", "x_k", "y_k", "z_k", "q_s", "z_lb_s", "z_ub_s", "D", "sigma", "rho_bar", "alpha"],
                    ["x_next", "y_next", "z_next", "b_next", "dx"], fn_opts)
FNS = [fn_asm, fn_r1, fn_r2, fn_r3, fn_r4, fn_r5, fn_it]
for f in FNS: log(f"  {f.name():16s} instr={f.n_instructions():7d} sz_w={f.sz_w():6d}")

# ---------------------------------------------------------------- CSR (lower, stage-interleaved) + gather map
def ccs_entries(sp):
    """(row, col) per nonzero in CasADi CCS order."""
    colind, row = sp.colind(), sp.row()
    return [(row[e], j) for j in range(sp.size2()) for e in range(colind[j], colind[j+1])]
ent = {k: ccs_entries(v) for k, v in SP5.items()}
off = {}; o = 0
for k in ("Ptriu", "Pup", "A", "Aprev"): off[k] = o; o += SP5[k].nnz()
off["diag"] = o; assert o + NG == NKKT
entries = []                                                       # (row, col, src) with src = k*NKKT + local
for k in range(N):
    base = k * NB
    for e, (i, j) in enumerate(ent["Ptriu"]): entries.append((base + j, base + i, k*NKKT + off["Ptriu"] + e))   # upper -> lower
    for e, (i, j) in enumerate(ent["A"]):     entries.append((base + NS + i, base + j, k*NKKT + off["A"] + e))
    for i in range(NG):                       entries.append((base + NS + i, base + NS + i, k*NKKT + off["diag"] + i))
    if k > 0:
        pb = (k - 1) * NB
        for e, (i, j) in enumerate(ent["Pup"]):   entries.append((base + j, pb + i, k*NKKT + off["Pup"] + e))   # (x_{k-1} i, x_k j) -> lower (x_k j, x_{k-1} i)
        for e, (i, j) in enumerate(ent["Aprev"]): entries.append((base + NS + i, pb + j, k*NKKT + off["Aprev"] + e))
assert all(r >= c_ for r, c_, _ in entries)
entries.sort()
assert len(set((r, c_) for r, c_, _ in entries)) == len(entries), "duplicate CSR entries"
n_kkt, nnz = N * NB, len(entries)
Ap = np.zeros(n_kkt + 1, np.int32)
for r, _, _ in entries: Ap[r + 1] += 1
Ap = np.cumsum(Ap).astype(np.int32)
Ai = np.array([c_ for _, c_, _ in entries], np.int32)
gmap = np.array([s for _, _, s in entries], np.int32)
log(f"KKT: n={n_kkt} nnz(lower)={nnz}  per-stage kkt_blk={NKKT}")

# ---------------------------------------------------------------- CPU verification vs the wip whole-horizon functions
f_kkt = ca.Function.load(os.path.join(REF, "KKT_matrix_mit_humanoid_mpc.casadi"))
f_ruiz = ca.Function.load(os.path.join(REF, "ruiz_scaling_mit_humanoid_mpc.casadi"))
f_ldl_fac = ca.Function.load(os.path.join(REF, "ldl_factorize_mit_humanoid_mpc.casadi"))
f_ldl_solve = ca.Function.load(os.path.join(REF, "ldl_solve_mit_humanoid_mpc.casadi"))
dx_ref = np.load(os.path.join(REF, "dx_custom.npy")).reshape(-1)
x0 = np.zeros(form.n_x)
# host-side inputs (batch of one), in the layouts the plans declare
dt = p_eval[par["dt"]]
x_pad = np.zeros((N + 2, NS)); x_pad[1:N+1] = x0.reshape(N, NS)          # [pad, x_0..x_{N-1}, pad]
dt_pad = np.concatenate([dt, [1.0]])                                      # pad 1.0: no /0 in the masked next-stage term
is_first_v = np.array([1.0] + [0.0] * (N - 1)); has_next_v = np.array([1.0] * (N - 1) + [0.0])
ps = np.stack([np.concatenate([p_eval[par[n]].reshape(sz, N, order='F')[:, k] for n, sz in P_STAGE]) for k in range(N)])
pg = np.concatenate([p_eval[par[n]] for n, _ in P_SHARED])
wip_x = lambda k, i: k * NS + i
wip_g = lambda k, j: form.n_x + k * NG + j
perm = np.zeros(n_kkt, int)                                              # mine -> wip index
for k in range(N):
    for i in range(NS): perm[k*NB + i] = wip_x(k, i)
    for j in range(NG): perm[k*NB + NS + j] = wip_g(k, j)

def npf(*a): return [np.asarray(v).reshape(-1) for v in a]
t0 = time.perf_counter()
asm = [fn_asm(x_pad[k], x_pad[k+1], x_pad[k+2], dt_pad[k], dt_pad[k+1], is_first_v[k], has_next_v[k], ps[k], pg) for k in range(N)]
BLK = np.stack([npf(a[0])[0] for a in asm]); Q = np.stack([npf(a[1])[0] for a in asm])
ZLB = np.stack([npf(a[2])[0] for a in asm]); ZUB = np.stack([npf(a[3])[0] for a in asm])
log(f"CPU assemble x{N}: {time.perf_counter()-t0:.2f}s")
# unscaled KKT check (wip KKT_matrix: triu of [P A^T; A 0] in [x; g] order)
kkt_w = f_kkt(x0, p_eval); spw = kkt_w.sparsity(); vw = np.asarray(kkt_w.nonzeros())
wip_entries = {(r, c_): v for (r, c_), v in zip(ccs_entries(spw), vw)}
ent0 = {k: ccs_entries(v) for k, v in SP.items()}
off0 = {}; o = 0
for k in ("Ptriu", "Pup", "A", "Aprev"): off0[k] = o; o += SP[k].nnz()
mine = {}
for k in range(N):
    for e, (i, j) in enumerate(ent0["Ptriu"]): mine[(wip_x(k, i), wip_x(k, j))] = BLK[k, off0["Ptriu"] + e]
    for e, (i, j) in enumerate(ent0["A"]):     mine[(wip_x(k, j), wip_g(k, i))] = BLK[k, off0["A"] + e]
    if k > 0:
        for e, (i, j) in enumerate(ent0["Pup"]):   mine[(wip_x(k-1, i), wip_x(k, j))] = BLK[k, off0["Pup"] + e]
        for e, (i, j) in enumerate(ent0["Aprev"]): mine[(wip_x(k-1, j), wip_g(k, i))] = BLK[k, off0["Aprev"] + e]
def cmp(name, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    fin = np.isfinite(a) & np.isfinite(b)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)) and np.array_equal(a[~fin], b[~fin]), name
    err = np.abs(a[fin] - b[fin]); scale = np.maximum(np.abs(b[fin]), 1e-300)
    exact = np.array_equal(a, b)
    log(f"  {name:28s} n={a.size:6d} max|d|={err.max():.3e} max rel={(err/scale).max():.3e} {'BIT-EXACT' if exact else ''}")
    return err.max(), exact
missing = set(wip_entries) - set(mine); extra = set(mine) - set(wip_entries)
log(f"unscaled KKT pattern: wip nnz {len(wip_entries)}, mine {len(mine)}, missing {len(missing)}, extra {len(extra)}"
    f"{'  (extra entries all zero: '+str(all(mine[e]==0 for e in extra))+')' if extra else ''}")
assert not missing
keys = sorted(wip_entries)
cmp("KKT_matrix nonzeros", [mine[k] for k in keys], [wip_entries[k] for k in keys])
# staged Ruiz on CPU
def padded(a, before, after): return np.concatenate([np.zeros((before,) + a.shape[1:]), a, np.zeros((after,) + a.shape[1:])])
BLKp = padded(BLK, 0, 1)
D1 = np.stack([npf(fn_r1(BLKp[k], BLKp[k+1]))[0] for k in range(N)]); D1p = padded(D1, 1, 1)
M1 = np.array([float(fn_r2(BLKp[k], BLKp[k+1], D1p[k], D1p[k+1], D1p[k+2], Q[k])) for k in range(N)])
G1 = 1 / np.max(M1)
D2 = np.stack([npf(fn_r3(BLKp[k], BLKp[k+1], D1p[k], D1p[k+1], D1p[k+2], G1))[0] for k in range(N)]); D2p = padded(D2, 1, 1)
M2 = np.array([float(fn_r4(BLKp[k], BLKp[k+1], D1p[k], D1p[k+1], D1p[k+2], D2p[k], D2p[k+1], D2p[k+2], G1, Q[k])) for k in range(N)])
G2 = 1 / np.max(M2)
r5 = [fn_r5(BLK[k], D1p[k], D1p[k+1], D2p[k], D2p[k+1], G1, G2, Q[k], ZLB[k], ZUB[k], SIGMA, RHO_BAR) for k in range(N)]
KB = np.stack([npf(r[0])[0] for r in r5]); QS = np.stack([npf(r[1])[0] for r in r5]); DD = np.stack([npf(r[2])[0] for r in r5])
ZLBs = np.stack([npf(r[3])[0] for r in r5]); ZUBs = np.stack([npf(r[4])[0] for r in r5]); B0 = np.stack([npf(r[5])[0] for r in r5])
kkt_nz_w = np.asarray(kkt_w.nonzeros()).reshape(-1)
rw = f_ruiz(kkt_nz_w, x0, p_eval, SIGMA, RHO_BAR)
ks_w = rw[0]; sp_ks = ks_w.sparsity(); vks = np.asarray(ks_w.nonzeros()).reshape(-1)
q_w, D_w, lb_w, ub_w = npf(rw[1], rw[2], rw[3], rw[4])
# my scaled values: lower CSR (row r, col c) in my order -> wip upper (perm[c], perm[r]) if perm[c] <= perm[r] else swapped
vals_mine = KB.reshape(-1)[gmap]
mine_s = {}
for (r, c_), v in zip([(r, c_) for r, c_, _ in entries], vals_mine):
    a, b = perm[r], perm[c_]
    mine_s[(min(a, b), max(a, b))] = v
wip_s = {(r, c_): v for (r, c_), v in zip(ccs_entries(sp_ks), vks)}
assert set(wip_s) == set(mine_s), (len(set(wip_s) - set(mine_s)), len(set(mine_s) - set(wip_s)))
keys = sorted(wip_s)
log("staged Ruiz vs wip ruiz_scaling:")
cmp("kkt_scaled_triu", [mine_s[k] for k in keys], [wip_s[k] for k in keys])
cmp("q_scaled", QS.reshape(-1), q_w)
cmp("D_scale", DD.reshape(-1), D_w)
cmp("z_lb_scaled", ZLBs.reshape(-1), lb_w)
cmp("z_ub_scaled", ZUBs.reshape(-1), ub_w)
# ADMM loop on CPU with the wip symbolic LDL (wip ordering), my iteration function per stage
triu_nz = np.array([mine_s[k] for k in ccs_entries(sp_ks)])
Dl, Ll, _ = f_ldl_fac(triu_nz)
Xk, Yk, Zk, Bm = np.zeros((N, NS)), np.zeros((N, NG)), np.zeros((N, NG)), B0.copy()
inv_perm = np.argsort(perm)
for it in range(MAX_ITER):
    b_w = np.zeros(n_kkt); b_w[perm] = Bm.reshape(-1)
    xs_w = npf(f_ldl_solve(b_w, Dl, Ll.nonzeros()))[0]
    XS = xs_w[perm].reshape(N, NB)
    outs = [fn_it(XS[k], Xk[k], Yk[k], Zk[k], QS[k], ZLBs[k], ZUBs[k], DD[k], SIGMA, RHO_BAR, ALPHA) for k in range(N)]
    Xk = np.stack([npf(o[0])[0] for o in outs]); Yk = np.stack([npf(o[1])[0] for o in outs])
    Zk = np.stack([npf(o[2])[0] for o in outs]); Bm = np.stack([npf(o[3])[0] for o in outs])
    DX = np.stack([npf(o[4])[0] for o in outs])
log("staged ADMM (CPU, wip LDL) vs wip dx_custom:")
err_dx, exact_dx = cmp("dx", DX.reshape(-1), dx_ref)

# ---------------------------------------------------------------- plans, kernels, driver inputs
STAGE, SHARED, COUPLED = "stage", "shared", "coupled"
def spec(*items):
    mode, offs, span = [], [], []
    for it in items:
        it = (it,) if isinstance(it, str) else it
        mode.append(it[0]); offs.append(it[1] if len(it) > 1 else 0); span.append(it[2] if len(it) > 2 else 0)
    return dict(in_mode=mode, in_offset=offs, in_span=span)
X2 = lambda off: (COUPLED, off, 2); X1 = lambda off: (COUPLED, off, 1)
SPECS = {
    "mpc_assemble":  spec(X2(0), X2(1), X2(2), X1(0), X1(1), STAGE, STAGE, STAGE, SHARED),
    "mpc_ruiz_r1":   spec(X1(0), X1(1)),
    "mpc_ruiz_r2":   spec(X1(0), X1(1), X2(0), X2(1), X2(2), STAGE),
    "mpc_ruiz_r3":   spec(X1(0), X1(1), X2(0), X2(1), X2(2), SHARED),
    "mpc_ruiz_r4":   spec(X1(0), X1(1), X2(0), X2(1), X2(2), X2(0), X2(1), X2(2), SHARED, STAGE),
    "mpc_ruiz_r5":   spec(STAGE, X2(0), X2(1), X2(0), X2(1), SHARED, SHARED, STAGE, STAGE, STAGE, SHARED, SHARED),
    "mpc_admm_iter": spec(STAGE, STAGE, STAGE, STAGE, STAGE, STAGE, STAGE, STAGE, SHARED, SHARED, SHARED),
}
plan_dir = os.path.join(OUT, "plans"); os.makedirs(plan_dir, exist_ok=True)
info = {}
for f in FNS:
    t0 = time.perf_counter()
    cusadi2.write_plan(f, os.path.join(plan_dir, f.name() + ".plan"), n_stages=N, **SPECS[f.name()])
    p = cusadi2.read_plan(os.path.join(plan_dir, f.name() + ".plan"))
    info[f.name()] = dict(n_instr=f.n_instructions(), sz_w=f.sz_w(), n_levels=len(p["level_widths"]),
                          peak_live_slots=p["peak_live_slots"], block_size=p["block_size"], plan_s=time.perf_counter() - t0)
    log(f"plan {f.name():16s} levels={len(p['level_widths']):5} peak_live={p['peak_live_slots']:6d} "
        f"(smem f64 {p['peak_live_slots']*8/1024:.1f} KB) block={p['block_size']} {info[f.name()]['plan_s']:.1f}s")
for dtype in DTYPES:
    cusadi2.write_kernel_source(os.path.join(OUT, f"kernel_{dtype}.cu"), dtype=dtype)
with open(os.path.join(OUT, "csr.bin"), "wb") as fh:                   # i32 n, nnz, N, NB, NKKT; Ap[n+1]; Ai[nnz]; map[nnz]
    fh.write(struct.pack("<5i", n_kkt, nnz, N, NB, NKKT)); fh.write(Ap.tobytes()); fh.write(Ai.tobytes()); fh.write(gmap.tobytes())
inp = os.path.join(OUT, "inputs"); os.makedirs(inp, exist_ok=True)
for name, arr in [("x_pad", x_pad), ("dt_pad", dt_pad), ("is_first", is_first_v), ("has_next", has_next_v),
                  ("p_stage", ps), ("p_shared", pg), ("scalars", np.array([SIGMA, RHO_BAR, ALPHA])), ("dx_ref", dx_ref),
                  ("dx_cpu_staged", DX.reshape(-1))]:
    np.ascontiguousarray(arr, dtype=np.float64).tofile(os.path.join(inp, name + ".f64"))
json.dump(dict(N=N, NS=NS, NG=NG, NB=NB, NBLK=NBLK, NKKT=NKKT, n_kkt=n_kkt, nnz=nnz, n_ps=n_ps, n_pg=n_pg,
               sigma=SIGMA, rho_bar=RHO_BAR, alpha=ALPHA, max_iter=MAX_ITER,
               cpu_staged_vs_ref_max_abs=float(err_dx), cpu_staged_vs_ref_exact=bool(exact_dx), functions=info),
          open(os.path.join(OUT, "meta.json"), "w"), indent=2)
log("BUILD STAGE DONE")
