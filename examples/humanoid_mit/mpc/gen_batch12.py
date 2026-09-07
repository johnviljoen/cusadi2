"""12 distinct MPC QP solutions (OSQP, same QP as the GPU benchmark) with per-instance
desired velocity and gait phase, plus the resulting q trajectories for rendering."""
import os, sys, numpy as np, casadi as ca
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "wip_pkg"), os.path.join(HERE, "stubs")]
from cusadi.controllers import MITHumanoidModelPredictiveController
c = MITHumanoidModelPredictiveController(os.path.join(HERE, "mit_humanoid_urdf/humanoid_full_sf_toeheel.urdf")); c.build()
c.setup_solver(qp_solver='osqp', qp_cfg={'max_iter': 25}, sqp_cfg={'max_iter': 1})
f = c.formulation; qp = c.solver.qp_backend; N, NV, NQ, NS = c.N_HORIZON, c.N_V, c.N_Q, c.N_STAGE
x0 = np.zeros((f.n_x, 1)); Q = []; P = []
rng = np.random.default_rng(0)
for i in range(12):
    vx = 0.6 * (i % 4) / 3.0                                # 0 .. 0.6 m/s forward
    wz = [-0.5, 0.0, 0.5][i // 4]                           # yaw rate
    v_des = np.zeros((NV, N)); v_des[0, :] = vx; v_des[5, :] = wz
    contact = np.ones((4, N)); phase = np.zeros((4, N))
    if i % 4:                                               # trot-like alternation for moving instances
        for k in range(N):
            swing = ((k + i) // 3) % 2
            contact[[0, 2] if swing else [1, 3], k] = 0.0
            phase[[0, 2] if swing else [1, 3], k] = ((k + i) % 3 + 0.5) / 3
    f.set_parameter("q_0", c.q_nominal); f.set_parameter("q_des", c.q_nominal_traj)
    f.set_parameter("v_des", v_des); f.set_parameter("contact_traj", contact); f.set_parameter("phi_traj", phase)
    p = np.asarray(f.get_parameter_vec()).reshape(-1, 1)
    dx = np.asarray(qp.solve_step(x0, p, None)).reshape(N, NS)     # stage-major: [dq(24) v(24) F(12)] per stage
    q_traj = np.stack([np.asarray(c.model.get_integrated_states(c.q_nominal, dx[k, :NV])).reshape(-1) for k in range(N)])
    Q.append(q_traj); P.append(p.reshape(-1))
    print(f"inst {i:2d} vx={vx:.2f} wz={wz:+.1f} |dx|={np.linalg.norm(dx):.3f}  base z {q_traj[:,2].min():.3f}..{q_traj[:,2].max():.3f}")
np.save("batch12_q.npy", np.stack(Q)); np.save("batch12_p.npy", np.stack(P)); np.save("batch12_dt.npy", P[0][f.parameters['dt'].idx])
print("joint order:", c.model.get_joint_labels() if hasattr(c.model, 'get_joint_labels') else None)
print("GEN DONE", np.stack(Q).shape)
