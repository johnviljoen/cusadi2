"""References for the 12-instance receding-horizon rollout (mpc_rollout.cu): per control step, instance and
stage, the stage parameters p_stage = [q_des, v_des, F_des, phi, contact] sampled on the MPC's geometric grid.

Everything here is a function of time only (commands, gait clock, heading), so it is computed once on the
host and the closed loop itself stays on the GPU. Control step s starts at t = s dt_0; its stage k describes
the state at t + t_k, t_k = dt_0 + ... + dt_k.

  command    vx = 0.6 (i % 4) / 3 m/s forward, wz = (-0.5, 0, 0.5)[i // 4] rad/s, both ramped from 0 over RAMP
             seconds once the instance's gait starts
  gait       walk: T_CYCLE cycle, each foot (toe and heel together) swings T_SWING, double support between;
             odd instances lead with the left foot; instances start stepping at 0.2 .. 0.3 s; the
             zero-command instance stands
  q_des      nominal joints and height, heading = integral of the ramped wz (absolute, from the clock) and the
             matching unicycle xy (xy carries no weight; it only keeps the SE(3) error local)

usage: python gen_batch12.py <build_dir> [seconds]     -> <build_dir>/rollout/{p_stage.f64, meta.json}
"""
import json, os, sys, numpy as np
OUT = sys.argv[1]; T_ROLL = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
meta = json.load(open(os.path.join(OUT, "meta.json")))
dt = np.array(meta["dt"]); N = meta["N"]; q_nom = np.array(meta["q_nominal"]); NQ = q_nom.size
sizes = dict(meta["p_stage"]); NV, NF = sizes["v_des"], sizes["F_des"]
B, T_CYCLE, T_SWING, RAMP = 12, 0.7, 0.28, 0.5
RIGHT, LEFT = [0, 2], [1, 3]                                   # contact order: right toe, left toe, right heel, left heel
n_steps = int(np.ceil(T_ROLL / dt[0])); t_stage = np.cumsum(dt)

fine = np.arange(0.0, n_steps * dt[0] + t_stage[-1] + 1e-3, 1e-3)
P = np.zeros((n_steps, B, N, NQ + NV + NF + 8)); cmds = []
for i in range(B):
    vx, wz = 0.6 * (i % 4) / 3.0, [-0.5, 0.0, 0.5][i // 4]
    start, lead = 0.2 + 0.05 * (i // 4), (0.5 * T_CYCLE if i % 2 else 0.0)
    moving = vx != 0 or wz != 0
    scale = np.clip((fine - start) / RAMP, 0, 1)
    yaw = np.cumsum(wz * scale) * 1e-3
    px, py = np.cumsum(vx * scale * np.cos(yaw)) * 1e-3, np.cumsum(vx * scale * np.sin(yaw)) * 1e-3
    for s in range(n_steps):
        for k in range(N):
            t = s * dt[0] + t_stage[k]
            sc, psi = np.interp(t, fine, scale), np.interp(t, fine, yaw)
            q_des = q_nom.copy(); q_des[:2] = np.interp(t, fine, px), np.interp(t, fine, py)
            q_des[3:7] = 0, 0, np.sin(psi / 2), np.cos(psi / 2)
            v_des = np.zeros(NV); v_des[0], v_des[5] = sc * vx, sc * wz
            phi, contact = np.zeros(4), np.ones(4)
            if moving and t >= start:
                for foot, off in ((RIGHT, 0.0), (LEFT, 0.5 * T_CYCLE)):
                    tau = (t - start + lead - off) % T_CYCLE
                    if tau < T_SWING: contact[foot] = 0.0; phi[foot] = tau / T_SWING
            P[s, i, k] = np.concatenate([q_des, v_des, np.zeros(NF), phi, contact])
    cmds.append(dict(vx=vx, wz=wz, start=start, lead_left=bool(i % 2)))
    print(f"inst {i:2d} vx={vx:.2f} wz={wz:+.1f} start={start:.2f} {'stand' if not moving else 'left first' if i % 2 else 'right first'}")
d = os.path.join(OUT, "rollout"); os.makedirs(d, exist_ok=True)
P.tofile(os.path.join(d, "p_stage.f64"))
json.dump(dict(B=B, n_steps=n_steps, dt0=float(dt[0]), T_CYCLE=T_CYCLE, T_SWING=T_SWING, RAMP=RAMP, commands=cmds),
          open(os.path.join(d, "meta.json"), "w"), indent=1)
print("GEN DONE", P.shape, f"{n_steps} steps = {n_steps * dt[0]:.3f} s")
