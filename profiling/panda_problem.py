"""Problem 1: Franka Panda OSCBF exactly as frax's examples/oscbf_demo.py.

`DemoOSCBFConfig` is imported from the frax examples unmodified (one sphere
obstacle at [0.5, 0.5, 0.5] r=0.3, joint limits, singularity avoidance, floor,
self-collision). The OSC controller is a verbatim copy of the body of
`operational_space_control` in that demo, minus the trailing safety_filter.
`cbf_utils.OSCBFTorqueConfig` hard-codes backend="elastiqp"; the backend is
switched on the config instance before `CBF.from_config` (no source change).
"""

import jax
import jax.numpy as jnp
import numpy as np

from cbfpy import CBF
from frax import load_panda
from frax.utils.rotation_utils import orientation_error_3D

# frax examples (on sys.path via common.ensure_frax). oscbf_demo sets
# JAX_PLATFORMS=cpu in os.environ at import; jax is already imported by then,
# so the env write has no effect on the backend.
assert "jax" in __import__("sys").modules
from oscbf_demo import DemoOSCBFConfig  # noqa: E402


class PandaProblem:
    name = "panda"

    def __init__(self, backend: str):
        assert backend in ("elastiqp", "qpax")
        robot = load_panda()
        self.robot = robot
        self.nv = robot.nv
        self.m = robot.nv
        collision_pos = np.array([[0.5, 0.5, 0.5]])
        collision_radii = np.array([0.3])
        cfg = DemoOSCBFConfig(robot, collision_pos, collision_radii)
        cfg.backend = backend  # instance attribute only; cbf_utils.py untouched
        self.cfg = cfg
        self.cbf = CBF.from_config(cfg)
        self.backend = backend
        self.penalty = np.asarray(cfg.constraint_relaxation_penalties)
        self.solver_tol = cfg.solver_tol
        self.num_cbf = cfg.num_cbf
        self.n_rows = cfg.num_cbf + 2 * cfg.m
        self.n_sc_pairs = int(robot.body_sc_pairs.shape[0]) + (
            int(robot.root_sc_pairs.shape[0]) if robot.has_root_collision_data else 0
        )
        self.barrier_groups = {
            "joint_limits": 2 * robot.nv,
            "singularity": 1,
            "obstacle": int(robot.flat_collision_radii.shape[0]) * len(collision_radii),
            "self_collision": self.n_sc_pairs,
            "floor": int(robot.flat_collision_radii.shape[0]),
        }
        self.z_des_dim = 18

        # ---- gains / targets copied from oscbf_demo.main ----
        kp_pos = 50.0 * np.ones(3)
        kp_rot = 20.0 * np.ones(3)
        kd_pos = 20.0 * np.ones(3)
        kd_rot = 10.0 * np.ones(3)
        self.kp_task = np.concatenate([kp_pos, kp_rot])
        self.kd_task = np.concatenate([kd_pos, kd_rot])
        self.kp_joint = 10.0 * np.ones(robot.nv)
        self.kd_joint = 5.0 * np.ones(robot.nv)
        self.is_redundant = True
        self.des_q = robot.default_configuration  # == env.q_init (panda_default_q)
        self.des_v = np.zeros(robot.nv)
        self.des_accel = np.zeros(3)
        self.des_alpha = np.zeros(3)

    # ------------------------------------------------------------------ #
    # 1. nominal controller: verbatim demo body, `torque_control_matrices` path
    # ------------------------------------------------------------------ #
    def nominal(self, z, z_ee_des):
        robot = self.robot
        q = z[: robot.nq]
        v = z[robot.nq :]
        des_pos = z_ee_des[:3]
        des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
        des_vel = z_ee_des[12:15]
        des_omega = z_ee_des[15:18]

        M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, v)
        pos = ee_tmat[:3, 3]
        rot = ee_tmat[:3, :3]

        twist = J @ v
        vel = twist[:3]
        omega = twist[3:]

        pos_error = pos - des_pos
        vel_error = vel - des_vel
        rot_error = orientation_error_3D(rot, des_rot)
        omega_error = omega - des_omega
        task_p_error = jnp.concatenate([pos_error, rot_error])
        task_d_error = jnp.concatenate([vel_error, omega_error])

        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia

        task_accel = (
            jnp.concatenate([self.des_accel, self.des_alpha])
            - self.kp_task * task_p_error
            - self.kd_task * task_d_error
        )
        task_wrench = task_inertia @ task_accel
        tau = J.T @ task_wrench
        tau += g + c

        if self.is_redundant:
            NT = jnp.eye(robot.nv) - J.T @ J_bar.T
            q_error = q - self.des_q
            v_error = v - self.des_v
            joint_accel = -self.kp_joint * q_error - self.kd_joint * v_error
            secondary_joint_torques = M @ joint_accel
            tau += NT @ secondary_joint_torques
        return tau

    # ------------------------------------------------------------------ #
    # 2a. dynamics terms as the config evaluates them (M, M^-1, bias, J)
    # ------------------------------------------------------------------ #
    def dyn_terms(self, z):
        robot = self.robot
        q = z[: self.nv]
        v = z[self.nv :]
        M = robot.mass_matrix(q)
        M_inv = jnp.linalg.inv(M)
        bias = robot.gravity_vector(q) + robot.centrifugal_coriolis_vector(q, v)
        J = robot.ee_jacobian(q)
        return M, M_inv, bias, J

    # 2b. barrier values with Lie derivatives (what G_qp / h_qp consume)
    def barriers(self, z):
        h, Lfh = self.cbf.h_and_Lfh(z)
        Lgh = self.cbf.Lgh(z)
        return h, Lfh, Lgh

    def h2_only(self, z):
        return self.cfg.h_2(z)

    def qp_data(self, z, u_des):
        return self.cbf.qp_data(z, u_des)

    def safety_filter(self, z, u_des):
        return self.cbf.safety_filter(z, u_des)

    def end_to_end(self, z, z_des):
        return self.cbf.safety_filter(z, self.nominal(z, z_des))

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def desired_from_q(self, q_des_batch):
        """z_ee_des = [pos, rot(9, row-major), vel(3), omega(3)] from FK of q."""
        T = jax.vmap(self.robot.ee_transform)(jnp.asarray(q_des_batch))
        T = np.asarray(T)
        pos = T[:, :3, 3]
        rot = T[:, :3, :3].reshape(len(T), 9)
        return np.concatenate([pos, rot, np.zeros((len(T), 6))], axis=1)

    def obstacle_clearance(self, q_batch):
        """min over spheres of (dist to obstacle centre - radii) per state."""
        robot = self.robot

        def f(q):
            p, r = robot.link_collision_data(q)
            d = jnp.linalg.norm(p - jnp.asarray(self.cfg.collision_positions[0]), axis=1)
            return jnp.min(d - r - self.cfg.collision_radii[0])

        return np.asarray(jax.vmap(f)(jnp.asarray(q_batch)))
