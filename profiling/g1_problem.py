"""Problem 2: Unitree G1, fixed root, 29 DOF. frax ships no G1 CBF demo.

`G1OSCBFTorqueConfig` is a copy of frax's examples/cbf_utils.py
`OSCBFTorqueConfig` adapted from `Manipulator` to `Humanoid`:
  * the task is both hands (12 rows: J = [J_left_hand; J_right_hand]),
  * barriers (all relative degree 2 on torque): joint limits, plus every one
    of frax's G1 self-collision pairs through frax's own
    `_self_collision_distances_from_link_data` on `link_collision_data`,
  * alpha / alpha_2 = 10 h, as in the Panda demo,
  * same l1-relaxed (elastic) QP, penalties and solver_tol=1e-5 as cbfpy / the demo.

The nominal controller is dual-hand OSC with a posture task in the nullspace,
the direct analogue of the demo's single-arm controller, built from the same
per-step terms as frax's tests/timing `osc_terms`.
"""

import jax
import jax.numpy as jnp
import numpy as np
from cbfpy import CBF, CBFConfig

from frax.core.humanoid import Humanoid
from frax.robots.unitree_g1 import load_fixed_root_g1
from frax.utils.rotation_utils import orientation_error_3D


class G1OSCBFTorqueConfig(CBFConfig):
    """cbf_utils.OSCBFTorqueConfig adapted to a fixed-root Humanoid (dual-hand task).

    State: z = [q, v] (2 * nv). Control: u = joint torques (nv).
    """

    def __init__(
        self,
        robot: Humanoid,
        pos_obj_weight: float = 1.0,
        rot_obj_weight: float = 1.0,
        joint_obj_weight: float = 1.0,
        compensate_centrifugal_coriolis: bool = True,
        backend: str = "elastiqp",
    ):
        assert isinstance(robot, Humanoid)
        assert robot.nv_floating == 0, "fixed-root model expected"
        self.robot = robot
        self.num_joints = robot.nv
        self.task_dim = 12  # two hand poses
        self.is_redundant = robot.nv > self.task_dim
        self.compensate_centrifugal_coriolis = compensate_centrifugal_coriolis
        self.pos_obj_weight = float(pos_obj_weight)
        self.rot_obj_weight = float(rot_obj_weight)
        self.joint_space_obj_weight = float(joint_obj_weight)
        self.W_T_W_task_diag = tuple(
            np.array(([self.pos_obj_weight] * 3 + [self.rot_obj_weight] * 3) * 2) ** 2
        )
        self.W_T_W_joint_diag = tuple(
            np.array([self.joint_space_obj_weight] * self.num_joints) ** 2
        )
        self.q_min = robot.actuated_joint_lower_limits
        self.q_max = robot.actuated_joint_upper_limits
        super().__init__(
            n=self.num_joints * 2,
            m=self.num_joints,
            u_min=-np.asarray(robot.actuated_joint_max_forces),
            u_max=np.asarray(robot.actuated_joint_max_forces),
            backend=backend,
            solver_tol=1e-5,
        )

    # --- dynamics (identical to OSCBFTorqueConfig) ---
    def f(self, z, *args, **kwargs):
        q = z[: self.num_joints]
        v = z[self.num_joints : self.num_joints * 2]
        M = self.robot.mass_matrix(q)
        M_inv = jnp.linalg.inv(M)
        bias = self.robot.gravity_vector(q)
        if self.compensate_centrifugal_coriolis:
            bias += self.robot.centrifugal_coriolis_vector(q, v)
        return jnp.concatenate([v, -M_inv @ bias])

    def g(self, z, *args, **kwargs):
        q = z[: self.num_joints]
        M = self.robot.mass_matrix(q)
        M_inv = jnp.linalg.inv(M)
        return jnp.block([[jnp.zeros((self.num_joints, self.m))], [M_inv]])

    def _task_jacobian(self, transforms):
        return jnp.concatenate(
            [
                self.robot._left_hand_jacobian(transforms),
                self.robot._right_hand_jacobian(transforms),
            ],
            axis=0,
        )

    def _P(self, z, *args, **kwargs):
        q = z[: self.num_joints]
        transforms = self.robot.joint_to_world_transforms(q)
        M = self.robot._mass_matrix(transforms)
        M_inv = jnp.linalg.inv(M)
        J = self._task_jacobian(transforms)
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia
        NT = jnp.eye(self.num_joints) - J.T @ J_bar.T
        W_T_W_joint = jnp.diag(jnp.asarray(self.W_T_W_joint_diag))
        W_T_W_task = jnp.diag(jnp.asarray(self.W_T_W_task_diag))
        return (
            NT.T @ M_inv.T @ W_T_W_joint @ M_inv @ NT
            + M_inv.T @ J.T @ W_T_W_task @ J @ M_inv
        )

    def P(self, z, u_des, *args, **kwargs):
        return self._P(z)

    def q(self, z, u_des, *args, **kwargs):
        return -u_des.T @ self._P(z)

    def _is_symmetric_psd(self, mat):
        """cbfpy's construction-time check (symmetry atol 1e-14, Cholesky of P + 1e-14 I)
        is written for float64. In the fp32 runs P(z=1) for the G1 (entries up to ~1e3,
        from M^-1 squared) fails it by rounding alone, so for float32 inputs the same
        test is applied with tolerances scaled to float32 eps. fp64 keeps cbfpy's check."""
        mat = np.asarray(mat)
        if mat.dtype != np.float32:
            return super()._is_symmetric_psd(mat)
        if mat.shape[0] != mat.shape[1]:
            return False
        scale = float(np.max(np.abs(mat)))
        tol = 1e-4 * scale  # TF32 matmuls on the GPU leave ~2e-5 relative asymmetry
        if not np.allclose(mat, mat.T, atol=tol):
            return False
        try:
            np.linalg.cholesky(0.5 * (mat + mat.T).astype(np.float64) + np.eye(mat.shape[0]) * tol)
        except np.linalg.LinAlgError:
            return False
        return True

    # --- barriers: joint limits + all frax G1 self-collision pairs ---
    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        h_joint_limits = jnp.concatenate(
            [jnp.asarray(self.q_max) - q, q - jnp.asarray(self.q_min)]
        )
        positions, radii = self.robot.link_collision_data(q)
        h_self_collision = self.robot._self_collision_distances_from_link_data(
            positions, radii
        )
        return jnp.concatenate([h_joint_limits, h_self_collision])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


class G1Problem:
    name = "g1"

    def __init__(self, backend: str):
        assert backend in ("elastiqp", "qpax")
        robot = load_fixed_root_g1()
        self.robot = robot
        self.nv = robot.nv
        self.m = robot.nv
        cfg = G1OSCBFTorqueConfig(robot, backend=backend)
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
            "self_collision": self.n_sc_pairs,
        }
        self.z_des_dim = 36  # two × [pos(3), rot(9), vel(3), omega(3)]

        # gains: same numbers as the Panda demo, per hand
        kp_pos = 50.0 * np.ones(3)
        kp_rot = 20.0 * np.ones(3)
        kd_pos = 20.0 * np.ones(3)
        kd_rot = 10.0 * np.ones(3)
        self.kp_task = np.tile(np.concatenate([kp_pos, kp_rot]), 2)
        self.kd_task = np.tile(np.concatenate([kd_pos, kd_rot]), 2)
        self.kp_joint = 10.0 * np.ones(robot.nv)
        self.kd_joint = 5.0 * np.ones(robot.nv)
        self.des_q = robot.default_configuration
        self.des_v = np.zeros(robot.nv)

    # ------------------------------------------------------------------ #
    # 1. nominal: dual-hand OSC + nullspace posture (frax `osc_terms` path)
    # ------------------------------------------------------------------ #
    def _hand_terms(self, q, v):
        robot = self.robot
        tfs = robot.joint_to_world_transforms(q)
        M = robot._mass_matrix(tfs)
        M_inv = robot.mass_matrix_inverse(M)
        g = robot._gravity_vector(tfs)
        c = robot._centrifugal_coriolis_vector(v, tfs)
        J = jnp.concatenate([robot._left_hand_jacobian(tfs), robot._right_hand_jacobian(tfs)], 0)
        T_l = robot._left_hand_transform(tfs)
        T_r = robot._right_hand_transform(tfs)
        return M, M_inv, g, c, J, T_l, T_r

    def nominal(self, z, z_des):
        robot = self.robot
        q = z[: robot.nq]
        v = z[robot.nq :]
        M, M_inv, g, c, J, T_l, T_r = self._hand_terms(q, v)

        twist = J @ v  # (12,)
        p_err = []
        d_err = []
        for k, T in enumerate((T_l, T_r)):
            d = z_des[18 * k : 18 * (k + 1)]
            des_pos, des_rot = d[:3], jnp.reshape(d[3:12], (3, 3))
            des_vel, des_omega = d[12:15], d[15:18]
            tw = twist[6 * k : 6 * (k + 1)]
            p_err += [T[:3, 3] - des_pos, orientation_error_3D(T[:3, :3], des_rot)]
            d_err += [tw[:3] - des_vel, tw[3:] - des_omega]
        task_p_error = jnp.concatenate(p_err)
        task_d_error = jnp.concatenate(d_err)

        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia
        task_accel = -self.kp_task * task_p_error - self.kd_task * task_d_error
        tau = J.T @ (task_inertia @ task_accel) + g + c

        NT = jnp.eye(robot.nv) - J.T @ J_bar.T
        joint_accel = -self.kp_joint * (q - self.des_q) - self.kd_joint * (v - self.des_v)
        tau += NT @ (M @ joint_accel)
        return tau

    # 2a. dynamics terms as the config evaluates them
    def dyn_terms(self, z):
        robot = self.robot
        q = z[: self.nv]
        v = z[self.nv :]
        M = robot.mass_matrix(q)
        M_inv = jnp.linalg.inv(M)
        bias = robot.gravity_vector(q) + robot.centrifugal_coriolis_vector(q, v)
        J = jnp.concatenate([robot.left_hand_jacobian(q), robot.right_hand_jacobian(q)], 0)
        return M, M_inv, bias, J

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
        robot = self.robot

        def fk(q):
            return robot.left_hand_transform(q), robot.right_hand_transform(q)

        T_l, T_r = jax.vmap(fk)(jnp.asarray(q_des_batch))
        out = []
        for T in (np.asarray(T_l), np.asarray(T_r)):
            out += [T[:, :3, 3], T[:, :3, :3].reshape(len(T), 9), np.zeros((len(T), 6))]
        return np.concatenate(out, axis=1)

    def self_collision_clearance(self, q_batch):
        return np.asarray(
            jnp.min(jax.vmap(self.robot.self_collision_distances)(jnp.asarray(q_batch)), axis=1)
        )
