"""Random test states for the profiling runs.

Mix per dataset (shuffled):
  * 50 %  q uniform inside the joint limits, v ~ N(0, 0.5^2) per joint,
  * 25 %  as above but ~15 % of the joints pushed to within 0.01 rad of a
          limit with velocity pointing outward (joint-limit barriers active),
  * 25 %  the lowest-clearance states (self-collision / obstacle / floor rows
          of h_2) out of a 10x larger random pool (collision barriers active).
Desired end-effector poses are forward kinematics of independently sampled
random q inside the limits, with zero desired velocity.
"""

import numpy as np

from common import push_to_joint_limits, sample_q, sample_v


def build_dataset(problem, n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    nv = problem.nv
    n_rand = n // 2
    n_lim = n // 4
    n_col = n - n_rand - n_lim

    q_r = sample_q(problem.robot, rng, n_rand)
    v_r = sample_v(rng, n_rand, nv)

    q_l, v_l = push_to_joint_limits(problem.robot, rng, sample_q(problem.robot, rng, n_lim), sample_v(rng, n_lim, nv))

    pool = sample_q(problem.robot, rng, 10 * n_col)
    clearance = collision_clearance(problem, pool)
    idx = np.argsort(clearance)[:n_col]
    q_c = pool[idx]
    v_c = sample_v(rng, n_col, nv)

    q = np.concatenate([q_r, q_l, q_c])
    v = np.concatenate([v_r, v_l, v_c])
    kind = np.concatenate([np.zeros(n_rand, int), np.ones(n_lim, int), 2 * np.ones(n_col, int)])
    perm = rng.permutation(n)
    q, v, kind = q[perm], v[perm], kind[perm]

    q_des = sample_q(problem.robot, rng, n)
    z_des = problem.desired_from_q(q_des)
    z = np.concatenate([q, v], axis=1)
    return {"z": z, "z_des": z_des, "kind": kind, "v_rms": float(np.sqrt(np.mean(v**2)))}


def collision_clearance(problem, q_batch):
    """min over the non-joint-limit rows of h_2 (collision-type barriers)."""
    import jax
    import jax.numpy as jnp

    n_jl = 2 * problem.nv
    h2 = jax.vmap(lambda q: problem.cfg.h_2(jnp.concatenate([q, jnp.zeros(problem.nv)])))(jnp.asarray(q_batch))
    h2 = np.asarray(h2)[:, n_jl:]
    if problem.name == "panda":
        h2 = h2[:, 1:]  # drop the singularity row
    return h2.min(axis=1)
