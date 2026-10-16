"""Recreate vendor/ (gitignored): everything the MPC example needs from outside this repo.

  vendor/cusadi_wip/         se-hwan/cusadi, branch wip ("CusADi 2.0"): the MIT humanoid MPC formulation, with the
                             left-leg length sign corrected (one-line patch, see below)
  vendor/arm_motion/         hojae-io/LearningHumanoidArmMotion-RAL2025-Code (sparse): humanoid_full_sf.urdf + meshes
  vendor/wip_pkg/cusadi      -> cusadi_wip/cusadi
  vendor/stubs/              import stubs for torch / viser / yourdfpy / trimesh (only the symbolic side is used)
  vendor/mit_humanoid_urdf/  humanoid_full_sf_toeheel.urdf (+ toe and heel fixed frames at the foot cylinder's
                             ends), humanoid_full_sf_toeheel_mj.urdf (same, with a MuJoCo compiler block)
  vendor/humanoid_mpc_fns/   the wip's whole-horizon reference functions, p_eval.npy, dx_custom.npy

usage: python vendor_setup.py [--no-ldl]      (then CUSADI_WIP defaults to vendor/ in the other scripts)
--no-ldl skips the symbolic LDL reference (1.28M instructions; build_mpc_stage.py needs it).
"""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(HERE, "vendor")
WIP = ("https://github.com/se-hwan/cusadi.git", "wip", "4003888")
ARM = ("https://github.com/hojae-io/LearningHumanoidArmMotion-RAL2025-Code.git", "main", "d176a14")
TOE, HEEL = "0.105 0 -0.04", "-0.045 0 -0.04"          # ends of the foot collision cylinder (centre 0.03 0 -0.03, length 0.15, r 0.01)
run = lambda *a, **k: subprocess.run(a, check=True, **k)
os.makedirs(V, exist_ok=True)

def clone(url, branch, commit, name, sparse=None):
    d = os.path.join(V, name)
    if not os.path.isdir(d):
        run("git", "clone", "--depth", "1", "--branch", branch, *(["--filter=blob:none", "--sparse"] if sparse else []), url, d)
        if sparse: run("git", "-C", d, "sparse-checkout", "set", sparse)
    head = subprocess.check_output(["git", "-C", d, "rev-parse", "--short=7", "HEAD"], text=True).strip()
    if head != commit: print(f"WARNING: {name} is at {head}, this example was built against {commit}")
    return d
wip = clone(*WIP, "cusadi_wip"); arm = clone(*ARM, "arm_motion", sparse="resources/mit_humanoid")

def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.lexists(dst): os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
link(os.path.join(wip, "cusadi"), os.path.join(V, "wip_pkg", "cusadi"))

# The wip's left-leg length row adds the hip position instead of subtracting it ("- - p_left_hip_yaw"), which
# bounds the distance from the world origin rather than the leg length. Corrected here and in build_mpc_stage.py.
ctl = os.path.join(wip, "cusadi", "controllers", "mit_humanoid_mpc.py"); src = open(ctl).read()
BUG, FIX = "- - fwd_kin_hip['p_left_hip_yaw']", "- fwd_kin_hip['p_left_hip_yaw']"
if BUG in src: open(ctl, "w").write(src.replace(BUG, FIX))
src = open(ctl).read(); assert BUG not in src and FIX in src, "left-leg row not found in the wip controller"

STUB = '''import sys
from unittest.mock import MagicMock
class Tensor: pass
_m = MagicMock(); _m.Tensor = Tensor
sys.modules[__name__] = _m
for _s in {subs!r}: sys.modules[__name__ + "." + _s] = MagicMock()
'''
for mod, subs in [("torch", ["utils", "utils.cpp_extension"]), ("viser", ["extras"]), ("yourdfpy", []), ("trimesh", [])]:
    os.makedirs(os.path.join(V, "stubs"), exist_ok=True)
    with open(os.path.join(V, "stubs", mod + ".py"), "w") as f: f.write(STUB.format(subs=subs))

# ---- URDF with toe/heel frames
src_dir = os.path.join(arm, "resources", "mit_humanoid", "urdf"); out_dir = os.path.join(V, "mit_humanoid_urdf")
link(os.path.join(src_dir, "meshes_v3"), os.path.join(out_dir, "meshes_v3"))
urdf = open(os.path.join(src_dir, "humanoid_full_sf.urdf")).read()
frames = "".join(f'''
  <link name="{side}_{part}"/>
  <joint name="{side}_{part}_fixed" type="fixed">
    <origin xyz="{xyz}" rpy="0 0 0"/>
    <parent link="{side}_foot"/>
    <child link="{side}_{part}"/>
  </joint>
''' for side in ("right", "left") for part, xyz in (("toe", TOE), ("heel", HEEL)))
assert urdf.rstrip().endswith("</robot>")
urdf = urdf.rstrip()[:-len("</robot>")] + frames + "\n</robot>\n"
open(os.path.join(out_dir, "humanoid_full_sf_toeheel.urdf"), "w").write(urdf)
mj = re.sub(r'(<robot name="humanoid">)', r'\1\n  <mujoco><compiler discardvisual="false" balanceinertia="true"/></mujoco>', urdf, count=1)
open(os.path.join(out_dir, "humanoid_full_sf_toeheel_mj.urdf"), "w").write(mj)

# ---- reference functions from the wip controller (x = 0, default parameters)
import numpy as np
sys.path[:0] = [os.path.join(V, "wip_pkg"), os.path.join(V, "stubs")]
from cusadi.controllers import MITHumanoidModelPredictiveController
ref = os.path.join(V, "humanoid_mpc_fns"); os.makedirs(ref, exist_ok=True)
c = MITHumanoidModelPredictiveController(os.path.join(out_dir, "humanoid_full_sf_toeheel.urdf")); c.build()
c.setup_solver(qp_solver='osqp', qp_cfg={'max_iter': 25}, sqp_cfg={'max_iter': 1})
form, qp = c.formulation, c.solver.qp_backend; name = form.name
p_eval = np.asarray(form.get_parameter_vec()).reshape(-1, 1); np.save(os.path.join(ref, "p_eval.npy"), p_eval)
x0 = np.zeros((form.n_x, 1))
np.save(os.path.join(ref, "dx_osqp.npy"), np.asarray(qp.solve_step(x0, p_eval, None)).reshape(-1))
if "--no-ldl" in sys.argv:
    qp._build_parallel_functions('cudss')
else:
    dx = np.asarray(qp.solve_step(x0, p_eval, 'custom')).reshape(-1); np.save(os.path.join(ref, "dx_custom.npy"), dx)
    print("custom ADMM vs OSQP, max abs:", np.abs(dx - np.load(os.path.join(ref, "dx_osqp.npy"))).max())
    qp.fn_LDL_fac.save(os.path.join(ref, f"ldl_factorize_{name}.casadi")); qp.fn_LDL_solve.save(os.path.join(ref, f"ldl_solve_{name}.casadi"))
for tag, f in [("KKT_matrix", qp.fn_KKT_matrix), ("ruiz_scaling", qp.fn_ruiz_scaling), ("KKT_vector", qp.fn_KKT_vector), ("ADMM_step", qp.fn_ADMM_step)]:
    f.save(os.path.join(ref, f"{tag}_{name}.casadi"))
print("VENDOR DONE", V)
