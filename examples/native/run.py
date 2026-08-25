#!/usr/bin/env python
"""Build and run the cusadi2 end-to-end GPU self-check.

Emits the function-independent kernel and two plan headers — one plain batch-LP
(`demo`), one staged with coupled+shared inputs (`dyn`) — into ./build, compiles
them against the runtime header, and runs the embedded reference check on the GPU.

    python examples/run.py

Env overrides: CUSADI2_GPU (device index, default "1"), NVCC (path to nvcc).
"""
import os
import subprocess
import sys

import casadi as ca

import cusadi2
from cusadi2.kernel_template import write_kernel_source
from cusadi2.generate import generate_native_plan, generate_reference
from cusadi2.dag import _demo_function

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = cusadi2.__path__[0]          # installed package dir (holds cusadi2.h)
BUILD = os.path.join(HERE, "build")

DEVICE = os.environ.get("CUSADI2_GPU", "1")
NVCC = os.environ.get("NVCC", "nvcc")

# casadi functions to pass through our system

def create_dyn_func():
    u = ca.SX.sym("u", 3)
    v = ca.SX.sym("v", 2)
    return ca.Function("dyn", [u, v], [ca.tanh(u) + ca.vertcat(v, v[0]), ca.sum1(u)])

def _floating_base_model(description):
    """Load a floating-base Pinocchio CasADi model; return (cmodel, cdata, q, v, tau)."""
    import pinocchio as pin
    import pinocchio.casadi as cpin
    from robot_descriptions.loaders.pinocchio import load_robot_description
    robot = load_robot_description(description, root_joint=pin.JointModelFreeFlyer())
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()
    q = ca.SX.sym("q", cmodel.nq)
    v = ca.SX.sym("v", cmodel.nv)
    tau = ca.SX.sym("tau", cmodel.nv)
    return cmodel, cdata, q, v, tau

def _aba(name, description):
    """Forward dynamics f(q, v, tau) -> ddq via Pinocchio ABA."""
    import pinocchio.casadi as cpin
    cmodel, cdata, q, v, tau = _floating_base_model(description)
    return ca.Function(name, [q, v, tau], [cpin.aba(cmodel, cdata, q, v, tau)])

def _aba_jac(name, description):
    """Analytical ABA derivatives f(q, v, tau) -> (ddq_dq, ddq_dv, ddq_dtau)."""
    import pinocchio.casadi as cpin
    cmodel, cdata, q, v, tau = _floating_base_model(description)
    return ca.Function(name, [q, v, tau], list(cpin.computeABADerivatives(cmodel, cdata, q, v, tau)))

def create_a1_func():       # Unitree A1 quadruped (floating base: nq=19, nv=18)
    return _aba("a1_aba", "a1_description")

def create_h1_func():       # Unitree H1 humanoid (floating base: nq=26, nv=25)
    return _aba("h1_aba", "h1_description")

def create_a1_jac_func():   # A1 analytical ABA derivatives (~41k instrs)
    return _aba_jac("a1_aba_jac", "a1_description")

def create_h1_jac_func():   # H1 analytical ABA derivatives (~78k instrs)
    return _aba_jac("h1_aba_jac", "h1_description")

def gpu_arch() -> str:
    cc = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", DEVICE]
    ).decode().splitlines()[0].strip()
    return "sm_" + cc.replace(".", "")

def main() -> int:
    arch = gpu_arch()
    rc = 0
    for dtype in ("double", "float"):       # f64 and f32 builds — one binary each
        outdir = os.path.join(BUILD, dtype)
        os.makedirs(outdir, exist_ok=True)

        write_kernel_source(os.path.join(outdir, "kernel.cu"), dtype=dtype, tiled=False)

        # plan + matching self-check reference (same fn/specs/prefix/dtype) per problem
        def emit(make_fn, name, prefix=None, **specs):
            fn = make_fn()
            generate_native_plan(fn, os.path.join(outdir, f"{name}_plan.h"),
                                 prefix=prefix, dtype=dtype, **specs)
            specs.pop("out_mode", None)     # reference needs the input specs only
            generate_reference(fn, os.path.join(outdir, f"{name}_ref.h"),
                               prefix=prefix, dtype=dtype, **specs)

        emit(_demo_function, "demo")                       # batch LP
        emit(create_dyn_func, "dyn",                       # staged
             in_mode=["coupled", "shared"], in_offset=[0, 0], in_span=[1, 0],
             out_mode=["stage", "stage"])
        # Pinocchio floating-base forward dynamics — heavier real-robot kernels (batch LP)
        emit(create_a1_func, "a1", prefix="A1")
        emit(create_h1_func, "h1", prefix="H1")
        # ...and their analytical ABA Jacobians (the MPC Gauss-Newton workload; 41k / 78k instrs)
        emit(create_a1_jac_func, "a1_jac", prefix="A1_JAC")
        emit(create_h1_jac_func, "h1_jac", prefix="H1_JAC")

        binpath = os.path.join(outdir, "selfcheck")
        cmd = [NVCC, "-std=c++14", f"-arch={arch}", f"-DCUSADI2_SCALAR={dtype}",
               f"-I{PKG}", f"-I{outdir}",
               os.path.join(HERE, "selfcheck.cu"), os.path.join(outdir, "kernel.cu"),
               "-o", binpath]
        print(f"=== {dtype} ===")
        if dtype == "float":
            print("note: f32 outputs are checked against the DOUBLE-precision CasADi reference,")
            print("      so maxrel reflects single-precision loss, not kernel error — well-")
            print("      conditioned funcs stay ~1e-6, ill-conditioned Jacobians (a1_jac) ~1e-1.")
        print("nvcc:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE)
        rc |= subprocess.run([binpath], env=env).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
