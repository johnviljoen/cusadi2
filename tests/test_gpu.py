"""GPU end-to-end: write the kernel and plan/reference files, build
selfcheck.cu against cusadi2.h with nvcc, run it; then the same for each
function's unrolled (original-cusadi-style) kernel. Skips without nvcc."""
import os
import shutil
import subprocess

import numpy as np
import pytest

import cusadi2
from cusadi2 import write_kernel_source, write_plan, write_reference, write_unrolled_source
from cusadi2.dag import _demo_function
from tests.test_plan import dyn_function, erfinv_function, late_ops_function, wide_function

HERE = os.path.dirname(os.path.abspath(__file__))
NVCC = os.environ.get("NVCC", shutil.which("nvcc"))

CASES = [  # (function, plan kwargs)
    (_demo_function, {}),
    (wide_function, {}),
    (late_ops_function, {}),
    (erfinv_function, {}),
    (dyn_function, dict(n_stages=4, in_mode=["coupled", "shared"], in_offset=[0, 0], in_span=[1, 0])),
]


def composed_dyn():
    """dyn(u, v) over 4 stages behind one whole-horizon interface: declared input x =
    u knots (5 x 3) knot-major, read at knot s + 2 (stage 3 falls off the end and reads
    zeros); v from per-stage constants baked into the plan; output 0 (3 per stage)
    stage-major except stage 2's second entry, which is discarded; output 1 (the scalar
    sum1(u)) summed over the stages, through the reduce."""
    from cusadi2.compose import DeclaredArg, compose, gather_block
    f = dyn_function()
    N = 4
    x = DeclaredArg.dense("x", 5 * 3)
    consts = np.linspace(-1, 1, N * 2)
    gathers = [(0, gather_block(N, 3, 0, 3, offset=2, count=5)), (-1, gather_block(N, 2, 0, 2))]
    rc0 = np.stack([np.repeat(np.arange(N), 3) * 3 + np.tile(np.arange(3), N), np.zeros(N * 3, int)], axis=1).reshape(N, 3, 2)
    rc0[2, 1] = -1
    rc1 = np.zeros((N, 1, 2), int)
    comp = compose(f, N, [x], gathers, [("y", N * 3, 1), ("total", 1, 1)], [(0, rc0), (1, rc1)], consts=consts)
    return f, comp


def _arch():
    cc = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    return "sm_" + cc.decode().split()[0].replace(".", "")


@pytest.mark.skipif(NVCC is None, reason="nvcc not found")
@pytest.mark.parametrize("dtype", ["double", "float"])
def test_kernel_matches_casadi(tmp_path, dtype):
    write_kernel_source(str(tmp_path / "kernel.cu"), dtype=dtype)
    args = []
    for make, kw in CASES:
        f = make()
        args += [write_plan(f, str(tmp_path / f"{f.name()}.plan"), **kw),
                 write_reference(f, str(tmp_path / f"{f.name()}.ref"),
                                 **{k: v for k, v in kw.items() if k != "out_mode"})]
    f, comp = composed_dyn()
    args += [write_plan(f, str(tmp_path / "composed.plan"), composition=comp),
             write_reference(f, str(tmp_path / "composed.ref"), composition=comp)]
    exe = str(tmp_path / "selfcheck")
    subprocess.run([NVCC, "-std=c++17", f"-arch={_arch()}", f"-DCUSADI2_SCALAR={dtype}",
                    f"-I{cusadi2.INCLUDE_DIR}", os.path.join(HERE, "selfcheck.cu"),
                    str(tmp_path / "kernel.cu"), "-o", exe], check=True)
    res = subprocess.run([exe] + args, capture_output=True, text=True)
    print(res.stdout)
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.skipif(NVCC is None, reason="nvcc not found")
@pytest.mark.parametrize("dtype", ["double", "float"])
def test_unrolled_kernels_match_casadi(tmp_path, dtype):
    for make, kw in CASES:
        f = make()
        specs = {k: v for k, v in kw.items() if k != "out_mode"}
        src = write_unrolled_source(f, str(tmp_path / f"{f.name()}_unrolled.cu"), dtype=dtype, **specs)
        ref = write_reference(f, str(tmp_path / f"{f.name()}.ref"), **specs)
        exe = str(tmp_path / f"check_{f.name()}")
        subprocess.run([NVCC, "-std=c++17", f"-arch={_arch()}", f"-DCUSADI2_SCALAR={dtype}",
                        f"-DFN={f.name()}", f"-I{cusadi2.INCLUDE_DIR}",
                        os.path.join(HERE, "unrolled_check.cu"), src, "-o", exe], check=True)
        res = subprocess.run([exe, ref], capture_output=True, text=True)
        print(res.stdout)
        assert res.returncode == 0, res.stdout + res.stderr
