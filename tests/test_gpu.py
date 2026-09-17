"""GPU end-to-end: write the kernel and plan/reference files, build
selfcheck.cu against cusadi2.h with nvcc, run it; then the same for each
function's unrolled (original-cusadi-style) kernel. Skips without nvcc."""
import os
import shutil
import subprocess

import pytest

import cusadi2
from cusadi2 import write_kernel_source, write_plan, write_reference, write_unrolled_source
from cusadi2.dag import _demo_function
from test_plan import dyn_function, erfinv_function, late_ops_function, wide_function

HERE = os.path.dirname(os.path.abspath(__file__))
NVCC = os.environ.get("NVCC", shutil.which("nvcc"))

CASES = [  # (function, plan kwargs)
    (_demo_function, {}),
    (wide_function, {}),
    (late_ops_function, {}),
    (erfinv_function, {}),
    (dyn_function, dict(n_stages=4, in_mode=["coupled", "shared"], in_offset=[0, 0], in_span=[1, 0])),
]


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
