"""Shared helpers for the FRAX / cbfpy CBF safety-filter profiling.

Import order matters: call `configure(platform, x64)` BEFORE anything imports
jax (JAX reads JAX_PLATFORMS / JAX_ENABLE_X64 / XLA_FLAGS once, at import).
`ensure_frax()` puts the pinned, unmodified frax checkout (and its examples
directory) at the front of sys.path, shadowing any pip-installed frax.
"""

import json
import os
import platform as _platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
THIRD_PARTY = HERE / "third_party"
FRAX_DIR = THIRD_PARTY / "frax"
FRAX_REPO = "https://github.com/danielpmorton/frax"
FRAX_COMMIT = "cc0a066b5b539c729838308496ae1b8a1d6aeada"  # 2026-10-07, version 0.1.0

CPU_XLA_FLAGS = (
    "--xla_cpu_multi_thread_eigen=false "
    "--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
)


def configure(platform: str, x64: bool) -> None:
    """Set the JAX environment the way frax's demos / timing scripts do."""
    assert platform in ("cpu", "gpu")
    assert "jax" not in sys.modules, "configure() must run before jax is imported"
    os.environ["JAX_ENABLE_X64"] = "True" if x64 else "False"
    if platform == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
    else:
        # Same flags as frax examples/oscbf_demo.py and tests/timing/timing_utils.py
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["XLA_FLAGS"] = CPU_XLA_FLAGS


def ensure_frax() -> Path:
    """Clone frax at the pinned commit if needed and put it first on sys.path."""
    if not (FRAX_DIR / "frax" / "__init__.py").exists():
        THIRD_PARTY.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", FRAX_REPO, str(FRAX_DIR)], check=True)
        subprocess.run(["git", "-C", str(FRAX_DIR), "checkout", "-q", FRAX_COMMIT], check=True)
    head = subprocess.run(
        ["git", "-C", str(FRAX_DIR), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if head != FRAX_COMMIT:
        print(f"WARNING: frax checkout at {head}, expected {FRAX_COMMIT}", file=sys.stderr)
    for p in (str(FRAX_DIR / "examples"), str(FRAX_DIR)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    import frax  # noqa: F401

    assert Path(frax.__file__).resolve().is_relative_to(FRAX_DIR), frax.__file__
    return FRAX_DIR


# --------------------------------------------------------------------------- #
# System / version info
# --------------------------------------------------------------------------- #
def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"<{e}>"


def sysinfo() -> dict:
    import importlib.metadata as md

    import jax

    def ver(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    cpu = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    gpu = _cmd(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    nvcc_bin = str(Path(sys.executable).parent / "nvcc") if (Path(sys.executable).parent / "nvcc").exists() else "nvcc"
    nvcc_out = _cmd([nvcc_bin, "--version"])
    nvcc = nvcc_out.splitlines()[-1] if nvcc_out and not nvcc_out.startswith("<") else nvcc_out
    cuda_rt = None
    try:
        from jax._src.lib import cuda_versions  # type: ignore

        if cuda_versions is not None:
            cuda_rt = {
                "cuda_runtime": cuda_versions.cuda_runtime_get_version(),
                "cuda_driver": cuda_versions.cuda_runtime_build_version(),
                "cudnn": cuda_versions.cudnn_get_version(),
            }
    except Exception:
        pass
    return {
        "hostname": _platform.node(),
        "os": _platform.platform(),
        "python": sys.version.split()[0],
        "cpu": cpu,
        "physical_cores": int(_cmd(["bash", "-c", "lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l"]) or 0),
        "logical_cpus": os.cpu_count(),
        "gpu": gpu,
        "nvcc": nvcc,
        "jax_cuda_versions": cuda_rt,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "versions": {
            k: ver(k)
            for k in ("jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt", "frax", "cbfpy", "qpax", "elastiqp", "numpy")
        },
        "frax_source": {"path": str(FRAX_DIR), "commit": FRAX_COMMIT, "pyproject_version": _frax_version()},
        "x64": jax.config.jax_enable_x64,
    }


def _frax_version():
    try:
        for line in (FRAX_DIR / "pyproject.toml").read_text().splitlines():
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def n_calls_for(batch: int) -> int:
    return 1000 if batch <= 128 else 200


def compile_time(jitted, *args) -> float:
    """Seconds to lower+compile `jitted` for these args (no execution)."""
    t0 = time.perf_counter()
    jitted.lower(*args).compile()
    return time.perf_counter() - t0


def time_calls(fn, arg_batches, n_calls: int, warmup: int = 10) -> dict:
    """Time `fn(*args)` cycling through `arg_batches` (a list of arg tuples).

    Every call is followed by block_until_ready. Returns median / p99 / mean /
    min / max in microseconds plus the number of calls.
    """
    import jax
    import numpy as np

    k = len(arg_batches)
    for i in range(warmup):
        jax.block_until_ready(fn(*arg_batches[i % k]))
    times = np.empty(n_calls)
    for i in range(n_calls):
        args = arg_batches[i % k]
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times[i] = time.perf_counter() - t0
    times *= 1e6
    return {
        "median_us": float(np.median(times)),
        "p99_us": float(np.percentile(times, 99)),
        "mean_us": float(times.mean()),
        "min_us": float(times.min()),
        "max_us": float(times.max()),
        "n_calls": int(n_calls),
    }


# --------------------------------------------------------------------------- #
# State sampling
# --------------------------------------------------------------------------- #
def sample_q(robot, rng, n, margin: float = 0.0):
    lo = robot.actuated_joint_lower_limits
    hi = robot.actuated_joint_upper_limits
    span = hi - lo
    return rng.uniform(lo + margin * span, hi - margin * span, size=(n, robot.nv))


def sample_v(rng, n, nv, rms: float = 0.5):
    return rng.normal(0.0, rms, size=(n, nv))


def push_to_joint_limits(robot, rng, q, v, frac_joints: float = 0.15, gap: float = 0.01):
    """Move ~frac_joints of each row's joints to within `gap` rad of a limit and
    give them velocity pointing outward (so the RD2 joint-limit barrier bites)."""
    q = q.copy()
    v = v.copy()
    lo = robot.actuated_joint_lower_limits
    hi = robot.actuated_joint_upper_limits
    n, nv = q.shape
    k = max(1, int(round(frac_joints * nv)))
    for i in range(n):
        js = rng.choice(nv, size=k, replace=False)
        for j in js:
            if rng.uniform() < 0.5:
                q[i, j] = hi[j] - rng.uniform(0, gap)
                v[i, j] = abs(rng.normal(0, 0.5)) + 0.2
            else:
                q[i, j] = lo[j] + rng.uniform(0, gap)
                v[i, j] = -abs(rng.normal(0, 0.5)) - 0.2
    return q, v


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
