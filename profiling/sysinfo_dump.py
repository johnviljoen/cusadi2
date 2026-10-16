"""Write results/system.json (GPU, CPU, driver/CUDA, package versions)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

common.configure("gpu", x64=True)
common.ensure_frax()
info = common.sysinfo()
common.save_json(common.RESULTS / "system.json", info)
print({k: v for k, v in info.items() if k in ("gpu", "cpu", "versions", "jax_cuda_versions", "nvcc")})
