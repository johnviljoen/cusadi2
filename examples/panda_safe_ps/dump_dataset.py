"""Write the timing/verification dataset as flat f64 files for the C++ timers: build/bench/{x,Y,rp,par}.f64 (n rows)."""
import os, sys
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common
common.configure("gpu", x64=True)
import numpy as np
from frax_model import FraxStep
from dataset import make_dataset
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096; T = 20
fs = FraxStep(T); ds = make_dataset(fs.base, n, 0, T)
out = HERE / "build" / "bench"; out.mkdir(parents=True, exist_ok=True)
for k in ("x", "Y", "rp", "par"):
    np.ascontiguousarray(ds[k], dtype=np.float64).tofile(out / f"{k}.f64")
np.asarray(ds["kind"], dtype=np.float64).tofile(out / "kind.f64")
print("wrote", out, ds["x"].shape)
