# cusadi2 examples

End-to-end self-check of the native (host-free C++/CUDA) backend.

```bash
python examples/run.py
```

This exercises the full pipeline — `dag.py` → `generate.py` → `cusadi2.h` +
generated `kernel.cu` — and validates it numerically on the GPU:

- **demo** — plain batch-only level parallelism (every arg `"stage"`, `n_stages=1`).
- **dyn** — the stage path with a `coupled` (span 1) and a `shared` input over 4 stages.

For each, `run.py` emits a plan header whose embedded reference (random inputs
evaluated through CasADi at build time) is compared against the kernel's GPU
output. Expected:

```
[demo  ] n_stages=1 nout=2  maxerr=0.000e+00 -> PASS
[dyn   ] n_stages=4 nout=16 maxerr=1.110e-16 -> PASS
ALL PASS
```

Generated artifacts land in `examples/build/` (transient). Env overrides:
`CUSADI2_GPU` (device index, default `1`), `NVCC` (path to `nvcc`).
