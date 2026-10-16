# Panda OSC + CBF on GPU (qpax, fp64): frax vs cusadi2 as the model-evaluation backend

Times in µs per batched call, median of 1000 calls (B ≤ 128) or 200 calls, jit + vmap, device-resident inputs, block_until_ready. `model` = nominal OSC torque + cbfpy's (P, q, G, h); `solve` = qpax.solve_qp_elastic_primal on precomputed data (identical for both backends); `e2e` = model + solve in one jit. Speed-up > 1 means cusadi2 is faster. `max|Δu|` is the largest difference in the safe torque between the two backends in the first batch window.

cusadi2 side: `panda_casadi.py` (Pinocchio-casadi on frax's panda.urdf; cbfpy's barrier / Lie-derivative algebra reproduced with forward-mode directional derivatives, `ca.jtimes`, as cbfpy does with `jax.jvp`, then `ca.cse`; verified to ≤1e-9 relative against frax + cbfpy on 64 random states). It runs inside the JAX program as one XLA FFI custom call on XLA's stream (`cusadi2_ffi.cu`); the dense P and G are scattered from cusadi2's column-major nonzeros inside the same jit. The bare kernel timed with CUDA events equals the jit time at every batch, so the FFI path adds no copies. `max_live` is cusadi2's live-slot budget (shared memory per block = 8 B × peak live slots); a smaller budget windows the plan into more levels but lets more blocks share an SM.

| cusadi2 variant | SX nodes | levels | peak live slots | shared memory / block |
|---|---:|---:|---:|---:|
| level, max_live 3000 | 37727 | 601 | 2082 | 16 KB |
| level, max_live 6000 | 37727 | 361 | 4397 | 34 KB |

| B | model frax | model cusadi2 (level, max_live 3000) | model cusadi2 (level, max_live 6000) | solve | e2e frax | e2e cusadi2 (level, max_live 3000) | e2e cusadi2 (level, max_live 6000) | best cusadi2 model speed-up | best cusadi2 e2e speed-up | model share frax | model share cusadi2 (best) | max\|Δu\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,425 | 283 | 213 | 2,154 | 3,469 | 2,362 | 2,300 | 6.69× | 1.51× | 38% | 6% | 7.9e-13 |
| 4 | 725 | 313 | 213 | 1,755 | 2,394 | 1,967 | 1,898 | 3.40× | 1.26× | 27% | 8% | 2.6e-11 |
| 16 | 776 | 319 | 221 | 1,951 | 2,711 | 2,270 | 2,087 | 3.51× | 1.30× | 28% | 7% | 3.3e-09 |
| 64 | 917 | 320 | 222 | 2,247 | 3,100 | 2,442 | 2,372 | 4.13× | 1.31× | 28% | 5% | 1.5e-08 |
| 256 | 1,907 | 357 | 378 | 2,849 | 4,713 | 3,118 | 3,144 | 5.34× | 1.51× | 40% | 9% | 8.3e-07 |
| 1024 | 5,553 | 946 | 1,134 | 4,490 | 9,917 | 5,263 | 5,526 | 5.87× | 1.88× | 55% | 15% | 1.6e-05 |
| 4096 | 21,335 | 3,027 | 4,198 | 12,786 | 33,732 | 15,731 | 16,930 | 7.05× | 2.14× | 62% | 19% | 1.1e-02 |
