# Panda OSC + CBF on GPU (qpax, fp64): frax vs cusadi2 model evaluation

cusadi2 plan: 485371 SX nodes, 2839 levels, peak live 11521 slots, runtime block size 1024, 92168 B shared memory, level kernel. CasADi model: Pinocchio-casadi on frax's panda.urdf, barriers + Lie derivatives via casadi.jacobian; matches frax+cbfpy to ≤1e-9 relative (q) and ≤1e-12 (others).

Median over 1000 calls (B ≤ 128) / 200 calls, vmapped, device-resident inputs. `model` = nominal OSC + (P, q, G, h); `solve` = qpax on precomputed data; model share = (e2e − solve)/e2e.

| B | model frax (µs) | model cusadi2 (µs) | model speedup | solve (µs) | e2e frax (µs) | e2e cusadi2 (µs) | e2e speedup | model share frax | model share cusadi2 | max\|Δu\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,427 | 1,166 | 1.22× | 2,154 | 3,477 | 3,257 | 1.07× | 38% | 34% | 7.9e-13 |
| 4 | 724 | 1,195 | 0.61× | 1,752 | 2,386 | 2,882 | 0.83× | 27% | 39% | 2.2e-11 |
| 16 | 776 | 1,193 | 0.65× | 1,954 | 2,695 | 3,060 | 0.88× | 28% | 36% | 1.0e-09 |
| 64 | 918 | 1,209 | 0.76× | 2,247 | 3,100 | 3,359 | 0.92× | 28% | 33% | 1.1e-08 |
| 256 | 1,928 | 4,399 | 0.44× | 2,859 | 4,736 | 7,130 | 0.66× | 40% | 60% | 8.3e-07 |
| 1024 | 5,629 | 15,446 | 0.36× | 4,454 | 9,936 | 19,563 | 0.51× | 55% | 77% | 1.6e-05 |
| 4096 | 21,310 | 59,327 | 0.36× | 12,780 | 33,716 | 71,718 | 0.47× | 62% | 82% | 1.1e-02 |

Compile times (s):

| B | model_frax | model_cusadi2 | solve | e2e_frax | e2e_cusadi2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0 |
| 4 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 |
| 16 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 |
| 64 | 0.0 | 0.1 | 0.3 | 0.0 | 0.0 |
| 256 | 0.0 | 0.1 | 0.3 | 0.0 | 0.0 |
| 1024 | 0.0 | 0.1 | 0.3 | 0.0 | 0.0 |
| 4096 | 0.0 | 0.1 | 0.6 | 0.0 | 0.0 |
