<!-- Loaded 18 JSON files from benchmark/mjx-comparison/results -->


## A1 — ABA Jacobian (ddq/dq, ddq/dv, ddq/dtau)

Median time per evaluation in µs.

| batch | LP (fp64) | Seq (fp64) | MJX (fp32) | MJX (fp64) |
|---:|---:|---:|---:|---:|
| 1 | 158.5 | 432.0 | 9,141.5 | 14,065 |
| 8 | 160.7 | 514.0 | 19,386 | 26,321 |
| 32 | 159.7 | 926.7 | 19,985 | 35,307 |
| 128 | 161.8 | 2,084.9 | 33,143 | 69,956 |
| 256 | 166.9 | 4,102.0 | 37,155 | 87,839 |
| 512 | 323.7 | 4,112.4 | 50,088 | 108,385 |

**Ratio to LP fp64** (>1 means the backend is slower than LP):

| batch | Seq / level_parallel-fp64 | MJX / level_parallel-fp64 | MJX / level_parallel-fp64 |
|---:|---:|---:|---:|
| 1 | — | — | — |
| 8 | — | — | — |
| 32 | — | — | — |
| 128 | — | — | — |
| 256 | — | — | — |
| 512 | — | — | — |

**First-call compile-and-run time** (s; one-off per batch size for MJX):

| batch | MJX first-call (s) |
|---:|---:|
| 1 | 49.6 |
| 8 | 34.0 |
| 32 | 39.1 |
| 128 | 43.6 |
| 256 | 47.0 |
| 512 | 56.7 |

LP data-driven kernel compiles once in ~1.5 s and is reused across all functions. Sequential CusADi kernels each require a one-off NVCC compile (20–60 s).

## A1 — Forward dynamics only (ABA, no Jacobian)

| batch | LP (fp64) | Seq (fp64) | MJX (fp32) | MJX (fp64) | mujoco-warp (fp32) |
|---:|---:|---:|---:|---:|---:|
| 1 | 93.1 | 77.8 | 1,717.8 | 2,411.4 | 2,829.9 |
| 8 | 93.4 | 78.8 | 3,370.4 | 4,550.5 | 3,240.7 |
| 32 | 94.2 | 98.2 | 3,125.3 | 4,792.9 | 3,461.0 |
| 128 | 94.0 | 224.3 | 3,105.6 | 6,007.1 | 3,652.0 |
| 256 | 94.2 | 436.2 | 4,445.9 | 7,149.8 | 3,912.9 |
| 512 | 96.3 | 436.2 | 8,628.9 | 9,400.1 | 3,880.3 |

## H1 — ABA Jacobian (ddq/dq, ddq/dv, ddq/dtau)

Median time per evaluation in µs.

| batch | LP (fp64) | Seq (fp64) | MJX (fp32) | MJX (fp64) |
|---:|---:|---:|---:|---:|
| 1 | 263.2 | 857.3 | 4,405.2 | 7,297.0 |
| 8 | 267.1 | 962.7 | 6,260.9 | 7,584.4 |
| 32 | 268.3 | 1,541.1 | 10,523 | 9,017.1 |
| 128 | 268.3 | 3,614.2 | 12,542 | 22,561 |
| 256 | 519.4 | 7,183.4 | 14,059 | 30,795 |
| 512 | 1,023.0 | 7,192.6 | 18,089 | 50,973 |

**Ratio to LP fp64** (>1 means the backend is slower than LP):

| batch | Seq / level_parallel-fp64 | MJX / level_parallel-fp64 | MJX / level_parallel-fp64 |
|---:|---:|---:|---:|
| 1 | — | — | — |
| 8 | — | — | — |
| 32 | — | — | — |
| 128 | — | — | — |
| 256 | — | — | — |
| 512 | — | — | — |

**First-call compile-and-run time** (s; one-off per batch size for MJX):

| batch | MJX first-call (s) |
|---:|---:|
| 1 | 32.5 |
| 8 | 46.4 |
| 32 | 58.1 |
| 128 | 62.6 |
| 256 | 64.0 |
| 512 | 160.4 |

LP data-driven kernel compiles once in ~1.5 s and is reused across all functions. Sequential CusADi kernels each require a one-off NVCC compile (20–60 s).

## H1 — Forward dynamics only (ABA, no Jacobian)

| batch | LP (fp64) | Seq (fp64) | MJX (fp32) | MJX (fp64) | mujoco-warp (fp32) |
|---:|---:|---:|---:|---:|---:|
| 1 | 148.5 | 123.9 | 1,113.8 | 1,325.0 | 2,381.4 |
| 8 | 150.5 | 128.8 | 1,552.4 | 1,259.1 | 2,610.5 |
| 32 | 151.6 | 165.9 | 2,132.4 | 1,268.7 | 2,621.3 |
| 128 | 151.6 | 389.1 | 1,625.2 | 2,720.4 | 2,632.3 |
| 256 | 151.5 | 754.7 | 1,696.4 | 2,801.9 | 2,878.2 |
| 512 | 154.6 | 757.6 | 1,751.3 | 3,138.4 | 2,864.4 |
