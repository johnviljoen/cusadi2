# Native CPU baselines: median speedup of cusadi2 (best config) across robots
(16 threads; n = robots with both numbers; MuJoCo CPU is fp64 on both dtype rows, no Hessian)

| func | dtype | B | Pinocchio ideal. | Pinocchio + transfer | MuJoCo CPU ideal. | MuJoCo CPU + transfer | CasADi CPU ideal. | CasADi CPU + transfer |
|---|---|---|---|---|---|---|---|---|
| fk | double | 256 | 8.20x (n=64) | 12.67x (n=64) | 14.43x (n=63) | 19.88x (n=63) | 2.56x (n=64) | 7.43x (n=64) |
| fk | double | 16384 | 18.67x (n=64) | 35.61x (n=64) | 34.61x (n=63) | 44.99x (n=63) | 5.68x (n=64) | 23.60x (n=64) |
| fk | float | 256 | 8.48x (n=63) | 11.42x (n=63) | 16.90x (n=63) | 22.78x (n=63) | 2.92x (n=63) | 6.43x (n=63) |
| fk | float | 16384 | 30.93x (n=63) | 42.42x (n=63) | 58.92x (n=63) | 85.22x (n=63) | 10.98x (n=63) | 22.54x (n=63) |
| fkjac | double | 256 | 10.65x (n=64) | 32.63x (n=64) | 12.05x (n=63) | 31.10x (n=63) | 2.27x (n=64) | 8.96x (n=64) |
| fkjac | double | 16384 | 15.92x (n=64) | 83.14x (n=59) | 22.51x (n=63) | 103.16x (n=59) | 3.59x (n=64) | 27.66x (n=64) |
| fkjac | float | 256 | 11.26x (n=63) | 27.25x (n=63) | 15.34x (n=63) | 41.27x (n=63) | 2.90x (n=63) | 7.68x (n=63) |
| fkjac | float | 16384 | 26.18x (n=63) | 84.12x (n=62) | 37.63x (n=63) | 182.21x (n=59) | 5.68x (n=63) | 19.81x (n=63) |
| fkhess | double | 256 | 6.74x (n=64) | 150.63x (n=62) | — | — | 2.04x (n=64) | 8.57x (n=64) |
| fkhess | double | 16384 | nanx (n=64) | 103.58x (n=30) | — | — | nanx (n=63) | nanx (n=61) |
| fkhess | float | 256 | 8.68x (n=63) | 117.19x (n=62) | — | — | 2.92x (n=63) | 7.59x (n=63) |
| fkhess | float | 16384 | 9.97x (n=63) | 90.36x (n=30) | — | — | 3.73x (n=63) | 15.52x (n=63) |
