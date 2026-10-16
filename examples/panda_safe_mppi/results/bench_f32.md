| evaluator (fp32) | K | T | tick median | tick p99 | rollout | model | integrate | project |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CusADi2 (ours) | 256 | 10 | 1.85 | 1.87 | 1.70 | 0.106 | 0.051 | 0.015 |
| CusADi2 (ours) | 256 | 20 | 3.02 | 3.56 | 3.04 | 0.096 | 0.046 | 0.013 |
| CusADi2 (ours) | 256 | 30 | 4.40 | 4.69 | 4.54 | 0.096 | 0.046 | 0.005 |
| CusADi2 (ours) | 256 | 50 | 7.20 | 7.70 | 7.63 | 0.096 | 0.046 | 0.013 |
| CusADi2 (ours) | 256 | 100 | 14.34 | 15.22 | 15.24 | 0.096 | 0.046 | 0.013 |
| CusADi2 (ours) | 64 | 100 | 13.58 | 14.57 | 13.69 | 0.091 | 0.046 | 0.003 |
| CusADi2 (ours) | 128 | 100 | 13.81 | 14.74 | 14.78 | 0.092 | 0.045 | 0.013 |
| CusADi2 (ours) | 512 | 100 | 24.05 | 24.79 | 24.87 | 0.192 | 0.047 | 0.014 |
| CusADi2 (ours) | 1024 | 100 | 34.91 | 35.67 | 35.56 | 0.295 | 0.051 | 0.014 |
| CusADi2 (ours) | 2048 | 100 | 59.48 | 60.32 | 59.85 | 0.487 | 0.100 | 0.015 |
| frax + cbfpy (JAX/XLA) | 256 | 10 | 7.77 | 8.29 | 6.99 | 0.522 | 0.150 | 0.003 |
| frax + cbfpy (JAX/XLA) | 256 | 20 | 14.59 | 15.63 | 13.83 | 0.515 | 0.142 | 0.003 |
| frax + cbfpy (JAX/XLA) | 256 | 30 | 21.38 | 22.22 | 20.92 | 0.521 | 0.149 | 0.003 |
| frax + cbfpy (JAX/XLA) | 256 | 50 | 35.23 | 35.99 | 34.50 | 0.536 | 0.146 | 0.003 |
| frax + cbfpy (JAX/XLA) | 256 | 100 | 69.94 | 71.90 | 68.87 | 0.518 | 0.148 | 0.003 |
| frax + cbfpy (JAX/XLA) | 64 | 100 | 50.98 | 52.21 | 50.75 | 0.350 | 0.125 | 0.003 |
| frax + cbfpy (JAX/XLA) | 128 | 100 | 57.08 | 58.92 | 56.32 | 0.403 | 0.137 | 0.003 |
| frax + cbfpy (JAX/XLA) | 512 | 100 | 87.44 | 89.89 | 86.65 | 0.686 | 0.153 | 0.003 |
| frax + cbfpy (JAX/XLA) | 1024 | 100 | 126.00 | 128.04 | 125.11 | 1.051 | 0.156 | 0.004 |
| frax + cbfpy (JAX/XLA) | 2048 | 100 | 202.62 | 204.67 | 200.41 | 1.794 | 0.173 | 0.004 |
| MuJoCo XLA + cbfpy (JAX/XLA) | 256 | 10 | 10.27 | 10.97 | 9.33 | 0.683 | 0.176 | 0.003 |
| MuJoCo XLA + cbfpy (JAX/XLA) | 256 | 20 | 19.63 | 20.91 | 19.00 | 0.690 | 0.184 | 0.003 |
| MuJoCo XLA + cbfpy (JAX/XLA) | 256 | 30 | 28.88 | 30.95 | 28.12 | 0.683 | 0.184 | 0.003 |
| MuJoCo XLA + cbfpy (JAX/XLA) | 256 | 50 | 47.81 | 49.36 | 47.09 | 0.683 | 0.184 | 0.003 |
| MuJoCo XLA + cbfpy (JAX/XLA) | 256 | 100 | 93.58 | 97.25 | 92.22 | 0.673 | 0.174 | 0.003 |
| MuJoCo XLA + cbfpy (JAX/XLA) | 64 | 100 | 77.73 | 81.42 | 77.97 | 0.532 | 0.167 | 0.003 |
| MuJoCo XLA + cbfpy (JAX/XLA) | 128 | 100 | 81.94 | 89.74 | 81.05 | 0.556 | 0.160 | 0.003 |
| MuJoCo XLA + cbfpy (JAX/XLA) | 512 | 100 | 106.43 | 108.85 | 105.43 | 0.809 | 0.172 | 0.003 |
| MuJoCo XLA + cbfpy (JAX/XLA) | 1024 | 100 | 144.03 | 148.00 | 142.08 | 1.162 | 0.188 | 0.004 |
| MuJoCo XLA + cbfpy (JAX/XLA) | 2048 | 100 | 216.92 | 229.21 | 215.37 | 1.839 | 0.213 | 0.004 |
| CusADi (original, one thread per instance) | 256 | 10 | 15.44 | 15.47 | 15.34 | 1.388 | 0.134 | 0.013 |
| CusADi (original, one thread per instance) | 256 | 20 | 30.62 | 30.80 | 30.64 | 1.389 | 0.134 | 0.013 |
| CusADi (original, one thread per instance) | 256 | 30 | 45.85 | 46.13 | 45.98 | 1.390 | 0.134 | 0.013 |
| CusADi (original, one thread per instance) | 256 | 50 | 76.28 | 76.78 | 76.76 | 1.390 | 0.134 | 0.013 |
| CusADi (original, one thread per instance) | 256 | 100 | 152.63 | 153.52 | 153.62 | 1.392 | 0.134 | 0.013 |
| CusADi (original, one thread per instance) | 64 | 100 | 36.89 | 37.90 | 36.90 | 0.333 | 0.036 | 0.003 |
| CusADi (original, one thread per instance) | 128 | 100 | 71.98 | 72.92 | 72.94 | 0.652 | 0.067 | 0.013 |
| CusADi (original, one thread per instance) | 512 | 100 | 151.86 | 152.62 | 152.72 | 1.383 | 0.133 | 0.014 |
| CusADi (original, one thread per instance) | 1024 | 100 | 152.45 | 153.26 | 153.09 | 1.387 | 0.134 | 0.014 |
| CusADi (original, one thread per instance) | 2048 | 100 | 153.06 | 153.92 | 153.56 | 1.389 | 0.134 | 0.014 |
| CasADi C codegen, CPU 16 threads + transfer | 256 | 10 | 2.80 | 64.22 | 2.00 | 0.162 | 0.031 | 0.004 |
| CasADi C codegen, CPU 16 threads + transfer | 256 | 20 | 4.42 | 8.12 | 3.99 | 0.163 | 0.031 | 0.004 |
| CasADi C codegen, CPU 16 threads + transfer | 256 | 30 | 6.35 | 6.91 | 6.08 | 0.163 | 0.031 | 0.004 |
| CasADi C codegen, CPU 16 threads + transfer | 256 | 50 | 10.48 | 43.00 | 10.11 | 0.163 | 0.031 | 0.004 |
| CasADi C codegen, CPU 16 threads + transfer | 256 | 100 | 20.78 | 94.06 | 20.19 | 0.163 | 0.031 | 0.004 |
| CasADi C codegen, CPU 16 threads + transfer | 64 | 100 | 8.74 | 104.66 | 8.47 | 0.081 | 0.035 | 0.004 |
| CasADi C codegen, CPU 16 threads + transfer | 128 | 100 | 12.62 | 21.47 | 12.15 | 0.091 | 0.023 | 0.004 |
| CasADi C codegen, CPU 16 threads + transfer | 512 | 100 | 37.86 | 114.48 | 36.71 | 0.305 | 0.046 | 0.004 |
| CasADi C codegen, CPU 16 threads + transfer | 1024 | 100 | 74.23 | 164.22 | 70.79 | 0.593 | 0.075 | 0.005 |
| CasADi C codegen, CPU 16 threads + transfer | 2048 | 100 | 143.44 | 311.99 | 141.38 | 1.238 | 0.146 | 0.005 |

cusadi2 vs frax + cbfpy (JAX/XLA): tick 4.9x, model call 5.4x, integrate 3.2x  
cusadi2 vs MuJoCo XLA + cbfpy (JAX/XLA): tick 6.5x, model call 7.0x, integrate 3.8x  
cusadi2 vs CusADi (original, one thread per instance): tick 10.6x, model call 14.5x, integrate 2.9x  
cusadi2 vs CasADi C codegen, CPU 16 threads + transfer: tick 1.4x, model call 1.7x, integrate 0.7x  
