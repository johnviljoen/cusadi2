"""CPU baseline for mppi_bench.cu: the same rollout sweep (B samples, horizon * substeps chained
evaluations of go2_step1) with CasADi's C code for the function, gcc -O3 -march=native, samples split
over OpenMP threads. Same inputs as the GPU benchmark, so the checksums agree.

    python cpu_bench.py <build_dir> <float|double> <threads> <n_timed> B [B ...]

Prints CSV rows  mode,dtype,B,launches,median_ms,min_ms,checksum  with mode = cpu<threads>.
"""
import os
import subprocess
import sys

import casadi as ca

DRIVER = r"""
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "go2_step1.h"
static double knot_value(long long b, int i, double lo, double hi) {      /* as in mppi_bench.cu */
    unsigned long long z = (unsigned long long)b * 0x9E3779B97F4A7C15ull + (unsigned long long)i * 0xBF58476D1CE4E5B9ull + 1;
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull; z ^= z >> 27; z *= 0x94D049BB133111EBull; z ^= z >> 31;
    double u = (double)(z >> 11) / 9007199254740992.0, v = 0.5 * u - 0.25;
    return v < lo ? lo : v > hi ? hi : v;
}
static double* read_f64(const char* dir, const char* name, int n) {
    char path[4096]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE* f = fopen(path, "rb"); double* v = malloc(n * sizeof(double));
    if (!f || fread(v, 8, n, f) != (size_t)n) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    fclose(f); return v;
}
static int cmp(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }
int main(int argc, char** argv) {
    const char* dir = argv[1]; int threads = atoi(argv[2]), n_timed = atoi(argv[3]), L = atoi(argv[4]); long B = atol(argv[5]);
    enum { NX = NX_, ND = 60, NP = NP_ };
    double *x0 = read_f64(dir, "bench/x0.f64", NX), *par = read_f64(dir, "bench/par.f64", NP), *ylim = read_f64(dir, "ylim.f64", 24);
    casadi_real *xs = malloc(B * NX * sizeof(casadi_real)), *x = malloc(B * NX * sizeof(casadi_real)), *Y = malloc(B * ND * sizeof(casadi_real)),
                *rp = calloc(B * 4, sizeof(casadi_real)), *pp = malloc(B * NP * sizeof(casadi_real));
    for (long b = 0; b < B; ++b) {
        for (int i = 0; i < NX; ++i) xs[b * NX + i] = (casadi_real)x0[i];
        for (int i = 0; i < NP; ++i) pp[b * NP + i] = (casadi_real)par[i];
        for (int i = 0; i < ND; ++i) Y[b * ND + i] = (casadi_real)knot_value(b, i, ylim[i % 12], ylim[12 + i % 12]);
        rp[b * 4] = (casadi_real)(x0[0] + 3.0); rp[b * 4 + 1] = (casadi_real)x0[1];
    }
    casadi_int sz_arg, sz_res, sz_iw, sz_w; go2_step1_work(&sz_arg, &sz_res, &sz_iw, &sz_w);
    omp_set_num_threads(threads);
    double* t = malloc(n_timed * sizeof(double));
    for (int k = -1; k < n_timed; ++k) {                                   /* k = -1: warm-up */
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(x, xs, B * NX * sizeof(casadi_real));
        #pragma omp parallel
        {
            const casadi_real** arg = calloc(sz_arg + 4, sizeof(*arg)); casadi_real** res = calloc(sz_res + 1, sizeof(*res));
            casadi_int* iw = calloc(sz_iw + 1, sizeof(*iw)); casadi_real* w = calloc(sz_w + 1, sizeof(*w)); casadi_real tmp[2][NX];
            #pragma omp for schedule(static)
            for (long b = 0; b < B; ++b) {
                memcpy(tmp[0], x + b * NX, sizeof tmp[0]);
                arg[1] = Y + b * ND; arg[2] = rp + b * 4; arg[3] = pp + b * NP;
                for (int s = 0; s < L; ++s) { arg[0] = tmp[s & 1]; res[0] = tmp[(s + 1) & 1]; go2_step1(arg, res, iw, w, 0); }
                memcpy(x + b * NX, tmp[L & 1], sizeof tmp[0]);
            }
            free(arg); free(res); free(iw); free(w);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (k >= 0) t[k] = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
    }
    double sum = 0; for (long b = 0; b < B; ++b) sum += x[b * NX + NX - 1];
    qsort(t, n_timed, sizeof(double), cmp);
    printf("cpu%d,%s,%ld,%d,%.4f,%.4f,%.9g\n", threads, DTYPE, B, L, t[n_timed / 2], t[0], sum / B);
    return 0;
}
"""

build, dtype, threads, n_timed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
meta = dict((k, float(v)) for k, v in (line.split() for line in open(os.path.join(build, "meta.txt"))))
out = os.path.join(build, "cpu_" + dtype)
os.makedirs(out, exist_ok=True)
exe = os.path.join(out, "cpu_bench")
if not os.path.exists(exe):
    f = ca.Function.load(os.path.join(build, "fns", "go2_step1.casadi"))
    gen = ca.CodeGenerator("go2_step1", {"with_header": True, "casadi_real": dtype})
    gen.add(f)
    gen.generate(out + os.sep)
    open(os.path.join(out, "driver.c"), "w").write(DRIVER)
    subprocess.run([os.environ.get("CC", "gcc"), "-O3", "-march=native", "-fopenmp", f"-DNX_={int(meta['nx'])}", f"-DNP_={int(meta['npar'])}",
                    f'-DDTYPE="{dtype}"', f"-I{out}", os.path.join(out, "driver.c"), os.path.join(out, "go2_step1.c"), "-lm", "-o", exe], check=True)
for B in sys.argv[5:]:
    sys.stdout.write(subprocess.check_output([exe, build, str(threads), str(n_timed), str(int(meta["horizon"] * meta["substeps"])), B]).decode())
    sys.stdout.flush()
