"""Sweep batch B for the MIT humanoid functions: cusadi2 level-parallel ('lp')
vs original-cusadi kernel ('batch'), kernel time (events) and wall time from
Python (ctypes), median of N_TIMED. CPU reference: CasADi C codegen, gcc -O3,
1 problem and B problems over CPU_THREADS OpenMP threads (bench.py's driver)."""
import ctypes, glob, json, os, subprocess, sys, time
import casadi as ca
sys.path.insert(0, "/home/jovi/code/cusecmpc/cusadi2/examples/paper_results")
from bench import CPU_DRIVER

HERE = os.path.dirname(os.path.abspath(__file__))
FN_DIR = os.path.join(HERE, "humanoid_fns")
BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]
N_WARMUP, N_TIMED = 5, 20
CPU_THREADS = [1, 16]
names = [os.path.basename(p)[:-7] for p in sorted(glob.glob(os.path.join(FN_DIR, "*.casadi")))]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rows = ["dtype,func,config,batch,latency_us,wall_us"]
for dtype in ("double", "float"):
    out = os.path.join(HERE, "humanoid_build", dtype)
    lib = ctypes.CDLL(os.path.join(out, "latency_lib.so"))
    lib.c2_setup.restype, lib.c2_setup.argtypes = ctypes.c_int, [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.c2_eval.restype, lib.c2_eval.argtypes = ctypes.c_double, [ctypes.c_int]
    lib.c2_teardown.argtypes = [ctypes.c_int]
    for n in names:
        for B in BATCHES:
            for cfg, serial in (("lp", 0), ("batch", 1)):
                h = lib.c2_setup(os.path.join(out, "plans", f"{n}.plan").encode(), serial, B)
                if h < 0:
                    rows.append(f"{dtype},{n},{cfg},{B},nan,nan"); continue
                for _ in range(N_WARMUP): lib.c2_eval(h)
                kus, wus = [], []
                for _ in range(N_TIMED):
                    t0 = time.perf_counter(); k = lib.c2_eval(h)
                    wus.append((time.perf_counter() - t0) * 1e6); kus.append(k * 1e6)
                lib.c2_teardown(h)
                km, wm = sorted(kus)[N_TIMED // 2], sorted(wus)[N_TIMED // 2]
                rows.append(f"{dtype},{n},{cfg},{B},{km:.3f},{wm:.3f}")
                print(f"[{dtype}] {n:38s} {cfg:5s} B={B:6d}: kernel {km:9.1f} us  wall {wm:9.1f} us", flush=True)
open(os.path.join(HERE, "humanoid_latency.csv"), "w").write("\n".join(rows) + "\n")

# CPU reference
cpu = {}
for dtype in ("double", "float"):
    outdir = os.path.join(HERE, "humanoid_build", dtype, "cpu"); os.makedirs(outdir, exist_ok=True)
    driver = os.path.join(outdir, "driver.c"); open(driver, "w").write(CPU_DRIVER)
    cpu[dtype] = {}
    for n in names:
        fn = ca.Function.load(os.path.join(FN_DIR, n + ".casadi"))
        gen = ca.CodeGenerator(n, {"with_header": True, "casadi_real": dtype}); gen.add(fn); gen.generate(outdir + os.sep)
        exe = os.path.join(outdir, n)
        subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", f"-DFN={n}", f'-DFN_H="{n}.h"', f"-I{outdir}",
                        driver, os.path.join(outdir, f"{n}.c"), "-lm", "-o", exe], check=True)
        def run(n_items, threads):
            reps = max(5, 2000 // n_items)
            return float(subprocess.check_output([exe, str(max(1, reps // 10)), str(reps), str(n_items), str(threads)]))
        cpu[dtype][n] = {"one_us": run(1, 1), "seq_us": {str(t): {str(B): run(B, t) for B in BATCHES} for t in CPU_THREADS}}
        print(f"[{dtype}] cpu {n:38s} 1 problem {cpu[dtype][n]['one_us']:8.2f} us; B=16384: "
              + "  ".join(f"{t}thr {cpu[dtype][n]['seq_us'][str(t)]['16384']/1e3:8.2f} ms" for t in CPU_THREADS), flush=True)
json.dump(cpu, open(os.path.join(HERE, "humanoid_cpu.json"), "w"), indent=2)
print("SWEEP DONE")
