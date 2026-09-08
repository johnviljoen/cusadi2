#!/usr/bin/env python
"""The CusADi paper's instruction-count benchmark (se-hwan/cusadi run_benchmarks.py
+ plotting/plot_benchmarking.m, media/fig_benchmark.png) reproduced on cusadi2.

The five released functions fn_1e1 .. fn_1e5 (10 .. 100k instructions, fetched
from the upstream repo at the pinned commit) are evaluated for batch sizes
1 .. 32768, N_EVALS times each, by upstream's six methods under their names:

    cusadi                 cusadi2 with its features off: the per-function kernel
                           emitted by cusadi2.unrolled (the original execution
                           model), kernel time from CUDA events as upstream
    pytorch                upstream's generated torch code (torch.compile + vmap),
                           run under CUSADI2_TORCH_PYTHON (a Python with a CUDA
                           torch; skipped with nan if it has none); the first of
                           the N_EVALS evals includes the compile, as upstream
    serial_cpu             upstream's C++ drivers (orig_cpu.cpp) against the
    parallel_cpu           CasADi-generated .so through libcasadi's external(),
    serial_cpu_transfer    one process per eval as upstream; *_transfer also
    parallel_cpu_transfer  moves one double per argument through cudaMalloc/
                           cudaMemcpy inside the timed region, as upstream

plus cusadi2 (features on: the level-parallel kernel, its plan windowed to the
GPU's shared memory where the longest-path plan does not fit, as fn_1e5's does
not). The only deliberate departure: the two GPU kernels get one untimed warm-up launch per
case so a first-launch module load cannot land in a 20-sample mean.

    python examples/paper_results/bench_orig.py [--runs N] [--plot]

Run k writes results/orig/runs/k/orig_bench.json. fig_benchmark.{png,pdf} is
upstream's figure (speedup over serial_cpu, and over serial_cpu_transfer, both
as the ratio of means pooled over runs); fig_benchmark_cusadi2.{png,pdf} adds
the level-parallel kernel as a fourth bar.
"""
import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build", "orig")
RESULTS = os.path.join(HERE, "results", "orig")
UPSTREAM = ("https://raw.githubusercontent.com/se-hwan/cusadi/"
            "bffea25849a901277a8f8dc4db16748e629f9053/src/benchmark_functions")
FNS = [f"fn_1e{k}" for k in range(1, 6)]
N_ENVS_SWEEP = [2 ** k for k in range(16)]
N_EVALS = 20
CPU_METHODS = ["serial_cpu", "serial_cpu_transfer", "parallel_cpu", "parallel_cpu_transfer"]
DEVICE = os.environ.get("CUSADI2_GPU", "0")
NVCC = os.environ.get("NVCC", "nvcc")
TORCH_PYTHON = os.environ.get("CUSADI2_TORCH_PYTHON", sys.executable)
TORCH_TIMEOUT = float(os.environ.get("CUSADI2_TORCH_TIMEOUT", 7200))

# upstream OP_PYTORCH_DICT restricted to CasADi's scalar ops (ids from casadi)
TORCH_OPS = {0: "work[%d] = work[%d]", 1: "work[%d] = work[%d] + work[%d]", 2: "work[%d] = work[%d] - work[%d]",
             3: "work[%d] = work[%d] * work[%d]", 4: "work[%d] = work[%d] / work[%d]", 5: "work[%d] = -work[%d]",
             6: "work[%d] = torch.exp(work[%d])", 7: "work[%d] = torch.log(work[%d])",
             8: "work[%d] = torch.pow(work[%d], work[%d])", 9: "work[%d] = torch.pow(work[%d], work[%d])",
             10: "work[%d] = torch.sqrt(work[%d])", 11: "work[%d] = work[%d] * work[%d]",
             12: "work[%d] = 2.*(work[%d])", 13: "work[%d] = torch.sin(work[%d])", 14: "work[%d] = torch.cos(work[%d])",
             15: "work[%d] = torch.tan(work[%d])", 44: "work[%d] = %f", 45: "work[%d] = inputs[%d][%d]",
             46: "outputs[%d][%d] = work[%d]"}


def functions():
    import casadi as ca
    d = os.path.join(BUILD, "functions")
    os.makedirs(d, exist_ok=True)
    fns = []
    for name in FNS:
        path = os.path.join(d, f"{name}.casadi")
        if not os.path.exists(path):
            print(f"fetching {UPSTREAM}/{name}.casadi")
            urllib.request.urlretrieve(f"{UPSTREAM}/{name}.casadi", path)
        fns.append(ca.Function.load(path))
    return fns


def gpu_arch():
    cc = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader",
                                  "-i", DEVICE]).decode().split()[0]
    return "sm_" + cc.replace(".", "")


def build_gpu(fns):
    """Plans (cusadi2) and unrolled kernels (cusadi) per function, linked into a
    ctypes build of latency.cu. Returns (lib, plan_dir, compile seconds)."""
    import cusadi2
    d = os.path.join(BUILD, "gpu")
    os.makedirs(d, exist_ok=True)
    cusadi2.write_kernel_source(os.path.join(d, "kernel.cu"), dtype="double")
    arch, comp, objs = f"-arch={gpu_arch()}", {}, []
    with open(os.path.join(d, "unrolled_table.h"), "w") as tbl:
        for f in fns:
            cusadi2.write_plan(f, os.path.join(d, f"{f.name()}.plan"),
                               max_live=cusadi2.device_max_live("double", int(DEVICE)))
            t0 = time.perf_counter()
            src = cusadi2.write_unrolled_source(f, os.path.join(d, f"{f.name()}.cu"), dtype="double")
            subprocess.run([NVCC, "-std=c++17", "-O3", arch, "-Xcompiler", "-fPIC", "-c", src, "-o", src[:-3] + ".o"],
                           check=True)
            comp[f.name()] = time.perf_counter() - t0
            objs.append(src[:-3] + ".o")
            print(f"  {f.name()}: {f.n_instructions()} instructions, codegen + nvcc {comp[f.name()]:.1f}s")
            n = f.name()
            tbl.write(f'extern "C" int {n}_unrolled_sz_w();\nextern "C" int {n}_unrolled_n_stages();\n'
                      f'extern "C" int {n}_unrolled_launch(const S* const*, S* const*, S*, int, int, cudaStream_t);\n')
        tbl.write("static const UnrolledFn UNROLLED[] = {\n" + "".join(
            f'    {{"{f.name()}", {f.name()}_unrolled_sz_w, {f.name()}_unrolled_n_stages, {f.name()}_unrolled_launch}},\n'
            for f in fns) + "};\n")
    lib = os.path.join(d, "latency_lib.so")
    subprocess.run([NVCC, "-std=c++17", "-O3", arch, "-DCUSADI2_SCALAR=double", f"-I{cusadi2.INCLUDE_DIR}", f"-I{d}",
                    os.path.join(HERE, "latency.cu"), os.path.join(d, "kernel.cu")] + objs
                   + ["-shared", "-Xcompiler", "-fPIC", "-DCUSADI2_BENCH_LIB", "-o", lib], check=True)
    return lib, d, comp


def run_gpu(lib_path, plan_dir, fns, t):
    lib = ctypes.CDLL(lib_path)
    lib.c2_setup.restype, lib.c2_setup.argtypes = ctypes.c_int, [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.c2_eval.restype, lib.c2_eval.argtypes = ctypes.c_double, [ctypes.c_int]
    lib.c2_teardown.argtypes = [ctypes.c_int]
    for f in fns:
        plan = os.path.join(plan_dir, f"{f.name()}.plan").encode()
        for method, serial in (("cusadi", 1), ("cusadi2", 0)):
            for B in N_ENVS_SWEEP:
                h = lib.c2_setup(plan, serial, B)
                if h >= 0:
                    lib.c2_eval(h)
                t[f.name()][method].append([lib.c2_eval(h) if h >= 0 else math.nan for _ in range(N_EVALS)])
                lib.c2_teardown(h)
            print(f"  {f.name()} {method}: B=32768 {1e6 * sum(t[f.name()][method][-1]) / N_EVALS:.1f} us")


def build_cpu(fns):
    """f.generate + gcc as upstream (compile_commands.md); the four drivers via nvcc."""
    d = os.path.join(BUILD, "cpu")
    os.makedirs(d, exist_ok=True)
    for f in fns:
        f.generate(f"{f.name()}.c")
        os.replace(f"{f.name()}.c", os.path.join(d, f"{f.name()}.c"))
        subprocess.run(["gcc", "-fPIC", "-shared", "-O3", "-march=native", os.path.join(d, f"{f.name()}.c"),
                        "-o", os.path.join(d, f"{f.name()}.so")], check=True)
    for m in CPU_METHODS:
        flags = ["-DPARALLEL"] * m.startswith("parallel") + ["-DTRANSFER"] * m.endswith("transfer")
        subprocess.run([NVCC, "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-Xcompiler", "-fopenmp"] + flags
                       + [os.path.join(HERE, "orig_cpu.cpp"), f"-I{sys.prefix}/include", f"-L{sys.prefix}/lib",
                          "-lcasadi", f"-Xlinker=-rpath={sys.prefix}/lib", "-o", os.path.join(d, m)], check=True)
    return d


def run_cpu(d, fns, t):
    for f in fns:
        for m in CPU_METHODS:
            for B in N_ENVS_SWEEP:
                t[f.name()][m].append([float(subprocess.check_output(
                    [os.path.join(d, m), f.name(), os.path.join(d, f"{f.name()}.so"), str(B)], cwd=d))
                    for _ in range(N_EVALS)])
            print(f"  {f.name()} {m}: B=32768 {1e3 * sum(t[f.name()][m][-1]) / N_EVALS:.1f} ms")


def torch_source(f, path):
    """upstream generatePytorchCode: one straight-line function over CasADi's instructions."""
    body = ""
    for k in range(f.n_instructions()):
        op, o, i = f.instruction_id(k), f.instruction_output(k), f.instruction_input(k)
        args = ((o[0], f.instruction_constant(k)) if op == 44 else (o[0], i[0], i[1]) if op == 45
                else (o[0], o[1], i[0]) if op == 46 else (o[0], i[0], i[0]) if op == 11
                else (o[0], i[0], i[1]) if TORCH_OPS[op].count("%d") == 3 else (o[0], i[0]))
        body += "\n    " + TORCH_OPS[op] % args
    with open(path, "w") as out:
        out.write("import torch\n\n@torch.compile\ndef evaluate(outputs, inputs, work):" + body + "\n")


def run_pytorch(fns, t):
    d = os.path.join(BUILD, "pytorch")
    os.makedirs(d, exist_ok=True)
    for f in fns:
        src = os.path.join(d, f"{f.name()}_PT.py")
        torch_source(f, src)
        spec = [",".join(str(f.nnz_in(i)) for i in range(f.n_in())),
                ",".join(str(f.nnz_out(i)) for i in range(f.n_out())), str(f.sz_w()), src + ".json"]
        try:
            subprocess.run([TORCH_PYTHON, __file__, "--pytorch", src] + spec, timeout=TORCH_TIMEOUT, check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE))
            with open(spec[-1]) as fh:
                t[f.name()]["pytorch"] = json.load(fh)
        except (subprocess.SubprocessError, OSError) as e:
            print(f"  {f.name()} pytorch failed ({type(e).__name__}): nan")
            t[f.name()]["pytorch"] = [[math.nan] * N_EVALS for _ in N_ENVS_SWEEP]
        print(f"  {f.name()} pytorch: B=32768 {1e3 * sum(t[f.name()]['pytorch'][-1]) / N_EVALS:.1f} ms")


def pytorch_main(src, nnz_in, nnz_out, sz_w, out):
    """(under TORCH_PYTHON) upstream runPytorchBenchmark + timeFunction."""
    import importlib.util
    import torch
    sys.setrecursionlimit(10000)
    spec = importlib.util.spec_from_file_location("fn_PT", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nnz_in, nnz_out = [int(s) for s in nnz_in.split(",")], [int(s) for s in nnz_out.split(",")]
    rows = []
    for N in N_ENVS_SWEEP:
        inputs = [torch.rand(N, n, device="cuda", dtype=torch.double).contiguous() for n in nnz_in]
        outputs = [torch.zeros(N, n, device="cuda", dtype=torch.double).contiguous() for n in nnz_out]
        work = torch.zeros(N, int(sz_w), device="cuda", dtype=torch.double).contiguous()
        row = []
        for _ in range(N_EVALS):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            torch.vmap(mod.evaluate, out_dims=None)(outputs, inputs, work)
            end.record()
            torch.cuda.synchronize()
            row.append(start.elapsed_time(end) / 1000)
        rows.append(row)
        print(f"    pytorch N={N}: {1e3 * sum(row) / N_EVALS:.2f} ms", flush=True)
    with open(out, "w") as fh:
        json.dump(rows, fh)


# ---------------------------------------------------------------------------
# upstream plot_benchmarking.m
# ---------------------------------------------------------------------------
def _pooled_mean(runs, fn, method):            # per batch size: mean over evals and runs
    import numpy as np
    return np.nanmean(np.concatenate([np.asarray(r["t"][fn][method], float) for r in runs], axis=1), axis=1)


def plot(extra=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    runs = []
    for k in sorted(os.listdir(os.path.join(RESULTS, "runs")), key=int):
        if os.path.exists(os.path.join(RESULTS, "runs", k, "orig_bench.json")):
            with open(os.path.join(RESULTS, "runs", k, "orig_bench.json")) as fh:
                runs.append(json.load(fh))
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    series = [("cusadi", "g", 1.0, "  CusADi (ours)"), ("pytorch", "#EB5036", 0.75, "  Pytorch"),
              ("parallel_cpu", "#332288", 0.5, "  CPU (parallel)")]
    if extra:
        series.append(("cusadi2", "#0072BD", 0.3, "  cusadi2 (level-parallel)"))
    markers = ["o", "+", (6, 2, 0), "s", "D"]
    bw, x = 0.17, np.arange(1, len(N_ENVS_SWEEP) + 1)
    for ax, base, suffix in ((axes[0], "serial_cpu", ""), (axes[1], "serial_cpu_transfer", "_transfer")):
        for i, fn in enumerate(FNS):
            pos = x + (i - (len(FNS) - 1) / 2) * bw
            top = np.zeros(len(x))
            for method, color, w, _ in series:
                m = method + suffix if method == "parallel_cpu" else method
                s = np.log10(_pooled_mean(runs, fn, base) / _pooled_mean(runs, fn, m))
                ax.bar(pos, s, w * bw, color=color, edgecolor=(0, 0, 0, 0.2), zorder=2)
                top = np.fmax(top, s)
            ax.plot(pos, top + 0.25, markers[i], mec="k", mfc="none", ms=5, ls="none", zorder=3)
        ax.set_yticks(range(-5, 6), [f"$10^{{{k}}}$" for k in range(-5, 6)])
        ax.set_xticks(x, [str(n) for n in N_ENVS_SWEEP])
        ax.set_xlim(9.43, len(x) + 0.5)
        ax.set_ylim(-4, 5)
        ax.grid(axis="y", zorder=0)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Speedup (x)")
    ax = axes[0]
    l1 = ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _, _ in series], [n for _, _, _, n in series],
                   title="Evaluation", loc="lower right")
    ax.add_artist(l1)
    ax.legend([plt.Line2D([], [], marker=m, mec="k", mfc="none", ls="none") for m in markers],
              ["1E1", "1E2", "1E3", "1E4", "1E5"], title="Num. instr.", loc="upper left")
    name = "fig_benchmark_cusadi2" if extra else "fig_benchmark"
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS, f"{name}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"wrote {RESULTS}/{name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--plot", action="store_true", help="only re-plot existing results")
    ap.add_argument("--pytorch", nargs=5, metavar=("PT_PY", "NNZ_IN", "NNZ_OUT", "SZ_W", "OUT"),
                    help="(internal, under CUSADI2_TORCH_PYTHON) time one generated torch function")
    args = ap.parse_args()
    if args.pytorch:
        return pytorch_main(*args.pytorch)
    if not args.plot:
        fns = functions()
        print("=== build ===")
        lib, plan_dir, comp = build_gpu(fns)
        cpu_dir = build_cpu(fns)
        for k in range(args.runs):
            out = os.path.join(RESULTS, "runs", str(k))
            os.makedirs(out, exist_ok=True)
            print(f"##### run {k + 1} / {args.runs} -> {out}")
            t = {f.name(): {m: [] for m in ["cusadi", "cusadi2", "pytorch"] + CPU_METHODS} for f in fns}
            run_gpu(lib, plan_dir, fns, t)
            run_cpu(cpu_dir, fns, t)
            run_pytorch(fns, t)
            with open(os.path.join(out, "orig_bench.json"), "w") as fh:
                json.dump({"n_envs": N_ENVS_SWEEP, "n_instr": {f.name(): f.n_instructions() for f in fns},
                           "compile_s": comp, "torch_python": TORCH_PYTHON, "t": t}, fh)
    plot()
    plot(extra=True)


if __name__ == "__main__":
    main()
