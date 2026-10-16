"""Run the whole profiling campaign (each JAX configuration in its own process).

    python profiling/run_all.py              # everything, skipping configs already in the CSVs
    python profiling/run_all.py --force      # re-run everything (CSV rows appended; dedupe in make_results)
    python profiling/run_all.py --only gpu   # subset: gpu | cpu | throughput | trace

Order: GPU sweeps (qpax f64, f32; both problems), CPU batch-1 (elastiqp,
qpax f64, qpax f32; both problems), CPU N-process throughput, GPU trace,
then make_results.py. Log: results/run_all.log.
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PY = sys.executable

GPU_BATCHES = [1, 16, 128, 1024, 4096]


def done_configs(path, keys):
    if not Path(path).exists():
        return set()
    with open(path) as f:
        return {tuple(r[k] for k in keys) for r in csv.DictReader(f)}


def run(cmd, log):
    line = f"\n$ {' '.join(cmd)}   [{time.strftime('%H:%M:%S')}]\n"
    print(line, end="", flush=True)
    log.write(line)
    log.flush()
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(HERE.parent))
    for out in p.stdout:
        if any(s in out for s in ("Warning", "check_env", "Consider using", "Memory-optimized", "optional, but")):
            continue
        print("  " + out, end="", flush=True)
        log.write(out)
        log.flush()
    p.wait()
    msg = f"  -> exit {p.returncode} after {time.time()-t0:.0f}s\n"
    print(msg, end="", flush=True)
    log.write(msg)
    log.flush()
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["gpu", "cpu", "throughput", "trace", "report"], default=None)
    ap.add_argument("--problems", nargs="+", default=["panda", "g1"])
    ap.add_argument("--seconds", type=float, default=10.0, help="throughput window per piece")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    log = open(RESULTS / "run_all.log", "a")
    log.write(f"\n==== run_all {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(args)}\n")

    timing_csv = RESULTS / "raw_timings.csv"
    done = set() if args.force else done_configs(timing_csv, ("problem", "backend", "platform", "dtype"))
    tp_done = set() if args.force else done_configs(RESULTS / "cpu_throughput.csv", ("problem", "backend", "dtype"))

    def want(section):
        return args.only in (None, section)

    if want("gpu"):
        for prob in args.problems:
            for dtype in ("f64", "f32"):
                if (prob, "qpax", "gpu", dtype) in done:
                    continue
                run([PY, str(HERE / "bench.py"), "--problem", prob, "--backend", "qpax", "--platform", "gpu",
                     "--dtype", dtype, "--batches", *map(str, GPU_BATCHES)], log)
    if want("cpu"):
        for prob in args.problems:
            for backend, dtype in (("elastiqp", "f64"), ("qpax", "f64"), ("qpax", "f32")):
                if (prob, backend, "cpu", dtype) in done:
                    continue
                run([PY, str(HERE / "bench.py"), "--problem", prob, "--backend", backend, "--platform", "cpu",
                     "--dtype", dtype, "--batches", "1"], log)
    if want("throughput"):
        for prob in args.problems:
            for backend, dtype in (("elastiqp", "f64"), ("qpax", "f64"), ("qpax", "f32")):
                if (prob, backend, dtype) in tp_done:
                    continue
                run([PY, str(HERE / "cpu_throughput.py"), "--problem", prob, "--backend", backend, "--dtype", dtype,
                     "--seconds", str(args.seconds)], log)
    if want("trace"):
        if args.force or not (RESULTS / "trace_g1_summary.json").exists() or args.only == "trace":
            run([PY, str(HERE / "trace_g1.py"), "--batch", "1024", "--calls", "5"], log)
    if want("report") or args.only is None:
        run([PY, str(HERE / "sysinfo_dump.py")], log)
        run([PY, str(HERE / "make_results.py")], log)
    log.write(f"==== finished {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
