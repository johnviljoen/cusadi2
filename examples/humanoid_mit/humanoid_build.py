"""Build cusadi2 plans + original-cusadi unrolled kernels + latency_lib.so for the
MIT humanoid CasADi functions (from hojae-io/cusadi dev_RAL2025). One-stage
functions: configs are 'lp' (cusadi2 level kernel) and 'batch' (original cusadi
kernel emitted by cusadi2.unrolled)."""
import glob, json, os, subprocess, sys, time
import casadi as ca
import cusadi2

HERE = os.path.dirname(os.path.abspath(__file__))
FN_DIR = os.path.join(HERE, "humanoid_fns")
REPO = "/home/jovi/code/cusecmpc/cusadi2"
LAT_CU = os.path.join(REPO, "examples/paper_results/latency.cu")
NVCC = os.path.join(sys.prefix, "bin", "nvcc")
cc = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", "0"]).decode().split()[0]
ARCH = "-arch=sm_" + cc.replace(".", "")

fns = [ca.Function.load(p) for p in sorted(glob.glob(os.path.join(FN_DIR, "*.casadi")))]
for dtype in ("double", "float"):
    out = os.path.join(HERE, "humanoid_build", dtype)
    plan_dir, unr_dir = os.path.join(out, "plans"), os.path.join(out, "unrolled")
    os.makedirs(plan_dir, exist_ok=True); os.makedirs(unr_dir, exist_ok=True)
    cusadi2.write_kernel_source(os.path.join(out, "kernel.cu"), dtype=dtype)
    info, unrolled = {}, []
    for fn in fns:
        name = fn.name()
        t0 = time.perf_counter()
        cusadi2.write_plan(fn, os.path.join(plan_dir, f"{name}.plan"))
        t_plan = time.perf_counter() - t0
        t0 = time.perf_counter()
        src = cusadi2.write_unrolled_source(fn, os.path.join(unr_dir, f"{name}.cu"), dtype=dtype)
        t_unr = time.perf_counter() - t0
        p = cusadi2.read_plan(os.path.join(plan_dir, f"{name}.plan"))
        info[name] = dict(n_instr=fn.n_instructions(), sz_w=fn.sz_w(), n_levels=p.get("n_levels"),
                          peak_live_slots=p["peak_live_slots"], block_size=p["block_size"],
                          codegen_plan_s=t_plan, codegen_unrolled_s=t_unr)
        unrolled.append((name, src))
        print(f"[{dtype}] {name:40s} instr={fn.n_instructions():6d} levels={p.get('n_levels')} "
              f"smem={p['peak_live_slots']*8/1024:.1f}KB block={p['block_size']} plan={t_plan:.2f}s", flush=True)
    with open(os.path.join(out, "unrolled_table.h"), "w") as f:
        for n, _ in unrolled:
            f.write(f'extern "C" int {n}_unrolled_sz_w();\nextern "C" int {n}_unrolled_n_stages();\n'
                    f'extern "C" int {n}_unrolled_launch(const S* const*, S* const*, S*, int, int, cudaStream_t);\n')
        f.write("static const UnrolledFn UNROLLED[] = {\n" + "".join(
            f'    {{"{n}", {n}_unrolled_sz_w, {n}_unrolled_n_stages, {n}_unrolled_launch}},\n' for n, _ in unrolled) + "};\n")
    objs = []
    for n, src in unrolled:
        obj = src[:-3] + ".o"; t0 = time.perf_counter()
        subprocess.run([NVCC, "-std=c++17", "-O3", ARCH, "-Xcompiler", "-fPIC", "-c", src, "-o", obj], check=True)
        info[n]["nvcc_unrolled_s"] = time.perf_counter() - t0; objs.append(obj)
        print(f"[{dtype}] nvcc {n}: {info[n]['nvcc_unrolled_s']:.1f}s", flush=True)
    t0 = time.perf_counter()
    subprocess.run([NVCC, "-std=c++17", "-O3", ARCH, f"-DCUSADI2_SCALAR={dtype}", f"-I{cusadi2.INCLUDE_DIR}", f"-I{out}",
                    LAT_CU, os.path.join(out, "kernel.cu")] + objs +
                   ["-shared", "-Xcompiler", "-fPIC", "-DCUSADI2_BENCH_LIB", "-o", os.path.join(out, "latency_lib.so")], check=True)
    info["_nvcc_lib_s"] = time.perf_counter() - t0
    print(f"[{dtype}] nvcc lib: {info['_nvcc_lib_s']:.1f}s", flush=True)
    json.dump(info, open(os.path.join(out, "build_info.json"), "w"), indent=2)
print("BUILD DONE")
