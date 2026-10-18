"""Closed-loop tuning sweep: run closed_loop.py (no video) over a list of configurations and seeds, sequentially, and
tabulate results/closed_loop_ps_<tag>_seed<s>.json.

    python sweep.py run  --grid 1 --seeds 1 2
    python sweep.py table [--grid 1]

Every configuration keeps the structure fixed (best sample applied directly, CBF only inside the rollouts, no re-filter,
cusadi2 evaluator, one tick <= 20 ms by the timing grid of results/timing_grid.json); only the horizon, sample count,
iteration count, knot noise (absolute std and per-knot scale) and the model parameters are tuned.
"""
import argparse, glob, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
# tag -> closed_loop.py arguments. Iteration counts are the ones that fit 20 ms at that (T, K) (timing_grid.py).
GRIDS = {
    1: {
        "T100_s010":      "--T 100 --K 256 --n-iter 1 --sigma 0.10",
        "T100_s020":      "--T 100 --K 256 --n-iter 1 --sigma 0.20",
        "T100_s010_k0":   "--T 100 --K 256 --n-iter 1 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1",
        "T100_s020_ramp": "--T 100 --K 256 --n-iter 1 --sigma 0.20 --knot-scale 0,0.5,1,1,1,1,1",
        "T100_mjpc":      "--T 100 --K 256 --n-iter 1",
        "T100_s010_kp30": "--T 100 --K 256 --n-iter 1 --sigma 0.10 --par kp=30",
        "T50_s010":       "--T 50 --K 256 --n-iter 2 --sigma 0.10",
        "T50_s020":       "--T 50 --K 256 --n-iter 2 --sigma 0.20",
        "T50_s010_k0":    "--T 50 --K 256 --n-iter 2 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1",
        "T50_s020_k0":    "--T 50 --K 256 --n-iter 2 --sigma 0.20 --knot-scale 0,1,1,1,1,1,1",
        "T50_s015_wv2":   "--T 50 --K 256 --n-iter 2 --sigma 0.15 --par w_v=2",
        "T50_K512_s015":  "--T 50 --K 512 --n-iter 1 --sigma 0.15",
        "T25_s010":       "--T 25 --K 256 --n-iter 5 --sigma 0.10",
        "T25_s020":       "--T 25 --K 256 --n-iter 5 --sigma 0.20",
        "T25_s010_k0":    "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1",
        "T25_K1024_s015": "--T 25 --K 1024 --n-iter 2 --sigma 0.15",
    },
    2: {   # around the grid-1 winner T25_s010_k0 (T 25, K 256, 5 iterations, sigma 0.10 rad, knot 0 frozen)
        "T25_s010_k0":       "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1",
        "T25_s007_k0":       "--T 25 --K 256 --n-iter 5 --sigma 0.07 --knot-scale 0,1,1,1,1,1,1",
        "T25_s015_k0":       "--T 25 --K 256 --n-iter 5 --sigma 0.15 --knot-scale 0,1,1,1,1,1,1",
        "T25_s010_ramp":     "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,0.5,1,1,1,1,1",
        "T25_s015_ramp":     "--T 25 --K 256 --n-iter 5 --sigma 0.15 --knot-scale 0,0.5,1,1,1,1,1",
        "T25_K512_s010_k0":  "--T 25 --K 512 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1",
        "T25_K1024_s010_k0": "--T 25 --K 1024 --n-iter 2 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1",
    },
    3: {   # cost / gain / exploration / horizon levers on the grid-2 base (T 25, K 256, 5 iterations, sigma 0.10, knot 0 frozen)
        "B_wpos100":   "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par w_pos=100",
        "B_wv05":      "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par w_v=0.5",
        "B_wv2":       "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par w_v=2",
        "B_wrot0":     "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par w_rot=0",
        "B_kp30":      "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=30",
        "B_kp80":      "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=80",
        "B_nonom":     "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --no-nominal",
        "T30_n4":      "--T 30 --K 256 --n-iter 4 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1",
        "T35_n3":      "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1",
    },
    4: {   # combinations of the grid-3 levers that helped on their own (kp 30, w_pos 100, T 35 with 3 iterations)
        "T35_kp30":          "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=30",
        "T35_wpos100_kp30":  "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=30,w_pos=100",
        "T25_wpos100_kp30":  "--T 25 --K 256 --n-iter 5 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=30,w_pos=100",
        "T35_wpos100":       "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par w_pos=100",
    },
    5: {   # around the grid-4 best (T 35, 3 iterations, kp 30, w_pos 100): barrier margin and the last timeouts
        "F_s007":     "--T 35 --K 256 --n-iter 3 --sigma 0.07 --knot-scale 0,1,1,1,1,1,1 --par kp=30,w_pos=100",
        "F_wv2":      "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=30,w_pos=100,w_v=2",
        "F_kp20":     "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=20,w_pos=100",
        "F_wpos150":  "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=30,w_pos=150",
        "F_T50_n2":   "--T 50 --K 256 --n-iter 2 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=30,w_pos=100",
    },
    6: {   # around F_kp20 (zero timeouts on 5 seeds): the barrier margin
        "G_proj96":   "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=20,w_pos=100 --proj-iters 96",
        "G_wpos80":   "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=20,w_pos=80",
        "G_s008":     "--T 35 --K 256 --n-iter 3 --sigma 0.08 --knot-scale 0,1,1,1,1,1,1 --par kp=20,w_pos=100",
        "G_wpos150":  "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=20,w_pos=150",
        "G_wu":       "--T 35 --K 256 --n-iter 3 --sigma 0.10 --knot-scale 0,1,1,1,1,1,1 --par kp=20,w_pos=100,w_u=1e-3",
    },
}


def run(grid, seeds, only=None):
    (HERE / "logs").mkdir(exist_ok=True)
    for tag, args in GRIDS[grid].items():
        if only and tag not in only:
            continue
        for s in seeds:
            out = HERE / "results" / f"closed_loop_ps_{tag}_seed{s}.json"
            if out.exists():
                print(f"skip {tag} seed {s} (exists)", flush=True); continue
            cmd = [PY, str(HERE / "closed_loop.py"), "--controller", "ps", "--no-video", "--seed", str(s), "--tag", f"_{tag}_seed{s}", *args.split()]
            with open(HERE / "logs" / f"sweep_{tag}_seed{s}.log", "w") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
            d = json.load(open(out)) if out.exists() else None
            print(f"{tag:16s} seed {s}: " + (f"reached {d['waypoints_reached']:2d} timed out {d['waypoints_timed_out']} h_min {d['h_min_all']:7.4f} v_max {d['v_max']:4.2f} "
                  f"t_reach {d['time_to_reach_mean_s'] or 0:.2f} tick {d['ours_ms_median']:.1f}/{d['ours_ms_p99']:.1f}" if d else f"FAILED rc {rc}"), flush=True)


def table(grid=None):
    rows = {}
    for f in sorted(glob.glob(str(HERE / "results" / "closed_loop_ps_*_seed*.json"))):
        d = json.load(open(f)); tag = Path(f).stem[len("closed_loop_ps_"):].rsplit("_seed", 1)[0]
        if grid is not None and tag not in GRIDS[grid]:
            continue
        rows.setdefault(tag, []).append(d)
    print(f"{'config':18s} {'seeds':5s} {'reached':>8s} {'timed out':>9s} {'h_min':>8s} {'v_max':>6s} {'t_reach':>8s} {'tick p99':>9s}  args")
    for tag, ds in sorted(rows.items(), key=lambda kv: (-sum(d["waypoints_reached"] for d in kv[1]) / len(kv[1]), sum(d["waypoints_timed_out"] for d in kv[1]))):
        n = len(ds)
        print(f"{tag:18s} {n:5d} {sum(d['waypoints_reached'] for d in ds) / n:8.1f} {sum(d['waypoints_timed_out'] for d in ds) / n:9.1f} "
              f"{min(d['h_min_all'] for d in ds):8.4f} {max(d['v_max'] for d in ds):6.2f} {sum((d['time_to_reach_mean_s'] or 6) for d in ds) / n:8.2f} "
              f"{max(d['ours_ms_p99'] for d in ds):9.1f}  {GRIDS.get(grid, {}).get(tag, '') if grid else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["run", "table"]); ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2]); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.what == "run":
        run(a.grid, a.seeds, a.only); table(a.grid)
    else:
        table(a.grid)
