import csv, json, os, sys
rows = list(csv.DictReader(open("humanoid_latency.csv")))
cpu = json.load(open("humanoid_cpu.json")) if os.path.exists("humanoid_cpu.json") else {}
def get(dt, fn, cfg, B, col="latency_us"):
    for r in rows:
        if (r["dtype"], r["func"], r["config"], r["batch"]) == (dt, fn, cfg, str(B)): return float(r[col])
    return float("nan")
fns = sorted({r["func"] for r in rows}, key=lambda n: -get("float", n, "batch", 16384))
for dt in ("float", "double"):
    print(f"\n=== {dt}: kernel-time medians, us (lp = cusadi2 level-parallel, orig = original cusadi kernel) ===")
    print(f"{'function':38s} {'B=1 lp':>8s} {'B=1 orig':>9s} {'x':>5s} | {'B=64 lp':>8s} {'B=64 orig':>9s} {'x':>5s} | {'B=1024 lp':>10s} {'orig':>9s} {'x':>5s} | {'B=16384 lp':>11s} {'orig':>9s} {'x':>5s} | {'cpu1 us':>8s} {'cpu16@16k ms':>12s}")
    for n in fns:
        c = cpu.get(dt, {}).get(n, {})
        c1 = c.get("one_us", float("nan")); c16 = c.get("seq_us", {}).get("16", {}).get("16384", float("nan")) / 1e3
        cells = []
        for B in (1, 64, 1024, 16384):
            lp, orig = get(dt, n, "lp", B), get(dt, n, "batch", B)
            cells += [lp, orig, orig / lp]
        print(f"{n.replace('_humanoid_full_sf',''):38s} {cells[0]:8.1f} {cells[1]:9.1f} {cells[2]:5.2f} | {cells[3]:8.1f} {cells[4]:9.1f} {cells[5]:5.2f} | {cells[6]:10.1f} {cells[7]:9.1f} {cells[8]:5.2f} | {cells[9]:11.1f} {cells[10]:9.1f} {cells[11]:5.2f} | {c1:8.2f} {c16:12.2f}")
    print(f"\n--- {dt}: wall-clock from Python (ctypes) at B=1, us ---")
    for n in fns:
        print(f"{n.replace('_humanoid_full_sf',''):38s} lp {get(dt,n,'lp',1,'wall_us'):8.1f}   orig {get(dt,n,'batch',1,'wall_us'):8.1f}")
