import csv, json
rows = list(csv.DictReader(open("humanoid_latency.csv"))); cpu = json.load(open("humanoid_cpu.json"))
fns = sorted({r["func"] for r in rows})
for dt in ("float", "double"):
    print(f"=== {dt}: sum over all 12 functions (one control step's worth), kernel us ===")
    for B in (1, 64, 1024, 4096, 16384):
        s = {c: sum(float(r["latency_us"]) for r in rows if r["dtype"] == dt and r["batch"] == str(B) and r["config"] == c) for c in ("lp", "batch")}
        c1 = sum(cpu[dt][f]["one_us"] for f in fns); c16 = sum(cpu[dt][f]["seq_us"]["16"][str(B)] for f in fns)
        print(f"  B={B:6d}: cusadi2 lp {s['lp']:9.1f}   orig cusadi {s['batch']:9.1f}   ratio {s['batch']/s['lp']:5.2f}   | CPU 16thr for B problems {c16:10.1f}  (1 problem, 1 thr {c1:6.2f})")
