import json, cusadi2
for dt in ("double","float"):
    info = json.load(open(f"humanoid_build/{dt}/build_info.json"))
    print(f"== {dt}: lib nvcc {info.pop('_nvcc_lib_s'):.1f}s")
    for n, d in info.items():
        p = cusadi2.read_plan(f"humanoid_build/{dt}/plans/{n}.plan")
        nl = len(p["level_widths"]); nn = len(p["node_ops"])
        print(f"  {n:38s} instr={d['n_instr']:6d} nodes={nn:6d} levels={nl:3d} avg/level={(nn/nl if nl else 0):6.1f} max_w={(max(p["level_widths"]) if nl else 0):5d} block={d['block_size']:3d} smem={d['peak_live_slots']*(8 if dt=='double' else 4)/1024:6.1f}KB plan={d['codegen_plan_s']:.2f}s unr_codegen={d['codegen_unrolled_s']:.2f}s nvcc={d['nvcc_unrolled_s']:.1f}s")
