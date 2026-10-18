"""Print the closed-loop summaries of results/closed_loop_ps*.json."""
import glob, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = []
for f in sorted(glob.glob(str(HERE / "results" / "closed_loop_ps*.json"))):
    d = json.load(open(f))
    keys = ("seed", "n_iter", "waypoints_reached", "waypoints_timed_out", "time_to_reach_mean_s", "time_to_reach_max_s", "h_min_all", "v_max",
            "ours_ms_median", "ours_ms_p99", "plan_kept_frac", "improved_iterations_mean", "cost_start_mean", "cost_min_mean")
    print(Path(f).name, {k: (round(d[k], 3) if isinstance(d[k], float) else d[k]) for k in keys if k in d})
    print("   events:", [(e["wp"], round(e["t_end"] - e["t_start"], 2), e["reached"]) for e in d["events"]])
    print("   h_min groups:", {k: round(v, 4) for k, v in d["h_min"].items()})
