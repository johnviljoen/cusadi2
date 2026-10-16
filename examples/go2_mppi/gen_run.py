"""Write a run directory for mppi_rollout: robots.f64, x0.f64 and par.f64 (one parameter row per robot).

    python gen_run.py <run_dir> <flat|sweep|course|east|north|west|south> [name=value | name=v0,v1,... ...]

Overrides are go2_model.PARAMS entries; a comma list gives each robot its own value (parameter sweeps
on the "sweep" scenario, where all robots have the same task).
"""
import os
import sys

import numpy as np

import course
import go2_model as G

run, scenario = sys.argv[1], sys.argv[2]
rob = (course.side_robots(scenario) if scenario in ("east", "north", "west", "south") else
       {"flat": course.flat_robots, "sweep": lambda: course.flat_robots(same=True), "course": course.robots}[scenario]())
par = np.tile(G.PAR_DEFAULT, (len(rob), 1))
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    par[:, G.PAR[k]] = [float(t) for t in v.split(",")] if "," in v else float(v)
os.makedirs(run, exist_ok=True)
rob.astype(np.float64).tofile(os.path.join(run, "robots.f64"))
np.array([G.initial_state(*r[:3]) for r in rob]).tofile(os.path.join(run, "x0.f64"))
par.tofile(os.path.join(run, "par.f64"))
