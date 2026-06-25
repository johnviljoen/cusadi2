#!/usr/bin/env python
"""Emit plan headers for the native-backend example: a 1-axis LP function and a
2-axis stage function. Build-time only.

    python gen_example.py        # writes generated/{lp,stage}_example_plan.h
"""
import os

import casadi as ca

from cusadi.native import generate_native_plan
from cusadi.level_parallel.stage import StageMode, StageSpec

HERE = os.path.dirname(os.path.realpath(__file__))
GEN = os.path.join(HERE, "generated")


def main():
    # --- 1-axis example: f(x) -> (sin(x)+x^2, sumsqr(x)) ------------------
    x = ca.SX.sym("x", 4)
    f_lp = ca.Function("lp_example", [x], [ca.sin(x) + x**2, ca.sumsqr(x)])
    generate_native_plan(f_lp, os.path.join(GEN, "lp_example_plan.h"), prefix="LP_EXAMPLE")

    # --- 2-axis stage example: per-knot running + stage*shared term -------
    xk = ca.SX.sym("xk", 3)
    xk1 = ca.SX.sym("xk1", 3)
    p = ca.SX.sym("p", 2)
    f_stage = ca.Function("stage_example", [xk, xk1, p],
                          [xk1 - 0.9 * xk, ca.vertcat(xk[0] * p[0] + xk[1] * p[1])])
    in_specs = [StageSpec(StageMode.STAGE),
                StageSpec(StageMode.COUPLED, offset=1, span=1),
                StageSpec(StageMode.SHARED)]
    out_specs = [StageSpec(StageMode.STAGE), StageSpec(StageMode.STAGE)]
    generate_native_plan(f_stage, os.path.join(GEN, "stage_example_plan.h"),
                         input_specs=in_specs, output_specs=out_specs,
                         prefix="STAGE_EXAMPLE", n_ref=4)
    print(f"wrote {GEN}/lp_example_plan.h and stage_example_plan.h")


if __name__ == "__main__":
    main()
