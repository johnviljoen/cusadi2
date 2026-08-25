"""
CLI driver for the CusADi validation harness (C0).

Usage:
    python run_validation.py --fn test --num_instances 8

The named function must already have a compiled .so in CUSADI_BUILD_DIR,
i.e. you must have run `python run_codegen.py --fn <name>` and built it first.
"""

import argparse
import os
import sys

import casadi

from cusadi import CUSADI_FUNCTION_DIR
from cusadi.pytorch.validation import validate_function


def main():
    parser = argparse.ArgumentParser(
        description="Validate a CusADi-compiled GPU kernel against CasADi's CPU oracle."
    )
    parser.add_argument("--fn", default="test",
                        help="Function name (file <name>.casadi). Default: test")
    parser.add_argument("--num_instances", type=int, default=8,
                        help="Batch instances to validate. Default: 8")
    parser.add_argument("--atol", type=float, default=1e-10,
                        help="Absolute tolerance. Default: 1e-10")
    parser.add_argument("--rtol", type=float, default=1e-10,
                        help="Relative tolerance. Default: 1e-10")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for random inputs. Default: 0")
    parser.add_argument("--benchmark", action="store_true",
                        help="(kept for compatibility) all generated functions, incl. fn_1eX, "
                             "now live in build/casadi_functions")
    args = parser.parse_args()

    fn_dir = CUSADI_FUNCTION_DIR
    fn_path = os.path.join(fn_dir, f"{args.fn}.casadi")
    if not os.path.exists(fn_path):
        print(f"Function file not found: {fn_path}", file=sys.stderr)
        sys.exit(2)

    f = casadi.Function.load(fn_path)  # type: ignore[arg-type]
    report = validate_function(
        f,
        num_instances=args.num_instances,
        atol=args.atol,
        rtol=args.rtol,
        seed=args.seed,
    )
    print(report.format())
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
