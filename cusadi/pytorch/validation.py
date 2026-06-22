"""
Validation harness: CusADi GPU kernels vs CasADi CPU oracle.

Entry point: `validate_function(f, num_instances=..., ...) -> ValidationReport`.

Flow:
    1. Generate (or accept) a batch of CUDA double inputs.
    2. Run the CusADi-compiled kernel on GPU over all instances.
    3. Run the same CasADi Function on CPU, per instance, as the oracle.
    4. Compare sparsity structure, then element-wise values with atol/rtol.
    5. Report per-entry mismatches with (output, instance, row, col) provenance.

Used from C0 onward. At C0 both sides derive sparsity from the same CasADi
Function, so the sparsity check is a structural sanity check only; at C2+ the
GPU side will carry independently-emitted COO index arrays and the same check
becomes a real comparison without changes to this module.
"""

import dataclasses
from typing import List, Optional, Tuple

import casadi
import numpy as np
import torch

from cusadi.pytorch.cusadi_function import CusadiFunction


@dataclasses.dataclass
class Mismatch:
    output_idx: int
    instance_idx: int
    nnz_idx: int
    row: int
    col: int
    cpu_val: float
    gpu_val: float
    abs_err: float
    rel_err: float


@dataclasses.dataclass
class OutputResult:
    output_idx: int
    shape: Tuple[int, int]
    nnz: int
    sparsity_ok: bool
    max_abs_err: float
    max_rel_err: float
    num_mismatches: int
    mismatches: List[Mismatch]

    @property
    def passed(self) -> bool:
        return self.sparsity_ok and self.num_mismatches == 0


@dataclasses.dataclass
class ValidationReport:
    function_name: str
    num_instances: int
    atol: float
    rtol: float
    per_output: List[OutputResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.per_output)

    def format(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{status}] {self.function_name}  "
            f"instances={self.num_instances}  "
            f"atol={self.atol:g}  rtol={self.rtol:g}"
        ]
        for r in self.per_output:
            rstatus = "ok" if r.passed else "FAIL"
            lines.append(
                f"  out[{r.output_idx}] shape={r.shape} nnz={r.nnz}  {rstatus}  "
                f"sparsity={'ok' if r.sparsity_ok else 'FAIL'}  "
                f"max|err|={r.max_abs_err:.2e}  max rel={r.max_rel_err:.2e}  "
                f"mismatches={r.num_mismatches}"
            )
            for m in r.mismatches:
                lines.append(
                    f"    inst={m.instance_idx} row={m.row} col={m.col} k={m.nnz_idx}  "
                    f"cpu={m.cpu_val:+.15e} gpu={m.gpu_val:+.15e}  "
                    f"|err|={m.abs_err:.2e} rel={m.rel_err:.2e}"
                )
        return "\n".join(lines)


def _random_inputs(
    f: casadi.Function, num_instances: int, generator: torch.Generator, device: str
) -> List[torch.Tensor]:
    return [
        torch.randn(
            num_instances, f.nnz_in(i),
            generator=generator, device=device, dtype=torch.double,
        ).contiguous()
        for i in range(f.n_in())
    ]


def _cpu_oracle(f: casadi.Function, inputs: List[torch.Tensor]) -> List[np.ndarray]:
    """Per-instance CasADi CPU evaluation. Returns list of (num_instances, nnz_out[i]) arrays."""
    num_instances = inputs[0].shape[0]
    n_out = f.n_out()
    out = [np.zeros((num_instances, f.nnz_out(i)), dtype=np.float64) for i in range(n_out)]
    for n in range(num_instances):
        inputs_np = [
            inputs[i][n, :].detach().cpu().numpy().astype(np.float64)
            for i in range(f.n_in())
        ]
        results = f.call(inputs_np)
        for i in range(n_out):
            out[i][n, :] = np.asarray(results[i].nonzeros(), dtype=np.float64)
    return out


def _sparsity_triplets(f: casadi.Function) -> List[Tuple[np.ndarray, np.ndarray]]:
    triplets = []
    for i in range(f.n_out()):
        rows, cols = f.sparsity_out(i).get_triplet()
        triplets.append(
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))
        )
    return triplets


def _compare_coo_indices(
    oracle_rows: np.ndarray, oracle_cols: np.ndarray,
    gpu_rows: np.ndarray, gpu_cols: np.ndarray,
) -> bool:
    return (
        oracle_rows.shape == gpu_rows.shape
        and oracle_cols.shape == gpu_cols.shape
        and np.array_equal(oracle_rows, gpu_rows)
        and np.array_equal(oracle_cols, gpu_cols)
    )


def _compare_output(
    output_idx: int,
    cpu_values: np.ndarray,
    gpu_values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    shape: Tuple[int, int],
    atol: float,
    rtol: float,
    max_mismatches: int,
    coo_indices_ok: bool = True,
) -> OutputResult:
    expected_nnz = int(rows.shape[0])
    sparsity_ok = (
        coo_indices_ok
        and cpu_values.shape == gpu_values.shape
        and cpu_values.shape[1] == expected_nnz
        and rows.shape == cols.shape
    )
    if not sparsity_ok:
        return OutputResult(
            output_idx=output_idx,
            shape=shape,
            nnz=expected_nnz,
            sparsity_ok=False,
            max_abs_err=float("inf"),
            max_rel_err=float("inf"),
            num_mismatches=0,
            mismatches=[],
        )

    abs_err = np.abs(cpu_values - gpu_values)
    denom = np.maximum(np.abs(cpu_values), np.abs(gpu_values))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(denom > 0, abs_err / denom, 0.0)
    tol = atol + rtol * np.abs(cpu_values)
    bad = abs_err > tol

    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    max_rel = float(rel_err.max()) if rel_err.size else 0.0

    mismatches: List[Mismatch] = []
    if bad.any():
        bad_coords = np.argwhere(bad)
        errs = abs_err[bad_coords[:, 0], bad_coords[:, 1]]
        order = np.argsort(-errs)
        for j in order[:max_mismatches]:
            inst_n = int(bad_coords[j, 0])
            k = int(bad_coords[j, 1])
            mismatches.append(Mismatch(
                output_idx=output_idx,
                instance_idx=inst_n,
                nnz_idx=k,
                row=int(rows[k]),
                col=int(cols[k]),
                cpu_val=float(cpu_values[inst_n, k]),
                gpu_val=float(gpu_values[inst_n, k]),
                abs_err=float(abs_err[inst_n, k]),
                rel_err=float(rel_err[inst_n, k]),
            ))

    return OutputResult(
        output_idx=output_idx,
        shape=shape,
        nnz=expected_nnz,
        sparsity_ok=sparsity_ok,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        num_mismatches=int(bad.sum()),
        mismatches=mismatches,
    )


def validate_function(
    f: casadi.Function,
    num_instances: int = 8,
    inputs: Optional[List[torch.Tensor]] = None,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    seed: int = 0,
    max_mismatches: int = 20,
    gpu_values_override: Optional[List[np.ndarray]] = None,
) -> ValidationReport:
    """
    Validate a CusADi-compiled GPU kernel for `f` against CasADi's CPU oracle.

    Args:
        f: loaded casadi.Function; its compiled .so must exist in CUSADI_BUILD_DIR
           (unless gpu_values_override is used).
        num_instances: batch size.
        inputs: optional list of CUDA double tensors shaped (num_instances, nnz_in[i]).
                Random inputs are generated from `seed` if omitted.
        atol, rtol: entry-wise tolerance, `|cpu-gpu| <= atol + rtol*|cpu|`.
        max_mismatches: max reported mismatches per output (sorted by |err|).
        gpu_values_override: harness self-test hook. If provided, use these arrays
                as the 'GPU' side instead of running the kernel.

    Returns:
        ValidationReport whose `.passed` is True iff every output passes.
    """
    use_cuda = gpu_values_override is None
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the GPU kernel")

    device = "cuda" if use_cuda else "cpu"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if inputs is None:
        inputs = _random_inputs(f, num_instances, gen, device)
    elif inputs[0].shape[0] != num_instances:
        raise ValueError(
            f"inputs[0] has {inputs[0].shape[0]} instances, expected {num_instances}"
        )

    cpu_outs = _cpu_oracle(f, inputs)

    cusadi_fn = None
    if use_cuda:
        cusadi_fn = CusadiFunction(f, num_instances)
        cusadi_fn.evaluate(inputs)
        torch.cuda.synchronize()
        gpu_outs = [
            cusadi_fn.outputs_sparse[i].detach().cpu().numpy().astype(np.float64)
            for i in range(f.n_out())
        ]
    else:
        gpu_outs = [np.asarray(a, dtype=np.float64) for a in gpu_values_override]

    triplets = _sparsity_triplets(f)

    per_output: List[OutputResult] = []
    for i in range(f.n_out()):
        oracle_rows, oracle_cols = triplets[i]
        coo_ok = True
        if cusadi_fn is not None:
            gpu_r, gpu_c, _ = cusadi_fn.get_coo(i)
            coo_ok = _compare_coo_indices(
                oracle_rows, oracle_cols,
                gpu_r.cpu().numpy(), gpu_c.cpu().numpy(),
            )
        shape = (f.size1_out(i), f.size2_out(i))
        per_output.append(_compare_output(
            output_idx=i,
            cpu_values=cpu_outs[i],
            gpu_values=gpu_outs[i],
            rows=oracle_rows,
            cols=oracle_cols,
            shape=shape,
            atol=atol,
            rtol=rtol,
            max_mismatches=max_mismatches,
            coo_indices_ok=coo_ok,
        ))

    return ValidationReport(
        function_name=f.name(),
        num_instances=num_instances,
        atol=atol,
        rtol=rtol,
        per_output=per_output,
    )
