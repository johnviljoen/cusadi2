"""CasADi scalar ops: their CUDA form (one expression per op, per dtype) and a
Python form for the plan verifier.

{0} is the first operand, {1} the second. The kernel's exec_op switch is built
from these maps (kernel_template._render), so the double and float interpreters
cannot drift: FLOAT uses the f-suffixed intrinsics and float literals so a float
kernel never silently promotes to double. IO ops (CONST/INPUT/OUTPUT) are not
op(a, b) expressions; the interpreter loop does those moves.

Coverage is every scalar op an SX instruction stream can carry: CasADi's ids
0..43 and the five later scalar additions (ERFINV, LOG1P, EXPM1, HYPOT,
REMAINDER; ids above OP_OUTPUT). The rest of calculus.hpp is MX-level
(matrix ops, calls, PRINTME) and never reaches a stream; ``dag._instructions``
rejects anything outside SUPPORTED_OPS at extraction, since the kernel cannot
check op ids at runtime.

Exhaustive list: https://github.com/casadi/casadi/blob/main/casadi/core/calculus.hpp
"""
import math

import casadi as ca

OP_EXPR_DOUBLE = {
    ca.OP_ASSIGN:       "{0}",
    ca.OP_ADD:          "{0} + {1}",
    ca.OP_SUB:          "{0} - {1}",
    ca.OP_MUL:          "{0} * {1}",
    ca.OP_DIV:          "{0} / {1}",
    ca.OP_NEG:          "-{0}",
    ca.OP_EXP:          "exp({0})",
    ca.OP_LOG:          "log({0})",
    ca.OP_POW:          "pow({0}, {1})",
    ca.OP_CONSTPOW:     "pow({0}, {1})",
    ca.OP_SQRT:         "sqrt({0})",
    ca.OP_SQ:           "{0} * {0}",
    ca.OP_TWICE:        "2.*({0})",
    ca.OP_SIN:          "sin({0})",
    ca.OP_COS:          "cos({0})",
    ca.OP_TAN:          "tan({0})",
    ca.OP_ASIN:         "asin({0})",
    ca.OP_ACOS:         "acos({0})",
    ca.OP_ATAN:         "atan({0})",
    ca.OP_LT:           "{0} < {1}",
    ca.OP_LE:           "{0} <= {1}",
    ca.OP_EQ:           "{0} == {1}",
    ca.OP_NE:           "{0} != {1}",
    ca.OP_NOT:          "!{0}",
    ca.OP_AND:          "{0} && {1}",
    ca.OP_OR:           "{0} || {1}",
    ca.OP_FLOOR:        "floor({0})",
    ca.OP_CEIL:         "ceil({0})",
    ca.OP_FMOD:         "fmod({0}, {1})",
    ca.OP_FABS:         "fabs({0})",
    ca.OP_SIGN:         "({0} > 0) - ({0} < 0)",
    ca.OP_COPYSIGN:     "copysign({0}, {1})",
    ca.OP_IF_ELSE_ZERO: "({0} == 0) ? 0 : {1}",
    ca.OP_ERF:          "erf({0})",
    ca.OP_FMIN:         "fmin({0}, {1})",
    ca.OP_FMAX:         "fmax({0}, {1})",
    ca.OP_INV:          "1./({0})",
    ca.OP_SINH:         "sinh({0})",
    ca.OP_COSH:         "cosh({0})",
    ca.OP_TANH:         "tanh({0})",
    ca.OP_ASINH:        "asinh({0})",
    ca.OP_ACOSH:        "acosh({0})",
    ca.OP_ATANH:        "atanh({0})",
    ca.OP_ATAN2:        "atan2({0}, {1})",
    ca.OP_ERFINV:       "erfinv({0})",
    ca.OP_LOG1P:        "log1p({0})",
    ca.OP_EXPM1:        "expm1({0})",
    ca.OP_HYPOT:        "hypot({0}, {1})",
    ca.OP_REMAINDER:    "remainder({0}, {1})",
}

OP_EXPR_FLOAT = {
    ca.OP_ASSIGN:       "{0}",
    ca.OP_ADD:          "{0} + {1}",
    ca.OP_SUB:          "{0} - {1}",
    ca.OP_MUL:          "{0} * {1}",
    ca.OP_DIV:          "{0} / {1}",
    ca.OP_NEG:          "-{0}",
    ca.OP_EXP:          "expf({0})",
    ca.OP_LOG:          "logf({0})",
    ca.OP_POW:          "powf({0}, {1})",
    ca.OP_CONSTPOW:     "powf({0}, {1})",
    ca.OP_SQRT:         "sqrtf({0})",
    ca.OP_SQ:           "{0} * {0}",
    ca.OP_TWICE:        "2.f*({0})",
    ca.OP_SIN:          "sinf({0})",
    ca.OP_COS:          "cosf({0})",
    ca.OP_TAN:          "tanf({0})",
    ca.OP_ASIN:         "asinf({0})",
    ca.OP_ACOS:         "acosf({0})",
    ca.OP_ATAN:         "atanf({0})",
    ca.OP_LT:           "{0} < {1}",
    ca.OP_LE:           "{0} <= {1}",
    ca.OP_EQ:           "{0} == {1}",
    ca.OP_NE:           "{0} != {1}",
    ca.OP_NOT:          "!{0}",
    ca.OP_AND:          "{0} && {1}",
    ca.OP_OR:           "{0} || {1}",
    ca.OP_FLOOR:        "floorf({0})",
    ca.OP_CEIL:         "ceilf({0})",
    ca.OP_FMOD:         "fmodf({0}, {1})",
    ca.OP_FABS:         "fabsf({0})",
    ca.OP_SIGN:         "({0} > 0.f) - ({0} < 0.f)",
    ca.OP_COPYSIGN:     "copysignf({0}, {1})",
    ca.OP_IF_ELSE_ZERO: "({0} == 0.f) ? 0.f : {1}",
    ca.OP_ERF:          "erff({0})",
    ca.OP_FMIN:         "fminf({0}, {1})",
    ca.OP_FMAX:         "fmaxf({0}, {1})",
    ca.OP_INV:          "1.f/({0})",
    ca.OP_SINH:         "sinhf({0})",
    ca.OP_COSH:         "coshf({0})",
    ca.OP_TANH:         "tanhf({0})",
    ca.OP_ASINH:        "asinhf({0})",
    ca.OP_ACOSH:        "acoshf({0})",
    ca.OP_ATANH:        "atanhf({0})",
    ca.OP_ATAN2:        "atan2f({0}, {1})",
    ca.OP_ERFINV:       "erfinvf({0})",
    ca.OP_LOG1P:        "log1pf({0})",
    ca.OP_EXPM1:        "expm1f({0})",
    ca.OP_HYPOT:        "hypotf({0}, {1})",
    ca.OP_REMAINDER:    "remainderf({0}, {1})",
}

SUPPORTED_OPS = frozenset(OP_EXPR_DOUBLE)
assert SUPPORTED_OPS == frozenset(OP_EXPR_FLOAT)


def _erfinv(y: float) -> float:
    """Inverse error function: Giles' initial guess refined by Newton on math.erf."""
    if y <= -1.0 or y >= 1.0:
        return math.copysign(math.inf, y) if abs(y) == 1.0 else math.nan
    w = -math.log((1.0 - y) * (1.0 + y))
    if w < 5.0:
        w -= 2.5
        x = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503,
                  -0.00417768164, 0.246640727, 1.50140941):
            x = c + x * w
    else:
        w = math.sqrt(w) - 3.0
        x = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613,
                  0.00943887047, 1.00167406, 2.83297682):
            x = c + x * w
    x *= y
    for _ in range(3):
        x -= (math.erf(x) - y) / (2.0 / math.sqrt(math.pi) * math.exp(-x * x))
    return x

# Python form, for dag.verify_plan (simulated parallel execution on the host).
OP_PY = {
    ca.OP_ASSIGN:       lambda a, b: a,
    ca.OP_ADD:          lambda a, b: a + b,
    ca.OP_SUB:          lambda a, b: a - b,
    ca.OP_MUL:          lambda a, b: a * b,
    ca.OP_DIV:          lambda a, b: a / b,
    ca.OP_NEG:          lambda a, b: -a,
    ca.OP_EXP:          lambda a, b: math.exp(a),
    ca.OP_LOG:          lambda a, b: math.log(a),
    ca.OP_POW:          lambda a, b: math.pow(a, b),
    ca.OP_CONSTPOW:     lambda a, b: math.pow(a, b),
    ca.OP_SQRT:         lambda a, b: math.sqrt(a),
    ca.OP_SQ:           lambda a, b: a * a,
    ca.OP_TWICE:        lambda a, b: 2.0 * a,
    ca.OP_SIN:          lambda a, b: math.sin(a),
    ca.OP_COS:          lambda a, b: math.cos(a),
    ca.OP_TAN:          lambda a, b: math.tan(a),
    ca.OP_ASIN:         lambda a, b: math.asin(a),
    ca.OP_ACOS:         lambda a, b: math.acos(a),
    ca.OP_ATAN:         lambda a, b: math.atan(a),
    ca.OP_LT:           lambda a, b: float(a < b),
    ca.OP_LE:           lambda a, b: float(a <= b),
    ca.OP_EQ:           lambda a, b: float(a == b),
    ca.OP_NE:           lambda a, b: float(a != b),
    ca.OP_NOT:          lambda a, b: float(not a),
    ca.OP_AND:          lambda a, b: float(bool(a) and bool(b)),
    ca.OP_OR:           lambda a, b: float(bool(a) or bool(b)),
    ca.OP_FLOOR:        lambda a, b: math.floor(a),
    ca.OP_CEIL:         lambda a, b: math.ceil(a),
    ca.OP_FMOD:         lambda a, b: math.fmod(a, b),
    ca.OP_FABS:         lambda a, b: abs(a),
    ca.OP_SIGN:         lambda a, b: float(a > 0) - float(a < 0),
    ca.OP_COPYSIGN:     lambda a, b: math.copysign(a, b),
    ca.OP_IF_ELSE_ZERO: lambda a, b: 0.0 if a == 0 else b,
    ca.OP_ERF:          lambda a, b: math.erf(a),
    ca.OP_FMIN:         lambda a, b: min(a, b),
    ca.OP_FMAX:         lambda a, b: max(a, b),
    ca.OP_INV:          lambda a, b: 1.0 / a,
    ca.OP_SINH:         lambda a, b: math.sinh(a),
    ca.OP_COSH:         lambda a, b: math.cosh(a),
    ca.OP_TANH:         lambda a, b: math.tanh(a),
    ca.OP_ASINH:        lambda a, b: math.asinh(a),
    ca.OP_ACOSH:        lambda a, b: math.acosh(a),
    ca.OP_ATANH:        lambda a, b: math.atanh(a),
    ca.OP_ATAN2:        lambda a, b: math.atan2(a, b),
    ca.OP_ERFINV:       lambda a, b: _erfinv(a),
    ca.OP_LOG1P:        lambda a, b: math.log1p(a),
    ca.OP_EXPM1:        lambda a, b: math.expm1(a),
    ca.OP_HYPOT:        lambda a, b: math.hypot(a, b),
    ca.OP_REMAINDER:    lambda a, b: math.remainder(a, b),
}
assert frozenset(OP_PY) == SUPPORTED_OPS
