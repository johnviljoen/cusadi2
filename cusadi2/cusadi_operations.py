
# CUDA definitions for all CasADi atomic operations
#   Nearly all symbolic, scalar operations are included here.
#   Exhaustive list: https://github.com/casadi/casadi/blob/main/casadi/core/calculus.hpp

# CUDA libraries may need to be manually added for compilation:
#   export PATH="/usr/local/cuda-12.3/bin:$PATH"
#   export LD_LIBRARY_PATH="/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH"

import casadi as ca

# Single source of truth for every compute op's CUDA form, as an operand-placeholder
# expression: {0} is the first operand, {1} the second. From these two dtype-specific
# maps codegen derives BOTH envelopes the op can appear in:
#   - the unrolled statement (OP_CUDA_DICT_{DOUBLE,FLOAT}, built below) — operands are
#     work[env_idx + %d] memory slots, with a leading dest slot and trailing `;`
#   - the data-driven exec_op switch RHS (codegen._build_exec_op) — operands are the
#     register args a, b
# Keeping one expression per op kills the double<->float and unrolled<->data-driven
# divergence that previously let bugs (e.g. the undefined sign()) live in one copy but
# not the others. DOUBLE uses exp/2./0; FLOAT uses the f-suffixed intrinsics (expf/...)
# and float literals (2.f/0.f) so a float kernel does not silently promote to double.
# IO ops (CONST/INPUT/OUTPUT) are not op(a,b) expressions and are added when building
# the unrolled dicts; exec_op never sees them (the interpreter loop does the moves).
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
}


# Build the unrolled per-instruction statement dict from an OP_EXPR map: each operand
# placeholder becomes a work[env_idx + %d] slot and the result is assigned into a
# leading dest slot. The %d order is (dest, operands...) — matching how the unrolled
# body builders fill the tuple. The IO ops carry their own statement form (CONST also
# differs in literal suffix: %.16f vs %.16ff). Output is byte-identical to the former
# hand-written OP_CUDA_DICT_{DOUBLE,FLOAT}.
def _unrolled_stmt(expr):
    rhs = expr.replace("{0}", "work[env_idx + %d]").replace("{1}", "work[env_idx + %d]")
    return "\n        work[env_idx + %d] = " + rhs + ";"


def _build_cuda_dict(expr_map, const_stmt):
    d = {op: _unrolled_stmt(expr) for op, expr in expr_map.items()}
    d[ca.OP_CONST]  = const_stmt
    d[ca.OP_INPUT]  = "\n        work[env_idx + %d] = inputs[%d][idx * nnz_in[%d] + %d];"
    d[ca.OP_OUTPUT] = "\n        outputs[%d][idx * nnz_out[%d] + %d] = work[env_idx + %d];"
    return d


OP_CUDA_DICT_DOUBLE = _build_cuda_dict(OP_EXPR_DOUBLE, "\n        work[env_idx + %d] = %.16f;")
OP_CUDA_DICT_FLOAT  = _build_cuda_dict(OP_EXPR_FLOAT,  "\n        work[env_idx + %d] = %.16ff;")


OP_PYTORCH_DICT = {
    ca.OP_ASSIGN:              "\n        work[%d] = work[%d]",
    ca.OP_ADD:                 "\n        work[%d] = work[%d] + work[%d]",
    ca.OP_SUB:                 "\n        work[%d] = work[%d] - work[%d]",
    ca.OP_MUL:                 "\n        work[%d] = work[%d] * work[%d]",
    ca.OP_DIV:                 "\n        work[%d] = work[%d] / work[%d]",
    ca.OP_NEG:                 "\n        work[%d] = -work[%d]",
    ca.OP_EXP:                 "\n        work[%d] = torch.exp(work[%d])",
    ca.OP_LOG:                 "\n        work[%d] = torch.log(work[%d])",
    ca.OP_POW:                 "\n        work[%d] = torch.pow(work[%d], work[%d])",
    ca.OP_CONSTPOW:            "\n        work[%d] = torch.pow(work[%d], work[%d])",
    ca.OP_SQRT:                "\n        work[%d] = torch.sqrt(work[%d])",
    ca.OP_SQ:                  "\n        work[%d] = work[%d] * work[%d]",
    ca.OP_TWICE:               "\n        work[%d] = 2.*(work[%d])",
    ca.OP_SIN:                 "\n        work[%d] = torch.sin(work[%d])",
    ca.OP_COS:                 "\n        work[%d] = torch.cos(work[%d])",
    ca.OP_TAN:                 "\n        work[%d] = torch.tan(work[%d])",
    ca.OP_ASIN:                "\n        work[%d] = torch.asin(work[%d])",
    ca.OP_ACOS:                "\n        work[%d] = torch.acos(work[%d])",
    ca.OP_ATAN:                "\n        work[%d] = torch.atan(work[%d])",
    ca.OP_LT:                  "\n        work[%d] = work[%d] < work[%d]",
    ca.OP_LE:                  "\n        work[%d] = work[%d] <= work[%d]",
    ca.OP_EQ:                  "\n        work[%d] = work[%d] == work[%d]",
    ca.OP_NE:                  "\n        work[%d] = work[%d] != work[%d]",
    ca.OP_NOT:                 "\n        work[%d] = torch.logical_not(work[%d])",
    ca.OP_AND:                 "\n        work[%d] = torch.logical_and(work[%d], work[%d])",
    ca.OP_OR:                  "\n        work[%d] = torch.logical_or(work[%d], work[%d])",
    ca.OP_FLOOR:               "\n        work[%d] = torch.floor(work[%d])",
    ca.OP_CEIL:                "\n        work[%d] = torch.ceil(work[%d])",
    ca.OP_FMOD:                "\n        work[%d] = torch.fmod(work[%d], work[%d])",
    ca.OP_FABS:                "\n        work[%d] = torch.abs(work[%d])",
    ca.OP_SIGN:                "\n        work[%d] = torch.sign(work[%d])",
    ca.OP_COPYSIGN:            "\n        work[%d] = torch.copysign(work[%d], work[%d])",
    ca.OP_IF_ELSE_ZERO:        "\n        work[%d] = (work[%d] == 0) ? 0 : work[%d]",
    ca.OP_ERF:                 "\n        work[%d] = torch.erf(work[%d])",
    ca.OP_FMIN:                "\n        work[%d] = torch.fmin(work[%d], work[%d])",
    ca.OP_FMAX:                "\n        work[%d] = torch.fmax(work[%d], work[%d])",
    ca.OP_INV:                 "\n        work[%d] = 1./(work[%d])",
    ca.OP_SINH:                "\n        work[%d] = torch.sinh(work[%d])",
    ca.OP_COSH:                "\n        work[%d] = torch.cosh(work[%d])",
    ca.OP_TANH:                "\n        work[%d] = torch.tanh(work[%d])",
    ca.OP_ASINH:               "\n        work[%d] = torch.asinh(work[%d])",
    ca.OP_ACOSH:               "\n        work[%d] = torch.acosh(work[%d])",
    ca.OP_ATANH:               "\n        work[%d] = torch.atanh(work[%d])",
    ca.OP_ATAN2:               "\n        work[%d] = torch.atan2(work[%d], work[%d])",
    ca.OP_CONST:               "\n        work[%d] = %f",
    ca.OP_INPUT:               "\n        work[%d] = inputs[%d][%d]",
    ca.OP_OUTPUT:              "\n        outputs[%d][%d] = work[%d]",
}


# JAX counterpart of OP_PYTORCH_DICT: one jax.numpy statement per CasADi instruction,
# operating on a Python list `work` of per-env scalars (SSA-style — each slot written
# before read). Two structural differences from the torch dict, both forced by JAX
# array immutability: OP_OUTPUT writes through outputs[i].at[j].set(...) (4 %d, not 3)
# rather than item assignment, and OP_IF_ELSE_ZERO uses jnp.where instead of the
# C-ternary the torch dict carries. Trig/inverse-hyperbolic use numpy spelling
# (arcsin/arcsinh/arctan2); erf comes from jax.scipy.special. Comparisons/logicals
# stay vectorizable. The generated function is per-env and meant to be jax.vmap'd over
# the batch then jax.jit'd — the apples-to-apples baseline against torch.compile.
OP_JAX_DICT = {
    ca.OP_ASSIGN:              "\n        work[%d] = work[%d]",
    ca.OP_ADD:                 "\n        work[%d] = work[%d] + work[%d]",
    ca.OP_SUB:                 "\n        work[%d] = work[%d] - work[%d]",
    ca.OP_MUL:                 "\n        work[%d] = work[%d] * work[%d]",
    ca.OP_DIV:                 "\n        work[%d] = work[%d] / work[%d]",
    ca.OP_NEG:                 "\n        work[%d] = -work[%d]",
    ca.OP_EXP:                 "\n        work[%d] = jnp.exp(work[%d])",
    ca.OP_LOG:                 "\n        work[%d] = jnp.log(work[%d])",
    ca.OP_POW:                 "\n        work[%d] = jnp.power(work[%d], work[%d])",
    ca.OP_CONSTPOW:            "\n        work[%d] = jnp.power(work[%d], work[%d])",
    ca.OP_SQRT:                "\n        work[%d] = jnp.sqrt(work[%d])",
    ca.OP_SQ:                  "\n        work[%d] = work[%d] * work[%d]",
    ca.OP_TWICE:               "\n        work[%d] = 2.*(work[%d])",
    ca.OP_SIN:                 "\n        work[%d] = jnp.sin(work[%d])",
    ca.OP_COS:                 "\n        work[%d] = jnp.cos(work[%d])",
    ca.OP_TAN:                 "\n        work[%d] = jnp.tan(work[%d])",
    ca.OP_ASIN:                "\n        work[%d] = jnp.arcsin(work[%d])",
    ca.OP_ACOS:                "\n        work[%d] = jnp.arccos(work[%d])",
    ca.OP_ATAN:                "\n        work[%d] = jnp.arctan(work[%d])",
    ca.OP_LT:                  "\n        work[%d] = work[%d] < work[%d]",
    ca.OP_LE:                  "\n        work[%d] = work[%d] <= work[%d]",
    ca.OP_EQ:                  "\n        work[%d] = work[%d] == work[%d]",
    ca.OP_NE:                  "\n        work[%d] = work[%d] != work[%d]",
    ca.OP_NOT:                 "\n        work[%d] = jnp.logical_not(work[%d])",
    ca.OP_AND:                 "\n        work[%d] = jnp.logical_and(work[%d], work[%d])",
    ca.OP_OR:                  "\n        work[%d] = jnp.logical_or(work[%d], work[%d])",
    ca.OP_FLOOR:               "\n        work[%d] = jnp.floor(work[%d])",
    ca.OP_CEIL:                "\n        work[%d] = jnp.ceil(work[%d])",
    ca.OP_FMOD:                "\n        work[%d] = jnp.fmod(work[%d], work[%d])",
    ca.OP_FABS:                "\n        work[%d] = jnp.abs(work[%d])",
    ca.OP_SIGN:                "\n        work[%d] = jnp.sign(work[%d])",
    ca.OP_COPYSIGN:            "\n        work[%d] = jnp.copysign(work[%d], work[%d])",
    ca.OP_IF_ELSE_ZERO:        "\n        work[%d] = jnp.where(work[%d] == 0, 0.0, work[%d])",
    ca.OP_ERF:                 "\n        work[%d] = erf(work[%d])",
    ca.OP_FMIN:                "\n        work[%d] = jnp.fmin(work[%d], work[%d])",
    ca.OP_FMAX:                "\n        work[%d] = jnp.fmax(work[%d], work[%d])",
    ca.OP_INV:                 "\n        work[%d] = 1./(work[%d])",
    ca.OP_SINH:                "\n        work[%d] = jnp.sinh(work[%d])",
    ca.OP_COSH:                "\n        work[%d] = jnp.cosh(work[%d])",
    ca.OP_TANH:                "\n        work[%d] = jnp.tanh(work[%d])",
    ca.OP_ASINH:               "\n        work[%d] = jnp.arcsinh(work[%d])",
    ca.OP_ACOSH:               "\n        work[%d] = jnp.arccosh(work[%d])",
    ca.OP_ATANH:               "\n        work[%d] = jnp.arctanh(work[%d])",
    ca.OP_ATAN2:               "\n        work[%d] = jnp.arctan2(work[%d], work[%d])",
    ca.OP_CONST:               "\n        work[%d] = %f",
    ca.OP_INPUT:               "\n        work[%d] = inputs[%d][%d]",
    ca.OP_OUTPUT:              "\n        outputs[%d] = outputs[%d].at[%d].set(work[%d])",
}
