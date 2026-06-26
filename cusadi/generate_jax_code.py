import os
import textwrap
import casadi as ca
import cusadi as cu


def generateJaxCode(f, filepath=None):
    """JAX counterpart of generatePytorchCode: emit one jax.numpy statement per CasADi
    instruction into a pure, jit-able function.

    The emitted `evaluate_<name>(inputs)` is the per-env version (work slots and
    inputs[i][j] are scalars); batch it with jax.vmap and wrap in jax.jit. Unlike the
    torch path it cannot mutate caller buffers under jit, so it allocates `work`
    (Python list, SSA) and `outputs` itself and returns `outputs`. Outputs are built
    functionally via .at[].set() (JAX arrays are immutable). dtype follows the inputs,
    so fp64 requires jax_enable_x64 in the caller.
    """
    print("Generating JAX code for CasADi function: ", f.name())
    codegen_filepath = filepath or os.path.join(cu.CUSADI_CODEGEN_DIR, f"{f.name()}.py")

    n_instr = f.n_instructions()
    nnz_in = [f.nnz_in(i) for i in range(f.n_in())]
    nnz_out = [f.nnz_out(i) for i in range(f.n_out())]
    n_w = f.sz_w()

    input_idx = [f.instruction_input(i) for i in range(n_instr)]
    output_idx = [f.instruction_output(i) for i in range(n_instr)]
    operations = [f.instruction_id(i) for i in range(n_instr)]
    const_instr = [f.instruction_constant(i) for i in range(n_instr)]

    codegen_strings = {}
    codegen_strings['header'] = "# ! AUTOMATICALLY GENERATED CODE FOR CUSADI\n"
    codegen_strings['includes'] = textwrap.dedent(
    '''
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import erf

    ''')
    codegen_strings["nnz_in"] = f"nnz_in = [{','.join(map(str, nnz_in))}]\n"
    codegen_strings["nnz_out"] = f"nnz_out = [{','.join(map(str, nnz_out))}]\n"
    codegen_strings["n_w"] = f"n_w = {n_w}\n\n"

    str_operations = "@jax.jit\n"
    str_operations += f"def evaluate_{f.name()}(inputs):"
    str_operations += "\n        work = [None] * n_w"
    str_operations += "\n        outputs = [jnp.zeros(n, dtype=inputs[0].dtype) for n in nnz_out]"

    for k in range(n_instr):
        op = operations[k]
        o_idx = output_idx[k]
        i_idx = input_idx[k]
        if op == ca.OP_CONST:
            str_operations += cu.OP_JAX_DICT[op] % (o_idx[0], const_instr[k])
        elif op == ca.OP_INPUT:
            str_operations += cu.OP_JAX_DICT[op] % (o_idx[0], i_idx[0], i_idx[1])
        elif op == ca.OP_OUTPUT:
            str_operations += cu.OP_JAX_DICT[op] % (o_idx[0], o_idx[0], o_idx[1], i_idx[0])
        elif op == ca.OP_SQ:
            str_operations += cu.OP_JAX_DICT[op] % (o_idx[0], i_idx[0], i_idx[0])
        elif cu.OP_JAX_DICT[op].count("%d") == 3:
            str_operations += cu.OP_JAX_DICT[op] % (o_idx[0], i_idx[0], i_idx[1])
        elif cu.OP_JAX_DICT[op].count("%d") == 2:
            str_operations += cu.OP_JAX_DICT[op] % (o_idx[0], i_idx[0])
        else:
            raise Exception('Unknown CasADi operation: ' + str(op))

    str_operations += "\n        return outputs\n"
    codegen_strings['jax_operations'] = str_operations

    with open(codegen_filepath, "w+") as codegen_file:
        for cg_str in codegen_strings.values():
            codegen_file.write(cg_str)
    print("JAX codegen complete for CasADi function: ", f.name())
    return codegen_filepath
