// Opti -> cusadi: C++ equivalent of the modeling side.
//
// This reproduces steps 1-2 of opti_to_cusadi.py -- the CasADi modeling layer,
// which has a full C++ API including Opti:
//
//   1. FORMULATE  build the same double-integrator OCP with the Opti stack.
//   2. EXTRACT    pull the sparse constraint Jacobian dg/dz out as a Function
//                 and serialize it to the same .casadi file.
//
// Step 3 (batched GPU evaluation via CusadiFunction2) is Python-only today:
// it depends on the level-plan extraction in cusadi/level_parallel/dag.py,
// which has no C++ counterpart. The .casadi file written here is the
// language-agnostic handoff -- run the Python script on the same file to
// batch it on the GPU:
//
//   ./build/opti_to_cusadi ../../build/casadi_functions
//   CUDA_VISIBLE_DEVICES=1 python opti_to_cusadi.py   # loads opti_con_jac.casadi
//
// Build: see CMakeLists.txt and README.md in this directory.

#include <casadi/casadi.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace casadi;

int main(int argc, char** argv) {
    const int N = 10;        // horizon
    const double dt = 0.1;   // timestep
    const std::string fn_name = "opti_con_jac";

    // Output dir: argv[1] if given, else the repo's build/casadi_functions.
    std::string out_dir = (argc > 1) ? argv[1] : "build/casadi_functions";
    std::string path = out_dir + "/" + fn_name + ".casadi";

    // --- Step 1: formulate the OCP with Opti -----------------------------
    Opti opti;
    MX X = opti.variable(2, N + 1);   // state trajectory  [pos; vel]
    MX U = opti.variable(1, N);       // control trajectory
    MX x0 = opti.parameter(2);        // initial state -- varies per instance

    MX J = 0;
    for (int k = 0; k < N; ++k) {
        MX xdot = vertcat(std::vector<MX>{X(1, k), U(0, k)});
        opti.subject_to(X(Slice(), k + 1) == X(Slice(), k) + dt * xdot);  // defect
        J = J + X(0, k) * X(0, k) + U(0, k) * U(0, k);
    }
    opti.subject_to(X(Slice(), 0) == x0);                                 // IC
    opti.minimize(J);

    // --- Step 2: extract the sparse constraint Jacobian, save ------------
    // In C++ the Opti pieces are accessor methods: x(), p(), g().
    MX z = opti.x();
    MX p = opti.p();
    MX Jg = jacobian(opti.g(), z);    // sparse, fixed structure

    Function f(fn_name, {z, p}, {Jg}, {"z", "p"}, {"Jg"});

    // Opti builds an MXFunction; the level-parallel interpreter reads the scalar
    // SX instruction tape, so expand MX -> SX before serializing for cusadi.
    f = f.expand();
    f.save(path);

    std::cout << "Saved: " << path << "\n";
    std::cout << "  " << f << "\n";
    std::cout << "  n_in=" << f.n_in() << " n_out=" << f.n_out()
              << " n_instructions=" << f.n_instructions()
              << " nnz_out=" << f.nnz_out(0)
              << " (sparse " << f.size1_out(0) << "x" << f.size2_out(0) << ")\n";
    std::cout << "\nHandoff: batch this on the GPU from Python --\n"
              << "  CUDA_VISIBLE_DEVICES=1 python opti_to_cusadi.py\n";
    return 0;
}
