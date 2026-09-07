// The CusADi paper's CPU reference drivers (se-hwan/cusadi, src/benchmark_functions/
// evaluate_{serial,parallel}_cpu[_transfer].cpp), kept verbatim up to the two switches
// that upstream compiled as four separate files:
//   -DPARALLEL   the batch loop is an OpenMP parallel for
//   -DTRANSFER   the timed region also moves the arguments GPU->CPU and the results
//                CPU->GPU one cudaMalloc/cudaMemcpy per pointer (one double each,
//                as upstream) and frees them
// Usage: orig_cpu <fn_name> <fn.so> <N_batch>; prints the timed region in seconds.
#include <chrono>
#include <casadi/casadi.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <omp.h>
#ifdef TRANSFER
#include <cuda_runtime.h>

void argsCPUtoGPU(const std::vector<std::vector<const double *>> &arg, std::vector<std::vector<double *>> &d_arg, int N_batch) {
    for (int i = 0; i < N_batch; ++i) {
        int inner_size = arg[i].size();
        d_arg[i].resize(inner_size);
        for (int j = 0; j < inner_size; ++j) {
            const double *h_ptr = arg[i][j];
            size_t bytes = sizeof(double);
            cudaMalloc(&d_arg[i][j], bytes);
            cudaMemcpy(d_arg[i][j], h_ptr, bytes, cudaMemcpyHostToDevice);
        }
    }
}

void argsGPUtoCPU(std::vector<std::vector<double *>> &d_arg, std::vector<std::vector<const double *>> &arg, int N_batch) {
    for (int i = 0; i < N_batch; ++i) {
        int inner_size = arg[i].size();
        for (int j = 0; j < inner_size; ++j) {
            double *d_ptr = d_arg[i][j];
            double *h_ptr = const_cast<double *>(arg[i][j]);
            size_t bytes = sizeof(double);
            cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost);
        }
    }
}

void resultsCPUtoGPU(const std::vector<std::vector<double *>> &res, std::vector<std::vector<double *>> &d_res, int N_batch) {
    for (int i = 0; i < N_batch; ++i) {
        int inner_size = res[i].size();
        d_res[i].resize(inner_size);
        for (int j = 0; j < inner_size; ++j) {
            double *h_ptr = res[i][j];
            size_t bytes = sizeof(double);
            cudaMalloc(&d_res[i][j], bytes);
            cudaMemcpy(d_res[i][j], h_ptr, bytes, cudaMemcpyHostToDevice);
        }
    }
}
#endif

int main(int argc, char* argv[]) {
    using namespace casadi;
    Function f = external(argv[1], argv[2]);
    int N_batch = std::stoi(argv[3]);

    std::vector<std::vector<const double *>> arg;
    std::vector<std::vector<double *>> res;
    arg.resize(N_batch);
    res.resize(N_batch);
    std::vector<std::vector<double *>> d_arg(N_batch);
    std::vector<std::vector<double *>> d_res(N_batch);

    for (int k = 0; k < N_batch; ++k) {
        for (int i = 0; i < f.n_in() ; ++i) {
            Eigen::Matrix<double, Eigen::Dynamic, 1>* x_vec = new Eigen::Matrix<double, Eigen::Dynamic, 1>(f.nnz_in(i));
            x_vec->setRandom();
            arg[k].push_back(x_vec->data());
        }
        for (int i = 0; i < f.n_out() ; ++i) {
            Eigen::Matrix<double, Eigen::Dynamic, 1>* y_vec = new Eigen::Matrix<double, Eigen::Dynamic, 1>(f.nnz_out(i));
            y_vec->setZero();
            res[k].push_back(y_vec->data());
        }
    }
#ifdef TRANSFER
    argsCPUtoGPU(arg, d_arg, N_batch);
#endif

    auto start = std::chrono::high_resolution_clock::now();
#ifdef TRANSFER
    argsGPUtoCPU(d_arg, arg, N_batch);
#endif
#ifdef PARALLEL
    #pragma omp parallel for
#endif
    for (int k = 0; k < N_batch; ++k) {
        f(arg[k], res[k]);
    }
#ifdef TRANSFER
    resultsCPUtoGPU(res, d_res, N_batch);
    for (int i = 0; i < N_batch; ++i) {
        for (size_t j = 0; j < d_arg[i].size(); ++j) cudaFree(d_arg[i][j]);
        for (size_t j = 0; j < d_res[i].size(); ++j) cudaFree(d_res[i][j]);
    }
#endif
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << static_cast<double>(duration) / 1e6;
    return 0;
}
