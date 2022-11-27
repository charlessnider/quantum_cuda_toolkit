// standards
#include <stdlib.h>

// input and outputs
#include <iostream>
#include <fstream>

// general utilities, types, etc
#include <cuComplex.h>
#include <string>
#include <assert.h>

// cuda stuff
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// solver and cublas
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// for code timing
#include <chrono>

// custom headers
#include "error_checking.h"
#include "custom_complex_float_arithmetic.h"
#include "read_write_matrix_utilities.h"
#include "quantum_cuda_toolkit.h"

// size of matrix in question
const int DIM = 4096;

int main(){

    /*
    // memory for matrices
    int dim_A = 10; int dim_B = 10;
    cuFloatComplex* h_A = new cuFloatComplex[dim_A * dim_A];
    cuFloatComplex* h_B = new cuFloatComplex[dim_B * dim_B];
    cuFloatComplex* h_C = new cuFloatComplex[dim_A * dim_B * dim_A * dim_B];
    cuFloatComplex* d_A; CUDA_CHECK(cudaMalloc(&d_A, dim_A * dim_A * sizeof(cuFloatComplex)));
    cuFloatComplex* d_B; CUDA_CHECK(cudaMalloc(&d_B, dim_B * dim_B * sizeof(cuFloatComplex)));
    cuFloatComplex* d_C; CUDA_CHECK(cudaMalloc(&d_C, dim_A * dim_B * dim_A * dim_B * sizeof(cuFloatComplex)));

    // load matrices A, B
    std::string a_name = "A"; std::string b_name = "B";
    read_array_from_file_C(h_A, a_name); read_array_from_file_C(h_B, b_name);

    // copy memory to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, dim_A * dim_A * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, dim_B * dim_B * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // start timing for performing product
    auto start = std::chrono::high_resolution_clock::now();

    // kronecker that ish
    int nblocks = dim_A * dim_A * dim_B * dim_B / 256;
    kron <<< nblocks + 1, 256 >>> (d_A, d_B, d_C, dim_A, dim_B);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // print time of execution
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "The total elapsed time to do the product was " << duration.count() << "s" << std::endl;

    // copy to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, dim_A * dim_B * dim_A * dim_B * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    // save to file
    std::string c_name = "C";
    write_matrix_to_file_C(h_C, c_name, dim_A * dim_B);

    // free memory
    delete [] h_A; delete [] h_B; delete [] h_C;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    */

    // start timing for loading hamiltonian
    auto start = std::chrono::high_resolution_clock::now();

    // get the hamiltonian
    cuFloatComplex* H = new cuFloatComplex[DIM * DIM];
    std::string h_name = "H";
    read_array_from_file_C(H, h_name);

    // print the time taken to get hamiltonian
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "The total elapsed time to fetch H was " << duration.count() << "s" << std::endl;

    // start timing for allocation
    start = std::chrono::high_resolution_clock::now();

    // host allocation for matrix to eigensolve, eigenvalues
    int dim = DIM;
    cuFloatComplex* h_A = new cuFloatComplex[dim * dim];
    cuFloatComplex* h_U = new cuFloatComplex[dim * dim];
    float* h_D = new float[dim]; // real valued eigenvalues since hermitian

    // device allocation for matrix to eigensolve, eigenvalues
    cuFloatComplex* d_A; CUDA_CHECK(cudaMalloc(&d_A, dim * dim * sizeof(cuFloatComplex)));
    float* d_D;          CUDA_CHECK(cudaMalloc(&d_D, dim * sizeof(float)));

    // copy hamiltonian to A on host and device
    memcpy(h_A, H, dim * dim * sizeof(cuFloatComplex));
    CUDA_CHECK(cudaMemcpy(d_A, H, dim * dim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // print the time taken to prepare dataStruct
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to prepare memory was " << duration.count() << "s" << std::endl;

    // start timing for eigensolving
    start = std::chrono::high_resolution_clock::now();

    // do the solving
    eigensolve(d_A, d_D, dim);

    // print the time taken to eigensolve
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to eigensolve was " << duration.count() << "s" << std::endl;

    // start timing for saving the result
    start = std::chrono::high_resolution_clock::now();

    // copy results to host
    CUDA_CHECK(cudaMemcpy(h_U, d_A, DIM * DIM * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D, d_D, DIM * sizeof(float), cudaMemcpyDeviceToHost));

    // write to files
    std::string d_name = "D"; std::string u_name = "U";
    write_vector_to_file_F(h_D, d_name, DIM);
    write_matrix_to_file_C(h_U, u_name, DIM);

    // print the time taken to eigensolve
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to fetch the result and write to a file was " << duration.count() << "s" << std::endl;

    // free all memory
    delete [] h_A; delete [] h_U; delete [] h_D; delete [] H;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_D));

    // return
    return 0;
}