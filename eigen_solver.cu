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

// size of matrix in question
const int DIM = 4096;

// DATA STRUCTURES
struct eigsolveStruct{

    // host pointers
    cuFloatComplex* h_A;    // matrix to diagonalize
    cuFloatComplex* h_U;    // matrix of eigenvectors
    float* h_D;             // vector of eigenvalues

    // device pointers
    cuFloatComplex* d_A;    // matrix to diagonalize (eigenvectors overwrite d_A, no need for d_U)
    float* d_D;             // vector of eigenvalues

    // cuSolver handle
    cusolverDnHandle_t cusolverH;

    // bits and pieces for solver
    int* devInfo;     // indicates type of success/failure of solver
    int n;            // dimension of matrix
    int lda;          // leading dimension of matrix
    
    // errors
    cusolverStatus_t cusolverStat;
    cudaError_t cudaStat;
};

eigsolveStruct prepare_eigsolveStruct(cuFloatComplex* H, const int dim){

    // create new data structure
    eigsolveStruct x;

    // host allocation
    x.h_A = new cuFloatComplex[dim * dim];
    x.h_U = new cuFloatComplex[dim * dim];
    x.h_D = new float[dim]; // real valued eigenvalues since hermitian

    // device allocation
    CUDA_CHECK(cudaMalloc(&x.d_A, dim * dim * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&x.d_D, dim * sizeof(float)));

    // copy values of H to h_A, d_A
    memcpy(x.h_A, H, dim * dim * sizeof(cuFloatComplex));
    CUDA_CHECK(cudaMemcpy(x.d_A, H, dim * dim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // initialize some values for the solver
    x.n = dim;
    x.lda = dim;

    // allocate some values for the solver
    CUDA_CHECK(cudaMalloc(&x.devInfo, sizeof(int)));

    // initialize errors
    x.cusolverStat = CUSOLVER_STATUS_SUCCESS;
    x.cudaStat = cudaSuccess;

    // create the solver handle
    CUSOLVER_CHECK(cusolverDnCreate(&x.cusolverH));

    // return the structure with data allocated
    return x;
}

// EIGENSOLVER
void eigensolve(eigsolveStruct x){

    // parameters for the solver here
    int lwork = 0;
    cuFloatComplex* work = NULL;

    // whether to save the eigenvectors
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    // probably solver only uses part of matrix (since symmetric) & this specifics which half you are supplying
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // compute amount of memory needed, and allocate on device
    CUSOLVER_CHECK(cusolverDnCheevd_bufferSize(x.cusolverH, jobz, uplo, x.n, x.d_A, x.lda, x.d_D, &lwork));
    CUDA_CHECK(cudaMalloc(&work, lwork * sizeof(cuFloatComplex)));

    // (palpatine voice) do it
    CUSOLVER_CHECK(cusolverDnCheevd(x.cusolverH, jobz, uplo, x.n, x.d_A, x.lda, x.d_D, work, lwork, x.devInfo));
}

// MATRIX OPERATIONS
__global__ void kron(cuFloatComplex* A, cuFloatComplex* B, cuFloatComplex* C, int dim_A, int dim_B){

    // one thread gets each element of the product
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dim_A * dim_B * dim_A * dim_B){

        // get indices of matrix C from flattened form
        int j_C = idx % (dim_A * dim_B);
        int i_C = (idx - j_C) / (dim_A * dim_B);

        // index of matrix A to fetch
        int i_B = i_C % dim_B;
        int j_B = j_C % dim_B; 

        // index of matrix B to fetch
        int i_A = (i_C - i_B) / dim_B;
        int j_A = (j_C - j_B) / dim_B;

        // C(i,j) = A(i_A, j_A) * B(i_B, j_B)
        C[idx] = my_cuCmulf(A[dim_A * i_A + j_A], B[dim_B * i_B + j_B]);
    }

}

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

    // create the data structure
    eigsolveStruct x = prepare_eigsolveStruct(H, DIM);

    // print the time taken to prepare dataStruct
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to prepare the dataStruct was " << duration.count() << "s" << std::endl;

    // start timing for eigensolving
    start = std::chrono::high_resolution_clock::now();

    // do the solving
    eigensolve(x);

    // print the time taken to eigensolve
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to eigensolve was " << duration.count() << "s" << std::endl;

    // start timing for saving the result
    start = std::chrono::high_resolution_clock::now();

    // copy results to host
    CUDA_CHECK(cudaMemcpy(x.h_U, x.d_A, DIM * DIM * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x.h_D, x.d_D, DIM * sizeof(float), cudaMemcpyDeviceToHost));

    // write to files
    std::string d_name = "D"; std::string u_name = "U";
    write_vector_to_file_F(x.h_D, d_name, DIM);
    write_matrix_to_file_C(x.h_U, u_name, DIM);

    // print the time taken to eigensolve
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to fetch the result and write to a file was " << duration.count() << "s" << std::endl;

    // free all memory
    delete [] x.h_A; delete [] x.h_U; delete [] x.h_D; delete [] H;
    CUDA_CHECK(cudaFree(x.d_A)); CUDA_CHECK(cudaFree(x.d_D)); CUDA_CHECK(cudaFree(x.devInfo));

    // return
    return 0;
}