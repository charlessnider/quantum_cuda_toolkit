#include <iostream>
#include <complex>
#include <cuda.h>
#include <cuComplex.h>
#include <stdlib.h>
#include <ctime>
#include <time.h>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <assert.h>
#include <chrono>
#include <typeinfo>
#include "device_launch_parameters.h"

// size of matrix in question
const int DIM = 1024;

// DATA STRUCTURES
struct dataStruct{

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

    /*
    // constructor and destructor
    dataStruct(){}
    dataStruct(cuFloatComplex* H, const int dim){

        // host allocation
        h_A = new cuFloatComplex[dim * dim];
        h_U = new cuFloatComplex[dim * dim];
        h_D = new float[dim]; // real valued eigenvalues since hermitian

        // device allocation
        cudaMalloc(&d_A, dim * dim * sizeof(cuFloatComplex));
        cudaMalloc(&d_D, dim * sizeof(float));

        // copy values of H to h_A, d_A
        memcpy(h_A, H, dim * dim * sizeof(cuFloatComplex));
        cudaMemcpy(d_A, H, dim * dim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

        // initialize some values for the solver
        n = dim;
        lda = dim;

        // allocate some values for the solver
        cudaMalloc(&devInfo, sizeof(int));

        // initialize errors
        cusolverStat = CUSOLVER_STATUS_SUCCESS;
        cudaStat = cudaSuccess;

        // create the solver handle
        cusolverDnCreate(&cusolverH);
    }
    ~dataStruct(){

        delete [] h_A; delete [] h_U; delete [] h_D;
        cudaFree(d_A); cudaFree(d_D); cudaFree(devInfo); // cudaFree(work);
    }
    */
};

dataStruct prepare_memory(cuFloatComplex* H, const int dim){

    // create new data structure
    dataStruct x;

    // host allocation
    x.h_A = new cuFloatComplex[dim * dim];
    x.h_U = new cuFloatComplex[dim * dim];
    x.h_D = new float[dim]; // real valued eigenvalues since hermitian

    // device allocation
    cudaMalloc(&x.d_A, dim * dim * sizeof(cuFloatComplex));
    cudaMalloc(&x.d_D, dim * sizeof(float));

    // copy values of H to h_A, d_A
    memcpy(x.h_A, H, dim * dim * sizeof(cuFloatComplex));
    cudaMemcpy(x.d_A, H, dim * dim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // initialize some values for the solver
    x.n = dim;
    x.lda = dim;

    // allocate some values for the solver
    cudaMalloc(&x.devInfo, sizeof(int));

    // initialize errors
    x.cusolverStat = CUSOLVER_STATUS_SUCCESS;
    x.cudaStat = cudaSuccess;

    // create the solver handle
    cusolverDnCreate(&x.cusolverH);

    // return the structure with data allocated
    return x;
}

// EIGENSOLVER
void eigensolve(dataStruct x){

    // parameters for the solver here
    int lwork = 0;
    cuFloatComplex* work = NULL;

    // whether to save the eigenvectors
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    // probably solver only uses part of matrix (since symmetric) & this specifics which half you are supplying
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // compute amount of memory needed, and allocate on device
    x.cusolverStat = cusolverDnCheevd_bufferSize(x.cusolverH, jobz, uplo, x.n, x.d_A, x.lda, x.d_D, &lwork);

    cudaMalloc(&work, lwork * sizeof(cuFloatComplex));

    // (palpatine voice) do it
    cusolverDnCheevd(x.cusolverH, jobz, uplo, x.n, x.d_A, x.lda, x.d_D, work, lwork, x.devInfo);
}

// UTILITY FUNCTIONS
void write_U_to_file(cuFloatComplex* U, int dim){

    // open files to write eigenvectors (real)
    std::ofstream r_output_U;
    r_output_U.open("real_U.txt");

    // open files to write eigenvectors (imag)
    std::ofstream i_output_U;
    i_output_U.open("imag_U.txt");

    // write to file
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            r_output_U << cuCrealf(U[dim * i + j]) << "\n";
            i_output_U << cuCimagf(U[dim * i + j]) << "\n";
        }
    }
}

void write_D_to_file(float* D, int dim){

    // open files to write eigenvectors (real)
    std::ofstream output_D;
    output_D.open("D.txt");

    // write to file
    for (int i = 0; i < dim; i++)
    {
        output_D << D[i] << "\n";
    }
}

void load_hamiltonian_from_file(cuFloatComplex* H){

    // load hamiltonian from file
    std::ifstream r_H; std::ifstream i_H;
    r_H.open("real_H.txt"); i_H.open("imag_H.txt");

    // temporary variables
    float hij;
    int idx = 0;

    // load real part
    while (r_H >> hij){

        H[idx] = make_cuFloatComplex(hij, float(0));
        idx += 1;
    }

    // load imaginary part
    idx = 0;
    while (i_H >> hij){

        H[idx] = cuCaddf(H[idx], make_cuFloatComplex(float(0), hij));
        idx += 1;
    }
}

int main(){

    // start timing for loading hamiltonian
    auto start = std::chrono::high_resolution_clock::now();

    // get the hamiltonian
    cuFloatComplex* H = new cuFloatComplex[DIM * DIM];
    load_hamiltonian_from_file(H);

    // print the time taken to get hamiltonian
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "The total elapsed time to fetch H was " << duration.count() << "s" << std::endl;

    // start timing for allocation
    start = std::chrono::high_resolution_clock::now();

    // create the dataStruct
    dataStruct x = prepare_memory(H, DIM);
    // dataStruct(H, DIM);

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
    cudaMemcpy(x.h_U, x.d_A, DIM * DIM * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(x.h_D, x.d_D, DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // write to files
    write_D_to_file(x.h_D, DIM);
    write_U_to_file(x.h_U, DIM);

    // print the time taken to eigensolve
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to fetch the result and write to a file was " << duration.count() << "s" << std::endl;

    // free all memory
    // delete &x;
    // delete [] H;    

    // return
    return 0;
}