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
int DIM = 1024;

// MATRIX TRACE
cuFloatComplex trace(cuFloatComplex* d_A, int dim, cuHandles x){

    // use dot product to calculate trace, idea stolen from scikit-cuda
    // https://scikit-cuda.readthedocs.io/en/latest/_modules/skcuda/linalg.html#trace
    
    // just a single value of 1
    cuFloatComplex h_one = make_cuFloatComplex(1,0);
    cuFloatComplex* one; CUDA_CHECK(cudaMalloc(&one, sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(one, &h_one, sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // trace value to return
    cuFloatComplex result;

    // increment step: for A, increment by matrix dimension dim, for B do not increment (incy = 0)
    int incx = dim + 1;
    int incy = 0;

    // crunch it
    CUBLAS_CHECK(cublasCdotu(x.cublasH, dim, d_A, incx, one, incy, &result));

    // free the memory just in case
    CUDA_CHECK(cudaFree(one));

    // return the trace
    return result;
}

__global__ void column_sum(cuFloatComplex* d_A, float* normA, int dim){

    // one thread gets each column
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // if threads < dim, run it
    if (idx < dim){

        float temp = 0;
        for (int i = 0; i < dim; i++){
            temp = temp + my_cuCabsf(d_A[dim * i + idx]);
        }
        normA[idx] = temp;
    }
}

// PREPROCESSING
cuFloatComplex pre_process(cuFloatComplex* d_A, int dim, cuHandles x, int* nsquares){

    // calculate trace
    cuFloatComplex TrA = trace(d_A, dim, x);

    // scale by matrix dimension
    TrA = cuCdivf(TrA, make_cuFloatComplex(float(dim), 0));

    // just a single value of -1
    cuFloatComplex h_one = make_cuFloatComplex(-1,0);
    cuFloatComplex* one; CUDA_CHECK(cudaMalloc(&one, sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(one, &h_one, sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // increment step: for A, increment by matrix dimension dim, for B do not increment (incy = 0)
    int incx = 0;
    int incy = dim + 1;

    // subtract off the trace using the same trick as when calculating the trace
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim, &TrA, one, incx, d_A, incy));

    // balance the matrix

    // calculate matrix norm (maximal column sum)
    float* normA; CUDA_CHECK(cudaMalloc(&normA, dim * sizeof(float)));
    column_sum <<< 1 + dim/32, 32 >>> (d_A, normA, dim);
    CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize()); // check errors for kernel

    // get maximal column sum to decide scale factor
    int idx;
    CUBLAS_CHECK(cublasIsamax(x.cublasH, dim, normA, 1, &idx));

    // copy over value of maximal column sum to host
    float nA; CUDA_CHECK(cudaMemcpy(&nA, normA + idx, sizeof(float), cudaMemcpyDeviceToHost));

    // calculate log2(scale factor) & save for later
    *nsquares = (int) ceilf(log2f(nA / 5.371920351148152));
    std::cout << *nsquares << std::endl;

    // get scale factor itself (2^n)
    cuFloatComplex s = make_cuFloatComplex(powf(2, -(*nsquares)), 0);
    std::cout << cuCrealf(s) << std::endl;

    // scale
    CUBLAS_CHECK(cublasCscal(x.cublasH, dim * dim, &s, d_A, 1));

    // free the memory just in case
    CUDA_CHECK(cudaFree(one)); CUDA_CHECK(cudaFree(normA));

    // return the trace, for use later
    return TrA;
}

// POSTPROCESSING
void post_process(cuFloatComplex* d_P, cuFloatComplex* d_X, cuFloatComplex TrA, int dim, cuHandles x, int* nsquares){

    // identity and zero values
    cuFloatComplex id = make_cuFloatComplex(1, 0); cuFloatComplex z = make_cuFloatComplex(0, 0);

    // intermediate storage for the calculation
    cuFloatComplex* d_x; CUDA_CHECK(cudaMalloc(&d_x, dim * dim * sizeof(cuFloatComplex)));
    cuFloatComplex* d_y; CUDA_CHECK(cudaMalloc(&d_y, dim * dim * sizeof(cuFloatComplex)));

    // first square, store in y
    CUBLAS_CHECK(cublasCgemm(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_P, dim, d_P, dim, &z, d_y, dim));

    // number of required squarings = value at nsquares
    int num_squares = *nsquares;

    // if only one square, copy to X right away
    if (num_squares == 1){
        CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, d_y, 1, d_X, 1));
    }

    // otherwise, loop through
    for (int idx = 0; idx < num_squares-1; idx++)
    {
        // ODD POWER (replace x with y * y)
        if (idx % 2 == 0 || idx == 0)
        {
            // calculate the product
            CUBLAS_CHECK(cublasCgemm(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_y, dim, d_y, dim, &z, d_x, dim));

            // if at the last index, copy to non-temporary memory
            if (idx == num_squares - 2)
            {
                CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, d_x, 1, d_X, 1));
            }
        }

        // EVEN POWER (replace y with x * x)
        else
        {
            // calculate the product
            CUBLAS_CHECK(cublasCgemm(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_x, dim, d_x, dim, &z, d_y, dim));

            // if at the last index, copy to non-temporary memory
            if (idx == num_squares - 2)
            {
                CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, d_y, 1, d_X, 1));
            }
        }
    }

    // undo balancing

    // calculate magnitude and argument of TrA for exponential
    float r = cuCabsf(TrA);
    float arg = atan2f(cuCimagf(TrA), cuCrealf(TrA));

    // put the values together to get exp(Tr A)
    // = exp(Tr A) = exp[ r exp(i arg) ] 
    // = exp[ r cos(arg) + i r sin(arg)] 
    // = exp[ r cos(arg) ] * [ cos(r sin(arg)) + i sin(r sin(arg)) ]
    cuFloatComplex exp_TrA = make_cuFloatComplex(expf(r * cosf(arg)) * cosf(r * sinf(arg)), 
                                                 expf(r * cosf(arg)) * sinf(r * sinf(arg)));

    // scale the matrix d_X which holds the result
    CUBLAS_CHECK(cublasCscal(x.cublasH, dim * dim, &exp_TrA, d_X, 1));

    // free all allocated cuda memory just in case
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
}


// LINSOLVE
void linsolve(cuFloatComplex* d_P, cuFloatComplex* d_Q, int dim, cuHandles x){
    
    // needed for the solver
    int* d_ipiv;  CUDA_CHECK(cudaMalloc(&d_ipiv, dim * sizeof(int)));
    int* devInfo; CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    // parameters for the solver here
    int lwork = 0;
    cuFloatComplex* work = NULL;

    // get size of buffer
    CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(x.cusolverH, dim, dim, d_Q, dim, &lwork));

    // allocate buffer
    CUDA_CHECK(cudaMalloc(&work, lwork * sizeof(int)));

    // factorize
    CUSOLVER_CHECK(cusolverDnCgetrf(x.cusolverH, dim, dim, d_Q, dim, work, d_ipiv, devInfo));

    // solve & overwrite P with solution X
    CUSOLVER_CHECK(cusolverDnCgetrs(x.cusolverH, CUBLAS_OP_N, dim, dim, d_Q, dim, d_ipiv, d_P, dim, devInfo));
}

// PADE APPROXIMANT POLYNOMIALS
// change this to reduce number of multiplications, see http://eprints.ma.man.ac.uk/634/1/high05e.pdf
void calc_PQ(cuFloatComplex* d_A, cuFloatComplex* d_P, cuFloatComplex* d_Q, int dim, cuHandles x){

    // identity and zero values
    cuFloatComplex id = make_cuFloatComplex(1, 0); cuFloatComplex z = make_cuFloatComplex(0, 0);

    // memory for pade approximant coefficients
    cuFloatComplex* coefP = new cuFloatComplex[14];
    cuFloatComplex* coefQ = new cuFloatComplex[14];

    // P polynomial coefficients
    coefP[0] = make_cuFloatComplex(float(64764752532480000), float(0));
    coefP[1] = make_cuFloatComplex(float(32382376266240000), float(0));
    coefP[2] = make_cuFloatComplex(float(7771770303897600), float(0));
    coefP[3] = make_cuFloatComplex(float(1187353796428800), float(0));
    coefP[4] = make_cuFloatComplex(float(129060195264000), float(0));
    coefP[5] = make_cuFloatComplex(float(10559470521600), float(0));
    coefP[6] = make_cuFloatComplex(float(670442572800), float(0));
    coefP[7] = make_cuFloatComplex(float(33522128640), float(0));
    coefP[8] = make_cuFloatComplex(float(1323241920), float(0));
    coefP[9] = make_cuFloatComplex(float(40840800), float(0));
    coefP[10] = make_cuFloatComplex(float(960960), float(0));
    coefP[11] = make_cuFloatComplex(float(16380), float(0));
    coefP[12] = make_cuFloatComplex(float(182), float(0));
    coefP[13] = make_cuFloatComplex(float(1), float(0));

    // Q polynomial coefficients: every other term is negative
    coefQ[0] = make_cuFloatComplex(float(64764752532480000), float(0));
    coefQ[1] = make_cuFloatComplex(float(-32382376266240000), float(0));
    coefQ[2] = make_cuFloatComplex(float(7771770303897600), float(0));
    coefQ[3] = make_cuFloatComplex(float(-1187353796428800), float(0));
    coefQ[4] = make_cuFloatComplex(float(129060195264000), float(0));
    coefQ[5] = make_cuFloatComplex(float(-10559470521600), float(0));
    coefQ[6] = make_cuFloatComplex(float(670442572800), float(0));
    coefQ[7] = make_cuFloatComplex(float(-33522128640), float(0));
    coefQ[8] = make_cuFloatComplex(float(1323241920), float(0));
    coefQ[9] = make_cuFloatComplex(float(-40840800), float(0));
    coefQ[10] = make_cuFloatComplex(float(960960), float(0));
    coefQ[11] = make_cuFloatComplex(float(-16380), float(0));
    coefQ[12] = make_cuFloatComplex(float(182), float(0));
    coefQ[13] = make_cuFloatComplex(float(-1), float(0));

    // initialize P and Q matrices properly
    cuFloatComplex* I = new cuFloatComplex[dim * dim];
    for (int i = 0; i < dim; i++){
        for (int j = 0; j < dim; j++){

            // fill P and Q with zeros
            I[dim * i + j] = z;
        }

        // punch in the proper diagonal value AFTER filling row
        I[dim * i + i] = cuCmulf(coefP[0], id);
    }
    CUDA_CHECK(cudaMemcpy(d_P, I, dim * dim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, I, dim * dim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // intermediate storage for the calculation
    cuFloatComplex* d_x; CUDA_CHECK(cudaMalloc(&d_x, dim * dim * sizeof(cuFloatComplex)));
    cuFloatComplex* d_y; CUDA_CHECK(cudaMalloc(&d_y, dim * dim * sizeof(cuFloatComplex)));

    // add/subtract A to/from P and Q
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefP[1], d_A, 1, d_P, 1));
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefQ[1], d_A, 1, d_Q, 1));

    // calculate A * A, store in x
    CUBLAS_CHECK(cublasCgemm(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_A, dim, d_A, dim, &z, d_x, dim));

    // add to Q and P
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefP[2], d_x, 1, d_P, 1));
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefQ[2], d_x, 1, d_Q, 1));

    // calculate the remaining powers
    for (int idx = 0; idx < 11; idx++)
    {
        // ODD POWER (replace y with a * x)
        if (idx % 2 == 0 || idx == 0)
        {
            // calculate the product
            CUBLAS_CHECK(cublasCgemm(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_A, dim, d_x, dim, &z, d_y, dim));

            // add to P or Q
            CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefP[idx + 3], d_y, 1, d_P, 1));
            CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefQ[idx + 3], d_y, 1, d_Q, 1));
        }

        // EVEN POWER (replace x with a * y)
        else
        {
            // calculate the product
            CUBLAS_CHECK(cublasCgemm(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_A, dim, d_y, dim, &z, d_x, dim));

            // add to P or Q
            CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefP[idx + 3], d_x, 1, d_P, 1));
            CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefQ[idx + 3], d_x, 1, d_Q, 1));
        }
    }

    // free all allocated cuda memory just in case
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
}

int main(){

    // size of matrices
    int dim = DIM;

    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    // load a matrix A to exponentiate
    cuFloatComplex* A = new cuFloatComplex[dim * dim];
    std::string a_name = "A";
    read_array_from_file_C(A, a_name);

    // print time of execution
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "The total elapsed time to read A into memory was " << duration.count() << "s" << std::endl;

    // start timing
    start = std::chrono::high_resolution_clock::now();

    // host pointers
    cuFloatComplex* h_X      = new cuFloatComplex[dim * dim]; // for the solution
    
    // device pointers
    cuFloatComplex* d_A;      // matrix to exponentiate
    cuFloatComplex* d_P;      // P = V + U, pade approximant function
    cuFloatComplex* d_Q;      // Q = V - U, pade approximant function
    cuFloatComplex* d_X;      // X the solution

    // allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, dim * dim * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_Q, dim * dim * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_P, dim * dim * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_X, dim * dim * sizeof(cuFloatComplex)));

    // copy memory to device
    CUDA_CHECK(cudaMemcpy(d_A, A, dim * dim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // create handles
    cuHandles x;

    // squaring step
    int* nsquares = new int;

    // print time of execution
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to allocate memory, copy A to device, and create handles was " << duration.count() << "s" << std::endl;
    
    // start timing
    start = std::chrono::high_resolution_clock::now();

    // start timing for the whole process
    auto net_start = std::chrono::high_resolution_clock::now();

    // pre-process matrix by scaling, subtracting the trace, etc
    cuFloatComplex TrA = pre_process(d_A, dim, x, nsquares);

    // print time of execution
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to pre-process was " << duration.count() << "s" << std::endl;

    // start timing
    start = std::chrono::high_resolution_clock::now();

    // calculate numerator and denominator P and Q of pade approximant
    calc_PQ(d_A, d_P, d_Q, dim, x);

    // print time of execution
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to calculate P and Q was " << duration.count() << "s" << std::endl;

    // start timing
    start = std::chrono::high_resolution_clock::now();

    // linsolve: overwrites P with solution of linsolve
    linsolve(d_P, d_Q, dim, x);

    // print time of execution
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to solve QX = P was " << duration.count() << "s" << std::endl;

    // start timing
    start = std::chrono::high_resolution_clock::now();

    // reverse scaling, multiply by exp(-trace)
    post_process(d_P, d_X, TrA, dim, x, nsquares);

    // print time of execution
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to post-process was " << duration.count() << "s" << std::endl;

    // grab total time of execution
    cudaDeviceSynchronize();
    auto net_end = std::chrono::high_resolution_clock::now();

    // start timing
    start = std::chrono::high_resolution_clock::now();

    // copy memory to host for error checking
    CUDA_CHECK(cudaMemcpy(h_X, d_X, dim * dim * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    // print time of execution
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to copy the solution to host was " << duration.count() << "s" << std::endl;

    // start timing
    start = std::chrono::high_resolution_clock::now();

    // write X to file for error checking
    std::string x_name = "X";
    write_matrix_to_file_C(h_X, x_name, dim);

    // print time of execution
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to write the solution to a file was " << duration.count() << "s" << std::endl;

    // print total execution time
    duration = net_end - net_start;
    std::cout << "\nThe total elapsed time to calculate expm(A) not counting preparatory steps was " << duration.count() << "s\n\n";

    // return
    return 0;
}

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

    /*
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

    */