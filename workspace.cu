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

// 

// MATRIX TRACE WITH CUBLAS
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

// EFFICIENT COLUMN MATRIX NORM CALCULATION (https://cuvilib.com/Reduction.pdf)
__global__ void gpu_column_norm128_double_load(cuFloatComplex* d_A, float* output, int row_dim, int col_dim, int itr){

    // variables to consider
    // row_dim = number of rows (gets cut by 128 after each kernel execution, consider intermediate outputs as matrices)
    // col_dim = number of columns (remains constant)

    // shared memory for the thread block for a chunk of A
    __shared__ float data[128];

    // indexing: 2d grid of 1d blocks-- each "row" of blocks (along x) works on one column
    unsigned int t_idx = threadIdx.x; // index in current block
    unsigned int col_idx = blockIdx.y; // which column we are working with = y index of grid
    unsigned int row_idx = blockIdx.x * (2 * blockDim.x) + threadIdx.x; // which element of the column (ie which row of A) we are working with

    // on the first iteration, move a chunk of A into shared memory & do one reduction
    data[t_idx] = 0.0; // by default set the memory to zero, basically zero padding the number of rows to a multiple of 128
    if (itr == 0){
        if (row_idx < row_dim) data[t_idx] = my_cuCabsf(d_A[row_dim * col_idx + row_idx]); // if within matrix bounds, load from d_A
        if (row_idx + blockDim.x < row_dim) data[t_idx] = __fadd_rn(data[t_idx], my_cuCabsf(d_A[row_dim * col_idx + row_idx + blockDim.x]));
    } else { // on the second, pull from the previous iteration's output
        if (row_idx < row_dim) data[t_idx] = output[row_dim * col_idx + row_idx];
        if (row_idx + blockDim.x < row_dim) data[t_idx] = __fadd_rn(data[t_idx], output[row_dim * col_idx + row_idx + blockDim.x]);
    }
    __syncthreads();

    // do reduction like in nvidia ppt
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        
        // only make the comparison if on a "zero" thread, ie one to replace with
        if (t_idx < s){

            // add the values
            data[t_idx] = __fadd_rn(data[t_idx], data[t_idx + s]);
        }
        __syncthreads();
    }

    // at the end of the process, save the result to the output: blockIdx.x is the new row index, col_idx remains the same
    if (t_idx == 0){
        output[gridDim.x * col_idx + blockIdx.x] = data[0];
    }
}

void column_norm(cuFloatComplex* d_A, float* output, int dim){

    // generate the initial grid: num_x = number of elements in the x direction, num_y = number of columns
    int num_x = dim, num_y = dim, numBlockperCol = 1 + (1 + dim / 128) / 2, itr = 0;

    // block & grid dimensions: each block = 1D w/ 128 threads
    dim3 block(128, 1), grid(numBlockperCol, num_y);

    // loop until down to one block (one block covers 2x number of threads with double loading)
    while (num_x > 256){
        
        // run the first reduction
        gpu_column_norm128_double_load <<< grid, block >>> (d_A, output, num_x, dim, itr);
        CUDA_CHECK(cudaPeekAtLastError());

        // number of elements along x is now equal to number of blocks per column
        num_x = numBlockperCol;

        // recalculate number of blocks per column
        numBlockperCol = 1 + (1 + num_x / 128) / 2;

        // change the grid size
        grid.x = numBlockperCol;

        // increment the iteration
        itr++;
    }

    // run once more to complete the reduction
    gpu_column_norm128_double_load <<< grid, block >>> (d_A, output, num_x, dim, itr);
    CUDA_CHECK(cudaPeekAtLastError());
}

// OLD INEFFICIENT (AND BASIC) COLUMN MATRIX NORM CALCULATION
__global__ void column_sum(cuFloatComplex* d_A, float* normA, int dim){

    // one thread gets each column
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim){

        float temp = 0;
        for (int i = 0; i < dim; i++){
            temp = __fadd_rn(temp, my_cuCabsf(d_A[dim * idx + i]));
        }
        normA[idx] = temp;
    }
}

// EFFICIENT ROW MATRIX NORM CALCULATION (https://cuvilib.com/Reduction.pdf)
__global__ void gpu_row_norm128_double_load(cuFloatComplex* d_A, float* output, int row_dim, int col_dim, int itr){

    // variables to consider
    // row_dim = number of rows (gets cut by 128 after each kernel execution, consider intermediate outputs as matrices)
    // col_dim = number of columns (remains constant)

    // shared memory for the thread block for a chunk of A
    __shared__ float data[128];

    // indexing: 2d grid of 1d blocks-- each "row" of blocks (along x) works on one column
    unsigned int t_idx = threadIdx.x; // index in current block
    unsigned int row_idx = blockIdx.y; // which row we are working with = y index of grid
    unsigned int col_idx = blockIdx.x * (2 * blockDim.x) + threadIdx.x; // which element of the row (ie which column of A) we are working with

    // on the first iteration, move a chunk of A into shared memory & do one reduction
    data[t_idx] = 0.0; // by default set the memory to zero, basically zero padding the number of rows to a multiple of 128
    if (itr == 0){
        if (col_idx < col_dim) data[t_idx] = my_cuCabsf(d_A[row_dim * col_idx + row_idx]); // if within matrix bounds, load from d_A
        if (col_idx + blockDim.x < col_dim) data[t_idx] = __fadd_rn(data[t_idx], my_cuCabsf(d_A[row_dim * (col_idx + blockDim.x) + row_idx]));
    } else { // on the second, pull from the previous iteration's output
        if (col_idx < col_dim) data[t_idx] = output[row_dim * col_idx + row_idx];
        if (col_idx + blockDim.x < col_dim) data[t_idx] = __fadd_rn(data[t_idx], output[row_dim * (col_idx + blockDim.x) + row_idx]);
    }
    __syncthreads();

    // do reduction like in nvidia ppt
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        
        // only make the comparison if on a "zero" thread, ie one to replace with
        if (t_idx < s){

            // add the values
            data[t_idx] = __fadd_rn(data[t_idx], data[t_idx + s]);
        }
        __syncthreads();
    }

    // at the end of the process, save the result to the output: blockIdx.x is the new row index, col_idx remains the same
    if (t_idx == 0){
        output[row_dim * blockIdx.x + row_idx] = data[0];
    }
}

void row_norm(cuFloatComplex* d_A, float* output, int dim){

    // generate the initial grid: num_x = number of elements in the x direction, num_y = number of columns
    int num_x = dim, num_y = dim, numBlockperRow = 1 + (1 + dim / 128) / 2, itr = 0;

    // block & grid dimensions: each block = 1D w/ 128 threads
    dim3 block(128, 1), grid(numBlockperRow, num_y);

    // loop until down to one block (one block covers 2x number of threads with double loading)
    while (num_x > 256){
        
        // run the first reduction
        gpu_row_norm128_double_load <<< grid, block >>> (d_A, output, dim, num_x, itr);
        CUDA_CHECK(cudaPeekAtLastError());

        // number of elements along x is now equal to number of blocks per column
        num_x = numBlockperRow;

        // recalculate number of blocks per column
        numBlockperRow = 1 + (1 + num_x / 128) / 2;

        // change the grid size
        grid.x = numBlockperRow;

        // increment the iteration
        itr++;
    }

    // run once more to complete the reduction
    gpu_row_norm128_double_load <<< grid, block >>> (d_A, output, dim, num_x, itr);
    CUDA_CHECK(cudaPeekAtLastError());
}

// OLD INEFFICIENT (AND BASIC) ROW MATRIX NORM CALCULATION
__global__ void row_sum(cuFloatComplex* d_A, float* normA, int dim){

    // one thread gets each row
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim){

        float temp = 0;
        for (int i = 0; i < dim; i++){
            temp = __fadd_rn(temp, my_cuCabsf(d_A[dim * i + idx]));
        }
        normA[idx] = temp;
    }
}

// FUNCTIONS FOR MATRIX BALANCING
__global__ void balance_matrix_calc_errors(float* cNorms, float* rNorms, float* err, int dim){

    // one thread gets one index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim){

        // assign quotient to error
        if (cNorms[idx] > rNorms[idx]){
            err[idx] = __fdiv_rn(cNorms[idx], rNorms[idx]);
        } else {
            err[idx] = __fdiv_rn(rNorms[idx], cNorms[idx]);
        }
    }
}

__global__ void balance_matrix_adjust_y(float* y, float* cNorms, float* rNorms, int* update_list, int batch_size){

    // just give each adjustment to a thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size){    
        int jdx = update_list[idx];
        float val = __fmul_rn(0.5, __fsub_rn(logf(cNorms[jdx]), logf(rNorms[jdx])));
        y[jdx] = __fadd_rn(y[jdx], val);
    }
}

__global__ void balance_matrix_zero_y(float* y, int dim){

    // one thread = one element of y to zero out
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim){
        y[idx] = 0.0;
    }
}

__global__ void balance_matrix_adjust_A(cuFloatComplex* d_A, cuFloatComplex* tempA, float* y, int dim){

    // use same configurations from matrix norm
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_idx = blockIdx.y;

    if (row_idx < dim){

        // calculate Dii x invDjj
        cuFloatComplex val = make_cuFloatComplex(expf(__fsub_rn(y[row_idx], y[col_idx])), 0.0);
    
        // adjust Aij
        tempA[dim * col_idx + row_idx] = my_cuCmulf(d_A[dim * col_idx + row_idx], val);
    }
}

// SORTING ALGORITHM: adapted from https://www.geeksforgeeks.org/cpp-program-for-quicksort/
int partition(float* vals, int* I, int start, int end){

    // I = array of indices, vals = array of values

    // find the correct position for pivot value by finding how many values are greater than or equal to pivot
    float pivot = vals[start];
    int di = 0;
    for (int i = start + 1; i <= end; i++){
        if (vals[i] >= pivot){
            di++;
        }
    }

    // move pivot to correct location
    int pivot_idx = start + di;
    std::swap(vals[pivot_idx], vals[start]);
    std::swap(I[pivot_idx], I[start]);

    // move all values greater than pivot to right of pivot, and all values less to the left
    int L = start, R = end, num = 0;;
    while (L < pivot_idx && R > pivot_idx){

        // increase L until find an element > pivot
        while (vals[L] >= pivot){
            L++;
        }

        // decrease R until find an element < pivot
        while(vals[R] < pivot){
            R--;
        }

        // if R, L stil on correct side of pivot, swap
        if (L < pivot_idx && R > pivot_idx){     
            std::swap(vals[L], vals[R]);
            std::swap(I[L], I[R]);
            L++; R--; num++;
        }        
    }

    return pivot_idx;
}

void quick_sort(float* vals, int* I, int start, int end){

    // kill if start is to right of end/no more sorting to do
    if (start >= end){
        return;
    }

    // sort around the pivot
    int p = partition(vals, I, start, end);

    // recursively do left and right parts
    quick_sort(vals, I, start, p - 1);
    quick_sort(vals, I, p + 1, end);
}

// MATRIX BALANCING
void batch_greedy_osborne(cuFloatComplex* d_A, float tol, int batch_size,
                          int* d_update, int* h_update, int* idx_list, cuHandles x,
                          float* cNorms, float* rNorms, float* tNorms, int dim,
                          float* d_err,  float* h_err, float* y, cuFloatComplex* tempA, int print_itr){

    // necessary parameters for matrix balancing
    dim3 block(128, 1), grid(1 + dim / 128, dim); // grid for adjustment of A to reduce necessity of modulus operators
    int counter = 0;                              // batch size (numer of row/col adjustments to make each iteration)

    // calculate column norm and copy to cNorms
    column_norm(d_A, tNorms, dim); CUBLAS_CHECK(cublasScopy(x.cublasH, dim, tNorms, 1, cNorms, 1));
    row_norm(d_A, tNorms, dim);    CUBLAS_CHECK(cublasScopy(x.cublasH, dim, tNorms, 1, rNorms, 1));

    // calculate errors
    balance_matrix_calc_errors <<< 1 + dim/128, 128 >>> (cNorms, rNorms, d_err, dim);
    CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // move to host for sorting
    CUDA_CHECK(cudaMemcpy(h_err, d_err, dim * sizeof(float), cudaMemcpyDeviceToHost));

    // sort the errors to get worst matches
    quick_sort(h_err, h_update, 0, dim - 1);

    // loop until within tolerance, or hit 5,000 iterations
    while (h_err[0] > 1 + tol){

        // if go too long, kill it
        if (counter > 5000){
            std::cout << "Unable to balance within 5,000 iterations. Ending balancing and outputting most recent balancing parameters." << std::endl;
            return;
        }

        // copy worst indices for update list to device
        CUDA_CHECK(cudaMemcpy(d_update, h_update, batch_size * sizeof(int), cudaMemcpyHostToDevice));

        // make the adjustment to y
        balance_matrix_adjust_y <<< 1 + batch_size / 128, 128 >>> (y, cNorms, rNorms, d_update, batch_size); // add a -1 if getting direct from greedy indexing
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

        // do the balancing step
        balance_matrix_adjust_A <<< grid, block >>> (d_A, tempA, y, dim);
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());     

        // calculate column norm and copy to cNorms
        column_norm(tempA, tNorms, dim); CUBLAS_CHECK(cublasScopy(x.cublasH, dim, tNorms, 1, cNorms, 1));
        row_norm(tempA, tNorms, dim);    CUBLAS_CHECK(cublasScopy(x.cublasH, dim, tNorms, 1, rNorms, 1));

        // calculate errors
        balance_matrix_calc_errors <<< 1 + dim/128, 128 >>> (cNorms, rNorms, d_err, dim);
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

        // copy errors to host
        CUDA_CHECK(cudaMemcpy(h_err, d_err, dim * sizeof(float), cudaMemcpyDeviceToHost));

        // reset update array for next sort
        memcpy(h_update, idx_list, dim * sizeof(int));

        // sort the errors to get worst matches
        quick_sort(h_err, h_update, 0, dim - 1);

        // update counter
        counter = counter + batch_size;
    }
    if (print_itr == 1) std::cout << "The number of iterations to balance was " << counter << std::endl;
}

// PREPROCESSING
cuFloatComplex pre_process(cuFloatComplex* d_A, int dim, cuHandles x, int* nsquares){

    // edits matrix A in place (d_A), outputs necessary values to undo changes at end of alg

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
    float* cNorms; CUDA_CHECK(cudaMalloc(&cNorms, (dim * (1 + (1 + dim / 128) / 2) * sizeof(float))));
    column_norm(d_A, cNorms, dim);

    // get maximal column sum to decide scale factor
    int idx;
    CUBLAS_CHECK(cublasIsamax(x.cublasH, dim, cNorms, 1, &idx));

    // copy over value of maximal column sum to host
    float nA; CUDA_CHECK(cudaMemcpy(&nA, cNorms + idx - 1, sizeof(float), cudaMemcpyDeviceToHost)); // cuBLAS using 1 indexing

    // calculate log2(scale factor) & save for later
    *nsquares = (int) ceilf(log2f(nA / 5.371920351148152));

    // get scale factor itself (2^n)
    cuFloatComplex s = make_cuFloatComplex(powf(2, -(*nsquares)), 0);

    // scale
    CUBLAS_CHECK(cublasCscal(x.cublasH, dim * dim, &s, d_A, 1));

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
    CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_P, dim, d_P, dim, &z, d_y, dim));

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
            CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_y, dim, d_y, dim, &z, d_x, dim));

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
            CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_x, dim, d_x, dim, &z, d_y, dim));

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
}

// LINSOLVE
void linsolve(cuFloatComplex* d_P, cuFloatComplex* d_Q, int dim, cuHandles x){
    
    // needed for the solver
    int* d_ipiv;  CUDA_CHECK(cudaMalloc(&d_ipiv, dim * sizeof(int)));
    int* devInfo; CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    // parameters for the solver here
    int lwork = 0;
    cuFloatComplex* work = nullptr;

    // get size of buffer
    CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(x.cusolverH, dim, dim, d_Q, dim, &lwork));

    // allocate buffer
    CUDA_CHECK(cudaMalloc(&work, lwork * sizeof(int)));

    // factorize
    CUSOLVER_CHECK(cusolverDnCgetrf(x.cusolverH, dim, dim, d_Q, dim, work, d_ipiv, devInfo));

    // solve & overwrite P with solution X (solves QX = P)
    CUSOLVER_CHECK(cusolverDnCgetrs(x.cusolverH, CUBLAS_OP_N, dim, dim, d_Q, dim, d_ipiv, d_P, dim, devInfo));
}

// PADE APPROXIMANT POLYNOMIALS (SERIAL CALCULATION, VARIABLE m)
void calc_PQ_seq(cuFloatComplex* d_A, cuFloatComplex* d_P, cuFloatComplex* d_Q, int dim, cuHandles x){

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
    CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_A, dim, d_A, dim, &z, d_x, dim));

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
            CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_A, dim, d_x, dim, &z, d_y, dim));

            // add to P or Q
            CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefP[idx + 3], d_y, 1, d_P, 1));
            CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefQ[idx + 3], d_y, 1, d_Q, 1));
        }

        // EVEN POWER (replace x with a * y)
        else
        {
            // calculate the product
            CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_A, dim, d_y, dim, &z, d_x, dim));

            // add to P or Q
            CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefP[idx + 3], d_x, 1, d_P, 1));
            CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &coefQ[idx + 3], d_x, 1, d_Q, 1));
        }
    }
}

// FASTER WAY TO CALCULATE P AND Q BUT STRICTLY FOR m = 13
void calc_PQ(cuFloatComplex* d_A, cuFloatComplex* d_P, cuFloatComplex* d_Q, int dim, cuHandles x){

    // identity and zero values
    cuFloatComplex id = make_cuFloatComplex(1, 0); cuFloatComplex mid = make_cuFloatComplex(-1,0); 
    cuFloatComplex z = make_cuFloatComplex(0, 0); 

    // need a copy of z, id
    cuFloatComplex* d_z; CUDA_CHECK(cudaMalloc(&d_z, sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_z, &z, sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    cuFloatComplex* d_id; CUDA_CHECK(cudaMalloc(&d_id, sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_id, &id, sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // memory for pade approximant coefficients
    cuFloatComplex* C = new cuFloatComplex[14];

    // load the coefficients
    C[0] = make_cuFloatComplex(float(64764752532480000), float(0));
    C[1] = make_cuFloatComplex(float(32382376266240000), float(0));
    C[2] = make_cuFloatComplex(float(7771770303897600), float(0));
    C[3] = make_cuFloatComplex(float(1187353796428800), float(0));
    C[4] = make_cuFloatComplex(float(129060195264000), float(0));
    C[5] = make_cuFloatComplex(float(10559470521600), float(0));
    C[6] = make_cuFloatComplex(float(670442572800), float(0));
    C[7] = make_cuFloatComplex(float(33522128640), float(0));
    C[8] = make_cuFloatComplex(float(1323241920), float(0));
    C[9] = make_cuFloatComplex(float(40840800), float(0));
    C[10] = make_cuFloatComplex(float(960960), float(0));
    C[11] = make_cuFloatComplex(float(16380), float(0));
    C[12] = make_cuFloatComplex(float(182), float(0));
    C[13] = make_cuFloatComplex(float(1), float(0));

    // memory for A2 = A * A, A4 = A2 * A2, A6 = A4 * A2
    cuFloatComplex* A2; CUDA_CHECK(cudaMalloc(&A2, dim * dim * sizeof(cuFloatComplex)));
    cuFloatComplex* A4; CUDA_CHECK(cudaMalloc(&A4, dim * dim * sizeof(cuFloatComplex)));
    cuFloatComplex* A6; CUDA_CHECK(cudaMalloc(&A6, dim * dim * sizeof(cuFloatComplex)));

    // initialize A2, A4, A6
    CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, d_z, 0, A2, 1));
    CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, d_z, 0, A4, 1));
    CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, d_z, 0, A6, 1));

    // intermediate storage
    cuFloatComplex* U1; CUDA_CHECK(cudaMalloc(&U1, dim * dim * sizeof(cuFloatComplex)));
    cuFloatComplex* U2; CUDA_CHECK(cudaMalloc(&U2, dim * dim * sizeof(cuFloatComplex)));
    cuFloatComplex* V1; CUDA_CHECK(cudaMalloc(&V1, dim * dim * sizeof(cuFloatComplex)));
    cuFloatComplex* V2; CUDA_CHECK(cudaMalloc(&V2, dim * dim * sizeof(cuFloatComplex)));

    // calculate A2 = A * A (store in A2)
    CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_A, dim, d_A, dim, &z, A2, dim));

    // calculate A4 = A2 * A2 (store in A4)
    CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, A2, dim, A2, dim, &z, A4, dim));

    // calculate A6 = A2 * A4 (store in A6)
    CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, A2, dim, A4, dim, &z, A6, dim));

    // calculate U1 = C13 * A6 + C11 * A4 + C9 * A2 (initialize to zero first)
    CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, d_z, 0, U1, 1));
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[13], A6, 1, U1, 1)); // add C13 * A6, overwriting
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[11], A4, 1, U1, 1)); // add C11 * A4, overwriting
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[9], A2, 1, U1, 1));  // add C9 * A2, overwriting

    // calculate U2 = C7 * A6 + C5 * A4 + C3 * A2 + C1 * I (initialize to zero first)
    CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, d_z, 0, U2, 1));
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[7], A6, 1, U2, 1));  // add C7 * A6, overwriting
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[5], A4, 1, U2, 1));  // add C5 * A4, overwriting
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[3], A2, 1, U2, 1));  // add C3 * A2, overwriting
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim, &C[1], d_id, 0, U2, dim + 1)); // add C1 * I, overwriting

    // calculate V1 = C12 * A6 + C10 * A4 + C8 * A2 (initialize to zero first)
    CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, d_z, 0, V1, 1));
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[12], A6, 1, V1, 1)); // add C12 * A6, overwriting
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[10], A4, 1, V1, 1)); // add C10 * A4, overwriting
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[8], A2, 1, V1, 1));  // add C8 * A2, overwriting

    // calculate V2 = C6 * A6 + C4 * A4 + C2 * A2 + C0 * I (initialize to zero first)
    CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, d_z, 0, V2, 1));
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[6], A6, 1, V2, 1));  // add C6 * A6, overwriting
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[4], A4, 1, V2, 1));  // add C4 * A4, overwriting
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &C[2], A2, 1, V2, 1));  // add C2 * A2, overwriting
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim, &C[0], d_id, 0, V2, dim + 1)); // add C0 * I, overwriting

    // left multiply U1, V1 by A6 (store A6 * U1 in A2, A6 * V1 in A4, since do not need these anymore)
    CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, A6, dim, U1, dim, &id, U2, dim));
    CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, A6, dim, V1, dim, &id, V2, dim));

    // last multiplication: left multiply A6 * U1 + U2 (stored in U2) by A to get U, store in U1
    CUBLAS_CHECK(cublasCgemm3m(x.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &id, d_A, dim, U2, dim, &z, U1, dim));

    // copy V (stored in V2) to P to calculate P = V + U
    CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, V2, 1, d_P, 1));

    // add U (stored in U1) to P
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &id, U1, 1, d_P, 1)); // P = V + U, overwrites P

    // copy V (stored in V2) to Q to calculate Q = V - U
    CUBLAS_CHECK(cublasCcopy(x.cublasH, dim * dim, V2, 1, d_Q, 1));

    // subtract U (stored in U1) from Q
    CUBLAS_CHECK(cublasCaxpy(x.cublasH, dim * dim, &mid, U1, 1, d_Q, 1)); // Q = V - U, overwrites Q
}

int main(){

    // size of matrix
    int dim = 1024;

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
    
    // create handles
    cuHandles x;

    // device pointers
    cuFloatComplex* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, dim * dim * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_A, A, dim * dim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // batch size and tolerance
    float tol = 0.01;
    int batch_size = dim / 5;

    // device memory for matrix balancing
    int* d_update;         CUDA_CHECK(cudaMalloc(&d_update, batch_size * sizeof(int)));                        // which indices to update at each step
    float* y;              CUDA_CHECK(cudaMalloc(&y, dim * sizeof(float)));                                    // balancing vector
    float* cNorms;         CUDA_CHECK(cudaMalloc(&cNorms, dim * sizeof(float)));                               // column norms
    float* rNorms;         CUDA_CHECK(cudaMalloc(&rNorms, dim * sizeof(float)));                               // row norms
    float* tNorms;         CUDA_CHECK(cudaMalloc(&tNorms, (dim * (1 + (1 + dim / 128) / 2) * sizeof(float)))); // temporary memory for reductions
    float* d_err;          CUDA_CHECK(cudaMalloc(&d_err, dim * sizeof(float)));                                // error in each row/col pair
    cuFloatComplex* tempA; CUDA_CHECK(cudaMalloc(&tempA, dim * dim * sizeof(cuFloatComplex)));                 // space for adjustment of A on each iteration
    
    // host memory for matrix balancing
    int* h_update = new int[dim];  // host copy of d_update for use with quicksort
    int* idx_list = new int[dim];  // ordered indexing to reset h_update for quicksorting
    float* h_err = new float[dim]; // host copy of d_err for use with quicksort

    // fill out idx_list, h_update
    for (int i = 0; i < dim; i++) idx_list[i] = i;
    memcpy(h_update, idx_list, dim * sizeof(int));

    // zero out vector y
    balance_matrix_zero_y <<< 1 + dim / 128, 128 >>> (y, dim);
    CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // start timing
    auto net_start = std::chrono::high_resolution_clock::now();

    // run the alg with tol = 0.01, dim/5 batch size, print out iterations
    batch_greedy_osborne_no_alloc(d_A, tol, batch_size, d_update, h_update, idx_list, x, cNorms, rNorms, tNorms, dim, d_err,  h_err, y, tempA, 1);

    // print time of execution
    cudaDeviceSynchronize();
    auto net_end = std::chrono::high_resolution_clock::now();
    duration = net_end - net_start;
    std::cout << "The total elapsed time to balance A was " << duration.count() << "s" << std::endl;

    // copy balancing vector to host
    float* h_y = new float[dim];
    CUDA_CHECK(cudaMemcpy(h_y, y, dim * sizeof(float), cudaMemcpyDeviceToHost));

    // write balancing vector to file
    std::string y_name = "Y";
    write_array_to_file_S(h_y, y_name, dim);

    // return
    return 0;
}