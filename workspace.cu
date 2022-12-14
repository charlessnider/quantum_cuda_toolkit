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

// MATRIX NORMS
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

    // just give it to a thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size){    
        int jdx = update_list[idx];
        float val = __fmul_rn(0.5, __fsub_rn(logf(cNorms[jdx]), logf(rNorms[jdx])));
        y[jdx] = __fadd_rn(y[jdx], val);
    }
}

__global__ void balance_matrix_zero_y(float* y, int dim){

    // one thread = one element of A to adjust
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim){
        y[idx] = 0.0;
    }
}

__global__ void balance_matrix_adjust_A(cuFloatComplex* d_A, cuFloatComplex* tempA, float* y, int dim){

    // one thread = one element of A to adjust
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim * dim){

        // get indices i, j from idx
        int i = idx % dim;
        int j = (idx - i) / dim;

        // calculate Dii x invDjj
        cuFloatComplex val = make_cuFloatComplex(expf(__fsub_rn(y[i], y[j])), 0.0);
    
        // adjust Aij
        tempA[idx] = my_cuCmulf(d_A[idx], val);
    }
}

__global__ void balance_matrix_calculate_weights(float* cNorms, float* rNorms, float* weights, int dim){

    // one thread = one comparison
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim){
        weights[idx] = fabsf(__fsub_rn(sqrtf(rNorms[idx]), sqrtf(cNorms[idx])));
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

// GPU INT REDUCTION: https://cuvilib.com/Reduction.pdf
__global__ void gpu_sum_int128(int* input, int* output, int dim){

    // shared memory for the thread block for a chunk of A
    __shared__ int data[128];

    // indexing: thread index in block as well as overall index for all threads/blocks
    unsigned int t_idx = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // only launch if in the right range
    if (idx < dim){

        // move a chunk of A into shared memory
        data[t_idx] = input[idx];
        __syncthreads();

        // s = spacing between sites: first iteration, compare neighbors, then compare 2 over, then 4 over, etc
        for(unsigned int s = 1; s < blockDim.x; s *= 2){
            
            // only make the comparison if on a "zero" thread, ie one to replace with
            if (t_idx % (2 * s) == 0){

                // add the values
                data[t_idx] = data[t_idx] + data[t_idx + s];

                // synchronize threads before continuing
                __syncthreads();
            }
        }

        // when reach the end, output the very final result
        if (t_idx == 0){
            output[blockIdx.x] = data[0];
        }
    }
}

// GET THE PIVOT SHIFT FOR QUICK SORT IN ONE KERNEL
__global__ void gpu_pivot_shift128(float* finput, int* input, int* output, float* pivot, int itr, int start, int end){

    // shared memory for the thread block for a chunk of A
    __shared__ int data[128];

    // indexing: thread index in block as well as overall index for all threads/blocks
    unsigned int t_idx = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // only launch if within range of interest (all but first index = pivot)
    if (idx < end - start){

        // if first iteration, assign directly
        int arr_idx = start + idx + 1; // index of subarray
        if (itr == 0){
            if (finput[arr_idx] >= *pivot){
                data[t_idx] = 1;
            } else {
                data[t_idx] = 0;
            }
        } else { // otherwise load from idx_input = output of previous iteration
            data[t_idx] = input[idx];
        }
        __syncthreads();

        // do a basic reduction
        for(unsigned int s = 1; s < blockDim.x; s *= 2){
            
            // only make the comparison if on a "zero" thread, ie one to replace with
            if (t_idx % (2 * s) == 0){

                // add the values
                data[t_idx] = data[t_idx] + data[t_idx + s];

                // synchronize threads before continuing
                __syncthreads();
            }
        }

        // when reach the end, save the result of each block to output
        if (t_idx == 0){
            output[blockIdx.x] = data[0];
        }
    }
}

// to swap floats, ints
__device__ void gpu_swap_float(float* a, float* b){
    float temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void gpu_swap_int(int* a, int* b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

// SWAPPING INDICES
__global__ void gpu_find_swaps(float* input, int* pivot_idx, int* swapL, int* swapR, 
                               int* idxL,    int* idxR,      int start,  int end){

    // pass each value for comparison to one thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // zero out updateL, updateR
    *idxL = 0;
    *idxR = 0;

    // swap the pivot before continuing
    int s = *pivot_idx;
    gpu_swap_float(input + start, input + start + s);
    __syncthreads();

    // find the elements to swap
    if (idx < (end - start + 1)){
        if (idx < s && input[start + idx] < input[start + s]){
            int t_idx = atomicAdd(idxL, 1);
            swapL[t_idx] = idx;
        } 
        if (idx > s && input[start + idx] >= input[start + s]){
            int t_idx = atomicAdd(idxR, 1);
            swapR[t_idx] = idx;
        } 
    }
}

__global__ void gpu_make_swaps(float* input, int* pivot_idx, int* swapL, int* swapR, 
                               int* idxL,    int* idxR,      int start,  int end){
    
    // pass each value for comparison to one thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // make the appropriate swaps: will have *idxL swaps to make
    if (idx < *idxL){
        gpu_swap_float(&input[start + swapL[idx]], &input[start + swapR[idx]]);
    }
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

void gpu_partition(float* vals, int start,  int end,   int* output, 
                   int* swapL,  int* swapR, int* idxL, int* idxR){

    // number of elements in subarray
    int dim = end - start + 1;

    // if dim > 128 (more than 1 block so need to run multiple times)
    if (dim > 128){

        // first pass: full sub-array of elements
        gpu_pivot_shift128 <<< 1 + dim / 128, 128 >>> (vals, output, output, vals + start, 0, start, end);
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize()); 

        // reduce dimension
        dim = dim / 128;

        // recursively if still too many for one block
        while (dim > 128){
            
            // first pass
            gpu_pivot_shift128 <<< 1 + dim / 128, 128 >>> (vals, output, output, vals + start, 1, start, end);
            CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());  

            // reduce dimension
            dim = dim / 128;

        }

        // run once more
        gpu_pivot_shift128 <<< 1 + dim / 128, 128 >>> (vals, output, output, vals + start, 1, start, end);
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());  

    } else {

        // run just once if dim small enough
        gpu_pivot_shift128 <<< 1 + dim / 128, 128 >>> (vals, output, output, vals + start, 0, start, end);
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());  

    }

    // reset dimension
    dim = (end - start + 1);

    // find which values to swap
    gpu_find_swaps <<< 1 + dim / 128, 128 >>> (vals, output, swapL, swapR, idxL, idxR, start, end);
    CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // swap them values
    gpu_make_swaps <<< 1 + dim / 128, 128 >>> (vals, output, swapL, swapR, idxL, idxR, start, end);
    CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // copy values to host for display
    int temp;
    CUDA_CHECK(cudaMemcpy(&temp, output, sizeof(int), cudaMemcpyDeviceToHost));
    float ttemp;
    CUDA_CHECK(cudaMemcpy(&ttemp, &vals[temp], sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Device pivot index = " << temp << std::endl;
    std::cout << "Device value at pivot = " << ttemp << std::endl << std::endl;
    CUDA_CHECK(cudaMemcpy(&temp, idxL, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "idxL = " << temp << std::endl;
    CUDA_CHECK(cudaMemcpy(&temp, idxR, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "idxR = " << temp << std::endl << std::endl;

    CUDA_CHECK(cudaMemcpy(&ttemp, vals, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "New value at vals[0] = " << ttemp << std::endl << std::endl;
}

void gpu_sort(float* vals, int start,  int end,   int* output, 
              int* swapL,  int* swapR, int* idxL, int* idxR){

    // kill if start is to right of end/no more sorting to do
    if (start >= end){
        return;
    }

    // sort around the pivot
    gpu_partition(vals, start, end, output, swapL, swapR, idxL, idxR);

    // get the partition index
    int p; CUDA_CHECK(cudaMemcpy(&p, output, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << p << std::endl;

    // recursively do left and right parts
    gpu_sort(vals, start, p - 1, output, swapL, swapR, idxL, idxR);
    gpu_sort(vals, p + 1, end, output, swapL, swapR, idxL, idxR);

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

    // get scale factor itself (2^n)
    cuFloatComplex s = make_cuFloatComplex(powf(2, -(*nsquares)), 0);

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
    cuFloatComplex* work = nullptr;

    // get size of buffer
    CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(x.cusolverH, dim, dim, d_Q, dim, &lwork));

    // allocate buffer
    CUDA_CHECK(cudaMalloc(&work, lwork * sizeof(int)));

    // factorize
    CUSOLVER_CHECK(cusolverDnCgetrf(x.cusolverH, dim, dim, d_Q, dim, work, d_ipiv, devInfo));

    // solve & overwrite P with solution X (solves QX = P)
    CUSOLVER_CHECK(cusolverDnCgetrs(x.cusolverH, CUBLAS_OP_N, dim, dim, d_Q, dim, d_ipiv, d_P, dim, devInfo));

    // free memory just in case
    CUDA_CHECK(cudaFree(d_ipiv)); CUDA_CHECK(cudaFree(devInfo)); CUDA_CHECK(cudaFree(work));
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

    // free all allocated cuda memory just in case
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
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

    // free all allocated cuda memory just in case
    CUDA_CHECK(cudaFree(U1));  CUDA_CHECK(cudaFree(U2));  CUDA_CHECK(cudaFree(V1));  CUDA_CHECK(cudaFree(V2));
    CUDA_CHECK(cudaFree(A2));  CUDA_CHECK(cudaFree(A4));  CUDA_CHECK(cudaFree(A6));
    CUDA_CHECK(cudaFree(d_z)); CUDA_CHECK(cudaFree(d_id));
}

int main(){

    // size of matrices
    int dim = DIM;

    /*
    // start timing
    // auto start = std::chrono::high_resolution_clock::now();

    // load a matrix A to exponentiate
    cuFloatComplex* A = new cuFloatComplex[dim * dim];
    std::string a_name = "A";
    read_array_from_file_C(A, a_name);

    // print time of execution
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float> duration = end - start;
    // std::cout << "The total elapsed time to read A into memory was " << duration.count() << "s" << std::endl;

    // start timing
    // start = std::chrono::high_resolution_clock::now();

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

    // memory for balancing
    float* y; CUDA_CHECK(cudaMalloc(&y, dim * sizeof(float)));

    // random seed
    srand(time(NULL));

    // zero out vector y
    balance_matrix_zero_y <<< 1 + dim/128, 128 >>> (y, dim);
    CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // memory for a copy of A for iterating
    cuFloatComplex* tempA; CUDA_CHECK(cudaMalloc(&tempA, dim * dim * sizeof(cuFloatComplex)));

    // memory for column, row norms
    float* cNorms;  CUDA_CHECK(cudaMalloc(&cNorms, dim * sizeof(float)));
    float* rNorms;  CUDA_CHECK(cudaMalloc(&rNorms, dim * sizeof(float)));

    */

    // memory for greedy indexing
    float* d_weights; CUDA_CHECK(cudaMalloc(&d_weights, dim * sizeof(float)));
    float* h_weights = new float[dim];
    
    // counter and batch information for iterating (batch = 20% of matrix at a time)
    int counter = 0, batch_size = dim / 5;

    // memory for finding indices to adjust in each interation
    int* h_update = new int[dim];
    int* d_update;  CUDA_CHECK(cudaMalloc(&d_update, batch_size * sizeof(int)));
    
    // index array for sorting
    int* idx_list = new int[dim];
    for (int i = 0; i < dim; i++) idx_list[i] = i;
    memcpy(h_update, idx_list, dim * sizeof(int));

    /*
    // memory for errors
    float* err; CUDA_CHECK(cudaMalloc(&err, dim * sizeof(float)));

    // start timing
    auto net_start = std::chrono::high_resolution_clock::now();

    // calculate norms
    column_sum <<< 1 + dim/128, 128 >>> (d_A, cNorms, dim);
    CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    row_sum <<< 1 + dim/128, 128 >>> (d_A, rNorms, dim);
    CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // calculate weights of columns, rows for greedy index picking
    balance_matrix_calculate_weights <<< 1 + dim/128, 128 >>> (cNorms, rNorms, d_weights, dim);
    CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // copy weights to host
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, dim * sizeof(float), cudaMemcpyDeviceToHost));

    // calculate errors
    balance_matrix_calc_errors <<< 1 + dim/128, 128 >>> (cNorms, rNorms, err, dim);
    CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // get the maximal error = epsilon
    float epsilon = 0.0; int result = 0;
    CUBLAS_CHECK(cublasIsamax(x.cublasH, dim, err, 1, &result));
    CUDA_CHECK(cudaMemcpy(&epsilon, err + result - 1, sizeof(float), cudaMemcpyDeviceToHost));

    // tolerance for balancing & value zero for weights
    float tol = 0.01;
    */
    
    // start and end values
    int start = 0;
    int end = dim - 1;

    // randomize h_weights
    srand(time(0));
    float pivot = 50.0;
    for (int i = 0; i < dim; i++) {
        float x = (float) (rand() % 10000) / 100.0;
        while (x == pivot) x = (float) (rand() % 10000) / 100.0;
        h_weights[i] = x;
    }
    h_weights[start] = pivot;
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, dim * sizeof(float), cudaMemcpyHostToDevice));

    // memory for swaps
    int* output; CUDA_CHECK(cudaMalloc(&output, dim * sizeof(int)));
    int* swapL; CUDA_CHECK(cudaMalloc(&swapL, dim * sizeof(int))); int* swapR; CUDA_CHECK(cudaMalloc(&swapR, dim * sizeof(int)));
    int* idxL; CUDA_CHECK(cudaMalloc(&idxL, sizeof(int)));         int* idxR; CUDA_CHECK(cudaMalloc(&idxR, sizeof(int)));

    // do the sorting
    // gpu_sort(d_weights, start, end, output, swapL, swapR, idxL, idxR);

    // on gpu
    gpu_partition(d_weights, start, end, output, swapL, swapR, idxL, idxR);

    // get pivot
    int p; CUDA_CHECK(cudaMemcpy(&p, output, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, dim * sizeof(float), cudaMemcpyHostToDevice));

    // run the partition on the left sub array
    gpu_partition(d_weights, start, end, output, swapL, swapR, idxL, idxR);

    // on cpu
    p = partition(h_weights, h_update, start, end);
    std::cout << "Pivot value = " << h_weights[p] << std::endl;
    std::cout << "Pivot index = " << p << std::endl;

    // save h to file
    std::string h_name = "host";
    write_array_to_file_S(h_weights, h_name, dim);

    // write d to file
    std::string d_name = "device";
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, dim * sizeof(float), cudaMemcpyDeviceToHost));
    write_array_to_file_S(h_weights, d_name, dim);

    /*

    // loop until within tolerance, or hit 5,000 iterations
    while (epsilon > 1 + tol){

        // if go too long, kill it
        if (counter > 5000){
            std::cout << "Unable to balance within 5,000 iterations. Ending balancing and outputting most recent balancing parameters." << std::endl;
            break;
        }

        // sort the weights to get worst matches
        quick_sort(h_weights, h_update, 0, dim - 1);   

        // copy worst indices for update list to device
        CUDA_CHECK(cudaMemcpy(d_update, h_update, batch_size * sizeof(int), cudaMemcpyHostToDevice));

        // make the adjustment to y
        balance_matrix_adjust_y <<< 1 + batch_size/128, 128 >>> (y, cNorms, rNorms, d_update, batch_size); // add a -1 if getting direct from greedy indexing
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

        // do the balancing step
        balance_matrix_adjust_A <<< 1 + (dim * dim)/128, 128 >>> (d_A, tempA, y, dim);
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());     

        // calculate all norms
        column_sum <<< 1 + dim/128, 128 >>> (tempA, cNorms, dim);
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row_sum <<< 1 + dim/128, 128 >>> (tempA, rNorms, dim);
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());   

        // calculate errors
        balance_matrix_calc_errors <<< 1 + dim/128, 128 >>> (cNorms, rNorms, err, dim);
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

        // get the maximal error = epsilon
        CUBLAS_CHECK(cublasIsamax(x.cublasH, dim, err, 1, &result));
        CUDA_CHECK(cudaMemcpy(&epsilon, err + result - 1, sizeof(float), cudaMemcpyDeviceToHost));

        // calculate new weights for greedy index picking
        balance_matrix_calculate_weights <<< 1 + dim/128, 128 >>> (cNorms, rNorms, d_weights, dim);
        CUDA_CHECK(cudaPeekAtLastError()); CUDA_CHECK(cudaDeviceSynchronize());

        // copy weights to host
        CUDA_CHECK(cudaMemcpy(h_weights, d_weights, dim * sizeof(float), cudaMemcpyDeviceToHost));

        // reset update array for next sort
        memcpy(h_update, idx_list, dim * sizeof(int));

        // update counter
        counter = counter + batch_size;
    }

    // print balancing iterations
    std::cout << "The number of iterations to balance was " << counter << std::endl;

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
    */

    /*

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
    write_array_to_file_C(h_X, x_name, dim * dim);

    // print time of execution
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to write the solution to a file was " << duration.count() << "s" << std::endl;

    // print total execution time
    duration = net_end - net_start;
    std::cout << "\nThe total elapsed time to calculate expm(A) not counting preparatory steps was " << duration.count() << "s\n\n";
    */

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
    write_array_to_file_C(h_C, c_name, dim_A * dim_B * dim_A * dim_B);

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
    write_array_to_file_S(h_D, d_name, DIM);
    write_array_to_file_C(h_U, u_name, DIM * DIM);

    // print the time taken to eigensolve
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "The total elapsed time to fetch the result and write to a file was " << duration.count() << "s" << std::endl;

    // free all memory
    delete [] h_A; delete [] h_U; delete [] h_D; delete [] H;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_D));

    */