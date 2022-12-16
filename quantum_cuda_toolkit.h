/* LINEAR ALGEBRA AND ASSORTED ESSENTIAL MATH FUNCTIONS FOR QUANTUM PROBLEMS */
//
//      void kron(cuFloatComplex* A, cuFloatComplex* B, cuFloatComplex* C, int dim_A, int dim_B)
//          compute the kronecker product of A, B and save the result to C
//
//      void eigensolve(cuFloatComplex* A, cuFloatComplex* D, int dim)
//          compute eigenvalues and eigenvectors of hermitian matrix A
//          eigenvalues are saved to real valued 1d array D, eigenvectors overwrite matrix A
//
//      class cuHandles: creates cuSolver and cuBLAS handles in one go, to avoid doing it multiple times


// CLASS TO HOLD CUBLAS, CUSOLVER HANDLES
class cuHandles{
    public:
        
        // cuBLAS and cuSolver handles
        cublasHandle_t cublasH;
        cusolverDnHandle_t cusolverH;

        // initialize the handles in constructor
        cuHandles(){
            CUBLAS_CHECK(cublasCreate(&cublasH));
            CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
        }
};

// STRUCTURE TO HOLD POINTERS, MUST MANUALLY FREE AND ALLOCATE
struct dataHolder{

    // FINAL OUTPUT OF expm ALGORITHM
    cuFloatComplex* d_X;

    // PRE/POST-PROCESSING POINTERS

        // whether to display how many iterations it takes to balance the matrix
        int* print_itr;

        // host pointers
        cuFloatComplex* TrA; // trace of the matrix being exponentiated
        float* h_err; // for matrix balancing: errors of each index in matrix for greedy indexing
        float* tol; // for matrix balancing: the error toleranace
        int* nsquares; // number of squarings necessary to undo scaling
        int* h_update; // for matrix balancing: which rows/column indices to udpate at each step
        int* idx_list; // for matrix balancing: base version of h_update to reset before sorting
        int* batch_size; // for matrix balancing: the how many indices you want to update at each step

        // device pointers
        cuFloatComplex* d_mid; // the value -1
        cuFloatComplex* d_x, * d_y; // intermediate storage for squaring in post-processing
        cuFloatComplex* tempA; // for matrix balancing: temporary storage for each update of A during balancing
        float* tNorms; // temporary memory for reductions for norm calculations
        float* cNorms; // for matrix balancing: column norms of matrix to exponentiate
        float* rNorms; // for matrix balancing: row norms of matrix to exponentiate
        float* d_err; // for matrix balancing: errors of each index in matrix for greedy indexing
        float* y; // for matrix balancing: the balancing vector, output of balancing algorithm
        int* d_update; // for matrix balancing: which rows/column indices to udpate at each step

    // CALCULATION OF P AND Q POINTERS

        // host pointers
        cuFloatComplex* C; // pade approximant coefficients

        // device pointers
        cuFloatComplex* d_z; // the value zero on the device
        cuFloatComplex* d_id; // the value 1 on the device
        cuFloatComplex* A2; // to hold the square of A
        cuFloatComplex* A4; // to hold A to the 4th
        cuFloatComplex* A6; // to hold A to the 6th
        cuFloatComplex* U1; // intermediate storage for P, Q calculation
        cuFloatComplex* U2; // intermediate storage for P, Q calculation
        cuFloatComplex* V1; // intermediate storage for P, Q calculation
        cuFloatComplex* V2; // intermediate storage for P, Q calculation
        cuFloatComplex* d_P; // numerator of pade approximant
        cuFloatComplex* d_Q; // denominator of pade approximant

    // LINSOLVE POINTERS

        // device pointers
        int* d_ipiv; // pivot info for solver
        int* devInfo; // output for success/failure of solver
};

dataHolder prep_expm_memory(int dim, float batch_frac = 0.2, float tol = 0.01, int print_balancing_iterations = 0){

    // CREATE THE STRUCTURE
    dataHolder x;

    // batch size
    int batch_size = (int) (dim * batch_frac);

    // FINAL OUTPUT OF expm ALGORITHM
    CUDA_CHECK(cudaMalloc(&x.d_X, dim * dim * sizeof(cuFloatComplex)));

    // PRE/POST-PROCESSING POINTERS

        // whether to display how many iterations it takes to balance the matrix
        x.print_itr = (int*) malloc(sizeof(int));
        *(x.print_itr) = print_balancing_iterations;

        // host pointers
        x.TrA = (cuFloatComplex*) malloc(sizeof(cuFloatComplex)); // trace of the matrix being exponentiated
        x.h_err = (float*) malloc(dim * sizeof(float)); // for matrix balancing: errors of each index in matrix for greedy indexing
        x.tol = (float*) malloc(sizeof(float)); *x.tol = tol; // for matrix balancing: the error toleranace, default value of 0.01;
        x.nsquares = (int*) malloc(sizeof(int)); // number of squarings necessary to undo scaling
        x.h_update = (int*) malloc(dim * sizeof(int)); // for matrix balancing: which rows/column indices to udpate at each step
        x.idx_list = (int*) malloc(dim * sizeof(int)); // for matrix balancing: base version of h_update to reset before sorting
        x.batch_size = (int*) malloc(sizeof(int)); *(x.batch_size) = batch_size; // for matrix balancing: the how many indices you want to update at each step, default value of 1/5 the matrix at a time

        // device pointers
        CUDA_CHECK(cudaMalloc(&x.d_mid, sizeof(cuFloatComplex))); // the value -1
        CUDA_CHECK(cudaMalloc(&x.d_x, dim * dim * sizeof(cuFloatComplex))); // intermediate storage for squaring in post-processing
        CUDA_CHECK(cudaMalloc(&x.d_y, dim * dim * sizeof(cuFloatComplex))); // intermediate storage for squaring in post-processing
        CUDA_CHECK(cudaMalloc(&x.tempA, dim * dim * sizeof(cuFloatComplex))); // for matrix balancing: temporary storage for each update of A during balancing
        CUDA_CHECK(cudaMalloc(&x.tNorms, (dim * (1 + (1 + dim / 128) / 2) * sizeof(float)))); // temporary memory for reductions for norm calculations
        CUDA_CHECK(cudaMalloc(&x.cNorms, dim * sizeof(float))); // for matrix balancing: column norms of matrix to exponentiate
        CUDA_CHECK(cudaMalloc(&x.rNorms, dim * sizeof(float))); // for matrix balancing: row norms of matrix to exponentiate
        CUDA_CHECK(cudaMalloc(&x.d_err, dim * sizeof(float))); // for matrix balancing: errors of each index in matrix for greedy indexing
        CUDA_CHECK(cudaMalloc(&x.y, dim * sizeof(float))); // for matrix balancing: the balancing vector, output of balancing algorithm
        CUDA_CHECK(cudaMalloc(&x.d_update, batch_size * sizeof(int))); // for matrix balancing: which rows/column indices to udpate at each step

    // CALCULATION OF P AND Q POINTERS

        // host pointers
        x.C = (cuFloatComplex*) malloc(14 * sizeof(cuFloatComplex)); // pade approximant coefficients

        // device pointers
        CUDA_CHECK(cudaMalloc(&x.d_z, sizeof(cuFloatComplex))); // the value zero on the device
        CUDA_CHECK(cudaMalloc(&x.d_id, sizeof(cuFloatComplex))); // the value 1 on the device
        CUDA_CHECK(cudaMalloc(&x.A2, dim * dim * sizeof(cuFloatComplex))); // to hold the square of A
        CUDA_CHECK(cudaMalloc(&x.A4, dim * dim * sizeof(cuFloatComplex))); // to hold A to the 4th
        CUDA_CHECK(cudaMalloc(&x.A6, dim * dim * sizeof(cuFloatComplex))); // to hold A to the 6th
        CUDA_CHECK(cudaMalloc(&x.U1, dim * dim * sizeof(cuFloatComplex))); // intermediate storage for P, Q calculation
        CUDA_CHECK(cudaMalloc(&x.U2, dim * dim * sizeof(cuFloatComplex))); // intermediate storage for P, Q calculation
        CUDA_CHECK(cudaMalloc(&x.V1, dim * dim * sizeof(cuFloatComplex))); // intermediate storage for P, Q calculation
        CUDA_CHECK(cudaMalloc(&x.V2, dim * dim * sizeof(cuFloatComplex))); // intermediate storage for P, Q calculation
        CUDA_CHECK(cudaMalloc(&x.d_P, dim * dim * sizeof(cuFloatComplex))); // numerator of pade approximant
        CUDA_CHECK(cudaMalloc(&x.d_Q, dim * dim * sizeof(cuFloatComplex))); // denominator of pade approximant

    // LINSOLVE POINTERS

        // device pointers
        CUDA_CHECK(cudaMalloc(&x.d_ipiv, dim * sizeof(int))); // pivot info for solver
        CUDA_CHECK(cudaMalloc(&x.devInfo, sizeof(int))); // output for success/failure of solver

    // INITIALIZE VALUES

        // value 1, -1, and 0
        cuFloatComplex h_mid = make_cuFloatComplex(-1,0); CUDA_CHECK(cudaMemcpy(x.d_mid, &h_mid, sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
        cuFloatComplex h_id = make_cuFloatComplex(1, 0); CUDA_CHECK(cudaMemcpy(x.d_id, &h_id, sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
        cuFloatComplex h_z = make_cuFloatComplex(0, 0); CUDA_CHECK(cudaMemcpy(x.d_z, &h_z, sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

        // pade approximant polynomial values
        x.C[0] = make_cuFloatComplex(float(64764752532480000), float(0));
        x.C[1] = make_cuFloatComplex(float(32382376266240000), float(0));
        x.C[2] = make_cuFloatComplex(float(7771770303897600), float(0));
        x.C[3] = make_cuFloatComplex(float(1187353796428800), float(0));
        x.C[4] = make_cuFloatComplex(float(129060195264000), float(0));
        x.C[5] = make_cuFloatComplex(float(10559470521600), float(0));
        x.C[6] = make_cuFloatComplex(float(670442572800), float(0));
        x.C[7] = make_cuFloatComplex(float(33522128640), float(0));
        x.C[8] = make_cuFloatComplex(float(1323241920), float(0));
        x.C[9] = make_cuFloatComplex(float(40840800), float(0));
        x.C[10] = make_cuFloatComplex(float(960960), float(0));
        x.C[11] = make_cuFloatComplex(float(16380), float(0));
        x.C[12] = make_cuFloatComplex(float(182), float(0));
        x.C[13] = make_cuFloatComplex(float(1), float(0));

        // index list for balancing
        for (int i = 0; i < dim; i++) x.idx_list[i] = i;

    // return the structure
    return x;
}

void free_expm_memory(dataHolder x){

    // FINAL OUTPUT OF expm ALGORITHM
    CUDA_CHECK(cudaFree(x.d_X));

    // PRE/POST-PROCESSING POINTERS

        // host pointers
        free(x.TrA); // trace of the matrix being exponentiated
        free(x.h_err); // for matrix balancing: errors of each index in matrix for greedy indexing
        free(x.tol); // for matrix balancing: the error toleranace
        free(x.nsquares); // number of squarings necessary to undo scaling
        free(x.h_update); // for matrix balancing: which rows/column indices to udpate at each step
        free(x.idx_list); // for matrix balancing: base version of h_update to reset before sorting
        free(x.batch_size); // for matrix balancing: the how many indices you want to update at each step

        // device pointers
        CUDA_CHECK(cudaFree(x.d_mid)); // the value -1
        CUDA_CHECK(cudaFree(x.d_x)); // intermediate storage for squaring in post-processing
        CUDA_CHECK(cudaFree(x.d_y)); // intermediate storage for squaring in post-processing
        CUDA_CHECK(cudaFree(x.tempA)); // for matrix balancing: temporary storage for each update of A during balancing
        CUDA_CHECK(cudaFree(x.tNorms)); // temporary memory for reductions for norm calculations
        CUDA_CHECK(cudaFree(x.cNorms)); // for matrix balancing: column norms of matrix to exponentiate
        CUDA_CHECK(cudaFree(x.rNorms)); // for matrix balancing: row norms of matrix to exponentiate
        CUDA_CHECK(cudaFree(x.d_err)); // for matrix balancing: errors of each index in matrix for greedy indexing
        CUDA_CHECK(cudaFree(x.y)); // for matrix balancing: the balancing vector, output of balancing algorithm
        CUDA_CHECK(cudaFree(x.d_update)); // for matrix balancing: which rows/column indices to udpate at each step

    // CALCULATION OF P AND Q POINTERS

        // host pointers
        free(x.C); // pade approximant coefficients

        // device pointers
        CUDA_CHECK(cudaFree(x.d_z)); // the value zero on the device
        CUDA_CHECK(cudaFree(x.d_id)); // the value 1 on the device
        CUDA_CHECK(cudaFree(x.A2)); // to hold the square of A
        CUDA_CHECK(cudaFree(x.A4)); // to hold A to the 4th
        CUDA_CHECK(cudaFree(x.A6)); // to hold A to the 6th
        CUDA_CHECK(cudaFree(x.U1)); // intermediate storage for P, Q calculation
        CUDA_CHECK(cudaFree(x.U2)); // intermediate storage for P, Q calculation
        CUDA_CHECK(cudaFree(x.V1)); // intermediate storage for P, Q calculation
        CUDA_CHECK(cudaFree(x.V2)); // intermediate storage for P, Q calculation
        CUDA_CHECK(cudaFree(x.d_P)); // numerator of pade approximant
        CUDA_CHECK(cudaFree(x.d_Q)); // denominator of pade approximant

    // LINSOLVE POINTERS

        // device pointers
        CUDA_CHECK(cudaFree(x.d_ipiv)); // pivot info for solver
        CUDA_CHECK(cudaFree(x.devInfo)); // output for success/failure of solver
}

// MATRIX OPERATIONS
__global__ void kron(cuFloatComplex* A, cuFloatComplex* B, cuFloatComplex* C, int dim_A, int dim_B){

    // one thread gets each element of the product
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dim_A * dim_B * dim_A * dim_B){

        // get indices of matrix C from flattened form
        int i_C = idx % (dim_A * dim_B);
        int j_C = (idx - i_C) / (dim_A * dim_B);

        // index of matrix A to fetch
        int i_B = i_C % dim_B;
        int j_B = j_C % dim_B; 

        // index of matrix B to fetch
        int i_A = (i_C - i_B) / dim_B;
        int j_A = (j_C - j_B) / dim_B;

        // C(i,j) = A(i_A, j_A) * B(i_B, j_B) in COLUMN MAJOR ORDER (i,j) -> dim * j + i
        C[idx] = my_cuCmulf(A[dim_A * j_A + i_A], B[dim_B * j_B + i_B]);
    }
}

// EIGENSOLVER
void eigensolve(cuFloatComplex* A, float* D, int dim, cuHandles x){

    // create the solver handle
    // cusolverDnHandle_t cusolverH;
    // CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    
    // bits and pieces for solver
    int* devInfo;           // indicates type of success/failure of solver
    int n = dim;            // dimension of matrix
    int lda = dim;          // leading dimension of matrix

    // allocate memory for devInfo
    CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    // parameters for the solver here
    int lwork = 0;
    cuFloatComplex* work = nullptr;

    // whether to save the eigenvectors
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    // probably solver only uses part of matrix (since symmetric) & this specifics which half you are supplying
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_FULL; //CUBLAS_FILL_MODE_LOWER;

    // compute amount of memory needed, and allocate on device
    CUSOLVER_CHECK(cusolverDnCheevd_bufferSize(x.cusolverH, jobz, uplo, n, A, lda, D, &lwork));
    CUDA_CHECK(cudaMalloc(&work, lwork * sizeof(cuFloatComplex)));

    // (palpatine voice) do it-- can maybe use cusolverDnCheevdx to calculate only SOME eigvecs if don't need all of them
    CUSOLVER_CHECK(cusolverDnCheevd(x.cusolverH, jobz, uplo, n, A, lda, D, work, lwork, devInfo));

    // free memory on device just in case, idk
    CUDA_CHECK(cudaFree(devInfo)); CUDA_CHECK(cudaFree(work));
}