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