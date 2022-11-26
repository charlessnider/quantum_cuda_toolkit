/*          CUDA FUNCTIONS FOR ERROR CHECKING, LOGGING (TAKEN BASICALLY STRAIGHT FROM STACKOVERFLOW)         */
/*                                    https://stackoverflow.com/a/14038590                                   */
/*                     LIST CUBLAS ERRORS TAKEN FROM https://stackoverflow.com/a/13041801                    */
/*                                     LIST CUSOLVER ERRORS TAKEN FROM                                       */
/* https://github.com/NVIDIA/cuda-samples/blob/81992093d2b8c33cab22dbf6852c070c330f1715/Common/helper_cuda.h */
//
//      CUDA_CHECK(function call)
//          check for cuda errors, abort program and print cuda error string & line of cuda error to console
//
//      CUSOLVER_CHECK(function call)
//          same as above, but for a cuSolver call
//
//      CUBLAS_CHECK(funciton call)
//          same as above, but for a cuBLAS call
// 
/*                                                                                                           */

// ERROR CHECKING FUNCTIONS (THANKS STACKOVERFLOW)

#define CUDA_CHECK(ans) { gpuAssert_cuda((ans), __FILE__, __LINE__); }
inline void gpuAssert_cuda(cudaError_t code, const char *file, int line, bool abort=true){

   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const char* cublasGetErrorString(cublasStatus_t status){

    switch(status){

        case CUBLAS_STATUS_SUCCESS: 
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: 
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: 
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: 
            return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: 
            return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: 
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: 
            return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: 
            return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }

    return "UNKNOWN ERROR";
}

#define CUBLAS_CHECK(ans) { gpuAssert_cublas((ans), __FILE__, __LINE__); }
inline void gpuAssert_cublas(cublasStatus_t code, const char *file, int line, bool abort=true){

   if (code != CUBLAS_STATUS_SUCCESS) 
   {
      fprintf(stderr,"cuBLAS Error: %s %s %d\n", cublasGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const char* cusolverGetErrorString(cusolverStatus_t status){

    switch (status){

      case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";
      case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
      case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
      case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
      case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
      case CUSOLVER_STATUS_MAPPING_ERROR:
        return "CUSOLVER_STATUS_MAPPING_ERROR";
      case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
      case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
      case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
      case CUSOLVER_STATUS_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_NOT_SUPPORTED ";
      case CUSOLVER_STATUS_ZERO_PIVOT:
        return "CUSOLVER_STATUS_ZERO_PIVOT";
      case CUSOLVER_STATUS_INVALID_LICENSE:
        return "CUSOLVER_STATUS_INVALID_LICENSE";
    }
  
    return "UNKNOWN ERROR";

}

#define CUSOLVER_CHECK(ans) { gpuAssert_cusolver((ans), __FILE__, __LINE__); }
inline void gpuAssert_cusolver(cusolverStatus_t code, const char *file, int line, bool abort=true){

   if (code != CUSOLVER_STATUS_SUCCESS) 
   {
      fprintf(stderr,"cuSolver Error: %s %s %d\n", cusolverGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}