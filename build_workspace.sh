#!/bin/bash

cp /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/real_A.txt ~/quantum_cuda_toolkit/real_A.txt -r
cp /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/imag_A.txt ~/quantum_cuda_toolkit/imag_A.txt -r

# cp /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/real_A.txt ~/quantum_cuda_toolkit/real_A.txt -r
# cp /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/imag_A.txt ~/quantum_cuda_toolkit/imag_A.txt -r
# cp /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/real_B.txt ~/quantum_cuda_toolkit/real_B.txt -r
# cp /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/imag_B.txt ~/quantum_cuda_toolkit/imag_B.txt -r

/usr/local/cuda-11.8/bin/nvcc workspace.cu /usr/local/cuda-11.8/lib64/libcusolver.so /usr/local/cuda-11.8/lib64/libcublas.so /usr/local/cuda-11.8/lib64/libcublasLt.so -o workspace
./workspace

cp ~/quantum_cuda_toolkit/real_X.txt /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/real_X.txt -r
cp ~/quantum_cuda_toolkit/imag_X.txt /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/imag_X.txt -r
cp ~/quantum_cuda_toolkit/normA.txt /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/normA.txt -r
# cp ~/quantum_cuda_toolkit/real_U.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/real_U.txt -r
# cp ~/quantum_cuda_toolkit/imag_U.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/imag_U.txt -r
# cp ~/quantum_cuda_toolkit/D.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/D.txt -r