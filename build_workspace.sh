#!/bin/bash

cp /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/real_A.txt ~/quantum_cuda_toolkit/real_A.txt -r
cp /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/imag_A.txt ~/quantum_cuda_toolkit/imag_A.txt -r

/usr/local/cuda-11.8/bin/nvcc workspace.cu /usr/local/cuda-11.8/lib64/libcusolver.so /usr/local/cuda-11.8/lib64/libcublas.so /usr/local/cuda-11.8/lib64/libcublasLt.so -o test_build
./test_build

cp ~/quantum_cuda_toolkit/Y.txt /mnt/c/Users/charl/Desktop/quantum_cuda_toolkit/Y.txt -r
