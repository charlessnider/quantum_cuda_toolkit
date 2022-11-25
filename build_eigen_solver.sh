#!/bin/bash

cp /mnt/c/Users/charl/Desktop/eigensolve_testing/real_H.txt ~/friendly_neighborhood_eigensolver/real_H.txt -r
cp /mnt/c/Users/charl/Desktop/eigensolve_testing/imag_H.txt ~/friendly_neighborhood_eigensolver/imag_H.txt -r

# cp /mnt/c/Users/charl/Desktop/eigensolve_testing/real_A.txt ~/friendly_neighborhood_eigensolver/real_A.txt -r
# cp /mnt/c/Users/charl/Desktop/eigensolve_testing/imag_A.txt ~/friendly_neighborhood_eigensolver/imag_A.txt -r
# cp /mnt/c/Users/charl/Desktop/eigensolve_testing/real_B.txt ~/friendly_neighborhood_eigensolver/real_B.txt -r
# cp /mnt/c/Users/charl/Desktop/eigensolve_testing/imag_B.txt ~/friendly_neighborhood_eigensolver/imag_B.txt -r

/usr/local/cuda-11.8/bin/nvcc eigen_solver.cu /usr/local/cuda-11.8/lib64/libcusolver.so /usr/local/cuda-11.8/lib64/libcublas.so /usr/local/cuda-11.8/lib64/libcublasLt.so -o eigen_solver
./eigen_solver

# cp ~/friendly_neighborhood_eigensolver/real_C.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/real_C.txt -r
# cp ~/friendly_neighborhood_eigensolver/imag_C.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/imag_C.txt -r
cp ~/friendly_neighborhood_eigensolver/real_C.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/real_C.txt -r
cp ~/friendly_neighborhood_eigensolver/imag_C.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/imag_C.txt -r
cp ~/friendly_neighborhood_eigensolver/D.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/D.txt -r