#!/bin/bash

cp /mnt/c/Users/charl/Desktop/eigensolve_testing/real_H.txt ~/friendly_neighborhood_eigensolver/real_H.txt -r
cp /mnt/c/Users/charl/Desktop/eigensolve_testing/imag_H.txt ~/friendly_neighborhood_eigensolver/imag_H.txt -r

/usr/local/cuda-11.8/bin/nvcc eigen_solver.cu /usr/local/cuda-11.8/lib64/libcusolver.so /usr/local/cuda-11.8/lib64/libcublas.so /usr/local/cuda-11.8/lib64/libcublasLt.so -o eigen_solver
./eigen_solver

cp ~/friendly_neighborhood_eigensolver/real_U.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/real_U.txt -r
cp ~/friendly_neighborhood_eigensolver/imag_U.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/imag_U.txt -r
cp ~/friendly_neighborhood_eigensolver/D.txt /mnt/c/Users/charl/Desktop/eigensolve_testing/D.txt -r