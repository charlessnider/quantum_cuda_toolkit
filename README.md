# quantum cuda toolkit

continuation of phd dissertation work.  pet project to write functions useful for quantum mechanics calculations that either a) don't exist in standard cuda libraries or b) are very cumbersome to use as is.

uses a lot of cuBLAS, cuSolver.

ultimately will start putting together algorithms, eg cuda based dmrg as practice in both cuda and dmrg

no promises how good or standard anything is, i'm self taught in both cuda and c++.

## current tasks

-documenting expm algorithm

## current working functions:

-expm (matrix exponent) for complex single precision: uses scaling & squaring algorithm with preprocessing using osborne's algo for matrix balancing.  current performance advantage over matlab for 5000 x 5000 = 48x faster run locally on a GTX 1080Ti.

-eigen solver for complex single precision (built in with cuSolver, but reorganized to be easier to use)

-kronecker/tensor product: parallelized with one thread => one element of the product

## using this code

if you want to use any of the code in this repo for your own projects, go ahead-- i just ask that you reference this repo when you do so.  you will note i have included links in the code for places i used for inspiration/instruction on various topics as well.
