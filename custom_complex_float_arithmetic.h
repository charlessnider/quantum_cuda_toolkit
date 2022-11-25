// CUSTOM ARITHMETIC FOR ACCURACY

__device__ static __inline__ cuFloatComplex my_cuCaddf(cuFloatComplex x, cuFloatComplex y){
    float re_z = __fadd_rn(cuCrealf(x), cuCrealf(y));
    float im_z = __fadd_rn(cuCimagf(x), cuCimagf(y));

    cuFloatComplex z = make_cuFloatComplex(re_z, im_z);
    return z;
}
__device__ static __inline__ cuFloatComplex my_cuCsubf(cuFloatComplex x, cuFloatComplex y){
    float re_z = __fsub_rn(cuCrealf(x), cuCrealf(y));
    float im_z = __fsub_rn(cuCimagf(x), cuCimagf(y));

    cuFloatComplex z = make_cuFloatComplex(re_z, im_z);
    return z;
}
__device__ static __inline__ cuFloatComplex my_cuCmulf(cuFloatComplex x, cuFloatComplex y){
    float re_z_L = __fmul_rn(cuCrealf(x), cuCrealf(y));
    float re_z_R = __fmul_rn(cuCimagf(x), cuCimagf(y));
    float re_z = __fsub_rn(re_z_L, re_z_R);

    float im_z_L = __fmul_rn(cuCrealf(x), cuCimagf(y));
    float im_z_R = __fmul_rn(cuCimagf(x), cuCrealf(y));
    float im_z = __fadd_rn(im_z_L, im_z_R);

    cuFloatComplex z = make_cuFloatComplex(re_z, im_z);
    return z;
}
__device__ static __inline__ cuFloatComplex my_cuCdivf(cuFloatComplex x, cuFloatComplex y){
    // scaling
    float re_s = (float) fabs((double) cuCrealf(y));
    float im_s = (float) fabs((double) cuCimagf(y));
    float s = __fadd_rn(re_s, im_s);
    float q = __fdiv_rn(1.0f, s);

    // terms
    float ars = __fmul_rn(cuCrealf(x), q);
    float ais = __fmul_rn(cuCimagf(x), q);
    float brs = __fmul_rn(cuCrealf(y), q);
    float bis = __fmul_rn(cuCimagf(y), q);

    // second scaling
    float s_L = __fmul_rn(brs, brs);
    float s_R = __fmul_rn(bis, bis);
    s = __fadd_rn(s_L, s_R);
    q = __fdiv_rn(1.0f, s);

    // terms for final result
    float re_z_L = __fmul_rn(ars, brs);
    float re_z_R = __fmul_rn(ais, bis);
    float re_z = __fmul_rn(__fadd_rn(re_z_L, re_z_R), q);
    float im_z_L = __fmul_rn(ais, brs);
    float im_z_R = __fmul_rn(ars, bis);
    float im_z = __fmul_rn(__fsub_rn(im_z_L, im_z_R), q);

    // result
    cuFloatComplex z = make_cuFloatComplex(re_z, im_z);
    return z;

}