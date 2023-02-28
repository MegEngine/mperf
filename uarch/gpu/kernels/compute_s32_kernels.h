R"(

// Avoiding auto-vectorize by using vector-width locked dependent code

#undef MAD_4
#undef MAD_16
#undef MAD_64

#if ISA_MAD24
#define MAD_4(x, y)     x = mad24(y, x, y); y = mad24(x, y, x); x = mad24(y, x, y); y = mad24(x, y, y);
#else
#define MAD_4(x, y)     x = (y*x) + y;      y = (x*y) + x;      x = (y*x) + y;      y = (x*y) + x;
#endif
#define MAD_16(x, y)    MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);
#define MAD_64(x, y)    MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);


// Note: some mali gpu driver may cost too long time to build_program
// when `compute_s32_v1` function have so too many integer comupte operations.
#if V1_CASE
__kernel void compute_s32_v1(__global int *ptr, int _A)
{
    int x = _A;
    int y = (int)get_local_id(0);
    
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    ptr[0] = y;
}
#else
__kernel void compute_s32_v1(__global int *ptr, int _A)
{
    int x = _A;
    int y = (int)get_local_id(0);

    MAD_64(x, y);   MAD_64(x, y); // reduce compute operation, you will got incorrect result about this test

    ptr[0] = y;
}
#endif


__kernel void compute_s32_v2(__global int *ptr, int _A)
{
    int2 x = (int2)(_A, (_A+1));
    int2 y = (int2)get_local_id(0);

    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);

    ptr[0] = (y.S0) + (y.S1);
}

__kernel void compute_s32_v4(__global int *ptr, int _A)
{
    int4 x = (int4)(_A, (_A+1), (_A+2), (_A+3));
    int4 y = (int4)get_local_id(0);

    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);

    ptr[0] = (y.S0) + (y.S1) + (y.S2) + (y.S3);
}


__kernel void compute_s32_v8(__global int *ptr, int _A)
{
    int8 x = (int8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    int8 y = (int8)get_local_id(0);

    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);

    ptr[0] = (y.S0) + (y.S1) + (y.S2) + (y.S3) + (y.S4) + (y.S5) + (y.S6) + (y.S7);
}

__kernel void compute_s32_v16(__global int *ptr, int _A)
{
    int16 x = (int16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    int16 y = (int16)get_local_id(0);

    MAD_64(x, y);
    MAD_64(x, y);

    int2 t = (y.S01) + (y.S23) + (y.S45) + (y.S67) + (y.S89) + (y.SAB) + (y.SCD) + (y.SEF);
    ptr[0] = t.S0 + t.S1;
}


)"
