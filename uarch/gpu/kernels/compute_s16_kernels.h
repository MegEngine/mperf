R"(

// Avoiding auto-vectorize by using vector-width locked dependent code

#undef MAD_4
#undef MAD_16
#undef MAD_64

#define MAD_4(x, y)     x = (y*x) + y;      y = (x*y) + x;      x = (y*x) + y;      y = (x*y) + x;
#define MAD_16(x, y)    MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);
#define MAD_64(x, y)    MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);

// Note: some mali gpu driver may cost too long time to build_program
// when `compute_s16_v1` function have so too many integer comupte operations.
#if V1_CASE
__kernel void compute_s16_v1(__global short *ptr, int _B)
{
    short _A = (short)_B;
    short x = _A;
    short y = (short)get_local_id(0);

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
__kernel void compute_s16_v1(__global short *ptr, int _B)
{
    short _A = (short)_B;
    short x = _A;
    short y = (short)get_local_id(0);

    MAD_64(x, y);   MAD_64(x, y); // reduce compute operation, you will got incorrect result about this test

    ptr[0] = y;
}
#endif

__kernel void compute_s16_v2(__global short *ptr, int _B)
{
    short _A = (short)_B;
    short2 x = (short2)(_A, (_A+1));
    short2 y = (short2)get_local_id(0);

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

__kernel void compute_s16_v4(__global short *ptr, int _B)
{
    short _A = (short)_B;
    short4 x = (short4)(_A, (_A+1), (_A+2), (_A+3));
    short4 y = (short4)get_local_id(0);

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


__kernel void compute_s16_v8(__global short *ptr, int _B)
{
    short _A = (short)_B;
    short8 x = (short8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    short8 y = (short8)get_local_id(0);

    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);

    ptr[0] = (y.S0) + (y.S1) + (y.S2) + (y.S3) + (y.S4) + (y.S5) + (y.S6) + (y.S7);
}

__kernel void compute_s16_v16(__global short *ptr, int _B)
{
    short _A = (short)_B;
    short16 x = (short16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    short16 y = (short16)get_local_id(0);

    MAD_64(x, y);
    MAD_64(x, y);

    short2 t = (y.S01) + (y.S23) + (y.S45) + (y.S67) + (y.S89) + (y.SAB) + (y.SCD) + (y.SEF);
    ptr[0] = t.S0 + t.S1;
}


)"
