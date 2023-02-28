
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


__kernel void compute_s32(__global int* ptr, int _A) {
    int4 x = (int4)(_A, (_A + 1), (_A + 2), (_A + 3));
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

__kernel void compute_s16(__global short* ptr, int _B) {
    short _A = (short)_B;
    short4 x = (short4)(_A, (_A + 1), (_A + 2), (_A + 3));
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

__kernel void compute_s8(__global char* ptr, int _B) {
    char _A = (char)_B;
    char4 x = (char4)(_A, (_A + 1), (_A + 2), (_A + 3));
    char4 y = (char4)get_local_id(0);

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
