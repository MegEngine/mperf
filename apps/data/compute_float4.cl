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

#undef FETCH_2
#undef FETCH_8

#define FETCH_2(sum, id, A, jumpBy)  sum += A[id];  id += jumpBy;   sum += A[id];   id += jumpBy;
#define FETCH_4(sum, id, A, jumpBy)  FETCH_2(sum, id, A, jumpBy);   FETCH_2(sum, id, A, jumpBy);
#define FETCH_8(sum, id, A, jumpBy)  FETCH_4(sum, id, A, jumpBy);   FETCH_4(sum, id, A, jumpBy);

#define FETCH_PER_WI  16

__kernel void compute(__global float* ptr, float _A) {
    float4 x = (float4)(_A, (_A + 1), (_A + 2), (_A + 3));
    float4 y = (float4)get_local_id(0);

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
