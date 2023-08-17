#include <stdio.h>
#if defined __ANDROID__ || defined __linux__
#include <sched.h>
#if defined __ANDROID__
#include <dlfcn.h>
#endif
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>

#include <arm_neon.h>
#include "mperf/cpu_affinity.h"
#include "mperf/exception.h"
#include "mperf/timer.h"
#include "mperf/tma/tma.h"

#define ENABLE_TMA 1

/* Block sizes */
#define kc 256
#define nc 252

/* Create macros so that the matrices are stored in row-major order */

#define A(i, j) a[(i)*lda + (j)]
#define B(i, j) b[(i)*ldb + (j)]
#define C(i, j) c[(i)*ldc + (j)]

#define min(i, j) ((i) < (j) ? (i) : (j))

void my_matmul_naive(int m, int n, int k, float* a, int lda, float* b, int ldb,
                     float* c, int ldc) {
    int i, j, p;

    for (i = 0; i < m; i++) {         /* Loop over the rows of C */
        for (j = 0; j < n; j++) {     /* Loop over the columns of C */
            for (p = 0; p < k; p++) { /* Update C( i,j ) with the inner
                                         product of the ith row of A and
                                         the jth column of B */
                C(i, j) = C(i, j) + A(i, p) * B(p, j);
            }
        }
    }
}
float check(float* a, float* b, float* c, int m, int n, int k) {
    const size_t buf_size = m * n * sizeof(float);
    float* ans;
    posix_memalign((void**)&ans, 4096, buf_size);
    memset(ans, 0, m * n * sizeof(float));
    my_matmul_naive(m, n, k, a, k, b, n, ans, n);
    float max_err = -1e6;
    for (int i = 0; i < m * n; i++)
        if (std::abs(c[i] - ans[i]) > max_err) {
            max_err = std::abs(c[i] - ans[i]);
        }
    delete[] ans;
    return max_err;
}

/* Routine for computing C = A * B + C */
void AddDot8x12(int, float*, int, float*, int, float*, int);
void AddDot4x12(int, float*, int, float*, int, float*, int);
void AddDot8x4(int, float*, int, float*, int, float*, int);
void AddDot4x4(int, float*, int, float*, int, float*, int);

void InnerKernel(int, int, int, float*, int, float*, int, float*, int);

void PackMatrixB(int, float*, int, float*);

void PackMatrixA(int, float*, int, float*);

void my_matmul_pack(int m, int n, int k, float* a, int lda, float* b, int ldb,
                    float* c, int ldc) {
    int j, p, pb, ib;
    for (p = 0; p < k; p += kc) {
        pb = min(k - p, kc);
        for (j = 0; j < n; j += nc) {
            ib = min(n - j, nc);
            InnerKernel(m, ib, pb, &A(0, p), lda, &B(p, j), ldb, &C(0, j), ldc);
        }
    }
}

void PackMatrixB_12(int k, float* b, int ldb, float* b_to) {
    int j;
    for (j = 0; j < k; ++j) {
        float* b_ij_pntr = &B(j, 0);
        *b_to++ = b_ij_pntr[0];
        *b_to++ = b_ij_pntr[1];
        *b_to++ = b_ij_pntr[2];
        *b_to++ = b_ij_pntr[3];
        *b_to++ = b_ij_pntr[4];
        *b_to++ = b_ij_pntr[5];
        *b_to++ = b_ij_pntr[6];
        *b_to++ = b_ij_pntr[7];
        *b_to++ = b_ij_pntr[8];
        *b_to++ = b_ij_pntr[9];
        *b_to++ = b_ij_pntr[10];
        *b_to++ = b_ij_pntr[11];
    }
}
void PackMatrixB_4(int k, float* b, int ldb, float* b_to) {
    int j;
    for (j = 0; j < k; ++j) {
        float* b_ij_pntr = &B(j, 0);
        *b_to++ = b_ij_pntr[0];
        *b_to++ = b_ij_pntr[1];
        *b_to++ = b_ij_pntr[2];
        *b_to++ = b_ij_pntr[3];
    }
}
void PackMatrixA_8(int k, float* a, int lda, float* a_to) {
    int i;
    float *a_0i_pntr = a, *a_1i_pntr = a + lda, *a_2i_pntr = a + (lda << 1),
          *a_3i_pntr = a + (3 * lda), *a_4i_pntr = a + (lda << 2),
          *a_5i_pntr = a + (lda * 5), *a_6i_pntr = a + (lda * 6),
          *a_7i_pntr = a + (lda * 7);

    for (i = 0; i < k; ++i) {
        *a_to++ = *a_0i_pntr++;
        *a_to++ = *a_1i_pntr++;
        *a_to++ = *a_2i_pntr++;
        *a_to++ = *a_3i_pntr++;
        *a_to++ = *a_4i_pntr++;
        *a_to++ = *a_5i_pntr++;
        *a_to++ = *a_6i_pntr++;
        *a_to++ = *a_7i_pntr++;
    }
}
void PackMatrixA_4(int k, float* a, int lda, float* a_to) {
    int i;
    float *a_0i_pntr = a, *a_1i_pntr = a + lda, *a_2i_pntr = a + (lda << 1),
          *a_3i_pntr = a + (3 * lda);

    for (i = 0; i < k; ++i) {
        *a_to++ = *a_0i_pntr++;
        *a_to++ = *a_1i_pntr++;
        *a_to++ = *a_2i_pntr++;
        *a_to++ = *a_3i_pntr++;
    }
}

void InnerKernel(int m, int n, int k, float* a, int lda, float* b, int ldb,
                 float* c, int ldc) {
    int i, j;
    float packedA[m * k];
    float packedB[k * n];

    for (j = 0; j < n; j += 12) {
        if (j + 12 > n)
            break;
        PackMatrixB_12(k, &B(0, j), ldb, packedB + j * k);
        for (i = 0; i < m; i += 8) {
            if (i + 8 > m)
                break;
            if (0 == j) {
                PackMatrixA_8(k, &A(i, 0), lda, packedA + i * k);
            }
            AddDot8x12(k, packedA + i * k, k, packedB + j * k, 12, &C(i, j),
                       ldc);
        }
        if (i != m) {
            PackMatrixA_4(k, &A(i, 0), lda, packedA + i * k);
            AddDot4x12(k, packedA + i * k, k, packedB + j * k, 12, &C(i, j),
                       ldc);
        }
    }
    if (j != n) {
        for (; j < n; j += 4) {
            PackMatrixB_4(k, &B(0, j), ldb, packedB + j * k);
            for (i = 0; i < m; i += 8) {
                if (i + 8 > m)
                    break;
                AddDot8x4(k, packedA + i * k, k, packedB + j * k, 12, &C(i, j),
                          ldc);
            }
            if (i != m) {
                AddDot4x4(k, packedA + i * k, k, packedB + j * k, 12, &C(i, j),
                          ldc);
            }
        }
    }
}
void AddDot4x4(int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    float32x4_t c_0p_sum = {0};
    float32x4_t c_1p_sum = {0};
    float32x4_t c_2p_sum = {0};
    float32x4_t c_3p_sum = {0};

    register float a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

    for (int p = 0; p < k; ++p) {
        float32x4_t b_reg = vld1q_f32(b);
        b += 4;

        a_0p_reg = a[0];
        a_1p_reg = a[1];
        a_2p_reg = a[2];
        a_3p_reg = a[3];
        a += 4;

        c_0p_sum = vmlaq_n_f32(c_0p_sum, b_reg, a_0p_reg);
        c_1p_sum = vmlaq_n_f32(c_1p_sum, b_reg, a_1p_reg);
        c_2p_sum = vmlaq_n_f32(c_2p_sum, b_reg, a_2p_reg);
        c_3p_sum = vmlaq_n_f32(c_3p_sum, b_reg, a_3p_reg);
    }

    float* c_pntr = 0;
    c_pntr = &C(0, 0);
    float32x4_t c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_0p_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(1, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_1p_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(2, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_2p_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(3, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_3p_sum);
    vst1q_f32(c_pntr, c_reg);
}
void AddDot4x12(int k, float* a, int lda, float* b, int ldb, float* c,
                int ldc) {
    float32x4_t c_p00_sum = {0};
    float32x4_t c_p04_sum = {0};
    float32x4_t c_p08_sum = {0};
    float32x4_t c_p10_sum = {0};
    float32x4_t c_p14_sum = {0};
    float32x4_t c_p18_sum = {0};
    float32x4_t c_p20_sum = {0};
    float32x4_t c_p24_sum = {0};
    float32x4_t c_p28_sum = {0};
    float32x4_t c_p30_sum = {0};
    float32x4_t c_p34_sum = {0};
    float32x4_t c_p38_sum = {0};

    register float a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

    for (int p = 0; p < k; ++p) {
        float32x4_t b_reg0 = vld1q_f32(b);
        b += 4;
        float32x4_t b_reg4 = vld1q_f32(b);
        b += 4;
        float32x4_t b_reg8 = vld1q_f32(b);
        b += 4;

        a_0p_reg = a[0];
        a_1p_reg = a[1];
        a_2p_reg = a[2];
        a_3p_reg = a[3];
        a += 4;

        c_p00_sum = vmlaq_n_f32(c_p00_sum, b_reg0, a_0p_reg);
        c_p04_sum = vmlaq_n_f32(c_p04_sum, b_reg4, a_0p_reg);
        c_p08_sum = vmlaq_n_f32(c_p08_sum, b_reg8, a_0p_reg);
        c_p10_sum = vmlaq_n_f32(c_p10_sum, b_reg0, a_1p_reg);
        c_p14_sum = vmlaq_n_f32(c_p14_sum, b_reg4, a_1p_reg);
        c_p18_sum = vmlaq_n_f32(c_p18_sum, b_reg8, a_1p_reg);
        c_p20_sum = vmlaq_n_f32(c_p20_sum, b_reg0, a_2p_reg);
        c_p24_sum = vmlaq_n_f32(c_p24_sum, b_reg4, a_2p_reg);
        c_p28_sum = vmlaq_n_f32(c_p28_sum, b_reg8, a_2p_reg);
        c_p30_sum = vmlaq_n_f32(c_p30_sum, b_reg0, a_3p_reg);
        c_p34_sum = vmlaq_n_f32(c_p34_sum, b_reg4, a_3p_reg);
        c_p38_sum = vmlaq_n_f32(c_p38_sum, b_reg8, a_3p_reg);
    }

    float* c_pntr = 0;
    c_pntr = &C(0, 0);
    float32x4_t c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p00_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    float32x4_t c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p04_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    float32x4_t c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p08_sum);
    vst1q_f32(c_pntr, c_reg8);

    c_pntr = &C(1, 0);
    c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p10_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p14_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p18_sum);
    vst1q_f32(c_pntr, c_reg8);

    c_pntr = &C(2, 0);
    c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p20_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p24_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p28_sum);
    vst1q_f32(c_pntr, c_reg8);

    c_pntr = &C(3, 0);
    c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p30_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p34_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p38_sum);
    vst1q_f32(c_pntr, c_reg8);
}
void AddDot8x12(int k, float* a, int lda, float* b, int ldb, float* c,
                int ldc) {
    float32x4_t c_p00_sum = {0};
    float32x4_t c_p04_sum = {0};
    float32x4_t c_p08_sum = {0};
    float32x4_t c_p10_sum = {0};
    float32x4_t c_p14_sum = {0};
    float32x4_t c_p18_sum = {0};
    float32x4_t c_p20_sum = {0};
    float32x4_t c_p24_sum = {0};
    float32x4_t c_p28_sum = {0};
    float32x4_t c_p30_sum = {0};
    float32x4_t c_p34_sum = {0};
    float32x4_t c_p38_sum = {0};
    float32x4_t c_p40_sum = {0};
    float32x4_t c_p44_sum = {0};
    float32x4_t c_p48_sum = {0};
    float32x4_t c_p50_sum = {0};
    float32x4_t c_p54_sum = {0};
    float32x4_t c_p58_sum = {0};
    float32x4_t c_p60_sum = {0};
    float32x4_t c_p64_sum = {0};
    float32x4_t c_p68_sum = {0};
    float32x4_t c_p70_sum = {0};
    float32x4_t c_p74_sum = {0};
    float32x4_t c_p78_sum = {0};

    register float a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, a_4p_reg, a_5p_reg,
            a_6p_reg, a_7p_reg;

    for (int p = 0; p < k; ++p) {
        float32x4_t b_reg0 = vld1q_f32(b);
        b += 4;
        float32x4_t b_reg4 = vld1q_f32(b);
        b += 4;
        float32x4_t b_reg8 = vld1q_f32(b);
        b += 4;

        a_0p_reg = a[0];
        a_1p_reg = a[1];
        a_2p_reg = a[2];
        a_3p_reg = a[3];
        a_4p_reg = a[4];
        a_5p_reg = a[5];
        a_6p_reg = a[6];
        a_7p_reg = a[7];
        a += 8;

        c_p00_sum = vmlaq_n_f32(c_p00_sum, b_reg0, a_0p_reg);
        c_p04_sum = vmlaq_n_f32(c_p04_sum, b_reg4, a_0p_reg);
        c_p08_sum = vmlaq_n_f32(c_p08_sum, b_reg8, a_0p_reg);
        c_p10_sum = vmlaq_n_f32(c_p10_sum, b_reg0, a_1p_reg);
        c_p14_sum = vmlaq_n_f32(c_p14_sum, b_reg4, a_1p_reg);
        c_p18_sum = vmlaq_n_f32(c_p18_sum, b_reg8, a_1p_reg);
        c_p20_sum = vmlaq_n_f32(c_p20_sum, b_reg0, a_2p_reg);
        c_p24_sum = vmlaq_n_f32(c_p24_sum, b_reg4, a_2p_reg);
        c_p28_sum = vmlaq_n_f32(c_p28_sum, b_reg8, a_2p_reg);
        c_p30_sum = vmlaq_n_f32(c_p30_sum, b_reg0, a_3p_reg);
        c_p34_sum = vmlaq_n_f32(c_p34_sum, b_reg4, a_3p_reg);
        c_p38_sum = vmlaq_n_f32(c_p38_sum, b_reg8, a_3p_reg);
        c_p40_sum = vmlaq_n_f32(c_p40_sum, b_reg0, a_4p_reg);
        c_p44_sum = vmlaq_n_f32(c_p44_sum, b_reg4, a_4p_reg);
        c_p48_sum = vmlaq_n_f32(c_p48_sum, b_reg8, a_4p_reg);
        c_p50_sum = vmlaq_n_f32(c_p50_sum, b_reg0, a_5p_reg);
        c_p54_sum = vmlaq_n_f32(c_p54_sum, b_reg4, a_5p_reg);
        c_p58_sum = vmlaq_n_f32(c_p58_sum, b_reg8, a_5p_reg);
        c_p60_sum = vmlaq_n_f32(c_p60_sum, b_reg0, a_6p_reg);
        c_p64_sum = vmlaq_n_f32(c_p64_sum, b_reg4, a_6p_reg);
        c_p68_sum = vmlaq_n_f32(c_p68_sum, b_reg8, a_6p_reg);
        c_p70_sum = vmlaq_n_f32(c_p70_sum, b_reg0, a_7p_reg);
        c_p74_sum = vmlaq_n_f32(c_p74_sum, b_reg4, a_7p_reg);
        c_p78_sum = vmlaq_n_f32(c_p78_sum, b_reg8, a_7p_reg);
    }

    float* c_pntr = 0;
    c_pntr = &C(0, 0);
    float32x4_t c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p00_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    float32x4_t c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p04_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    float32x4_t c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p08_sum);
    vst1q_f32(c_pntr, c_reg8);

    c_pntr = &C(1, 0);
    c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p10_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p14_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p18_sum);
    vst1q_f32(c_pntr, c_reg8);

    c_pntr = &C(2, 0);
    c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p20_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p24_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p28_sum);
    vst1q_f32(c_pntr, c_reg8);

    c_pntr = &C(3, 0);
    c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p30_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p34_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p38_sum);
    vst1q_f32(c_pntr, c_reg8);

    c_pntr = &C(4, 0);
    c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p40_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p44_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p48_sum);
    vst1q_f32(c_pntr, c_reg8);

    c_pntr = &C(5, 0);
    c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p50_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p54_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p58_sum);
    vst1q_f32(c_pntr, c_reg8);

    c_pntr = &C(6, 0);
    c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p60_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p64_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p68_sum);
    vst1q_f32(c_pntr, c_reg8);

    c_pntr = &C(7, 0);
    c_reg0 = vld1q_f32(c_pntr);
    c_reg0 = vaddq_f32(c_reg0, c_p70_sum);
    vst1q_f32(c_pntr, c_reg0);
    c_pntr += 4;
    c_reg4 = vld1q_f32(c_pntr);
    c_reg4 = vaddq_f32(c_reg4, c_p74_sum);
    vst1q_f32(c_pntr, c_reg4);
    c_pntr += 4;
    c_reg8 = vld1q_f32(c_pntr);
    c_reg8 = vaddq_f32(c_reg8, c_p78_sum);
    vst1q_f32(c_pntr, c_reg8);
}
void AddDot8x4(int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    float32x4_t c_0p_sum = {0};
    float32x4_t c_1p_sum = {0};
    float32x4_t c_2p_sum = {0};
    float32x4_t c_3p_sum = {0};
    float32x4_t c_4p_sum = {0};
    float32x4_t c_5p_sum = {0};
    float32x4_t c_6p_sum = {0};
    float32x4_t c_7p_sum = {0};

    register float a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, a_4p_reg, a_5p_reg,
            a_6p_reg, a_7p_reg;

    for (int p = 0; p < k; ++p) {
        float32x4_t b_reg = vld1q_f32(b);
        b += 4;

        a_0p_reg = a[0];
        a_1p_reg = a[1];
        a_2p_reg = a[2];
        a_3p_reg = a[3];
        a_4p_reg = a[4];
        a_5p_reg = a[5];
        a_6p_reg = a[6];
        a_7p_reg = a[7];
        a += 8;

        c_0p_sum = vmlaq_n_f32(c_0p_sum, b_reg, a_0p_reg);
        c_1p_sum = vmlaq_n_f32(c_1p_sum, b_reg, a_1p_reg);
        c_2p_sum = vmlaq_n_f32(c_2p_sum, b_reg, a_2p_reg);
        c_3p_sum = vmlaq_n_f32(c_3p_sum, b_reg, a_3p_reg);
        c_4p_sum = vmlaq_n_f32(c_4p_sum, b_reg, a_4p_reg);
        c_5p_sum = vmlaq_n_f32(c_5p_sum, b_reg, a_5p_reg);
        c_6p_sum = vmlaq_n_f32(c_6p_sum, b_reg, a_6p_reg);
        c_7p_sum = vmlaq_n_f32(c_7p_sum, b_reg, a_7p_reg);
    }

    float* c_pntr = 0;
    c_pntr = &C(0, 0);
    float32x4_t c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_0p_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(1, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_1p_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(2, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_2p_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(3, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_3p_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(4, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_4p_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(5, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_5p_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(6, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_6p_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(7, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_7p_sum);
    vst1q_f32(c_pntr, c_reg);
}

void gettma(int m, int n, int k) {
    printf("----------m:%d, k:%d, n:%d----------\n", m, k, n);

    size_t iter_num = 10;

    const size_t buf_size_a = m * k * sizeof(float);
    const size_t buf_size_b = k * n * sizeof(float);
    const size_t buf_size_c = m * n * sizeof(float);
    float* a, *b, *c;
    posix_memalign((void**)&a, 4096, buf_size_a);
    posix_memalign((void**)&b, 4096, buf_size_b);
    posix_memalign((void**)&c, 4096, buf_size_c);

    for (int i = 0; i < m * k; i++)
        a[i] = (float)(rand() % 10 + rand() % 5);
    for (int i = 0; i < n * k; i++)
        b[i] = (float)(rand() % 10 + rand() % 3);
    memset(c, 0, m * n * sizeof(float));
    // warm up
    my_matmul_pack(m, n, k, a, k, b, n, c, n);
    float max_err = check(a, b, c, m, n, k);
    printf("max_err:%f\n", max_err);
    if (max_err > 0.1f) {
        mperf_throw(mperf::MperfError, "ERROR: result check error.");
    }

#if ENABLE_TMA
#if defined(__aarch64__)
    mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::A55);
    // clang-format off
    mpf_tma.init({"Frontend_Bound",
                      "Fetch_Latency", 
                          "ICache_Misses",
                          "ITLB_Misses",
                          "Predecode_Error",
                      "Fetch_Bandwidth",
                  "Bad_Speculation",
                      "Branch_Mispredicts",
                  "Backend_Bound",
                      "Memory_Bound",
                          "Load_Bound",
                              "Load_DTLB",
                              "Load_Cache",
                          "Store_Bound",
                              "Store_TLB",
                              "Store_Buffer",
                      "Core_Bound",
                          "Interlock_Bound",
                              "Interlock_AGU",
                              "Interlock_FPU",
                          "Core_Bound_Others",
                  "Retiring",
                      "LD_Retiring",
                      "ST_Retiring",
                      "DP_Retiring",
                      "ASE_Retiring",
                      "VFP_Retiring",
                      "PC_Write_Retiring",
                          "BR_IMMED_Retiring",
                          "BR_RETURN_Retiring",
                          "BR_INDIRECT_Retiring",
                "Metric_L1D_Miss_Ratio",	
	                    "Metric_L1D_RD_Miss_Ratio",
	                    "Metric_L1D_WR_Miss_Ratio",
                    "Metric_L2D_Miss_Ratio",	
	                    "Metric_L2D_RD_Miss_Ratio",
	                    "Metric_L2D_WR_Miss_Ratio",
                    "Metric_L3D_Miss_Ratio",	
	                "Metric_L3D_RD_Miss_Ratio",
                    "Metric_BR_Mispred_Ratio",
                    "Metric_L1I_TLB_Miss_Ratio",
                    "Metric_L1D_TLB_Miss_Ratio",
                    "Metric_L2_TLB_Miss_Ratio",
                    "Metric_ITLB_Table_Walk_Ratio",
                    "Metric_DTLB_Table_Walk_Ratio",
                    "Metric_Load_Port_Util",
                    "Metric_Store_Port_Util",
                    "Metric_FPU_Util",
                    "Metric_GFLOPs_Use"});
    // clang-format on
#else
    mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::HSX_SERVER);
    mpf_tma.init(
            {"Frontend_Bound", "Bad_Speculation", "Backend_Bound", "Retiring"});
#endif

    size_t gn = mpf_tma.group_num();
    size_t uncore_evt_num = mpf_tma.uncore_events_num();

    for (size_t i = 0; i < gn; ++i) {
        mpf_tma.start(i);
#endif
        for (size_t j = 0; j < iter_num; ++j) {
            my_matmul_pack(m, n, k, a, k, b, n, c, n);
        }
#if ENABLE_TMA
        mpf_tma.sample_and_stop(iter_num);
    }

    for (size_t i = 0; i < uncore_evt_num; ++i) {
        mpf_tma.start_uncore(i);
#endif
        for (size_t j = 0; j < iter_num; ++j) {
            my_matmul_pack(m, n, k, a, k, b, n, c, n);

#if ENABLE_TMA
            mpf_tma.sample(1);
#endif
        }
#if ENABLE_TMA
        // mpf_tma.sample_and_stop(iter_num);
    }
    mpf_tma.deinit();
#endif

    delete[] a;
    delete[] b;
    delete[] c;
}

int main() {
    int dev_id = 0;
    if (set_cpu_thread_affinity_spec_core(dev_id)) {
        printf("faild set thread affinity(core %d)\n", dev_id);
    }
    gettma(100, 100, 100);
    gettma(200, 200, 200);
    gettma(300, 300, 300);
    gettma(500, 500, 500);
    gettma(700, 700, 700);
    gettma(900, 900, 900);
    return 0;
}
