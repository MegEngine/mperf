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

#define A(i, j) a[(i)*lda + (j)]
#define B(i, j) b[(i)*ldb + (j)]
#define C(i, j) c[(i)*ldc + (j)]

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
    posix_memalign((void**)(&ans), 4096, buf_size);
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

void my_matmul_unroll(int m, int n, int k, float* a, int lda, float* b, int ldb,
                      float* c, int ldc) {
    int i, j;

    for (j = 0; j < n; j += 12) {
        if (j + 12 > n)
            break;
        for (i = 0; i < m; i += 8) {
            if (i + 8 > m)
                break;
            AddDot8x12(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
        if (i != m) {
            AddDot4x12(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
    if (j != n) {
        for (; j < n; j += 4) {
            for (i = 0; i < m; i += 8) {
                if (i + 8 > m)
                    break;
                AddDot8x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
            }
            if (i != m) {
                AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
            }
        }
    }
}

void AddDot8x12(int k, float* a, int lda, float* b, int ldb, float* c,
                int ldc) {
    float *a_0p_pntr, *a_1p_pntr, *a_2p_pntr, *a_3p_pntr, *a_4p_pntr,
            *a_5p_pntr, *a_6p_pntr, *a_7p_pntr;

    a_0p_pntr = &A(0, 0);
    a_1p_pntr = &A(1, 0);
    a_2p_pntr = &A(2, 0);
    a_3p_pntr = &A(3, 0);
    a_4p_pntr = &A(4, 0);
    a_5p_pntr = &A(5, 0);
    a_6p_pntr = &A(6, 0);
    a_7p_pntr = &A(7, 0);

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
        float32x4_t b_reg0 = vld1q_f32(&B(p, 0));
        float32x4_t b_reg4 = vld1q_f32(&B(p, 4));
        float32x4_t b_reg8 = vld1q_f32(&B(p, 8));

        a_0p_reg = *a_0p_pntr++;
        a_1p_reg = *a_1p_pntr++;
        a_2p_reg = *a_2p_pntr++;
        a_3p_reg = *a_3p_pntr++;
        a_4p_reg = *a_4p_pntr++;
        a_5p_reg = *a_5p_pntr++;
        a_6p_reg = *a_6p_pntr++;
        a_7p_reg = *a_7p_pntr++;

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
void AddDot4x12(int k, float* a, int lda, float* b, int ldb, float* c,
                int ldc) {
    float *a_0p_pntr, *a_1p_pntr, *a_2p_pntr, *a_3p_pntr;

    a_0p_pntr = &A(0, 0);
    a_1p_pntr = &A(1, 0);
    a_2p_pntr = &A(2, 0);
    a_3p_pntr = &A(3, 0);

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
        float32x4_t b_reg0 = vld1q_f32(&B(p, 0));
        float32x4_t b_reg4 = vld1q_f32(&B(p, 4));
        float32x4_t b_reg8 = vld1q_f32(&B(p, 8));

        a_0p_reg = *a_0p_pntr++;
        a_1p_reg = *a_1p_pntr++;
        a_2p_reg = *a_2p_pntr++;
        a_3p_reg = *a_3p_pntr++;

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
void AddDot4x4(int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    /* So, this routine computes a 4x4 block of matrix A
             C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).
             C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).
             C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).
             C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).
       Notice that this routine is called with c = C( i, j ) in the
       previous routine, so these are actually the elements
             C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 )
             C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 )
             C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 )
             C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 )
       in the original matrix C
       In this version, we use registers for elements in the current row
       of B as well */

    float
            /* Point to the current elements in the four rows of A */
            *a_0p_pntr,
            *a_1p_pntr, *a_2p_pntr, *a_3p_pntr;

    a_0p_pntr = &A(0, 0);
    a_1p_pntr = &A(1, 0);
    a_2p_pntr = &A(2, 0);
    a_3p_pntr = &A(3, 0);

    float32x4_t c_p0_sum = {0};
    float32x4_t c_p1_sum = {0};
    float32x4_t c_p2_sum = {0};
    float32x4_t c_p3_sum = {0};

    register float a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

    for (int p = 0; p < k; ++p) {
        float32x4_t b_reg = vld1q_f32(&B(p, 0));

        a_0p_reg = *a_0p_pntr++;
        a_1p_reg = *a_1p_pntr++;
        a_2p_reg = *a_2p_pntr++;
        a_3p_reg = *a_3p_pntr++;

        c_p0_sum = vmlaq_n_f32(c_p0_sum, b_reg, a_0p_reg);
        c_p1_sum = vmlaq_n_f32(c_p1_sum, b_reg, a_1p_reg);
        c_p2_sum = vmlaq_n_f32(c_p2_sum, b_reg, a_2p_reg);
        c_p3_sum = vmlaq_n_f32(c_p3_sum, b_reg, a_3p_reg);
    }

    float* c_pntr = 0;
    c_pntr = &C(0, 0);
    float32x4_t c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p0_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(1, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p1_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(2, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p2_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(3, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p3_sum);
    vst1q_f32(c_pntr, c_reg);
}
void AddDot8x4(int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    float *a_0p_pntr, *a_1p_pntr, *a_2p_pntr, *a_3p_pntr, *a_4p_pntr,
            *a_5p_pntr, *a_6p_pntr, *a_7p_pntr;

    a_0p_pntr = &A(0, 0);
    a_1p_pntr = &A(1, 0);
    a_2p_pntr = &A(2, 0);
    a_3p_pntr = &A(3, 0);
    a_4p_pntr = &A(4, 0);
    a_5p_pntr = &A(5, 0);
    a_6p_pntr = &A(6, 0);
    a_7p_pntr = &A(7, 0);

    float32x4_t c_p0_sum = {0};
    float32x4_t c_p1_sum = {0};
    float32x4_t c_p2_sum = {0};
    float32x4_t c_p3_sum = {0};
    float32x4_t c_p4_sum = {0};
    float32x4_t c_p5_sum = {0};
    float32x4_t c_p6_sum = {0};
    float32x4_t c_p7_sum = {0};

    register float a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, a_4p_reg, a_5p_reg,
            a_6p_reg, a_7p_reg;

    for (int p = 0; p < k; ++p) {
        float32x4_t b_reg = vld1q_f32(&B(p, 0));

        a_0p_reg = *a_0p_pntr++;
        a_1p_reg = *a_1p_pntr++;
        a_2p_reg = *a_2p_pntr++;
        a_3p_reg = *a_3p_pntr++;
        a_4p_reg = *a_4p_pntr++;
        a_5p_reg = *a_5p_pntr++;
        a_6p_reg = *a_6p_pntr++;
        a_7p_reg = *a_7p_pntr++;

        c_p0_sum = vmlaq_n_f32(c_p0_sum, b_reg, a_0p_reg);
        c_p1_sum = vmlaq_n_f32(c_p1_sum, b_reg, a_1p_reg);
        c_p2_sum = vmlaq_n_f32(c_p2_sum, b_reg, a_2p_reg);
        c_p3_sum = vmlaq_n_f32(c_p3_sum, b_reg, a_3p_reg);
        c_p4_sum = vmlaq_n_f32(c_p4_sum, b_reg, a_4p_reg);
        c_p5_sum = vmlaq_n_f32(c_p5_sum, b_reg, a_5p_reg);
        c_p6_sum = vmlaq_n_f32(c_p6_sum, b_reg, a_6p_reg);
        c_p7_sum = vmlaq_n_f32(c_p7_sum, b_reg, a_7p_reg);
    }

    float* c_pntr = 0;
    c_pntr = &C(0, 0);
    float32x4_t c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p0_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(1, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p1_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(2, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p2_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(3, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p3_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(4, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p4_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(5, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p5_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(6, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p6_sum);
    vst1q_f32(c_pntr, c_reg);

    c_pntr = &C(7, 0);
    c_reg = vld1q_f32(c_pntr);
    c_reg = vaddq_f32(c_reg, c_p7_sum);
    vst1q_f32(c_pntr, c_reg);
}

void gettma(int m, int n, int k) {
    printf("----------m:%d, k:%d, n:%d----------\n", m, k, n);

    size_t iter_num = 10;

    const size_t buf_size_a = m * k * sizeof(float);
    const size_t buf_size_b = k * n * sizeof(float);
    const size_t buf_size_c = m * n * sizeof(float);
    float *a, *b, *c;
    posix_memalign((void**)&a, 4096, buf_size_a);
    posix_memalign((void**)&b, 4096, buf_size_b);
    posix_memalign((void**)&c, 4096, buf_size_c);

    for (int i = 0; i < m * k; i++)
        a[i] = (float)(rand() % 10);
    for (int i = 0; i < n * k; i++)
        b[i] = (float)(rand() % 10);
    memset(c, 0, m * n * sizeof(float));
    // warm up
    my_matmul_unroll(m, n, k, a, k, b, n, c, n);
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
            my_matmul_unroll(m, n, k, a, k, b, n, c, n);
        }
#if ENABLE_TMA
        mpf_tma.sample_and_stop(iter_num);
    }

    for (size_t i = 0; i < uncore_evt_num; ++i) {
        mpf_tma.start_uncore(i);
#endif
        for (size_t j = 0; j < iter_num; ++j) {
            my_matmul_unroll(m, n, k, a, k, b, n, c, n);

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
