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
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            for (p = 0; p < k; p++) {
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

void Kern_8x12(const float* packA, const float* packB, int K, float* output,
               int LDC, bool is_first_k);
void Kern_4x12(const float* packA, const float* packB, int K, float* output,
               int LDC, bool is_first_k, int m_remain);
void Kern_4x4(const float* packA, const float* packB, int K, float* output,
              int LDC, bool is_first_k, int m_remain, int n_remain);
void Kern_8x4(const float* packA, const float* packB, int K, float* output,
              int LDC, bool is_first_k, int n_remain);

void InnerKernel(int, int, int, float*, int, float*, int, float*, int, bool);

void PackMatrixB_12(int, float*, int, float*);
void PackMatrixB_4(int, float*, int, float*);
void PackMatrixA_8(int, float*, int, float*);
void PackMatrixA_4(int, float*, int, float*);

void PackMatrixA(int, float*, int, float*);

void my_matmul_asm(int m, int n, int k, float* a, int lda, float* b, int ldb,
                   float* c, int ldc) {
    int j, p, pb, ib;
    for (p = 0; p < k; p += kc) {
        pb = min(k - p, kc);
        bool is_first_k = (p == 0) ? 1 : 0;
        for (j = 0; j < n; j += nc) {
            ib = min(n - j, nc);
            InnerKernel(m, ib, pb, &A(0, p), lda, &B(p, j), ldb, &C(0, j), ldc,
                        is_first_k);
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
                 float* c, int ldc, bool is_first_k) {
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
            Kern_8x12(packedA + i * k, packedB + j * k, k, &C(i, j), ldc,
                      is_first_k);
        }
        if (i != m) {
            if (0 == j)
                PackMatrixA_4(k, &A(i, 0), lda, packedA + i * k);
            Kern_4x12(packedA + i * k, packedB + j * k, k, &C(i, j), ldc,
                      is_first_k, 4);
        }
    }
    if (j != n) {
        for (; j < n; j += 4) {
            PackMatrixB_4(k, &B(0, j), ldb, packedB + j * k);
            for (i = 0; i < m; i += 8) {
                if (i + 8 > m)
                    break;
                Kern_8x4(packedA + i * k, packedB + j * k, k, &C(i, j), ldc,
                         is_first_k, n - j);
            }
            if (i != m) {
                Kern_4x4(packedA + i * k, packedB + j * k, k, &C(i, j), ldc,
                         is_first_k, m - i, n - j);
            }
        }
    }
}

void Kern_8x12(const float* packA, const float* packB, int K, float* output,
               int LDC, bool is_first_k) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;
    LDC = LDC * sizeof(float);
    register float* outptr asm("x0") = reinterpret_cast<float*>(output);
// clang-format off
#define LOAD_LINE(v0, v1, v2, n)                            \
    "ld1 {v" v0 ".4s, v" v1 ".4s, v" v2 ".4s}, [x" n "]\n"  \

#define LOAD_C                        \
    LOAD_LINE("8", "9", "10", "0")    \
    LOAD_LINE("11", "12", "13", "1")  \
    LOAD_LINE("14", "15", "16", "2")  \
    LOAD_LINE("17", "18", "19", "3")  \
    LOAD_LINE("20", "21", "22", "4")  \
    LOAD_LINE("23", "24", "25", "5")  \
    LOAD_LINE("26", "27", "28", "6")  \
    LOAD_LINE("29", "30", "31", "7")

    // clang-format on
    asm volatile(
            // load accumulator C
            "add x1, x0, %x[LDC]\n"
            "prfm pldl1keep, [%[a_ptr]]\n"
            "add x2, x1, %x[LDC]\n"
            "prfm pldl1keep, [%[b_ptr]]\n"
            "add x3, x2, %x[LDC]\n"
            "prfm pldl1keep, [%[a_ptr], #64]\n"
            "add x4, x3, %x[LDC]\n"
            "prfm pldl1keep, [%[a_ptr], #128]\n"
            "add x5, x4, %x[LDC]\n"
            "prfm pldl1keep, [%[a_ptr], #192]\n"
            "add x6, x5, %x[LDC]\n"
            "prfm pldl1keep, [%[a_ptr], #256]\n"
            "add x7, x6, %x[LDC]\n"

            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C
            "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "b 2f\n"

            "1:\n"
            "eor v8.16b, v8.16b, v8.16b\n"
            "ldr d2, [%[b_ptr]]\n"

            "eor v9.16b, v9.16b, v9.16b\n"
            "ldr x10, [%[b_ptr], #8]\n"

            "eor v10.16b, v10.16b, v10.16b\n"
            "ldr d3, [%[b_ptr], #16]\n"

            "eor v11.16b, v11.16b, v11.16b\n"
            "ldr x11, [%[b_ptr], #24]\n"

            "eor v12.16b, v12.16b, v12.16b\n"
            "ldr d4, [%[b_ptr], #32]\n"

            "eor v13.16b, v13.16b, v13.16b\n"
            "ldr x12, [%[b_ptr], #40]\n"

            "eor v14.16b, v14.16b, v14.16b\n"
            "ldr d0, [%[a_ptr]]\n"

            "eor v15.16b, v15.16b, v15.16b\n"
            "ldr x9, [%[a_ptr], #8]\n"

            "eor v16.16b, v16.16b, v16.16b\n"
            "add %[b_ptr], %[b_ptr], #48\n"

            "eor v17.16b, v17.16b, v17.16b\n"
            "add %[a_ptr], %[a_ptr], #16\n"

            "eor v18.16b, v18.16b, v18.16b\n"
            "ins v2.d[1], x10\n"

            "eor v19.16b, v19.16b, v19.16b\n"
            "ins v3.d[1], x11\n"

            "eor v20.16b, v20.16b, v20.16b\n"
            "ins v4.d[1], x12\n"

            "eor v21.16b, v21.16b, v21.16b\n"
            "ins v0.d[1], x9\n"

            "eor v22.16b, v22.16b, v22.16b\n"
            "prfm pldl1keep, [%[a_ptr], #384]\n"

            "eor v23.16b, v23.16b, v23.16b\n"
            "prfm pldl1keep, [%[b_ptr]]\n"

            "eor v24.16b, v24.16b, v24.16b\n"
            "prfm pldl1keep, [%[b_ptr], #64]\n"

            "eor v25.16b, v25.16b, v25.16b\n"
            "prfm pldl1keep, [%[b_ptr], #128]\n"

            "eor v26.16b, v26.16b, v26.16b\n"
            "prfm pldl1keep, [%[b_ptr], #192]\n"

            "eor v27.16b, v27.16b, v27.16b\n"
            "prfm pldl1keep, [%[b_ptr], #256]\n"

            "eor v28.16b, v28.16b, v28.16b\n"
            "prfm pldl1keep, [%[b_ptr], #320]\n"

            "eor v29.16b, v29.16b, v29.16b\n"
            "prfm pldl1keep, [%[b_ptr], #384]\n"

            "eor v30.16b, v30.16b, v30.16b\n"
            "prfm pldl1keep, [%[b_ptr], #448]\n"

            "eor v31.16b, v31.16b, v31.16b\n"
            "prfm pldl1keep, [%[b_ptr], #512]\n"

            "2: \n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"

            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "ldr d1, [%[a_ptr]]\n"

            "fmla v9.4s,  v3.4s, v0.s[0]\n"
            "subs %w[K], %w[K], #1\n"

            "fmla v10.4s, v4.4s, v0.s[0]\n"
            "ldr x8, [%[a_ptr], #8]\n"

            "fmla v11.4s, v2.4s, v0.s[1]\n"

            "fmla v12.4s, v3.4s, v0.s[1]\n"
            "ldr d5, [%[b_ptr]]\n"

            "fmla v13.4s, v4.4s, v0.s[1]\n"
            "ins v1.d[1], x8\n"

            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "ldr x10, [%[b_ptr], #8]\n"

            "fmla v15.4s, v3.4s, v0.s[2]\n"

            "fmla v16.4s, v4.4s, v0.s[2]\n"
            "ldr d6, [%[b_ptr], #16]\n"

            "fmla v17.4s, v2.4s, v0.s[3]\n"
            "ins v5.d[1], x10\n"

            "fmla v18.4s, v3.4s, v0.s[3]\n"
            "ldr x11, [%[b_ptr], #24]\n"

            "fmla v19.4s, v4.4s, v0.s[3]\n"

            "fmla v20.4s, v2.4s, v1.s[0]\n"
            "ldr d7, [%[b_ptr], #32]\n"

            "fmla v21.4s, v3.4s, v1.s[0]\n"
            "ins v6.d[1], x11\n"

            "fmla v22.4s, v4.4s, v1.s[0]\n"
            "ldr d0, [%[a_ptr], #16]\n"

            "fmla v23.4s, v2.4s, v1.s[1]\n"

            "fmla v24.4s, v3.4s, v1.s[1]\n"
            "ldr x12, [%[b_ptr], #40]\n"

            "fmla v25.4s, v4.4s, v1.s[1]\n"

            "fmla v26.4s, v2.4s, v1.s[2]\n"
            "ldr x9, [%[a_ptr], #24]\n"

            "fmla v27.4s, v3.4s, v1.s[2]\n"
            "ins v7.d[1], x12\n"

            "fmla v28.4s, v4.4s, v1.s[2]\n"
            "prfm pldl1keep, [%[a_ptr], #448]\n"

            "fmla v29.4s, v2.4s, v1.s[3]\n"
            "ins v0.d[1], x9\n"

            "fmla v30.4s, v3.4s, v1.s[3]\n"
            "prfm pldl1keep, [%[b_ptr], #576]\n"

            "fmla v31.4s, v4.4s, v1.s[3]\n"

            //! UNROLL
            "fmla v8.4s,  v5.4s, v0.s[0]\n"
            "ldr d1, [%[a_ptr], #32]\n"

            "fmla v9.4s,  v6.4s, v0.s[0]\n"

            "fmla v10.4s, v7.4s, v0.s[0]\n"
            "ldr x8, [%[a_ptr], #40]\n"

            "fmla v11.4s, v5.4s, v0.s[1]\n"

            "fmla v12.4s, v6.4s, v0.s[1]\n"
            "ldr d2, [%[b_ptr], #48]\n"

            "fmla v13.4s, v7.4s, v0.s[1]\n"
            "ins v1.d[1], x8\n"

            "fmla v14.4s, v5.4s, v0.s[2]\n"
            "ldr x10, [%[b_ptr], #56]\n"

            "fmla v15.4s, v6.4s, v0.s[2]\n"

            "fmla v16.4s, v7.4s, v0.s[2]\n"
            "ldr d3, [%[b_ptr], #64]\n"

            "fmla v17.4s, v5.4s, v0.s[3]\n"
            "ins v2.d[1], x10\n"

            "fmla v18.4s, v6.4s, v0.s[3]\n"
            "ldr x11, [%[b_ptr], #72]\n"

            "fmla v19.4s, v7.4s, v0.s[3]\n"

            "fmla v20.4s, v5.4s, v1.s[0]\n"
            "ldr d4, [%[b_ptr], #80]\n"

            "fmla v21.4s, v6.4s, v1.s[0]\n"
            "ins v3.d[1], x11\n"

            "fmla v22.4s, v7.4s, v1.s[0]\n"
            "ldr x12, [%[b_ptr], #88]\n"

            "fmla v23.4s, v5.4s, v1.s[1]\n"
            "add %[b_ptr], %[b_ptr], #96\n"

            "fmla v24.4s, v6.4s, v1.s[1]\n"
            "ldr d0, [%[a_ptr], #48]\n"

            "fmla v25.4s, v7.4s, v1.s[1]\n"
            "ins v4.d[1], x12\n"

            "fmla v26.4s, v5.4s, v1.s[2]\n"
            "ldr x9, [%[a_ptr], #56]\n"

            "fmla v27.4s, v6.4s, v1.s[2]\n"
            "add %[a_ptr], %[a_ptr], #64\n"

            "fmla v28.4s, v7.4s, v1.s[2]\n"
            "prfm pldl1keep, [%[b_ptr], #640]\n"

            "fmla v29.4s, v5.4s, v1.s[3]\n"
            "ins v0.d[1], x9\n"

            "fmla v30.4s, v6.4s, v1.s[3]\n"

            "fmla v31.4s, v7.4s, v1.s[3]\n"

            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "prfm pstl1keep, [x0]\n"

            "fmla v9.4s,  v3.4s, v0.s[0]\n"
            "ldr d1, [%[a_ptr]] \n"

            "fmla v10.4s, v4.4s, v0.s[0]\n"
            "prfm pstl1keep, [x1]\n"

            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "ldr x8, [%[a_ptr], #8] \n"

            "fmla v12.4s, v3.4s, v0.s[1]\n"
            "prfm pstl1keep, [x2]\n"

            "fmla v13.4s, v4.4s, v0.s[1]\n"
            "ldr d5, [%[b_ptr]]\n"

            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "ins v1.d[1], x8\n"

            "fmla v15.4s, v3.4s, v0.s[2]\n"
            "ldr x10, [%[b_ptr], #8]\n"

            "fmla v16.4s, v4.4s, v0.s[2]\n"

            "fmla v17.4s, v2.4s, v0.s[3]\n"
            "ldr d6, [%[b_ptr], #16]\n"

            "fmla v18.4s, v3.4s, v0.s[3]\n"
            "ins v5.d[1], x10\n"

            "fmla v19.4s, v4.4s, v0.s[3]\n"
            "ldr x11, [%[b_ptr], #24]\n"

            "fmla v20.4s, v2.4s, v1.s[0]\n"

            "fmla v21.4s, v3.4s, v1.s[0]\n"
            "ldr d0, [%[a_ptr], #16]\n"

            "fmla v22.4s, v4.4s, v1.s[0]\n"
            "ins v6.d[1], x11\n"

            "fmla v23.4s, v2.4s, v1.s[1]\n"
            "ldr x9, [%[a_ptr], #24]\n"

            "fmla v24.4s, v3.4s, v1.s[1]\n"

            "fmla v25.4s, v4.4s, v1.s[1]\n"
            "ldr d7, [%[b_ptr], #32]\n"

            "fmla v26.4s, v2.4s, v1.s[2]\n"
            "ins v0.d[1], x9\n"

            "fmla v27.4s, v3.4s, v1.s[2]\n"
            "ldr x12, [%[b_ptr], #40]\n"

            "fmla v28.4s, v4.4s, v1.s[2]\n"

            "fmla v29.4s, v2.4s, v1.s[3]\n"

            "fmla v30.4s, v3.4s, v1.s[3]\n"
            "ins v7.d[1], x12\n"

            "fmla v31.4s, v4.4s, v1.s[3]\n"

            "fmla v8.4s,  v5.4s, v0.s[0]\n"
            "ldr d1, [%[a_ptr], #32]\n"

            "fmla v9.4s,  v6.4s, v0.s[0]\n"

            "fmla v10.4s, v7.4s, v0.s[0]\n"
            "ldr x8, [%[a_ptr], #40]\n"

            "fmla v11.4s, v5.4s, v0.s[1]\n"

            "fmla v12.4s, v6.4s, v0.s[1]\n"
            "str q8, [x0]\n"

            "fmla v13.4s, v7.4s, v0.s[1]\n"
            "ins v1.d[1], x8\n"

            "fmla v14.4s, v5.4s, v0.s[2]\n"
            "str q9, [x0, #16]\n"

            "fmla v15.4s, v6.4s, v0.s[2]\n"
            "str q10, [x0, #32]\n"

            "fmla v16.4s, v7.4s, v0.s[2]\n"
            "str q11, [x1]\n"

            "fmla v17.4s, v5.4s, v0.s[3]\n"
            "str q12, [x1, #16]\n"

            "fmla v18.4s, v6.4s, v0.s[3]\n"
            "str q13, [x1, #32]\n"

            "fmla v19.4s, v7.4s, v0.s[3]\n"
            "str q14, [x2]\n"

            "fmla v20.4s, v5.4s, v1.s[0]\n"
            "str q15, [x2, #16]\n"

            "fmla v21.4s, v6.4s, v1.s[0]\n"
            "str q16, [x2, #32]\n"

            "fmla v22.4s, v7.4s, v1.s[0]\n"
            "str q17, [x3]\n"

            "fmla v23.4s, v5.4s, v1.s[1]\n"
            "str q18, [x3, #16]\n"

            "fmla v24.4s, v6.4s, v1.s[1]\n"
            "str q19, [x3, #32]\n"

            "fmla v25.4s, v7.4s, v1.s[1]\n"
            "str q20, [x4]\n"

            "fmla v26.4s, v5.4s, v1.s[2]\n"
            "str q21, [x4, #16]\n"

            "fmla v27.4s, v6.4s, v1.s[2]\n"
            "str q22, [x4, #32]\n"

            "fmla v28.4s, v7.4s, v1.s[2]\n"
            "str q23, [x5]\n"

            "fmla v29.4s, v5.4s, v1.s[3]\n"
            "str q24, [x5, #16]\n"

            "fmla v30.4s, v6.4s, v1.s[3]\n"
            "str q25, [x5, #32]\n"

            "fmla v31.4s, v7.4s, v1.s[3]\n"

            "st1 {v26.4s, v27.4s, v28.4s}, [x6]\n"
            "st1 {v29.4s, v30.4s, v31.4s}, [x7]\n"
            "b 6f\n"

            // odd tail
            "5:\n"
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "ldr d1, [%[a_ptr]]\n"

            "fmla v9.4s,  v3.4s, v0.s[0]\n"
            "ldr x8, [%[a_ptr], #8]\n"

            "fmla v10.4s, v4.4s, v0.s[0]\n"
            "str q8, [x0]\n"

            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "str q9, [x0, #16]\n"

            "fmla v12.4s, v3.4s, v0.s[1]\n"
            "str q10, [x0, #32]\n"

            "fmla v13.4s, v4.4s, v0.s[1]\n"
            "ins v1.d[1], x8\n"

            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "str q11, [x1]\n"

            "fmla v15.4s, v3.4s, v0.s[2]\n"
            "str q12, [x1, #16]\n"

            "fmla v16.4s, v4.4s, v0.s[2]\n"
            "str q13, [x1, #32]\n"

            "fmla v17.4s, v2.4s, v0.s[3]\n"
            "str q14, [x2]\n"

            "fmla v18.4s, v3.4s, v0.s[3]\n"
            "str q15, [x2, #16]\n"

            "fmla v19.4s, v4.4s, v0.s[3]\n"
            "str q16, [x2, #32]\n"

            "fmla v20.4s, v2.4s, v1.s[0]\n"
            "str q17, [x3]\n"

            "fmla v21.4s, v3.4s, v1.s[0]\n"
            "str q18, [x3, #16]\n"

            "fmla v22.4s, v4.4s, v1.s[0]\n"
            "str q19, [x3, #32]\n"

            "fmla v23.4s, v2.4s, v1.s[1]\n"
            "str q20, [x4]\n"

            "fmla v24.4s, v3.4s, v1.s[1]\n"
            "str q21, [x4, #16]\n"

            "fmla v25.4s, v4.4s, v1.s[1]\n"
            "str q22, [x4, #32]\n"

            "fmla v26.4s, v2.4s, v1.s[2]\n"
            "str q23, [x5]\n"

            "fmla v27.4s, v3.4s, v1.s[2]\n"
            "str q24, [x5, #16]\n"

            "fmla v28.4s, v4.4s, v1.s[2]\n"
            "str q25, [x5, #32]\n"

            "fmla v29.4s, v2.4s, v1.s[3]\n"
            "str q26, [x6]\n"

            "fmla v30.4s, v3.4s, v1.s[3]\n"
            "str q27, [x6, #16]\n"

            "fmla v31.4s, v4.4s, v1.s[3]\n"
            "str q28, [x6, #32]\n"

            "st1 {v29.4s, v30.4s, v31.4s}, [x7]\n"

            "6:\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
              "x8", "x9", "x10", "x11", "x12", "x13", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
}
void Kern_4x12(const float* packA, const float* packB, int K, float* output,
               int LDC, bool is_first_k, int m_remain) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    LDC = LDC * sizeof(float);
    register float* outptr asm("x0") = output;

// clang-format off
#define LOAD_LINE(v0, v1, v2, n)                            \
    "cmp x10, #0\n"                                         \
    "beq 102f\n"                                            \
    "ld1 {v" v0 ".4s, v" v1 ".4s, v" v2 ".4s}, [x" n "]\n"  \
    "subs x10, x10, #1\n"


#define LOAD_C                      \
    "mov x10, %x[m_remain]\n"       \
    LOAD_LINE("8","9","10", "0")    \
    LOAD_LINE("11","12","13", "1")  \
    LOAD_LINE("14","15","16", "2")  \
    LOAD_LINE("17","18","19", "3")  \
    "102:\n"

#define STORE_LINE(v0, v1, v2, n)                           \
    "cmp x10, #0 \n"                                        \
    "beq 105f\n"                                            \
    "st1 {v" v0 ".4s, v" v1 ".4s, v" v2 ".4s}, [x" n "]\n"  \
    "subs x10, x10, #1\n"

#define STORE_C                          \
    "mov x10, %x[m_remain]\n"            \
    STORE_LINE("8","9","10", "0")        \
    STORE_LINE("11","12","13", "1")      \
    STORE_LINE("14","15","16", "2")      \
    STORE_LINE("17","18","19", "3")      \
    "105:\n"

    // clang-format on

    asm volatile(
            // load accumulator C
            "add x1, x0, %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"

            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "eor v8.16b, v8.16b, v8.16b\n"
            "eor v9.16b, v9.16b, v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v12.16b, v12.16b, v12.16b\n"
            "eor v13.16b, v13.16b, v13.16b\n"
            "eor v14.16b, v14.16b, v14.16b\n"
            "eor v15.16b, v15.16b, v15.16b\n"
            "eor v16.16b, v16.16b, v16.16b\n"
            "eor v17.16b, v17.16b, v17.16b\n"
            "eor v18.16b, v18.16b, v18.16b\n"
            "eor v19.16b, v19.16b, v19.16b\n"

            "2: \n"
            "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "ldr d5, [%[b_ptr]]\n"
            "fmla v9.4s,  v3.4s, v0.s[0]\n"
            "ldr x20, [%[b_ptr], #8]\n"
            "fmla v10.4s, v4.4s, v0.s[0]\n"
            "ldr d6, [%[b_ptr], #16]\n"
            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "ldr x21, [%[b_ptr], #24]\n"
            "fmla v12.4s, v3.4s, v0.s[1]\n"
            "ins v5.d[1], x20\n"

            "fmla v13.4s, v4.4s, v0.s[1]\n"
            "ldr d7, [%[b_ptr], #32]\n"
            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "ldr x22, [%[b_ptr], #40]\n"

            "ld1 {v1.4s}, [%[a_ptr]], 16\n"

            "fmla v15.4s, v3.4s, v0.s[2]\n"
            "ins v6.d[1], x21\n"
            "fmla v16.4s, v4.4s, v0.s[2]\n"
            "ins v7.d[1], x22\n"
            "fmla v17.4s, v2.4s, v0.s[3]\n"
            "fmla v18.4s, v3.4s, v0.s[3]\n"
            "fmla v19.4s, v4.4s, v0.s[3]\n"

            "fmla v8.4s,  v5.4s, v1.s[0]\n"
            "ldr d2, [%[b_ptr], #48]\n"
            "fmla v9.4s,  v6.4s, v1.s[0]\n"
            "ldr x20, [%[b_ptr], #56]\n"
            "fmla v10.4s, v7.4s, v1.s[0]\n"
            "ldr d3, [%[b_ptr], #64]\n"
            "fmla v11.4s, v5.4s, v1.s[1]\n"
            "ldr x21, [%[b_ptr], #72]\n"
            "fmla v12.4s, v6.4s, v1.s[1]\n"
            "ldr d4, [%[b_ptr], #80]\n"
            "fmla v13.4s, v7.4s, v1.s[1]\n"
            "ldr x22, [%[b_ptr], #88]\n"
            "fmla v14.4s, v5.4s, v1.s[2]\n"
            "ins v2.d[1], x20\n"
            "fmla v15.4s, v6.4s, v1.s[2]\n"
            "ins v3.d[1], x21\n"

            "ld1 {v0.4s}, [%[a_ptr]], 16\n"

            "fmla v16.4s, v7.4s, v1.s[2]\n"
            "ins v4.d[1], x22\n"
            "fmla v17.4s, v5.4s, v1.s[3]\n"
            "add %[b_ptr], %[b_ptr], #96\n"
            "fmla v18.4s, v6.4s, v1.s[3]\n"
            "subs %w[K], %w[K], #1\n"
            "fmla v19.4s, v7.4s, v1.s[3]\n"

            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "ldr d5, [%[b_ptr]]\n"
            "fmla v9.4s,  v3.4s, v0.s[0]\n"
            "ldr x20, [%[b_ptr], #8]\n"
            "fmla v10.4s, v4.4s, v0.s[0]\n"
            "ldr d6, [%[b_ptr], #16]\n"
            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "ldr x21, [%[b_ptr], #24]\n"

            "ld1 {v1.4s}, [%[a_ptr]], 16\n"

            "fmla v12.4s, v3.4s, v0.s[1]\n"
            "ldr d7, [%[b_ptr], #32]\n"
            "fmla v13.4s, v4.4s, v0.s[1]\n"
            "ins v5.d[1], x20\n"
            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "ldr x22, [%[b_ptr], #40]\n"
            "fmla v15.4s, v3.4s, v0.s[2]\n"
            "ins v6.d[1], x21\n"

            "fmla v16.4s, v4.4s, v0.s[2]\n"
            "fmla v17.4s, v2.4s, v0.s[3]\n"
            "fmla v18.4s, v3.4s, v0.s[3]\n"
            "ins v7.d[1], x22\n"
            "fmla v19.4s, v4.4s, v0.s[3]\n"

            "fmla v8.4s,  v5.4s, v1.s[0]\n"
            "fmla v9.4s,  v6.4s, v1.s[0]\n"
            "fmla v10.4s, v7.4s, v1.s[0]\n"
            "fmla v11.4s, v5.4s, v1.s[1]\n"
            "fmla v12.4s, v6.4s, v1.s[1]\n"
            "fmla v13.4s, v7.4s, v1.s[1]\n"
            "fmla v14.4s, v5.4s, v1.s[2]\n"
            "fmla v15.4s, v6.4s, v1.s[2]\n"
            "fmla v16.4s, v7.4s, v1.s[2]\n"
            "fmla v17.4s, v5.4s, v1.s[3]\n"
            "fmla v18.4s, v6.4s, v1.s[3]\n"
            "fmla v19.4s, v7.4s, v1.s[3]\n"

            "b 6f\n"

            // odd tail
            "5:\n"
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "fmla v9.4s,  v3.4s, v0.s[0]\n"
            "fmla v10.4s, v4.4s, v0.s[0]\n"
            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "fmla v12.4s, v3.4s, v0.s[1]\n"
            "fmla v13.4s, v4.4s, v0.s[1]\n"
            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "fmla v15.4s, v3.4s, v0.s[2]\n"
            "fmla v16.4s, v4.4s, v0.s[2]\n"
            "fmla v17.4s, v2.4s, v0.s[3]\n"
            "fmla v18.4s, v3.4s, v0.s[3]\n"
            "fmla v19.4s, v4.4s, v0.s[3]\n"

            "6:\n" STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
              [outptr] "+r"(outptr), [m_remain] "+r"(m_remain)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "x1", "x2", "x3", "x10", "x20", "x21", "x22", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}
void Kern_4x4(const float* packA, const float* packB, int K, float* output,
              int LDC, bool is_first_k, int m_remain, int n_remain) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    LDC = LDC * sizeof(float);
    register float* outptr asm("x0") = output;

// clang-format off
#define LOAD_LINE(v0, n)                \
    "cmp x10, #0\n"                     \
    "beq 102f\n"                        \
    "cmp %w[n_remain], #4\n"            \
    "blt 100" n "f\n"                   \
    "ld1 {v" v0 ".4s}, [x" n "]\n"  \
    "b 101" n "f\n"                     \
    "100" n ":\n"                       \
    "cmp %w[n_remain], #0\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[0], [x" n "], 4\n" \
    "cmp %w[n_remain], #1\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[1], [x" n "], 4\n" \
    "cmp %w[n_remain], #2\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[2], [x" n "], 4\n" \
    "101" n ":\n"                       \
    "subs x10, x10, #1\n"

#define LOAD_C                  \
    "mov x10, %x[m_remain]\n"   \
    LOAD_LINE("8", "0")         \
    LOAD_LINE("11", "1")        \
    LOAD_LINE("14", "2")        \
    LOAD_LINE("17", "3")        \
    "102:\n"

#define STORE_LINE(v0, n)               \
    "cmp x10, #0 \n"                    \
    "beq 105f\n"                        \
    "cmp %w[n_remain], #4\n"            \
    "blt 103" n "f\n"                   \
    "st1 {v" v0 ".4s}, [x" n " ],16\n" \
    "b 104" n "f\n"                     \
    "103" n ":\n"                       \
    "cmp %w[n_remain], #0\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[0], [x" n "], 4\n" \
    "cmp %w[n_remain], #1\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[1], [x" n "], 4\n" \
    "cmp %w[n_remain], #2\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[2], [x" n "], 4\n" \
    "104" n ":\n"                       \
    "subs x10, x10, #1\n"


#define STORE_C                 \
    "mov x10, %x[m_remain]\n"   \
    STORE_LINE("8", "0")        \
    STORE_LINE("11", "1")       \
    STORE_LINE("14", "2")       \
    STORE_LINE("17", "3")       \
    "105:\n"
    // clang-format on

    asm volatile(
            // load accumulator C
            "add x1, x0, %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"

            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "eor v8.16b, v8.16b, v8.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v14.16b, v14.16b, v14.16b\n"
            "eor v17.16b, v17.16b, v17.16b\n"

            "2: \n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "ld1 {v5.4s}, [%[b_ptr]], 16\n"
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "fmla v17.4s, v2.4s, v0.s[3]\n"

            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "fmla v8.4s,  v5.4s, v1.s[0]\n"
            "fmla v11.4s, v5.4s, v1.s[1]\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "fmla v14.4s, v5.4s, v1.s[2]\n"
            "fmla v17.4s, v5.4s, v1.s[3]\n"

            "subs %w[K], %w[K], #1\n"
            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "ld1 {v5.4s}, [%[b_ptr]], 16\n"
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "fmla v17.4s, v2.4s, v0.s[3]\n"

            "fmla v8.4s,  v5.4s, v1.s[0]\n"
            "fmla v11.4s, v5.4s, v1.s[1]\n"
            "fmla v14.4s, v5.4s, v1.s[2]\n"
            "fmla v17.4s, v5.4s, v1.s[3]\n"

            "b 6f\n"

            // odd tail
            "5:\n"
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "fmla v17.4s, v2.4s, v0.s[3]\n"
            "fmla v29.4s, v2.4s, v1.s[3]\n"

            "6:\n" STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
              [outptr] "+r"(outptr), [n_remain] "+r"(n_remain),
              [m_remain] "+r"(m_remain)
            :
            : "v0", "v1", "v2", "v5", "v8", "v11", "v14", "v17", "x1", "x2",
              "x3", "x10", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}
void Kern_8x4(const float* packA, const float* packB, int K, float* output,
              int LDC, bool is_first_k, int n_remain) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    LDC = LDC * sizeof(float);
    register float* outptr asm("x0") = reinterpret_cast<float*>(output);

// clang-format off
#define LOAD_LINE(v0, n)                \
    "cmp %w[n_remain], #4\n"            \
    "blt 100" n "f\n"                   \
    "ld1 {v" v0 ".4s}, [x" n "]\n"  \
    "b 101" n "f\n"                     \
    "100" n ":\n"                       \
    "cmp %w[n_remain], #0\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[0], [x" n "],#4\n" \
    "cmp %w[n_remain], #1\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[1], [x" n "],#4\n" \
    "cmp %w[n_remain], #2\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[2], [x" n "],#4\n" \
    "101" n ":\n"                       \

#define LOAD_C                   \
    LOAD_LINE("8", "0")          \
    LOAD_LINE("11", "1")         \
    LOAD_LINE("14", "2")         \
    LOAD_LINE("17", "3")         \
    LOAD_LINE("20", "4")         \
    LOAD_LINE("23", "5")         \
    LOAD_LINE("26", "6")         \
    LOAD_LINE("29", "7")         \


#define STORE_LINE(v0, n)               \
    "cmp %w[n_remain], #4\n"            \
    "blt 103" n "f\n"                   \
    "st1 {v" v0 ".4s}, [x" n " ],#16\n" \
    "b 104" n "f\n"                     \
    "103" n ":\n"                       \
    "cmp %w[n_remain], #0\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[0], [x" n "],#4\n" \
    "cmp %w[n_remain], #1\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[1], [x" n "],#4\n" \
    "cmp %w[n_remain], #2\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[2], [x" n "],#4\n" \
    "104" n ":\n"                       \


#define STORE_C                  \
    STORE_LINE("8", "0")         \
    STORE_LINE("11", "1")        \
    STORE_LINE("14", "2")        \
    STORE_LINE("17", "3")        \
    STORE_LINE("20", "4")        \
    STORE_LINE("23", "5")        \
    STORE_LINE("26", "6")        \
    STORE_LINE("29", "7") \
    // clang-format on

    asm volatile(
            // load accumulator C
            "add x1, x0, %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"
            "add x4, x3, %x[LDC]\n"
            "add x5, x4, %x[LDC]\n"
            "add x6, x5, %x[LDC]\n"
            "add x7, x6, %x[LDC]\n"

            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "eor v8.16b, v8.16b, v8.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v14.16b, v14.16b, v14.16b\n"
            "eor v17.16b, v17.16b, v17.16b\n"
            "eor v20.16b, v20.16b, v20.16b\n"
            "eor v23.16b, v23.16b, v23.16b\n"
            "eor v26.16b, v26.16b, v26.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"

            "2: \n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "fmla v17.4s, v2.4s, v0.s[3]\n"
            "ld1 {v5.4s}, [%[b_ptr]], 16\n"
            "fmla v20.4s, v2.4s, v1.s[0]\n"
            "fmla v23.4s, v2.4s, v1.s[1]\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "fmla v26.4s, v2.4s, v1.s[2]\n"
            "fmla v29.4s, v2.4s, v1.s[3]\n"

            "fmla v8.4s,  v5.4s, v0.s[0]\n"
            "fmla v11.4s, v5.4s, v0.s[1]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v14.4s, v5.4s, v0.s[2]\n"
            "fmla v17.4s, v5.4s, v0.s[3]\n"
            "fmla v20.4s, v5.4s, v1.s[0]\n"
            "fmla v23.4s, v5.4s, v1.s[1]\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "fmla v26.4s, v5.4s, v1.s[2]\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "fmla v29.4s, v5.4s, v1.s[3]\n"

            "subs %w[K], %w[K], #1\n"
            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "fmla v17.4s, v2.4s, v0.s[3]\n"
            "ld1 {v5.4s}, [%[b_ptr]], 16\n"
            "fmla v20.4s, v2.4s, v1.s[0]\n"
            "fmla v23.4s, v2.4s, v1.s[1]\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "fmla v26.4s, v2.4s, v1.s[2]\n"
            "fmla v29.4s, v2.4s, v1.s[3]\n"

            "fmla v8.4s,  v5.4s, v0.s[0]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v11.4s, v5.4s, v0.s[1]\n"
            "fmla v14.4s, v5.4s, v0.s[2]\n"
            "fmla v17.4s, v5.4s, v0.s[3]\n"
            "fmla v20.4s, v5.4s, v1.s[0]\n"
            "fmla v23.4s, v5.4s, v1.s[1]\n"
            "fmla v26.4s, v5.4s, v1.s[2]\n"
            "fmla v29.4s, v5.4s, v1.s[3]\n"

            "b 6f\n"

            // odd tail
            "5:\n"
            "fmla v8.4s,  v2.4s, v0.s[0]\n"
            "fmla v11.4s, v2.4s, v0.s[1]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v14.4s, v2.4s, v0.s[2]\n"
            "fmla v17.4s, v2.4s, v0.s[3]\n"
            "fmla v20.4s, v2.4s, v1.s[0]\n"
            "fmla v23.4s, v2.4s, v1.s[1]\n"
            "fmla v26.4s, v2.4s, v1.s[2]\n"
            "fmla v29.4s, v2.4s, v1.s[3]\n"

            "6:\n" STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
              [outptr] "+r"(outptr), [n_remain] "+r"(n_remain)
            :
            : "v0", "v1", "v2", "v5", "v8", "v11", "v14", "v17", "v20", "v23",
              "v26", "v29", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "cc",
              "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
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
        a[i] = (float)(rand() % 10 + rand() % 5);
    for (int i = 0; i < n * k; i++)
        b[i] = (float)(rand() % 10 + rand() % 3);
    memset(c, 0, m * n * sizeof(float));

    // warm up
    my_matmul_asm(m, n, k, a, k, b, n, c, n);
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
                    "Metric_GFLOPs_Use"
                    });
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
            my_matmul_asm(m, n, k, a, k, b, n, c, n);
        }
#if ENABLE_TMA
        mpf_tma.sample_and_stop(iter_num);
    }
    for (size_t i = 0; i < uncore_evt_num; ++i) {
        mpf_tma.start_uncore(i);
#endif
        for (size_t j = 0; j < iter_num; ++j) {
            my_matmul_asm(m, n, k, a, k, b, n, c, n);
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
