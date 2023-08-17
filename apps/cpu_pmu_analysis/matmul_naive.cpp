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

#include "mperf/cpu_affinity.h"
#include "mperf/timer.h"
#include "mperf/tma/tma.h"

#define ENABLE_TMA 1
void my_matmul_naive(int m, int n, int k, float* a, int lda, float* b, int ldb,
                     float* c, int ldc) {
#define A(i, j) a[(i)*k + (j)]
#define B(i, j) b[(i)*n + (j)]
#define C(i, j) c[(i)*n + (j)]

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
#undef A
#undef B
#undef C
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
    my_matmul_naive(m, n, k, a, k, b, n, c, n);

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
            my_matmul_naive(m, n, k, a, k, b, n, c, n);
        }
#if ENABLE_TMA
        mpf_tma.sample_and_stop(iter_num);
    }

    for (size_t i = 0; i < uncore_evt_num; ++i) {
        mpf_tma.start_uncore(i);
#endif
        for (size_t j = 0; j < iter_num; ++j) {
            my_matmul_naive(m, n, k, a, k, b, n, c, n);

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

    return 0;
}
