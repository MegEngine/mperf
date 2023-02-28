#include <stdio.h>
#include "mperf/cpu_affinity.h"
#include "mperf/cpu_march_probe.h"
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

#include <arm_neon.h>

#define ENABLE_TMA 1
void use_ldq(float* b_ptr) {
    for (int i = 0; i < 100; i++) {
        asm volatile(
                "fmla v1.4s, v1.4s, v1.s[0]\n"
                "ld1 {v0.4s}, [%[b_ptr]]\n"
                "fmla v2.4s, v2.4s, v2.s[0]\n"
                "ld1 {v7.4s}, [%[b_ptr]]\n"
                "fmla v3.4s, v3.4s, v3.s[0]\n"
                "fmla v4.4s, v4.4s, v4.s[0]\n"
                "fmla v5.4s, v5.4s, v5.s[0]\n"
                "fmla v6.4s, v6.4s, v6.s[0]\n"
                : [b_ptr] "+r"(b_ptr)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }
}
void split_ldq(float* b_ptr) {
    for (int i = 0; i < 100; i++) {
        asm volatile(
                "fmla v1.4s, v1.4s, v1.s[0]\n"
                "ldr d0, [%[b_ptr]]\n"
                "fmla v2.4s, v2.4s, v2.s[0]\n"
                "ldr x0, [%[b_ptr], #8]\n"
                "fmla v3.4s, v3.4s, v3.s[0]\n"
                "ins v0.d[1], x0\n"
                "fmla v4.4s, v4.4s, v4.s[0]\n"
                "ldr d7, [%[b_ptr]]\n"
                "fmla v5.4s, v5.4s, v5.s[0]\n"
                "ldr x0, [%[b_ptr], #8]\n"
                "fmla v6.4s, v6.4s, v6.s[0]\n"
                "ins v7.d[1], x0\n"
                : [b_ptr] "+r"(b_ptr)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }
}
void gettma(bool optimize) {
    printf("----------optimize:%d----------\n", optimize);
    float a[4] = {1.0, 1.0, 1.0, 1.0};
    float b[2] = {0, 0};
    float* a_ptr = &a[0];
    float* b_ptr = &b[0];
    asm volatile(

            "ld1 {v1.4s}, [%[a_ptr]]\n"
            "ld1 {v2.4s}, [%[a_ptr]]\n"
            "ld1 {v3.4s}, [%[a_ptr]]\n"
            "ld1 {v4.4s}, [%[a_ptr]]\n"
            "ld1 {v5.4s}, [%[a_ptr]]\n"
            "ld1 {v6.4s}, [%[a_ptr]]\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr)
            :
            : "v1", "v2", "v3", "v4", "v5", "v6");
    size_t iter_num = 10;

    if (optimize)
        split_ldq(b_ptr);
    else
        use_ldq(b_ptr);

#if ENABLE_TMA
#if defined(__aarch64__)
    mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::A55);
#else
    mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::HSX_SERVER);
#endif

#if defined(__aarch64__)
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
    mpf_tma.init(
            {"Frontend_Bound", "Bad_Speculation", "Backend_Bound", "Retiring"});
#endif

    size_t gn = mpf_tma.group_num();
    size_t uncore_evt_num = mpf_tma.uncore_events_num();
    for (size_t i = 0; i < gn; ++i) {
        mpf_tma.start(i);
#endif
        if (optimize)
            for (size_t j = 0; j < iter_num; ++j) {
                split_ldq(b_ptr);
            }
        else
            for (size_t j = 0; j < iter_num; ++j) {
                use_ldq(b_ptr);
            }
#if ENABLE_TMA
        mpf_tma.sample_and_stop(iter_num);
    }

    for (size_t i = 0; i < uncore_evt_num; ++i) {
        mpf_tma.start_uncore(i);
#endif
        if (optimize) {
            for (size_t j = 0; j < iter_num; ++j) {
                split_ldq(b_ptr);

#if ENABLE_TMA
                mpf_tma.sample(1);
#endif
            }
        } else {
            for (size_t j = 0; j < iter_num; ++j) {
                use_ldq(b_ptr);

#if ENABLE_TMA
                mpf_tma.sample(1);
#endif
            }
        }

#if ENABLE_TMA
        // mpf_tma.sample_and_stop(iter_num);
    }
    mpf_tma.deinit();
#endif
}

int main() {
    gettma(0);
    gettma(1);
}