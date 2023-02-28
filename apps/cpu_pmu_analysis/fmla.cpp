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

namespace mperf {
constexpr static uint32_t RUNS = 800000;
}
#define UNROLL_RAW1(cb, v0, a...) cb(0, ##a)
#define UNROLL_RAW2(cb, v0, a...) cb(0, ##a) cb(1, ##a)
#define UNROLL_RAW5(cb, v0, a...) \
    cb(0, ##a) cb(1, ##a) cb(2, ##a) cb(3, ##a) cb(4, ##a)
#define UNROLL_RAW10(cb, v0, a...) \
    UNROLL_RAW5(cb, v0, ##a)       \
    cb(5, ##a) cb(6, ##a) cb(7, ##a) cb(8, ##a) cb(9, ##a)
#define UNROLL_RAW20(cb, v0, a...)                                          \
    UNROLL_RAW10(cb, v0, ##a)                                               \
    cb(10, ##a) cb(11, ##a) cb(12, ##a) cb(13, ##a) cb(14, ##a) cb(15, ##a) \
            cb(16, ##a) cb(17, ##a) cb(18, ##a) cb(19, ##a)

#define UNROLL_RAW5_START6(cb, v0, a...) \
    cb(6, ##a) cb(7, ##a) cb(8, ##a) cb(9, ##a) cb(10, ##a)
#define UNROLL_RAW10_START6(cb, v0, a...) \
    UNROLL_RAW5_START6(cb, v0, ##a)       \
    cb(11, ##a) cb(12, ##a) cb(13, ##a) cb(14, ##a) cb(15, ##a)
#define UNROLL_RAW20_START6(cb, v0, a...)                                   \
    UNROLL_RAW10_START6(cb, v0, ##a)                                        \
    cb(16, ##a) cb(17, ##a) cb(18, ##a) cb(19, ##a) cb(20, ##a) cb(21, ##a) \
            cb(22, ##a) cb(23, ##a) cb(24, ##a) cb(25, ##a)

#define UNROLL_CALL0(step, cb, v...) UNROLL_RAW##step(cb, 0, ##v)
#define UNROLL_CALL(step, cb, v...) UNROLL_CALL0(step, cb, ##v)
//! As some arm instruction, the second/third operand must be [d0-d7], so the
//! iteration should start from a higher number, otherwise may cause data
//! dependence
#define UNROLL_CALL0_START6(step, cb, v...) \
    UNROLL_RAW##step##_START6(cb, 0, ##v)
#define UNROLL_CALL_START6(step, cb, v...) UNROLL_CALL0_START6(step, cb, ##v)

#if MPERF_AARCH64
#include <arm_neon.h>
#define eor(i) "eor v" #i ".16b, v" #i ".16b, v" #i ".16b\n"

//#define UNROLL_NUM 20

#define THROUGHPUT(cb, func, UNROLL_NUM)                                      \
    static int func##_throughput_##UNROLL_NUM() {                             \
        asm volatile(                                                       \
        UNROLL_CALL(UNROLL_NUM, eor)                                                \
        "mov x0, %x[RUNS]\n"                                                \
        "1:\n"                                                              \
        UNROLL_CALL(UNROLL_NUM, cb)                                                 \
        "subs  x0, x0, #1 \n"                                               \
        "bne 1b \n"                                                         \
        :                                                                   \
        : [RUNS] "r"(mperf::RUNS)                                         \
        : "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", \
          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",    \
          "v19", "x0"); \
        return mperf::RUNS * UNROLL_NUM;                                      \
    }

#define LATENCY(cb, func, UNROLL_NUM)          \
    static int func##_latency_##UNROLL_NUM() { \
        asm volatile(                  \
        "eor v0.16b, v0.16b, v0.16b\n" \
        "mov x0, #0\n"                 \
        "1:\n"                         \
        UNROLL_CALL(UNROLL_NUM, cb)            \
        "add  x0, x0, #1 \n"           \
        "cmp x0, %x[RUNS] \n"          \
        "blt 1b \n"                    \
        :                              \
        : [RUNS] "r"(mperf::RUNS)    \
        : "cc", "v0", "x0");       \
        return mperf::RUNS * UNROLL_NUM;       \
    }

#define cb(i) "fmla v" #i ".4s, v" #i ".4s, v" #i ".4s\n"
THROUGHPUT(cb, fmla, 20)
THROUGHPUT(cb, fmla, 10)
THROUGHPUT(cb, fmla, 5)
THROUGHPUT(cb, fmla, 2)
THROUGHPUT(cb, fmla, 1)
#undef cb
#define cb(i) "fmla v0.4s, v0.4s, v0.4s\n"
LATENCY(cb, fmla, 20)
LATENCY(cb, fmla, 10)
LATENCY(cb, fmla, 5)
LATENCY(cb, fmla, 2)
LATENCY(cb, fmla, 1)
#undef cb

void fmla_benchmark(std::function<int()> throughtput_func,
                    std::function<int()> latency_func, const char* inst,
                    size_t inst_simd = 4, float* throuphput_used = NULL,
                    float* latency_used = NULL) {
    mperf::Timer timer;
    auto runs = throughtput_func();
    *throuphput_used = timer.get_nsecs() / runs;
    timer.reset();
    runs = latency_func();
    *latency_used = timer.get_nsecs() / runs;
}

void call_fmla(float* throuphput_used, float* latency_used, int num) {
    //! warmup
    if (num == 20) {
        for (size_t i = 0; i < 100; i++) {
            fmla_throughput_20();
        }
        fmla_benchmark(fmla_throughput_20, fmla_latency_20, "fmla", 8,
                       throuphput_used, latency_used);
    }
    if (num == 10) {
        for (size_t i = 0; i < 100; i++) {
            fmla_throughput_10();
        }
        fmla_benchmark(fmla_throughput_10, fmla_latency_10, "fmla", 8,
                       throuphput_used, latency_used);
    }
    if (num == 5) {
        for (size_t i = 0; i < 100; i++) {
            fmla_throughput_5();
        }
        fmla_benchmark(fmla_throughput_5, fmla_latency_5, "fmla", 8,
                       throuphput_used, latency_used);
    }
    if (num == 2) {
        for (size_t i = 0; i < 100; i++) {
            fmla_throughput_2();
        }
        fmla_benchmark(fmla_throughput_2, fmla_latency_2, "fmla", 8,
                       throuphput_used, latency_used);
    }
    if (num == 1) {
        for (size_t i = 0; i < 100; i++) {
            fmla_throughput_1();
        }
        fmla_benchmark(fmla_throughput_1, fmla_latency_1, "fmla", 8,
                       throuphput_used, latency_used);
    }
}
#else
void call_fmla(float* throuphput_used, float* latency_used, int num) {}
#endif

void gettma(int num) {
    printf("---------------UNROLL_NUM:%d------------\n", num);
    float throuphput_used;
    float latency_used;
    size_t iter_num = 10;

    // warm up
    call_fmla(&throuphput_used, &latency_used, num);

#if defined(__aarch64__)
    mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::A55);
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

        for (size_t j = 0; j < iter_num; ++j) {
            // mperf::aarch64();
            call_fmla(&throuphput_used, &latency_used, num);
        }

        mpf_tma.sample_and_stop(iter_num);
    }
    for (size_t i = 0; i < uncore_evt_num; ++i) {
        mpf_tma.start_uncore(i);

        for (size_t j = 0; j < iter_num; ++j) {
            // mperf::aarch64();
            call_fmla(&throuphput_used, &latency_used, num);
            mpf_tma.sample(1);
        }
        // mpf_tma.sample_and_stop(iter_num);
    }
    mpf_tma.deinit();

    const char* inst = "fmla";
    size_t inst_simd = 8;
    printf("%s throughput_%d: %f ns %f GFlops latency: %f ns\n", inst, num,
           throuphput_used, 1.f / throuphput_used * inst_simd, latency_used);
}
#define ENABLE_TMA 1

int main(int ac, char** av) {
    if (ac < 2) {
        fprintf(stderr, "sample usage:\n");
        fprintf(stderr, "./fmla coreid\n");
        return -1;
    }
    int dev_id = atoi(av[1]);
    if (set_cpu_thread_affinity_spec_core(dev_id)) {
        return -1;
    }

    gettma(20);
    gettma(10);
    gettma(5);
    gettma(2);
    gettma(1);

    return 0;
}