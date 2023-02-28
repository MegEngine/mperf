/**
 * MegPeaK is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied.
 * ---------------------------------------------------------------------------
 * \file uarch/cpu/compute/x86_sse.cpp
 *
 * Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2022-2023 Megvii Inc. All rights
 * reserved.
 *
 * ---------------------------------------------------------------------------
 */

#include "./common.h"
#include "./x86_utils.h"
#include "mperf_build_config.h"

#if MPERF_X86

#define eor(i) "xorps %%xmm" #i ", %%xmm" #i "\n"

// clang-format off
#define THROUGHPUT(cb, func, simd)                                 \
    MPERF_ATTRIBUTE_TARGET(simd)                                 \
    static int func##_throughput() {                               \
        asm volatile(                                              \
        UNROLL_CALL(10, eor)                                       \
        "movl %[RUNS], %%eax \n"                                   \
        "1:\n"                                                     \
        UNROLL_CALL(10, cb)                                        \
        "sub  $0x01, %%eax\n"                                      \
        "jne 1b \n"                                                \
        :                                                          \
        :[RUNS] "r"(mperf::RUNS * 100)                           \
        : "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5",    \
           "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%eax", "cc");      \
        return mperf::RUNS * 100 * 10;                           \
    }

#define LATENCY(cb, func, simd)          \
    MPERF_ATTRIBUTE_TARGET(simd)       \
    static int func##_latency() {        \
        asm volatile(                    \
        "xorpd %%xmm0, %%xmm0\n"         \
        "movl %[RUNS], %%eax \n"         \
        "1:\n"                           \
        UNROLL_CALL(10, cb)              \
        "sub  $0x01, %%eax\n"            \
        "jne 1b \n"                      \
        :                                \
        :[RUNS] "r"(mperf::RUNS * 100) \
        : "%xmm0", "%eax", "cc");        \
        return mperf::RUNS * 100 * 10; \
    }
// clang-format on
#define cb(i) "mulps %%xmm" #i ", %%xmm" #i "\n"
THROUGHPUT(cb, mulps, "sse4.2")
#undef cb
#define cb(i) "mulps %%xmm0, %%xmm0\n"
LATENCY(cb, mulps, "sse4.2")
#undef cb

#define cb(i) "mulpd %%xmm" #i ", %%xmm" #i "\n"
THROUGHPUT(cb, mulpd, "sse4.2")
#undef cb
#define cb(i) "mulpd %%xmm0, %%xmm0\n"
LATENCY(cb, mulpd, "sse4.2")
#undef cb

#define cb(i) "vfnmadd132ps %%xmm" #i ", %%xmm" #i ", %%xmm" #i "\n"
THROUGHPUT(cb, vfmadd132ps, "fma")
#undef cb
#define cb(i) "vfmadd132ps %%xmm0, %%xmm0, %%xmm0\n"
LATENCY(cb, vfmadd132ps, "fma")
#undef cb
// clang-format off
#define cb(i)                                          \
    "vpmaddwd %%xmm" #i ", %%xmm" #i ", %%xmm" #i "\n" \
    "vpaddd %%xmm" #i ", %%xmm" #i ", %%xmm" #i "\n"
THROUGHPUT(cb, vpmaddwd_vpaddd, "sse4.2")
#undef cb
// clang-format on
#define cb(i)                           \
    "vpmaddwd %%xmm0, %%xmm0, %%xmm0\n" \
    "vpaddd %%xmm0, %%xmm0, %%xmm0\n"
LATENCY(cb, vpmaddwd_vpaddd, "sse4.2")
#undef cb

void mperf::x86_sse() {
    if (is_supported(SIMDType::SSE2)) {
        //! warmup
        for (size_t i = 0; i < 10; i++) {
            mulps_throughput();
        }
        //! sse2 is 128 width, mul + add = 2 ops
        benchmark(mulps_throughput, mulps_latency, "mulps_sse", 4 * 2);
        benchmark(mulpd_throughput, mulpd_latency, "mulpd_sse", 2 * 2);
        benchmark(vfmadd132ps_throughput, vfmadd132ps_latency,
                  "vfmadd132ps_sse", 4 * 2);
        benchmark(vpmaddwd_vpaddd_throughput, vpmaddwd_vpaddd_latency,
                  "vpmaddwd_vpaddd_sse", 8 + 4 + 4);
    } else {
        printf("this x86 CPU does not support sse2.\n");
    }
}
#else
void mperf::x86_sse() {}
#endif

// vim: syntax=cpp.doxygen
