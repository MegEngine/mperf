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
 * \file uarch/cpu/compute/common.h
 *
 * Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2022-2023 Megvii Inc. All rights
 * reserved.
 *
 * ---------------------------------------------------------------------------
 */
#pragma once
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include "mperf/timer.h"
#include "mperf/utils.h"

namespace mperf {
constexpr static uint32_t RUNS = 800000;

/**
 * latency:
 *
 *       xor reg, reg
 *       mov r0, count
 * loop:
 *       op reg, reg
 *       op reg, reg
 *       ...
 *       op reg, reg
 *       dec r0
 *       jne loop
 *
 *
 * throughput
 *
 *       xor reg0, reg0
 *       ....
 *       xor reg19, reg19
 *       mov r0, count
 * loop:
 *       op reg0, reg0
 *       op reg1, reg1
 *       ...
 *       op reg19, reg19
 *       dec 0r
 *       jne loop
 *
 */
inline static void benchmark(std::function<int()> throughtput_func,
                             std::function<int()> latency_func,
                             const char* inst, size_t inst_simd = 4) {
    mperf::Timer timer;
    auto runs = throughtput_func();
    float throuphput_used = timer.get_nsecs() / runs;
    timer.reset();
    runs = latency_func();
    float latency_used = timer.get_nsecs() / runs;
    printf("%s throughput: %f ns %f GFlops latency: %f ns\n", inst,
           throuphput_used, 1.f / throuphput_used * inst_simd, latency_used);
}
#define UNROLL_RAW1(cb, v0, a...) \
    cb(0, ##a) 
#define UNROLL_RAW2(cb, v0, a...) \
    cb(0, ##a) cb(1, ##a) 
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

void aarch64();
void armv7();
void x86_avx();
void x86_sse();
}  // namespace mperf

// vim: syntax=cpp.doxygen
