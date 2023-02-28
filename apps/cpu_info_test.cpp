/**
 * \file apps/cpu_info_test.cpp
 *
 * This file is part of mperf.
 *
 * \brief abstraction of cpu intrincis
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */
#include "mperf/cpu_info.h"
#include <stdio.h>
#include "mperf/timer.h"

int main(int argc, char** argv) {
    mperf::WallTimer t0;
    mperf::CPUTimer t1;
    auto res = mperf::cpu_info_support_features();
    printf("Support CPU features: %s\n", res.c_str());
    printf("cpu count:%d\n", mperf::cpu_info_get_cpu_count());
    for (int i = 0; i < mperf::cpu_info_get_cpu_count(); ++i) {
        printf("khz: %d\n", mperf::cpu_info_get_max_freq_khz(i));
    }
    printf("litte cpu count:%d\n", mperf::cpu_info_get_little_cpu_count());
    printf("middle cpu count:%d\n", mperf::cpu_info_get_middle_cpu_count());
    printf("big cpu count:%d\n", mperf::cpu_info_get_big_cpu_count());

    auto ref_freq = mperf::cpu_info_ref_freq(0);
    printf("cpu_info_ref_freq:%lu\n", ref_freq);
    printf("wall time:%f ms\n", t0.get_msecs());
    auto ref_cycle = t1.get_cycles();
    auto cpu_time = t1.get_msecs();
    printf("cpu time:%f ms, cycles:%lu, ref-time(not accurate):%f ms, "
           "dynamic_mhz:%f khz\n",
           cpu_time, ref_cycle, (1.0 * ref_cycle / ref_freq) * 1e3,
           ref_cycle / cpu_time);

    return 0;
}