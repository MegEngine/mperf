/**
 * \file include/mperf/cpu_march_probe.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */
#pragma once
#include "mperf/utils.h"

namespace mperf {

struct BenchParam {
    int parallel;
    int warmup;
    int repetitions;
};

/***** memory ******/
float cpu_mem_bw(BenchParam param, int aligned, int nbytes, char* mop,
                 char* core_list);

// dram bandwidth(method2)
float cpu_dram_bandwidth();

/***** compute ******/
void cpu_insts_gflops_latency();

}  // namespace mperf