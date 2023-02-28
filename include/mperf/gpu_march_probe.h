/**
 * \file include/mperf/gpu_march_probe.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */
#pragma once
#include "mperf/utils.h"

namespace mperf {
int gpu_max_reg_num_per_thread();
int gpu_warp_size();

double gpu_mem_bw();
float gpu_local_memory_bw();
float gpu_global_memory_bw();

int gpu_unified_cache_latency();
int gpu_unified_cacheline_size();
int gpu_unified_cache_hierarchy_pchase();

double gpu_texture_cache_bw();
int gpu_texture_cacheline_size();
int gpu_texture_cache_hierachy_pchase();

float gpu_insts_gflops_latency();
}  // namespace mperf