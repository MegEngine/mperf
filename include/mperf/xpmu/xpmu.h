/**
 * \file include/mperf/xpmu/xpmu.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#pragma once

#include "mperf/xpmu/cpu_profiler.h"
#include "mperf/xpmu/gpu_profiler.h"
#include "mperf_build_config.h"

#include <functional>
#include <memory>
#include <string>

namespace mperf {
struct Measurements {
    const CpuMeasurements* cpu{nullptr};
    const GpuMeasurements* gpu{nullptr};
};

/** A class that collects CPU/GPU performance data. */
class XPMU {
public:
#if MPERF_WITH_PFM
    XPMU(CpuCounterSet enabled_cpu_counters,
         GpuCounterSet enabled_gpu_counters);
#endif
    XPMU(CpuCounterSet2 enabled_cpu_counters);
    XPMU(GpuCounterSet enabled_gpu_counters);

#if MPERF_WITH_PFM
    // Sets the enabled counters for the CPU profiler
    void set_enabled_cpu_counters(CpuCounterSet counters);
#endif
    // Sets the enabled counters for the CPU profiler
    void set_enabled_cpu_counters(CpuCounterSet2 counters);

    // Sets the enabled counters for the GPU profiler
    void set_enabled_gpu_counters(GpuCounterSet counters);

    // set the cpu uncore event enabled
    void set_cpu_uncore_event_enabled();

    // Starts a profiling session
    void run();

    // Sample the counters. The function returns pointers to the CPU and GPU
    // measurements maps, if the corresponding profiler is enabled.
    // The entries in the maps are the counters that are both available and
    // enabled. A profiling session must be running when sampling the counters.
    Measurements sample();

    // Stops the active profiling session
    void stop();

    CpuProfiler* cpu_profiler() { return cpu_profiler_.get(); }
    GpuProfiler* gpu_profiler() { return gpu_profiler_.get(); }

private:
    std::unique_ptr<CpuProfiler> cpu_profiler_{};
    std::unique_ptr<GpuProfiler> gpu_profiler_{};

#if MPERF_WITH_PFM
    void create_profilers(CpuCounterSet enabled_cpu_counters,
                          GpuCounterSet enabled_gpu_counters);
#endif
    void create_profilers(GpuCounterSet enabled_gpu_counters);
    void create_profilers(CpuCounterSet2 enabled_cpu_counters);
};

}  // namespace mperf
