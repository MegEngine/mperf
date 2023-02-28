/**
 * \file include/mperf/xpmu/cpu_profiler.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#pragma once

#include <string>
#include <vector>
#include "mperf/pmu_types.h"
#include "mperf/utils.h"
#include "mperf_build_config.h"
#include "value.h"

namespace mperf {
#if MPERF_WITH_PFM
typedef std::string CpuCounterSet;
#endif
typedef std::vector<mperf::EventAttr> CpuCounterSet2;
typedef std::vector<std::pair<std::string, uint64_t>> CpuMeasurements;

/** An interface for classes that collect CPU performance data. */
class CpuProfiler {
public:
    virtual ~CpuProfiler() = default;

#if MPERF_WITH_PFM
    // Sets the enabled counters after initialization
    virtual void set_enabled_counters(
            const std::vector<std::string>& counter_names) = 0;
#endif
    virtual void set_enabled_counters(
            const std::vector<mperf::EventAttr>& event_attrs) = 0;

    // Starts a profiling session
    virtual MPERF_ALWAYS_INLINE void run() = 0;

    // Sample the counters. Returns a map of measurements for the counters
    // that are both available and enabled.
    // A profiling session must be running when sampling the counters.
    virtual MPERF_ALWAYS_INLINE const CpuMeasurements& sample() = 0;

    // Stops the active profiling session
    virtual MPERF_ALWAYS_INLINE void stop() = 0;

    virtual MPERF_ALWAYS_INLINE void set_uncore_event_enabled() = 0;
};

}  // namespace mperf
