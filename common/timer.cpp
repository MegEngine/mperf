/**
 * \file common/timer.cpp
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#include "mperf/timer.h"
#include "mperf/cpu_info.h"
namespace mperf {

WallTimer::WallTimer() {
    reset();
}

void WallTimer::reset() {
    m_start_point = clock::now();
}

// wall-clock-time
double WallTimer::get_msecs() const {
    // clock_gettime(CLOCK_REALTIME)
    auto now = clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now -
                                                                m_start_point)
                   .count() *
           1e-6;
}
double WallTimer::get_nsecs() const {
    return get_msecs() * 1e6;
}

CPUTimer::CPUTimer() {
    reset();
}

void CPUTimer::reset() {
    m_start_point = mperf::cpu_process_time_ms();
    m_start_cycle = mperf::cpu_info_ref_cycles();
}

// wall-clock-time
double CPUTimer::get_msecs() const {
    return mperf::cpu_process_time_ms() - m_start_point;
}
double CPUTimer::get_nsecs() const {
    return get_msecs() * 1e6;
}

uint64_t CPUTimer::get_cycles() const {
    return mperf::cpu_info_ref_cycles() - m_start_cycle;
}

}  // namespace mperf