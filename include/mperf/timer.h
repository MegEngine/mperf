/**
 * \file include/mperf/timer.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#pragma once
#include <chrono>

namespace mperf {
// wall time
class WallTimer {
private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    time_point m_start_point;

public:
    WallTimer();

    void reset();

    //! get milliseconds (one thousandth of a second)
    double get_msecs() const;

    //! get nanoseconds
    double get_nsecs() const;
};

using Timer = WallTimer;

// cpu time & cycles
class CPUTimer {
private:
    double m_start_point;
    uint64_t m_start_cycle;

public:
    CPUTimer();

    void reset();

    //! get milliseconds (one thousandth of a second)
    double get_msecs() const;

    //! get nanoseconds
    double get_nsecs() const;

    uint64_t get_cycles() const;
};

}  // namespace mperf