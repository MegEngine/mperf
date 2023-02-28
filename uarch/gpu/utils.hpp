// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
/**
 * ---------------------------------------------------------------------------
 * \file uarch/gpu/utils.hpp
 *
 * Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2022-2023 Megvii Inc. All rights
 * reserved.
 *
 * ---------------------------------------------------------------------------
 */

#include <stdio.h>
#include "mperf_build_config.h"
#include "mperf/opencl_driver.h"
#include "stats.hpp"

using namespace mperf::stats;

namespace mperf {
namespace {

template <uint32_t NTap>
struct DtJumpFinder {
private:
    NTapAvgStats<double, NTap> time_avg_;
    AvgStats<double> dtime_avg_;
    double compensation_;
    double threshold_;

public:
    // Compensation is a tiny additive to give on delta time so that the
    // algorithm works smoothly when a sequence of identical timing is ingested,
    // which is pretty common in our tests. Threshold is simply how many times
    // the new delta has to be to be recognized as a deviation.
    DtJumpFinder(double compensation = 0.01, double threshold = 10)
            : time_avg_(),
              dtime_avg_(),
              compensation_(compensation),
              threshold_(threshold) {}

    // Returns true if the delta time regarding to the last data point seems
    // normal; returns false if it seems the new data point is too much away
    // from the historical records.
    bool push(double time) {
        if (time_avg_.has_value()) {
            double dtime =
                    std::abs(time - time_avg_) + (compensation_ * time_avg_);
            if (dtime_avg_.has_value()) {
                double ddtime = std::abs(dtime - dtime_avg_);
                if (ddtime > threshold_ * dtime_avg_) {
                    return true;
                }
            }
            dtime_avg_.push(dtime);
        }
        time_avg_.push(time);
        return false;
    }

    double dtime_avg() const { return dtime_avg_; }
    double compensate_time() const { return compensation_ * time_avg_; }
};

namespace utils {
template <typename... TArgs>
struct format_impl_t;
template <>
struct format_impl_t<> {
    static inline void format_impl(std::stringstream&) {}
};
template <typename T>
struct format_impl_t<T> {
    static inline void format_impl(std::stringstream& ss, const T& x) {
        ss << x;
    }
};
template <typename T, typename... TArgs>
struct format_impl_t<T, TArgs...> {
    static inline void format_impl(std::stringstream& ss, const T& x,
                                   const TArgs&... others) {
        format_impl_t<T>::format_impl(ss, x);
        format_impl_t<TArgs...>::format_impl(ss, others...);
    }
};

template <typename... TArgs>
inline std::string format(const TArgs&... args) {
    std::stringstream ss{};
    format_impl_t<TArgs...>::format_impl(ss, args...);
    return ss.str();
}
}  // namespace utils

#if MPERF_WITH_LOGGING > 0
std::string pretty_data_size(size_t size) {
    const size_t K = 1024;
    if (size < K) {
        return utils::format(size, "B");
    }
    size /= K;
    if (size < K) {
        return utils::format(size, "KB");
    }
    size /= K;
    if (size < K) {
        return utils::format(size, "MB");
    }
    size /= K;
    if (size < K) {
        return utils::format(size, "GB");
    }
    size /= K;
    if (size < K) {
        return utils::format(size, "TB");
    }
    size /= K;
    printf("unsupported data size");
    return {};
}
#endif

cl_channel_order channel_order_by_ncomp(uint32_t ncomp) {
    switch (ncomp) {
        case 1:
            return CL_R;
        case 2:
            return CL_RG;
        case 4:
            return CL_RGBA;
        default:
            printf("image component count must be 1, 2 or 4");
    }
    return CL_RGBA;
}

std::string vec_name_by_ncomp(const char* scalar_name, uint32_t ncomp) {
    return scalar_name + (ncomp == 1 ? "" : std::to_string(ncomp));
}
}  // namespace

}  // namespace mperf