/**
 * \file include/mperf/tma/tma.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#pragma once
#include <stdint.h>
#include <chrono>
#include <set>
#include <string>
#include <vector>
#include "mperf/pmu_types.h"
#include "mperf/xpmu/xpmu.h"

namespace mperf {
namespace tma {

class ArchRatioSetup;
class MPFTMA {
public:
    explicit MPFTMA(MPFXPUType t);

    int init(std::vector<std::string> metrics);
    // int init(const std::string& metric_group);
    // call after init
    size_t group_num() const;
    size_t uncore_events_num() const;
    int start(size_t group_id = 0);
    int start_uncore(size_t evt_idx = 0);
    int sample(size_t iter_num);
    int sample_and_stop(size_t iter_num);
    int deinit();

private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    time_point m_start_point;

    MPFXPUType m_xpu_type;
    ArchRatioSetup* m_ratio_setup;
    XPMU* m_xpmu;
    std::vector<std::string> m_metrics;
    std::set<EventAttr> m_events;
    std::set<EventAttr> m_uncore_events;
    std::vector<std::pair<std::string, float>> m_duration;
    size_t m_group_num;
    size_t m_group_id;
    size_t m_uncore_events_num;
    bool m_binit;

    float ev_collect(EventAttr event, int level);
    float ev_query(EventAttr event, int level);
    bool is_cpu() const;
};

}  // namespace tma
}  // namespace mperf