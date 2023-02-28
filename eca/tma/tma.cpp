#include "mperf/tma/tma.h"
#include <cmath>
#include "arch_ratios/arch_ratios.h"
#include "mperf/exception.h"
#include "mperf/utils.h"
#include "mperf/xpmu/xpmu.h"
namespace mperf {
namespace tma {

typedef float (*TMACompute)(FEV EV);

// collect events
float MPFTMA::ev_collect(EventAttr attr, int level) {
#if 0  
    // TODO. Maybe make some modifications to attr in the future.
    auto adjust_ev = [&]() -> int {
        return 0;
    };
#endif

    if (attr.is_uncore) {
        m_uncore_events.insert(attr);
    } else {
        if (attr.name.compare("time_interval") != 0) {
            m_events.insert(attr);
        }
    }
    return 1;
}

// execute formula
float MPFTMA::ev_query(EventAttr attr, int level) {
    size_t ss = m_duration.size();
    for (size_t i = 0; i < ss; ++i) {
        if (attr.name == m_duration[i].first)
            return m_duration[i].second;
    }
    mperf_throw(MperfError,
                "Internal Error. No execution result of event(%s) was found.",
                attr.name.c_str());
    return 0;
}

MPFTMA::MPFTMA(MPFXPUType t) {
    m_xpu_type = t;

    m_ratio_setup = ArchRatioSetup::inst(t);

    mperf::CpuCounterSet2 cpuset;
    m_xpmu = new XPMU(cpuset);

    m_binit = false;
    m_group_id = -1;
}

int MPFTMA::init(std::vector<std::string> metrics) {
    // collect all events
    size_t mz = metrics.size();
    for (size_t i = 0; i < mz; ++i) {
        auto pmetric = m_ratio_setup->metric(metrics[i]);
        if (!pmetric) {
            mperf_throw(MperfError, "can not found metirc(%s)\n",
                        metrics[i].c_str());
        } else {
            auto fec = std::bind(&MPFTMA::ev_collect, this,
                                 std::placeholders::_1, std::placeholders::_2);
            ((TMACompute)(pmetric->func_compute))(fec);
        }
    }
    m_metrics = metrics;

    size_t counter_num = m_ratio_setup->counter_num();
    size_t events_num = m_events.size();
    mperf_log("the initial events num is %zu.", events_num);
    // FIXME(hc): when events_num is zero and m_uncore_events_num is not zero,
    // we will get wrong time_interval result
    m_group_num = std::ceil(1.0f * events_num / counter_num);
    m_uncore_events_num = m_uncore_events.size();
    m_binit = true;
    return 0;
}

size_t MPFTMA::group_num() const {
    if (!m_binit) {
        mperf_throw(MperfError, "You should call `init` first.");
    }
    return m_group_num;
}

size_t MPFTMA::uncore_events_num() const {
    if (!m_binit) {
        mperf_throw(MperfError, "You should call `init` first.");
    }
    return m_uncore_events_num;
}

int MPFTMA::start(size_t group_id) {
    m_group_id = group_id;
    mperf_log_debug("the group_id %zu\n", m_group_id);
    size_t counter_num = m_ratio_setup->counter_num();
    size_t events_num = m_events.size();
    size_t ts = (group_id != m_group_num - 1)
                        ? counter_num
                        : (events_num - counter_num * group_id);
    ts = std::min(ts, events_num);

    std::vector<EventAttr> all_tev(events_num);
    std::copy(m_events.begin(), m_events.end(), all_tev.begin());

    if (is_cpu()) {
        std::vector<EventAttr> part_tev(ts);
        std::copy(all_tev.begin() + group_id * counter_num,
                  all_tev.begin() + group_id * counter_num + ts,
                  part_tev.begin());
        m_xpmu->set_enabled_cpu_counters(part_tev);
    } else {
#if 0
        m_xpmu->set_enabled_gpu_counters();
#endif
        mperf_throw(MperfError,
                    "ERROR: unsupport gpu in tma analysis currently.\n");
    }
    m_start_point = clock::now();
    m_xpmu->run();

    return 0;
}

int MPFTMA::start_uncore(size_t evt_idx) {
    if (is_cpu()) {
        std::vector<EventAttr> part_tev(1);
        std::vector<EventAttr> all_tev(m_uncore_events_num);
        std::copy(m_uncore_events.begin(), m_uncore_events.end(),
                  all_tev.begin());
        std::copy(all_tev.begin() + evt_idx, all_tev.begin() + evt_idx + 1,
                  part_tev.begin());
        m_xpmu->set_enabled_cpu_counters(part_tev);
        m_xpmu->set_cpu_uncore_event_enabled();
    } else {
        mperf_throw(MperfError,
                    "ERROR: this interface only support on cpu platform.\n");
    }
    m_xpmu->run();

    return 0;
}

// you should call "sample" when you want repeat sample the event values on the
// same event_list
int MPFTMA::sample(size_t iter_num) {
    if (is_cpu()) {
        auto cpu_measurements = m_xpmu->sample().cpu;
        for (size_t k = 0; k < cpu_measurements->size(); k++) {
            auto iter = (*cpu_measurements)[k];
            mperf_log_debug("sample: %s:%lu, \n", iter.first.c_str(),
                            iter.second / iter_num);
            m_duration.push_back(
                    {iter.first.c_str(), (float)iter.second / iter_num});
        }
        if (m_group_id == 0) {
            auto now = clock::now();
            float time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 now - m_start_point)
                                 .count() *
                         1e-6;

            // add a fake event entry names time_interval to record sample duration
            m_duration.push_back({"time_interval", time / iter_num});
            m_group_id = -1;
            mperf_log_debug("sample: time_interval:%f, and iter_num %zu\n",
                            time / iter_num, iter_num);
        }
        return 0;
    } else {
        mperf_throw(MperfError, "unsupport gpu currently.\n");
    }
}

// you should use "sample_and_stop" before you want to change the event_list in
// the XPMU through set_enabled_cpu_counters() interface
int MPFTMA::sample_and_stop(size_t iter_num) {
    sample(iter_num);
    m_xpmu->stop();
    return 0;
}

int MPFTMA::deinit() {
    std::vector<std::pair<std::string, float>> res;
    size_t mz = m_metrics.size();
    for (size_t i = 0; i < mz; ++i) {
        auto pmetric = m_ratio_setup->metric(m_metrics[i]);
        auto feq = std::bind(&MPFTMA::ev_query, this, std::placeholders::_1,
                             std::placeholders::_2);
        float val = ((TMACompute)(pmetric->func_compute))(feq);
        // FIXME. The value and usage of thresh need to be fixed.
        res.push_back({pmetric->name, val});
    }

    printf("TAM STATIS RESULTS:\n");
    for (size_t i = 0; i < mz; ++i) {
        printf("%10s : %10.5f\n", res[i].first.c_str(), res[i].second);
    }

    delete m_xpmu;
    m_xpmu = nullptr;
    m_metrics.clear();
    m_events.clear();
    m_uncore_events.clear();
    m_duration.clear();
    m_group_num = 0;
    m_binit = false;

    return 0;
}

bool MPFTMA::is_cpu() const {
#if 0  // TODO. support GPU later.
    if(m_xpu_type == MALI || m_xpu_type == ADRENO) {
        return false;
    }
#endif
    return true;
}

}  // namespace tma
}  // namespace mperf