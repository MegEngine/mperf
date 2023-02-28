#pragma once

#include <string>
#include <vector>
#include "mperf/exception.h"
#include "mperf/utils.h"
#include "mperf/xpmu/cpu_profiler.h"
#include "pmu_counter.h"

#if __cplusplus >= 201103L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)
#include <atomic>
inline MPERF_ALWAYS_INLINE void ClobberMemory() {
    std::atomic_signal_fence(std::memory_order_acq_rel);
}
#else
inline MPERF_ALWAYS_INLINE void ClobberMemory() {
    asm volatile("" : : : "memory");
}
#endif

namespace mperf {
/** A CPU profiler that uses PMU counter data. */
class PmuProfiler final : public CpuProfiler {
public:
#if MPERF_WITH_PFM
    PmuProfiler(const std::vector<std::string>& counter_names,
                bool is_uncore = false)
            : counters_(PerfCounters::Create(counter_names)),
              start_values_(PerfCounterValues::kMaxCounters, is_uncore),
              end_values_(PerfCounterValues::kMaxCounters, is_uncore) {}
#endif
    PmuProfiler(const std::vector<mperf::EventAttr>& event_attrs,
                bool is_uncore = false)
            : counters_(PerfCounters::Create(event_attrs)),
              start_values_(PerfCounterValues::kMaxCounters, is_uncore),
              end_values_(PerfCounterValues::kMaxCounters, is_uncore) {}

#if MPERF_WITH_PFM
    void set_enabled_counters(
            const std::vector<std::string>& counter_names) override {
        counters_ = PerfCounters::Create(counter_names);
    }
#endif

    void set_enabled_counters(
            const std::vector<mperf::EventAttr>& event_attrs) override {
        counters_ = PerfCounters::Create(event_attrs);
    }

    MPERF_ALWAYS_INLINE void run() override {
        assert(IsValid());

        if (ioctl(counters_.counter_id(0), PERF_EVENT_IOC_ENABLE,
                  PERF_IOC_FLAG_GROUP) != 0) {
            mperf_throw(MperfError, "Failed to enable counters\n");
        }

        if (ioctl(counters_.counter_id(0), PERF_EVENT_IOC_RESET,
                  PERF_IOC_FLAG_GROUP) != 0) {
            mperf_throw(MperfError, "Failed to reset counters\n");
        }

        // Tell the compiler to not move instructions above/below where we take
        // the snapshot.
        ClobberMemory();
        counters_.Snapshot(&start_values_);
        ClobberMemory();

        for (size_t i = 0; i < PerfCounterValues::kMaxCounters; ++i) {
            mperf_log_debug("the start_calues_[%zu] : %lu\n", i,
                            start_values_[i]);
        }
    }

    MPERF_ALWAYS_INLINE const CpuMeasurements& sample() override {
        assert(IsValid());
        // Tell the compiler to not move instructions above/below where we take
        // the snapshot.
        ClobberMemory();
        counters_.Snapshot(&end_values_);
        ClobberMemory();

        for (size_t i = 0; i < PerfCounterValues::kMaxCounters; ++i) {
            mperf_log_debug("the end_values_[%zu] : %lu\n", i, end_values_[i]);
        }
        results.clear();
        for (size_t i = 0; i < counters_.names().size(); ++i) {
            uint64_t measurement = static_cast<uint64_t>(end_values_[i]) -
                                   static_cast<uint64_t>(start_values_[i]);
            results.push_back({counters_.names()[i], measurement});
            start_values_.set_value(i, end_values_[i]);
            mperf_log_debug("the measurement[%zu] : %lu\n", i, measurement);
        }

        return results;
    }

    MPERF_ALWAYS_INLINE void stop() override {
        ioctl(counters_.counter_id(0), PERF_EVENT_IOC_DISABLE,
              PERF_IOC_FLAG_GROUP);
    }

    MPERF_ALWAYS_INLINE void set_uncore_event_enabled() override {
        start_values_.set_uncore_enebaled();
        end_values_.set_uncore_enebaled();
    }

private:
    bool IsValid() const { return counters_.IsValid(); }

    PerfCounters counters_;
    PerfCounterValues start_values_;
    PerfCounterValues end_values_;
    CpuMeasurements results;
};

// TODO(zxb). change to manual startup mode.
__attribute__((unused)) static bool perf_init_anchor =
        PerfCounters::Initialize();

}  // namespace mperf
