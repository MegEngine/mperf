#pragma once

#include <functional>
#include <vector>
#include "adreno_core.hpp"
#include "mperf/utils.h"
#include "mperf/xpmu/gpu_profiler.h"

namespace mperf {

class AdrenoProfiler final : public GpuProfiler {
public:
    // The maximum number of events using by one sample calls
    static constexpr size_t MAXIMUM_SAMPLE_EVNETS = 125;

    explicit AdrenoProfiler(const GpuCounterSet& counters);
    virtual ~AdrenoProfiler() = default;

    virtual void set_enabled_counters(GpuCounterSet counters) override {
        split_and_fill(counters);
    };

    virtual void run() override;
    virtual const GpuMeasurements& sample() override;
    virtual void stop() override;
    virtual void set_kern_time(uint64_t kern_time) override;
    virtual void set_dtype_sz(uint64_t dtype_size) override {
        MPERF_MARK_USED_VAR(dtype_size);
    };

private:
    std::vector<std::pair<std::string, std::vector<std::string>>>
            enabled_events_{};
    std::unordered_map<std::string, int> event_idx_record{};
    typedef std::function<double(void)> AdrenoValueGetter;
    std::unordered_map<std::string, AdrenoValueGetter> mappings_{};
    unsigned int total_event_num_;
    const char* const device_{"/dev/kgsl-3d0"};
    int fd_{-1};
    uint64_t events_snapshot_begin[MAXIMUM_SAMPLE_EVNETS];
    uint64_t events_snapshot_end[MAXIMUM_SAMPLE_EVNETS];
    struct adreno_event_read_entry event_read_group[MAXIMUM_SAMPLE_EVNETS];
    uint64_t kern_time{0};  // nano seconds
    GpuMeasurements measurements_{};

    void split_and_fill(const GpuCounterSet& counters);
    bool check_is_series6() const;
    int series6_group_index(const std::string& group) const;
    int series6_event_index(int group_id, const std::string& event) const;
    double get_event_value(const std::string& event);
    void sample_events(uint64_t* snapshot_values);
    uint64_t get_kern_time() const;
};
}  // namespace mperf
