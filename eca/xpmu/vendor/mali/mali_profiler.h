#pragma once

#include "mperf/xpmu/gpu_profiler.h"

#include "mali_core.hpp"

#include <functional>
#include <vector>

namespace mperf {
/** A Gpu profiler that uses Mali counter data. */
class MaliProfiler final : public GpuProfiler {
public:
    explicit MaliProfiler(const GpuCounterSet& enabled_counters);
    virtual ~MaliProfiler() = default;

    virtual void set_enabled_counters(GpuCounterSet counters) override {
        enabled_counters_ = std::move(counters);
    };

    virtual void run() override;
    virtual const GpuMeasurements& sample() override;
    virtual void stop() override;
    virtual void set_kern_time(uint64_t kern_time) override;
    virtual void set_dtype_sz(uint64_t dtype_size) override {
        dtype_sz = dtype_size;
    };

private:
    GpuCounterSet enabled_counters_{};

    typedef std::function<double(void)> MaliValueGetter;
    std::unordered_map<GpuCounter, MaliValueGetter, GpuCounterHash> mappings_{};

    const char* const device_{"/dev/mali0"};
    int num_cores_{0};
    int num_l2_slices_{0};
    int gpu_id_{0};
    uint32_t hw_ver_{0};
    int buffer_count_{16};
    size_t buffer_size_{0};
    size_t dtype_sz{4};
    uint8_t* sample_data_{nullptr};
    uint64_t timestamp_old_{0};
    uint64_t timestamp_{0};
    uint64_t sample_duration{0};
    uint64_t kern_time{0};  // nano seconds
    const char* const* names_lut_{nullptr};
    std::vector<uint32_t> raw_counter_buffer_{};
    std::vector<unsigned int> core_index_remap_{};
    int fd_{-1};
    int xpmu_fd_{-1};

    GpuMeasurements measurements_{};

    void init();
    void sample_counters();
    void wait_next_event();
    const uint32_t* get_counters(mali_userspace::MaliCounterBlockName block,
                                 int index = 0) const;
    uint64_t get_counter_value(mali_userspace::MaliCounterBlockName block,
                               const char* name) const;
    int find_counter_index_by_name(mali_userspace::MaliCounterBlockName block,
                                   const char* name) const;
    uint64_t get_sample_duration() const;
    uint64_t get_kern_time() const;
};

}  // namespace mperf
