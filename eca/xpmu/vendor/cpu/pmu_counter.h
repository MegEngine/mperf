#pragma once

#include <assert.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <vector>

#include "mperf/pmu_types.h"
#include "mperf/utils.h"
#include "mperf_build_config.h"
#if MPERF_WITH_PFM
#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"
#else
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif

class PerfCounterValues {
public:
    // TODO(zxb). Confirm kMaxCounters(x86/ARM)
#if defined(__aarch64__) || defined(__arm__)
    static constexpr size_t kMaxCounters = 3;
#else
    static constexpr size_t kMaxCounters = 2;
#endif
    // static constexpr size_t kMaxUncCounters = 6; // uncore counter nums is
    // greater than core counters usually
    explicit PerfCounterValues(size_t nr_counters, bool is_uncore = false)
            : nr_counters_(nr_counters), is_uncore_(is_uncore) {
        if (!is_uncore && nr_counters_ > kMaxCounters) {
            mperf_log_warn("The maximum number of counters is %zu\n",
                           kMaxCounters);
            nr_counters_ = kMaxCounters;
        }
        // if (is_uncore && nr_counters_ > kMaxUncCounters) {
        //    printf("WARNING: The maximum number of uncore counters is %zu\n",
        //    kMaxUncCounters); nr_counters_ = kMaxUncCounters;
        //}
    }

    uint64_t operator[](size_t pos) const { return values_[kPadding + pos]; }

    void set_value(size_t pos, uint64_t val) { values_[kPadding + pos] = val; }

    bool is_uncore() { return is_uncore_; }

    void set_uncore_enebaled() { is_uncore_ = true; }

private:
    friend class PerfCounters;
    // Get the byte buffer in which perf counters can be captured.
    // This is used by PerfCounters::Read
    std::pair<char*, size_t> get_data_buffer() {
        return {reinterpret_cast<char*>(values_.data()),
                sizeof(uint64_t) * (kPadding + nr_counters_)};
    }

    // the return value in values_[0] is the event nums in the event group
    // ref:
    // https://elixir.bootlin.com/linux/v4.2/source/include/uapi/linux/perf_event.h#L231
    static constexpr size_t kPadding = 1;
    std::array<uint64_t, kPadding + kMaxCounters> values_;
    size_t nr_counters_;
    bool is_uncore_;
};

// Collect PMU counters. The object, once constructed, is ready to be used by
// calling read(). PMU counter collection is enabled from the time create() is
// called, to obtain the object, until the object's destructor is called.
class PerfCounters final {
public:
    // True iff this platform supports performance counters.
    static const bool kSupported;

    bool IsValid() const { return is_valid_; }
    static PerfCounters NoCounters() { return PerfCounters(); }

    ~PerfCounters();
    PerfCounters(PerfCounters&&) = default;
    PerfCounters(const PerfCounters&) = delete;

    // Platform-specific implementations may choose to do some library
    // initialization here.
    static bool Initialize();

    static PerfCounters Create(const std::vector<mperf::EventAttr>& attrs);

#if MPERF_WITH_PFM
    // Return a PerfCounters object ready to read the counters with the names
    // specified. The values are user-mode only. The counter name format is
    // implementation and OS specific.
    // TODO: once we move to C++-17, this should be a std::optional, and then
    // the IsValid() boolean can be dropped.
    static PerfCounters Create(const std::vector<std::string>& counter_names);
#endif

    // Take a snapshot of the current value of the counters into the provided
    // valid PerfCounterValues storage. The values are populated such that:
    // names()[i]'s value is (*values)[i]
    __attribute__((always_inline)) bool Snapshot(
            PerfCounterValues* values) const {
        assert(values != nullptr);
        assert(IsValid());

        auto buffer = values->get_data_buffer();
#if 0  // no group perf_event read
    auto read_bytes = ::read(counter_ids_[0], buffer.first + sizeof(uint64_t), sizeof(uint64_t));
    return static_cast<size_t>(read_bytes) == sizeof(uint64_t);
#endif

        size_t read_bytes = 0;
        if (values->is_uncore()) {
            for (size_t i = 0; i < counter_ids_.size(); ++i) {
                read_bytes =
                        ::read(counter_ids_[i],
                               buffer.first + sizeof(uint64_t) * (i + 1), 8);
                assert(read_bytes == 8);
                mperf_log_debug(
                        "uncore event read: idx %zu event, the value %lu\n", i,
                        *reinterpret_cast<uint64_t*>(
                                buffer.first + sizeof(uint64_t) * (i + 1)));
            }
        } else {
            read_bytes = ::read(counter_ids_[0], buffer.first, buffer.second);
            return static_cast<size_t>(read_bytes) == buffer.second;
        }
        return true;
    }

    const std::vector<std::string>& names() const { return counter_names_; }
    size_t num_counters() const { return counter_names_.size(); }
    int counter_id(int i) const { return counter_ids_[i]; }

    PerfCounters& operator=(const PerfCounters& pc) {
        if (this == &pc)
            return *this;

        // just the fds of last event group is not close
        for (int fd : counter_ids_) {
            close(fd);
        }
        counter_ids_ = pc.counter_ids_;
        counter_names_ = pc.counter_names_;
        is_valid_ = pc.is_valid_;
        return *this;
    }

private:
    PerfCounters(const std::vector<std::string>& counter_names,
                 std::vector<int>&& counter_ids)
            : counter_ids_(std::move(counter_ids)),
              counter_names_(counter_names),
              is_valid_(true) {}
    PerfCounters() : is_valid_(false) {}

    std::vector<int> counter_ids_;
    std::vector<std::string> counter_names_;
    bool is_valid_;
};