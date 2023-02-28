#include "pmu_counter.h"
#include "mperf/exception.h"
#include "mperf/utils.h"
#include "mperf_build_config.h"

#include <errno.h>
#include <string.h>
#include <string>
#include <vector>

constexpr size_t PerfCounterValues::kMaxCounters;
// constexpr size_t PerfCounterValues::kMaxUncCounters;
const bool PerfCounters::kSupported = true;

bool PerfCounters::Initialize() {
#if MPERF_WITH_PFM
    return pfm_initialize() == PFM_SUCCESS;
#else
    return true;
#endif
}

#if !MPERF_WITH_PFM
// clang-format off
/*
 * perf_event_open() syscall stub
 */
static inline int
perf_event_open(
	struct perf_event_attr		*hw_event_uptr,
	pid_t				pid,
	int				cpu,
	int				group_fd,
	unsigned long			flags)
{
	return syscall(
		__NR_perf_event_open, hw_event_uptr, pid, cpu, group_fd, flags);
}
// clang-format on
#endif

PerfCounters PerfCounters::Create(
        const std::vector<mperf::EventAttr>& event_attrs) {
    if (event_attrs.empty()) {
        // intentional not to use mperf_throw, because the xpmu ctor may receive
        // an enpty CpuCounterSet.
        return NoCounters();
    }
    size_t tsize = event_attrs.size();
    if (tsize > PerfCounterValues::kMaxCounters) {
        mperf_throw(
                mperf::MperfError, "%zu,%s,%zu\n", tsize,
                " counters were requested. The minimum is 1, the maximum is ",
                PerfCounterValues::kMaxCounters);
    }
    std::vector<int> counter_ids(tsize);
    std::vector<std::string> counter_names(tsize);

    for (size_t i = 0; i < tsize; ++i) {
        const bool is_first = i == 0;
        struct perf_event_attr attr {};
        attr.size = sizeof(attr);
        const int group_id = !is_first ? counter_ids[0] : -1;
        const auto& name = event_attrs[i].name;
        if (name.empty()) {
            mperf_throw(mperf::MperfError,
                        "A counter name was the empty string\n");
        }

        attr.config = event_attrs[i].config;
        attr.config1 = event_attrs[i].config1;
        attr.type = event_attrs[i].type;
        attr.exclude_user = event_attrs[i].exclude_user;
        mperf_log_debug("name %s, attr.type %d, and attr.config %llu\n",
                        name.c_str(), attr.type, attr.config);
        attr.disabled = 1;

        if (event_attrs[i].is_uncore) {
            attr.inherit = true;
            attr.sample_type = 1U << 16;  // PERF_SAMPLE_IDENTIFIER = 1U << 16
            attr.read_format = 0;
        } else {
            // Note: the man page for perf_event_create suggests inerit = true
            // and read_format = PERF_FORMAT_GROUP don't work together, but
            // that's not the case.
            attr.inherit = true;
            attr.pinned = is_first;
            attr.exclude_kernel = true;
            attr.exclude_hv = true;
            // TODO(hc) check the attr.exclude_guest usage.
            // attr.exclude_guest = 1;)
            // Read all counters in one read.
            attr.read_format = PERF_FORMAT_GROUP;
        }

        int id = -1;
        static constexpr size_t kNrOfSyscallRetries = 5;
        // Retry syscall as it was interrupted often (b/64774091).
        for (size_t num_retries = 0; num_retries < kNrOfSyscallRetries;
             ++num_retries) {
            if (event_attrs[i].is_uncore) {
                // TODO(hc): we need sample the uncore event on each sochet if
                // it's x86 platform
                id = perf_event_open(&attr, -1, 0, -1, 0);
            } else {
                id = perf_event_open(&attr, 0, -1, group_id, 0);
            }

            if (id >= 0 || errno != EINTR) {
                break;
            }
        }

        if (id < 0) {
            mperf_throw(mperf::MperfError,
                        "Failed to get a file descriptor for %s\n",
                        name.c_str());
        }

        counter_ids[i] = id;
        counter_names[i] = name;
        mperf_log_debug("the counter id %d\n", id);
    }

    if (event_attrs[0].is_uncore) {
        for (size_t i = 0; i < tsize; ++i) {
            if (ioctl(counter_ids[i], PERF_EVENT_IOC_RESET, 0) != 0) {
                mperf_throw(mperf::MperfError, "Failed to reset counters\n");
            }

            if (ioctl(counter_ids[i], PERF_EVENT_IOC_ENABLE, 0) != 0) {
                mperf_throw(mperf::MperfError, "Failed to enable counters\n");
            }
        }
    } else {
        if (ioctl(counter_ids[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) !=
            0) {
            mperf_throw(mperf::MperfError, "Failed to reset counters\n");
        }

        if (ioctl(counter_ids[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) !=
            0) {
            mperf_throw(mperf::MperfError, "Failed to enable counters\n");
        }
    }

    return PerfCounters(counter_names, std::move(counter_ids));
}

#if MPERF_WITH_PFM
PerfCounters PerfCounters::Create(
        const std::vector<std::string>& counter_names) {
    if (counter_names.empty()) {
        // intentional not to use mperf_throw, because the xpmu ctor may receive
        // an enpty CpuCounterSet.
        return NoCounters();
    }
    if (counter_names.size() > PerfCounterValues::kMaxCounters) {
        mperf_throw(mperf::MperfError,
                    "%zu counters were requested. The minimum is 1, the "
                    "maximum is %zu",
                    counter_names.size(), PerfCounterValues::kMaxCounters);
    }
    std::vector<int> counter_ids(counter_names.size());

    const int mode = PFM_PLM3;  // user mode only
    for (size_t i = 0; i < counter_names.size(); ++i) {
        const bool is_first = i == 0;
        struct perf_event_attr attr {};
        attr.size = sizeof(attr);
        const int group_id = !is_first ? counter_ids[0] : -1;
        const auto& name = counter_names[i];
        if (name.empty()) {
            mperf_throw(mperf::MperfError,
                        "A counter name was the empty string\n");
        }
        pfm_perf_encode_arg_t arg{};
        arg.attr = &attr;

        const int pfm_get = pfm_get_os_event_encoding(name.c_str(), mode,
                                                      PFM_OS_PERF_EVENT, &arg);
        if (pfm_get != PFM_SUCCESS) {
            mperf_throw(mperf::MperfError, "Unknown counter name: %s\n",
                        name.c_str());
        }
        mperf_log_debug("name %s, attr.type %d, and attr.config %llu\n",
                        name.c_str(), attr.type, attr.config);

        attr.disabled = 1;
        // Note: the man page for perf_event_create suggests inerit = true and
        // read_format = PERF_FORMAT_GROUP don't work together, but that's not
        // the case.
        attr.inherit = true;
        attr.pinned = is_first;
        attr.exclude_kernel = true;
        attr.exclude_user = false;
        attr.exclude_hv = true;
        // Read all counters in one read.
        attr.read_format = PERF_FORMAT_GROUP;

        int id = -1;
        static constexpr size_t kNrOfSyscallRetries = 5;
        // Retry syscall as it was interrupted often (b/64774091).
        for (size_t num_retries = 0; num_retries < kNrOfSyscallRetries;
             ++num_retries) {
            id = perf_event_open(&attr, 0, -1, group_id, 0);
            if (id >= 0 || errno != EINTR) {
                break;
            }
        }

        if (id < 0) {
            mperf_throw(mperf::MperfError,
                        "Failed to get a file descriptor for %s\n",
                        name.c_str());
        }

        counter_ids[i] = id;
        mperf_log_debug("the counter id %d\n", id);
    }

    if (ioctl(counter_ids[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        mperf_throw(mperf::MperfError, "Failed to reset counters\n");
    }

    if (ioctl(counter_ids[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) !=
        0) {
        mperf_throw(mperf::MperfError, "Failed to enable counters\n");
    }

    return PerfCounters(counter_names, std::move(counter_ids));
}
#endif

PerfCounters::~PerfCounters() {
    if (counter_ids_.empty()) {
        return;
    }
    // intentional not to close fd here, because the operator= interface call
    // this dctor
    // ioctl(counter_ids_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    // for (int fd : counter_ids_) {
    //    close(fd);
    //}
}
