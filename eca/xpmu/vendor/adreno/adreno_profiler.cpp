#include "adreno_profiler.h"
#include "adreno.h"
#include "series/adreno6_events.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

using namespace adreno_userspace;

namespace mperf {
// GpuCounterSet initial usage:
// {"GROUP1,EVT1,EVT2,EVT3,...;GROUP2,EVT1,EVT2,EVT3,...; ...
// ;GROUP_CUSTOM,EVT1,EVT2,EVT3,..."}
AdrenoProfiler::AdrenoProfiler(const GpuCounterSet& counters) {
    if (!check_is_series6()) {
        throw std::runtime_error(
                "invalid adreno gpu version(only support series6).");
    }

    fd_ = open(device_, O_RDWR | O_CLOEXEC | O_NONBLOCK);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open /dev/kgsl-3d0.");
    }

    memset(&events_snapshot_end, 0, sizeof(events_snapshot_end));
    memset(&events_snapshot_begin, 0, sizeof(events_snapshot_begin));
    memset(&event_read_group, 0, sizeof(event_read_group));

    mappings_ = {
            {"GFLOPs",
             [this]() {
                 return (get_event_value("SP_CS_INSTRUCTIONS") * 8) /
                        std::max<float>(1e-5, get_kern_time());
             }},
            {"GBPs",
             [this]() {
                 return ((get_event_value("UCHE_VBIF_READ_BEATS_SP") +
                          get_event_value("UCHE_VBIF_READ_BEATS_TP")) *
                                 32 +
                         (get_event_value("SP_GM_STORE_INSTRUCTIONS") * 64)) /
                        std::max<float>(1e-5, get_kern_time());
             }},
            {"GPUCycles",
             [this]() {
                 int sp_num = 2;
                 return (get_event_value("SP_BUSY_CYCLES") +
                         get_event_value("SP_NON_EXECUTION_CYCLES")) /
                        (float)sp_num;
             }},
            {"ShaderLoadStoreCycles",
             [this]() {
                 return get_event_value("SP_LM_LOAD_INSTRUCTIONS") +
                        get_event_value("SP_LM_STORE_INSTRUCTIONS") +
                        get_event_value("SP_LM_ATOMICS") +
                        get_event_value("SP_GM_LOAD_INSTRUCTIONS") +
                        get_event_value("SP_GM_STORE_INSTRUCTIONS") +
                        get_event_value("SP_GM_ATOMICS");
             }},
            {"ShaderComputeCycles",
             [this]() {
                 return get_event_value("SP_CS_INSTRUCTIONS") / 64 / 2;
             }},

            // TODO(hc): need check about adreno texture about events
            {"ShaderTextureCycles",
             [this]() { return get_event_value("TP_BUSY_CYCLES"); }},

            // TODO(hc): AluUtil cannot reach 100% in pure compute testcase,
            // need check.
            {"AluUtil",
             [this]() {
                 int sp_num = 2;
                 return (get_event_value("SP_CS_INSTRUCTIONS") / 64 / 2) /
                        (float)((get_event_value("SP_BUSY_CYCLES") +
                                 get_event_value("SP_NON_EXECUTION_CYCLES")) /
                                sp_num);
             }},
            {"LoadStoreUtil",
             [this]() {
                 int sp_num = 2;
                 return (get_event_value("SP_LM_LOAD_INSTRUCTIONS") +
                         get_event_value("SP_LM_STORE_INSTRUCTIONS") +
                         get_event_value("SP_LM_ATOMICS") +
                         get_event_value("SP_GM_LOAD_INSTRUCTIONS") +
                         get_event_value("SP_GM_STORE_INSTRUCTIONS") +
                         get_event_value("SP_GM_ATOMICS")) /
                        ((get_event_value("SP_BUSY_CYCLES") +
                          get_event_value("SP_NON_EXECUTION_CYCLES")) /
                         (float)sp_num);
             }},
            {"TextureUtil",
             [this]() {
                 int sp_num = 2;
                 return get_event_value("TP_BUSY_CYCLES") /
                        ((get_event_value("SP_BUSY_CYCLES") +
                          get_event_value("SP_NON_EXECUTION_CYCLES")) /
                         (float)sp_num);
             }},
            {"FullAluRatio",
             [this]() {
                 return (get_event_value("SP_FS_STAGE_FULL_ALU_INSTRUCTIONS") +
                         get_event_value("SP_VS_STAGE_FULL_ALU_INSTRUCTIONS")) /
                        (float)get_event_value("SP_CS_INSTRUCTIONS");
             }},
            {"ShaderBusyRatio",
             [this]() {
                 return get_event_value("SP_BUSY_CYCLES") /
                        (float)(get_event_value("SP_BUSY_CYCLES") +
                                get_event_value("SP_NON_EXECUTION_CYCLES"));
             }},
            {"ShaderStalledRatio",
             [this]() {
                 return (get_event_value("SP_STALL_CYCLES_VPC") +
                         get_event_value("SP_STALL_CYCLES_TP") +
                         get_event_value("SP_STALL_CYCLES_UCHE") +
                         get_event_value("SP_STALL_CYCLES_RB")) /
                        (float)(get_event_value("SP_BUSY_CYCLES") +
                                get_event_value("SP_NON_EXECUTION_CYCLES"));
             }},
            {"TexturePipesBusyRatio",
             [this]() {
                 return get_event_value("TP_BUSY_CYCLES") /
                        (float)(get_event_value("TP_BUSY_CYCLES") +
                                get_event_value("TP_STALL_CYCLES_UCHE") +
                                get_event_value("TP_LATENCY_CYCLES") +
                                get_event_value("TP_STARVE_CYCLES_SP") +
                                get_event_value("TP_STARVE_CYCLES_UCHE"));
             }},
            {"InstructionCacheMissRatio",
             [this]() {
                 return get_event_value("SP_ICL1_MISSES") /
                        (float)get_event_value("SP_ICL1_REQUESTS");
             }},
            {"TextureL1MissRatio",
             [this]() {
                 return get_event_value("TP_L1_CACHELINE_MISSES") /
                        (float)get_event_value("TP_L1_CACHELINE_REQUESTS");
             }},
            {"TextureL2ReadMissRatio",
             [this]() {
                 return get_event_value("UCHE_VBIF_READ_BEATS_TP") /
                        (float)get_event_value("UCHE_READ_REQUESTS_TP");
             }},
            {"L2ReadMissRatio",
             [this]() {
                 return get_event_value("UCHE_VBIF_READ_BEATS_SP") /
                        (float)get_event_value("UCHE_READ_REQUESTS_SP");
             }},
            {"L1TextureMissPerPixel",
             [this]() {
                 return get_event_value("TP_L1_CACHELINE_MISSES") /
                        (float)(get_event_value("TP_OUTPUT_PIXELS_POINT") +
                                get_event_value("TP_OUTPUT_PIXELS_BILINEA") +
                                get_event_value("TP_OUTPUT_PIXELS_MIP") +
                                get_event_value("TP_OUTPUT_PIXELS_ANISO") +
                                get_event_value("TP_OUTPUT_PIXELS_ZERO_LOD"));
             }},

    };

    split_and_fill(counters);
}

void AdrenoProfiler::split_and_fill(const GpuCounterSet& counters) {
    enabled_events_.clear();
    total_event_num_ = 0;
    auto str_group = StrSplit(counters, ';');
    size_t str_group_size = str_group.size();
    if (str_group_size < 1) {
        throw std::runtime_error(
                "invalid pmu sample input. Must have more than one group.");
    }

    enabled_events_.resize(str_group_size);

    size_t event_idx = 0;
    for (size_t g = 0; g < str_group_size; g++) {
        auto str_events = StrSplit(str_group[g], ',');
        size_t str_events_len = str_events.size();
        if (str_events_len < 2) {
            throw std::runtime_error(
                    "invalid pmu sample input. Must have more than one event "
                    "in each group.");
        }
        total_event_num_ += str_events_len - 1;
        std::string group_name = str_events[0];
        enabled_events_[g].first = group_name;
        int event_group = series6_group_index(group_name);
        for (size_t i = 1; i < str_events_len; ++i, ++event_idx) {
            event_idx_record[str_events[i]] = event_idx;
            enabled_events_[g].second.push_back(str_events[i]);
            if (event_group == EVENT_GROUP_CUSTOM) {
                continue;
            }
            if (adreno_activate_event(
                        fd_, event_group,
                        series6_event_index(event_group, str_events[i])) ==
                -1) {
                throw std::runtime_error("Failed to active event.");
            }
            event_read_group[event_idx].event_group = event_group;
            event_read_group[event_idx].event_selector =
                    series6_event_index(event_group, str_events[i]);
        }
    }
}

void AdrenoProfiler::sample_events(uint64_t* snapshot_values) {
    if (adreno_read_events(fd_, total_event_num_, event_read_group,
                           snapshot_values) == -1) {
        throw std::runtime_error("Failed to read the value of an event set.");
    }
}

void AdrenoProfiler::run() {
    sample_events(events_snapshot_begin);
}

const GpuMeasurements& AdrenoProfiler::sample() {
    sample_events(events_snapshot_end);
    measurements_.clear();

    size_t group_size = enabled_events_.size();
    size_t i = 0;
    for (size_t g = 0; g < group_size; g++) {
        auto events = enabled_events_[g].second;
        size_t len = events.size();
        int event_group = series6_group_index(enabled_events_[g].first);
        for (size_t j = 0; j < len; ++j) {
            if (event_group == EVENT_GROUP_CUSTOM) {
                std::string event_name = enabled_events_[g].second[j];
                auto mapping = mappings_.find(event_name);
                if (mapping == mappings_.end()) {
                    continue;
                }
                measurements_.push_back({event_name, mapping->second()});
            } else {
                measurements_.push_back({enabled_events_[g].second[j],
                                         (double)(events_snapshot_end[i] -
                                                  events_snapshot_begin[i])});
                events_snapshot_begin[i] = events_snapshot_end[i];
            }
            i++;
        }
    }
    return measurements_;
}

void AdrenoProfiler::stop() {
    size_t group_size = enabled_events_.size();
    for (size_t g = 0; g < group_size; g++) {
        auto group_name = enabled_events_[g].first;
        int event_group = series6_group_index(group_name);
        if (event_group == EVENT_GROUP_CUSTOM) {
            continue;
        }
        auto events = enabled_events_[g].second;
        size_t len = events.size();
        for (size_t i = 0; i < len; ++i) {
            if (adreno_deactivate_event(
                        fd_, event_group,
                        series6_event_index(event_group, events[i])) == -1) {
                throw std::runtime_error("Failed to deactive an event.");
            }
        }
    }
    close(fd_);
}

bool AdrenoProfiler::check_is_series6() const {
    // TODO(hc): AdrenoProfiler only support adreno series 6 now
    return true;
}

int AdrenoProfiler::series6_group_index(const std::string& group) const {
    auto iter = group_name_to_id.find(group.c_str());
    if (iter != group_name_to_id.end()) {
        return iter->second;
    }
    return -1;
}

int AdrenoProfiler::series6_event_index(int event_group,
                                        const std::string& event) const {
    auto map_events = series6_total_events[event_group].events;
    auto iter = map_events->find(event);
    if (iter != map_events->end()) {
        return iter->second;
    }
    return -1;
}

double AdrenoProfiler::get_event_value(const std::string& event) {
    std::string str = event;
    return measurements_[event_idx_record[str]].second.get<double>();
}

uint64_t AdrenoProfiler::get_kern_time() const {
    if (kern_time == 0) {
        mperf_log_warn("please set kern_time before use it");
    }

    return kern_time;
}

void AdrenoProfiler::set_kern_time(uint64_t kern_time_) {
    kern_time = kern_time_;
}

}  // namespace mperf
