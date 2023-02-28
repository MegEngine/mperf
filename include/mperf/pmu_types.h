#pragma once
#include <stdint.h>
#include <string>

namespace mperf {

// TODO. Support DEFALUT(like. simple_ratios.py)
// TODO. Support MALI/ADRENO GPU
enum MPFXPUType {
    DEFAULT,
    SNB_CLIENT,
    // JKT_SERVER,
    IVB_CLIENT,
    IVB_SERVER,
    HSW_CLIENT,
    HSX_SERVER,
    BDW_CLIENT,
    BDX_SERVER,
    SKL_CLIENT,
    SKX_SERVER,
    // CLX_SERVER,
    ICL_CLIENT,
    ICX_SERVER,
    // ADL_GLC,
    // ADL_GRT,
    // SPR_SERVER,
    A55,
    A510
};

struct EventAttr {
public:
    // ref: https://man7.org/linux/man-pages/man2/perf_event_open.2.html
    /*! \param[_name] the event name, e.g. CPU_CYCLES, INST_RETIRED.
     *  \param[_config] the event configuration, used as the unique
     *  identifier in the event set of the specific event type.
     *  \param[_t] the event type. e.g. PERF_TYPE_RAW(enum value is 4).
     *  \param[_eu] exclude user, the count excludes events that happen
     *  in user space.
     *  \param[_is_uncore] if the flag is true, means to sample an uncore
     *  event, e.g. event of arm_dsu pmu.
     *  \param[_config1] is used for setting events that need an extra
     *  register or otherwise do not fit in the regular config field,
     *  e.g. the offcore event of intel_x86 offcore pmu.
     */
    EventAttr(std::string _name = "", uint64_t _config = 0, uint32_t _t = 4,
              int _eu = 0, bool _is_uncore = false, uint64_t _config1 = 0)
            : name(_name),
              config(_config),
              type(_t),
              exclude_user(_eu),
              is_uncore(_is_uncore),
              config1(_config1) {}

    bool operator!=(const EventAttr& rhs) const {
        return name != rhs.name || config != rhs.config || type != rhs.type ||
               exclude_user != rhs.exclude_user;
    }

    bool operator==(const EventAttr& rhs) const {
        return name == rhs.name && config == rhs.config && type == rhs.type &&
               exclude_user == rhs.exclude_user;
    }

    bool operator<(const EventAttr& rhs) const { return name < rhs.name; }

    std::string name;
    uint64_t config;
    uint32_t type;
    int exclude_user;
    bool is_uncore;
    uint64_t config1;
};

}  // namespace mperf