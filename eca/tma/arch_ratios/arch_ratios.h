/**
 * \file eca/tma/arch_ratios/arch_ratios.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#pragma once
#include <stdio.h>
#include <functional>
#include <string>
#include "mperf/exception.h"
#include "mperf/tma/tma.h"

namespace mperf {
namespace tma {

class MetricBase {
public:
    std::string name;
    std::string domain;
    std::string area;
    int level;
    std::string desc;
    static float val;
    static bool thresh;

    static MetricBase* parent;
    void* func_compute;
};

using FEV = std::function<float(EventAttr attr, int level)>;

inline float g_ev_error(std::string name, int level) {
    MPERF_MARK_USED_VAR(name);
    mperf_throw(MperfError, "unsupport yet.\n");
}

inline float g_ev_process(std::function<float()> func, int level) {
    return func();
}

class SetUpBase {
public:
    SetUpBase() {}
    virtual ~SetUpBase() {}
    MetricBase* metric(std::string name) const;
    std::vector<std::pair<std::string, MetricBase*>> m_vmtc_core;
    std::vector<std::pair<std::string, MetricBase*>> m_vmtc_extra;
};

#define DEF_SETUP_CLS(NAME)                    \
    class NAME##SetUpImpl : public SetUpBase { \
    public:                                    \
        NAME##SetUpImpl();                     \
        ~NAME##SetUpImpl();                    \
    };

class ArchRatioSetup {
public:
    static ArchRatioSetup* inst(MPFXPUType t);

    DEF_SETUP_CLS(SNBCLIENT)
    DEF_SETUP_CLS(JKTSERVER)
    DEF_SETUP_CLS(IVBCLIENT)
    DEF_SETUP_CLS(IVBSERVER)
    DEF_SETUP_CLS(HSWCLIENT)
    DEF_SETUP_CLS(HSXSERVER)
    DEF_SETUP_CLS(BDWCLIENT)
    DEF_SETUP_CLS(BDXSERVER)
    DEF_SETUP_CLS(SKLCLIENT)
    DEF_SETUP_CLS(SKXSERVER)
    DEF_SETUP_CLS(CLXSERVER)
    DEF_SETUP_CLS(ICLCLIENT)
    DEF_SETUP_CLS(ICXSERVER)
    DEF_SETUP_CLS(ADLGLC)
    DEF_SETUP_CLS(ADLGRT)
    DEF_SETUP_CLS(SPRSERVER)
    DEF_SETUP_CLS(A55)
    DEF_SETUP_CLS(A510)

    MetricBase* metric(std::string name) const;
    size_t counter_num() const;

private:
    ArchRatioSetup(MPFXPUType t);

    size_t m_counter_num;
    std::unique_ptr<SetUpBase> m_setup;
};
#undef DEF_SETUP_CLS

#define DEClARE_SETUPIMPL(NAME) \
    using NAME##SetUpImpl = ArchRatioSetup::NAME##SetUpImpl;

// clang-format off
DEClARE_SETUPIMPL(SNBCLIENT)
DEClARE_SETUPIMPL(JKTSERVER)
DEClARE_SETUPIMPL(IVBCLIENT)
DEClARE_SETUPIMPL(IVBSERVER)
DEClARE_SETUPIMPL(HSWCLIENT)
DEClARE_SETUPIMPL(HSXSERVER)
DEClARE_SETUPIMPL(BDWCLIENT)
DEClARE_SETUPIMPL(BDXSERVER)
DEClARE_SETUPIMPL(SKLCLIENT)
DEClARE_SETUPIMPL(SKXSERVER)
DEClARE_SETUPIMPL(CLXSERVER)
DEClARE_SETUPIMPL(ICLCLIENT)
DEClARE_SETUPIMPL(ICXSERVER)
DEClARE_SETUPIMPL(ADLGLC)
DEClARE_SETUPIMPL(ADLGRT)
DEClARE_SETUPIMPL(SPRSERVER)
DEClARE_SETUPIMPL(A55)
DEClARE_SETUPIMPL(A510)
// clang-format on

#undef DEClARE_SETUPIMPL

}  // namespace tma
}  // namespace mperf