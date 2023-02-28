/**
 * \file eca/tma/arch_ratios/arch_ratios.cpp
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#include "arch_ratios.h"

namespace mperf {
namespace tma {

MetricBase* MetricBase::parent = nullptr;
float MetricBase::val = 0;
bool MetricBase::thresh = 0;

MetricBase* SetUpBase::metric(std::string name) const {
    size_t cz = m_vmtc_core.size();
    for (size_t i = 0; i < cz; ++i) {
        if (m_vmtc_core[i].first == name) {
            return m_vmtc_core[i].second;
        }
    }
    size_t ez = m_vmtc_extra.size();
    for (size_t i = 0; i < ez; ++i) {
        if (m_vmtc_extra[i].first == name) {
            return m_vmtc_extra[i].second;
        }
    }
    return nullptr;
}

ArchRatioSetup* ArchRatioSetup::inst(MPFXPUType t) {
    static ArchRatioSetup instance(t);
    return &instance;
}

ArchRatioSetup::ArchRatioSetup(MPFXPUType t) {
    // FIXME. Confirm the correct number of counters.
    m_counter_num = 3;
    switch (t) {
#if defined(__aarch64__) || defined(__arm__)
        case A55: {
            m_setup = std::make_unique<A55SetUpImpl>();
            // FIXME(hc): the counter num in a55 trm is six, but less than six
            // counters can be use for some unknown reasons
            m_counter_num = 3;
            break;
        }
        case A510: {
            m_setup = std::make_unique<A510SetUpImpl>();
            m_counter_num = 3;
            break;
        }
#else
        // case ADL_GLC: {
        //     m_setup = std::make_unique<ADLGLCSetUpImpl>();
        //     break;
        // }
        // case ADL_GRT: {
        //     m_setup = std::make_unique<ADLGRTSetUpImpl>();
        //     break;
        // }
        // case BDW_CLIENT: {
        //     m_setup = std::make_unique<BDWCLIENTSetUpImpl>();
        //     break;
        // }
        // case BDX_SERVER: {
        //     m_setup = std::make_unique<BDXSERVERSetUpImpl>();
        //     break;
        // }
        // case CLX_SERVER: {
        //     m_setup = std::make_unique<CLXSERVERSetUpImpl>();
        //     break;
        // }
        // case HSW_CLIENT: {
        //    m_setup = std::make_unique<HSWCLIENTSetUpImpl>();
        //    break;
        //}
        case HSX_SERVER: {
            m_setup = std::make_unique<HSXSERVERSetUpImpl>();
            // TODO(hc): encountered a situation that resulting in wront result
            // with m_counter_num 3 and sample set of
            // "INST_RETIRED.ALL,MEM_LOAD_UOPS_RETIRED.L1_MISS,MEM_LOAD_UOPS_RETIRED.L2_MISS",
            // so set the m_counter_num equal 2 temporarily
            m_counter_num = 2;
            break;
        }
            // case ICL_CLIENT: {
            //     m_setup = std::make_unique<ICLCLIENTSetUpImpl>();
            //     break;
            // }
            // case ICX_SERVER: {
            //     m_setup = std::make_unique<ICXSERVERSetUpImpl>();
            //     break;
            // }
            // case IVB_CLIENT: {
            //     m_setup = std::make_unique<IVBCLIENTSetUpImpl>();
            //     break;
            // }
            // case IVB_SERVER: {
            //     m_setup = std::make_unique<IVBSERVERSetUpImpl>();
            //     break;
            // }
            // case JKT_SERVER: {
            //     m_setup = std::make_unique<JKTSERVERSetUpImpl>();
            //     break;
            // }
            // case SKL_CLIENT: {
            //     m_setup = std::make_unique<SKLCLIENTSetUpImpl>();
            //     break;
            // }
            // case SKX_SERVER: {
            //     m_setup = std::make_unique<SKXSERVERSetUpImpl>();
            //     break;
            // }
            // case SNB_CLIENT: {
            //     m_setup = std::make_unique<SNBCLIENTSetUpImpl>();
            //     break;
            // }
            // case SPR_SERVER: {
            //     m_setup = std::make_unique<SPRSERVERSetUpImpl>();
            //     break;
            // }
#endif
        case DEFAULT:  // SIMPLE
        default:
            mperf_throw(MperfError, "unknown CPU type.");
    }
}

MetricBase* ArchRatioSetup::metric(std::string name) const {
    if (m_setup) {
        return m_setup->metric(name);
    } else {
        return nullptr;
    }
}

size_t ArchRatioSetup::counter_num() const {
    return m_counter_num;
}

}  // namespace tma
}  // namespace mperf
