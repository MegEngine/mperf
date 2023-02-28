/**
 * \file include/mperf/opencl_public_api.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void mperfOpenCLEnableProfile(bool flag);

void mperfOpenCLSetCacheFileName(const char* filename);

//! brief clear all global cache data
void mperfOpenCLClearGlobalData();

enum class MPERFGPUPerfHint { PERF_LOW, PERF_NORMAL, PERF_HIGH };

enum class MPERFGPUPriorityHint {
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    PRIORITY_HIGH
};

/*!
 * \brief set qcom perf hint.
 * \note The hint only usable in QCom Device.
 * \warning The default value is PERF_HIGH and PRIORITY_HIGH as before.
 */
void mperfSetQcomHint(MPERFGPUPerfHint perf_hint,
                      MPERFGPUPriorityHint priority_hint);

#ifdef __cplusplus
}
#endif
