#include "mperf/xpmu/xpmu.h"
#include "mperf_build_config.h"

#ifdef __linux__
#include "vendor/cpu/pmu_profiler.h"
#if MPERF_WITH_MALI
#include "vendor/mali/mali_profiler.h"
#endif
#if MPERF_WITH_ADRENO
#include "vendor/adreno/adreno_profiler.h"
#endif
#endif

#include <memory>

namespace mperf {
#if MPERF_WITH_PFM
XPMU::XPMU(CpuCounterSet enabled_cpu_counters,
           GpuCounterSet enabled_gpu_counters) {
    create_profilers(std::move(enabled_cpu_counters),
                     std::move(enabled_gpu_counters));
}
#endif

XPMU::XPMU(CpuCounterSet2 enabled_cpu_counters) {
    create_profilers(std::move(enabled_cpu_counters));
}

XPMU::XPMU(GpuCounterSet enabled_gpu_counters) {
    create_profilers(std::move(enabled_gpu_counters));
}

#if MPERF_WITH_PFM
void XPMU::set_enabled_cpu_counters(CpuCounterSet counters) {
    if (cpu_profiler_) {
        cpu_profiler_->set_enabled_counters(std::move(StrSplit(counters, ',')));
    } else {
        create_profilers(counters, {});
    }
}
#endif

void XPMU::set_enabled_cpu_counters(CpuCounterSet2 event_attrs) {
    if (cpu_profiler_) {
        cpu_profiler_->set_enabled_counters(event_attrs);
    } else {
        create_profilers(event_attrs);
    }
}

void XPMU::set_enabled_gpu_counters(GpuCounterSet counters) {
    if (gpu_profiler_) {
        gpu_profiler_->set_enabled_counters(std::move(counters));
    } else {
        create_profilers(counters);
    }
}

void XPMU::set_cpu_uncore_event_enabled() {
    if (cpu_profiler_) {
        cpu_profiler_->set_uncore_event_enabled();
    }
}

void XPMU::run() {
    if (cpu_profiler_) {
        cpu_profiler_->run();
    }
    if (gpu_profiler_) {
        gpu_profiler_->run();
    }
}

Measurements XPMU::sample() {
    Measurements m;
    if (cpu_profiler_) {
        m.cpu = &cpu_profiler_->sample();
    }
    if (gpu_profiler_) {
        m.gpu = &gpu_profiler_->sample();
    }
    return m;
}

void XPMU::stop() {
    if (cpu_profiler_) {
        cpu_profiler_->stop();
    }
    if (gpu_profiler_) {
        gpu_profiler_->stop();
    }
}

#if MPERF_WITH_PFM
void XPMU::create_profilers(CpuCounterSet enabled_cpu_counters,
                            GpuCounterSet enabled_gpu_counters) {
#ifdef __linux__
    if (enabled_cpu_counters.size() != 0) {
        cpu_profiler_ = std::unique_ptr<PmuProfiler>(
                new PmuProfiler(StrSplit(enabled_cpu_counters, ',')));
    }

    if (enabled_gpu_counters.size() != 0) {
#if MPERF_GPU_MALI
        gpu_profiler_ = std::unique_ptr<MaliProfiler>(
                new MaliProfiler(enabled_gpu_counters));
#elif MPERF_GPU_ADRENO
        gpu_profiler_ = std::unique_ptr<AdrenoProfiler>(
                new AdrenoProfiler(enabled_gpu_counters));
#endif
    }

#else
    mperf_throw(MperfError, "unsupport platform.");
#endif
}
#endif

void XPMU::create_profilers(GpuCounterSet enabled_gpu_counters) {
#ifdef __linux__
    if (enabled_gpu_counters.size() != 0) {
#if MPERF_WITH_MALI
        gpu_profiler_ = std::unique_ptr<MaliProfiler>(
                new MaliProfiler(enabled_gpu_counters));
#elif MPERF_WITH_ADRENO
        gpu_profiler_ = std::unique_ptr<AdrenoProfiler>(
                new AdrenoProfiler(enabled_gpu_counters));
#endif
    }
#else
    mperf_thorw(MperfError, "ERROR: unsupport platform.");
#endif
}

void XPMU::create_profilers(CpuCounterSet2 enabled_cpu_counters) {
#ifdef __linux__
    if (enabled_cpu_counters.size() != 0) {
        cpu_profiler_ = std::unique_ptr<PmuProfiler>(
                new PmuProfiler(enabled_cpu_counters));
    }
#else
    mperf_thorw(MperfError, "ERROR: unsupport platform.");
#endif
}
}  // namespace mperf
