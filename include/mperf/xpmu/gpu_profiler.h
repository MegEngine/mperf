/**
 * \file include/mperf/xpmu/gpu_profiler.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#pragma once

#include "mperf_build_config.h"
#include "value.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mperf {

#if MPERF_WITH_ADRENO
typedef std::string GpuCounterSet;
typedef std::vector<std::pair<std::string, Value>> GpuMeasurements;
#else
// The supported GPU counters on mali platform. MaliProfiler will sample a
// subset of them.
enum class GpuCounter {
    GpuCycles,
    ComputeCycles,
    VertexCycles,
    VertexComputeCycles,
    FragmentCycles,
    TilerCycles,
    ComputeJobs,
    VertexJobs,
    VertexComputeJobs,
    FragmentJobs,
    Pixels,
    CulledPrimitives,
    VisiblePrimitives,
    InputPrimitives,
    Tiles,
    TransactionEliminations,
    EarlyZTests,
    EarlyZKilled,
    LateZTests,
    LateZKilled,
    Instructions,
    DivergedInstructions,
    ShaderComputeCycles,
    ShaderFragmentCycles,
    ShaderCycles,
    ShaderArithmeticCycles,
    ShaderInterpolatorCycles,
    ShaderLoadStoreCycles,
    ShaderTextureCycles,
    CacheReadLookups,
    CacheWriteLookups,
    ExternalMemoryReadAccesses,
    ExternalMemoryWriteAccesses,
    ExternalMemoryReadStalls,
    ExternalMemoryWriteStalls,
    ExternalMemoryReadBytes,
    ExternalMemoryWriteBytes,
    ComputeActive,
    ComputeTasks,
    ComputeWarps,
    ComputeStarving,
    WarpRegSize64,
    LoadStoreReadFull,
    LoadStoreReadPartial,
    LoadStoreWriteFull,
    LoadStoreWritePartial,
    LscRdBeats,
    LscRdExtBeats,
    AluUtil,
    LoadStoreUtil,
    PartialReadRatio,
    PartialWriteRatio,
    GFLOPs,
    GBPs,
    L2ReadMissRatio,
    L2WriteMissRatio,
    FullRegWarpRatio,
    WarpDivergenceRatio,
    ShaderTextureCyclesRatio,
    ShaderArithmeticCyclesRatio,
    ShaderLoadStoreCyclesRatio,
    OtherCyclesRatio,
    MaxValue
};

// A hash function for GpuCounter values
struct GpuCounterHash {
    template <typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

// Mapping from GPU counter enum values to counter string name. Used for log
// print.
const std::unordered_map<GpuCounter, std::string, GpuCounterHash>
        counter_enum_to_names{
                {GpuCounter::GpuCycles, "GpuCycles"},
                {GpuCounter::ComputeCycles, "ComputeCycles"},
                {GpuCounter::VertexCycles, "VertexCycles"},
                {GpuCounter::VertexComputeCycles, "VertexComputeCycles"},
                {GpuCounter::FragmentCycles, "FragmentCycles"},
                {GpuCounter::TilerCycles, "TilerCycles"},
                {GpuCounter::ComputeJobs, "ComputeJobs"},
                {GpuCounter::VertexJobs, "VertexJobs"},
                {GpuCounter::VertexComputeJobs, "VertexComputeJobs"},
                {GpuCounter::FragmentJobs, "FragmentJobs"},
                {GpuCounter::Pixels, "Pixels"},
                {GpuCounter::CulledPrimitives, "CulledPrimitives"},
                {GpuCounter::VisiblePrimitives, "VisiblePrimitives"},
                {GpuCounter::InputPrimitives, "InputPrimitives"},
                {GpuCounter::Tiles, "Tiles"},
                {GpuCounter::TransactionEliminations,
                 "TransactionEliminations"},
                {GpuCounter::EarlyZTests, "EarlyZTests"},
                {GpuCounter::EarlyZKilled, "EarlyZKilled"},
                {GpuCounter::LateZTests, "LateZTests"},
                {GpuCounter::LateZKilled, "LateZKilled"},
                {GpuCounter::Instructions, "Instructions"},
                {GpuCounter::DivergedInstructions, "DivergedInstructions"},
                {GpuCounter::ShaderComputeCycles, "ShaderComputeCycles"},
                {GpuCounter::ShaderFragmentCycles, "ShaderFragmentCycles"},
                {GpuCounter::ShaderCycles, "ShaderCycles"},
                {GpuCounter::ShaderArithmeticCycles, "ShaderArithmeticCycles"},
                {GpuCounter::ShaderInterpolatorCycles,
                 "ShaderInterpolatorCycles"},
                {GpuCounter::ShaderLoadStoreCycles, "ShaderLoadStoreCycles"},
                {GpuCounter::ShaderTextureCycles, "ShaderTextureCycles"},
                {GpuCounter::CacheReadLookups, "CacheReadLookups"},
                {GpuCounter::CacheWriteLookups, "CacheWriteLookups"},
                {GpuCounter::ExternalMemoryReadAccesses,
                 "ExternalMemoryReadAccesses"},
                {GpuCounter::ExternalMemoryWriteAccesses,
                 "ExternalMemoryWriteAccesses"},
                {GpuCounter::ExternalMemoryReadStalls,
                 "ExternalMemoryReadStalls"},
                {GpuCounter::ExternalMemoryWriteStalls,
                 "ExternalMemoryWriteStalls"},
                {GpuCounter::ExternalMemoryReadBytes,
                 "ExternalMemoryReadBytes"},
                {GpuCounter::ExternalMemoryWriteBytes,
                 "ExternalMemoryWriteBytes"},
                {GpuCounter::ComputeActive, "ComputeActive"},
                {GpuCounter::ComputeTasks, "ComputeTasks"},
                {GpuCounter::ComputeWarps, "ComputeWarps"},
                {GpuCounter::ComputeStarving, "ComputeStarving"},
                {GpuCounter::WarpRegSize64, "WarpRegSize64"},
                {GpuCounter::LoadStoreReadFull, "LoadStoreReadFull"},
                {GpuCounter::LoadStoreReadPartial, "LoadStoreReadPartial"},
                {GpuCounter::LoadStoreWriteFull, "LoadStoreWriteFull"},
                {GpuCounter::LoadStoreWritePartial, "LoadStoreWritePartial"},
                {GpuCounter::LscRdBeats, "LscRdBeats"},
                {GpuCounter::LscRdExtBeats, "LscRdExtBeats"},
                {GpuCounter::AluUtil, "AluUtil"},
                {GpuCounter::LoadStoreUtil, "LoadStoreUtil"},
                {GpuCounter::PartialReadRatio, "PartialReadRatio"},
                {GpuCounter::PartialWriteRatio, "PartialWriteRatio"},
                {GpuCounter::GFLOPs, "GFLOPs"},
                {GpuCounter::GBPs, "GBPs"},
                {GpuCounter::L2ReadMissRatio, "L2ReadMissRatio"},
                {GpuCounter::L2WriteMissRatio, "L2WriteMissRatio"},
                {GpuCounter::FullRegWarpRatio, "FullRegWarpRatio"},
                {GpuCounter::WarpDivergenceRatio, "WarpDivergenceRatio"},
                {GpuCounter::ShaderTextureCyclesRatio,
                 "ShaderTextureCyclesRatio"},
                {GpuCounter::ShaderArithmeticCyclesRatio,
                 "ShaderArithmeticCyclesRatio"},
                {GpuCounter::ShaderLoadStoreCyclesRatio,
                 "ShaderLoadStoreCyclesRatio"},
                {GpuCounter::OtherCyclesRatio, "OtherCyclesRatio"}};

typedef std::unordered_set<GpuCounter, GpuCounterHash> GpuCounterSet;
typedef std::unordered_map<GpuCounter, Value, GpuCounterHash> GpuMeasurements;
#endif

/** An interface for classes that collect GPU performance data. */
class GpuProfiler {
public:
    virtual ~GpuProfiler() = default;

    // Sets the enabled counters after initialization
    virtual void set_enabled_counters(GpuCounterSet counters) = 0;

    // Starts a profiling session
    virtual void run() = 0;

    // Sample the counters. Returns a map of measurements for the counters
    // that are both available and enabled.
    // A profiling session must be running when sampling the counters.
    virtual const GpuMeasurements& sample() = 0;

    // Stops the active profiling session
    virtual void stop() = 0;

    virtual void set_kern_time(uint64_t kern_time) = 0;

    virtual void set_dtype_sz(uint64_t dtype_size) = 0;
};

}  // namespace mperf
