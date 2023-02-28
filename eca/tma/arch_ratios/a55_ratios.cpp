/**
 * \file eca/tma/arch_ratios/a55_ratios.cpp
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#include "arch_ratios.h"

namespace mperf {
namespace tma {
namespace {
static std::string version = "v1.0.0";
static float Pipeline_Width = 2;

static float OneMillion = 1000000;
static float OneBillion = 1000000000;

static float Cacheline_size = 64;    // unit: bytes
static float ACE_bus_bitwidth = 16;  // unit: bytes
// refer: a55 software opt guide P11
static float LS_rd_bitwidth = 8;   // unit: bytes
static float LS_wr_bitwidth = 16;  // unit: bytes

// if the major simd is fmul, but not fmla, the op_nums_per_ase_inst should set
// to 4, because the event "ASE_SPEC" cannot distinguish fmul and fmla operation
static float op_nums_per_ase_inst = 8;

float Frontend_Latency_Cycles(FEV EV, int level, bool& thresh);
float HighIPC(FEV EV, int level, bool& thresh);
float Retired_Slots(FEV EV, int level, bool& thresh);
float IPC(FEV EV, int level, bool& thresh);
float UPI(FEV EV, int level, bool& thresh);
float CPI(FEV EV, int level, bool& thresh);
float CLKS(FEV EV, int level, bool& thresh);
float SLOTS(FEV EV, int level, bool& thresh);
float CoreIPC(FEV EV, int level, bool& thresh);
float CORE_CLKS(FEV EV, int level, bool& thresh);
float IpLoad(FEV EV, int level, bool& thresh);
float IpStore(FEV EV, int level, bool& thresh);
float IpBranch(FEV EV, int level, bool& thresh);
float Instructions(FEV EV, int level, bool& thresh);
float IpMispredict(FEV EV, int level, bool& thresh);
float L1MPKI(FEV EV, int level, bool& thresh);
float L2MPKI(FEV EV, int level, bool& thresh);
float L3MPKI(FEV EV, int level, bool& thresh);
float L1_BW_Use(FEV EV, int level, bool& thresh);
float L2_BW_Use(FEV EV, int level, bool& thresh);
float L3_BW_Use(FEV EV, int level, bool& thresh);
float DRAM_BW_Use(FEV EV, int level, bool& thresh);
float GFLOPs_Use(FEV EV, int level, bool& thresh);
float Time(FEV EV, int level, bool& thresh);
float LD_Ratio(FEV EV, int level, bool& thresh);
float ST_Ratio(FEV EV, int level, bool& thresh);
float ASE_Ratio(FEV EV, int level, bool& thresh);
float VFP_Ratio(FEV EV, int level, bool& thresh);
float DP_Ratio(FEV EV, int level, bool& thresh);
float BR_IMMED_Ratio(FEV EV, int level, bool& thresh);
float BR_RETURN_Ratio(FEV EV, int level, bool& thresh);
float BR_INDIRECT_Ratio(FEV EV, int level, bool& thresh);
float L1I_Miss_Ratio(FEV EV, int level, bool& thresh);
float L1D_Miss_Ratio(FEV EV, int level, bool& thresh);
float L1D_RD_Miss_Ratio(FEV EV, int level, bool& thresh);
float L1D_WR_Miss_Ratio(FEV EV, int level, bool& thresh);
float L2D_Miss_Ratio(FEV EV, int level, bool& thresh);
float L2D_RD_Miss_Ratio(FEV EV, int level, bool& thresh);
float L2D_WR_Miss_Ratio(FEV EV, int level, bool& thresh);
float L3D_Miss_Ratio(FEV EV, int level, bool& thresh);
float L3D_RD_Miss_Ratio(FEV EV, int level, bool& thresh);
float L3D_WR_Miss_Ratio(FEV EV, int level, bool& thresh);
float BR_Mispred_Ratio(FEV EV, int level, bool& thresh);
float L1D_TLB_Miss_Ratio(FEV EV, int level, bool& thresh);
float L1I_TLB_Miss_Ratio(FEV EV, int level, bool& thresh);
float L2_TLB_Miss_Ratio(FEV EV, int level, bool& thresh);
float DTLB_Table_Walk_Ratio(FEV EV, int level, bool& thresh);
float ITLB_Table_Walk_Ratio(FEV EV, int level, bool& thresh);
float Unaligned_LDST_Ratio(FEV EV, int level, bool& thresh);
float FPU_Util(FEV EV, int level, bool& thresh);
float Load_Port_Util(FEV EV, int level, bool& thresh);
float Store_Port_Util(FEV EV, int level, bool& thresh);

class Frontend_Bound : public MetricBase {
public:
    Frontend_Bound() {
        name = "Frontend_Bound";
        domain = "Slots";
        area = "FE";
        level = 1;
        desc = "No operation issued because of the frontend.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Fetch_Latency : public MetricBase {
public:
    Fetch_Latency() {
        name = "Fetch_Latency";
        domain = "Slots";
        area = "FE";
        level = 2;
        desc = "This metric represents fraction of slots the CPU was stalled "
               "due to Frontend latency issues.  For example; instruction- "
               "cache misses; iTLB misses or predecode errors. In such "
               "cases; the Frontend eventually delivers no uops for some "
               "period.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class ICache_Misses : public MetricBase {
public:
    ICache_Misses() {
        name = "ICache_Misses";
        domain = "Clocks";
        area = "FE";
        level = 3;
        desc = "No operation issued due to the frontend, cache miss.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class ITLB_Misses : public MetricBase {
public:
    ITLB_Misses() {
        name = "ITLB_Misses";
        domain = "Clocks";
        area = "FE";
        level = 3;
        desc = "No operation issued due to the frontend, TLB miss.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Predecode_Error : public MetricBase {
public:
    Predecode_Error() {
        name = "Predecode_Error";
        domain = "Clocks";
        area = "FE";
        level = 3;
        desc = "No operation issued due to the frontend, pre-decode error.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Fetch_Bandwidth : public MetricBase {
public:
    Fetch_Bandwidth() {
        name = "Fetch_Bandwidth";
        domain = "Slots";
        area = "FE";
        level = 2;
        desc = "the value of Fetch_Bandwidth is zero on a55 core";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Bad_Speculation : public MetricBase {
public:
    Bad_Speculation() {
        name = "Bad_Speculation";
        domain = "Slots";
        area = "BAD";
        level = 1;
        desc = "This category represents fraction of slots wasted due to "
               "incorrect speculations.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Branch_Mispredicts : public MetricBase {
public:
    Branch_Mispredicts() {
        name = "Branch_Mispredicts";
        domain = "Count";
        area = "BAD";
        level = 2;
        desc = "This metric represents the number of Branch Misprediction, "
               "because don't know the average cycle penalty for single branch "
               "misprediction.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Backend_Bound : public MetricBase {
public:
    Backend_Bound() {
        name = "Backend_Bound";
        domain = "Slots";
        area = "BE";
        level = 1;
        desc = "No operation issued because of the backend.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Memory_Bound : public MetricBase {
public:
    Memory_Bound() {
        name = "Memory_Bound";
        domain = "Slots";
        area = "BE/Mem";
        level = 2;
        desc = "This metric represents fraction of slots the Memory subsystem "
               "within the Backend was a bottleneck. Memory Bound estimates "
               "fraction of slots where pipeline is likely stalled due to "
               "demand load or store instructions.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Load_Bound : public MetricBase {
public:
    Load_Bound() {
        name = "Load_Bound";
        domain = "Stalls";
        area = "BE/Mem";
        level = 3;
        desc = "No operation issued due to the backend, load.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Load_DTLB : public MetricBase {
public:
    Load_DTLB() {
        name = "Load_DTLB";
        domain = "Stalls";
        area = "BE/Mem";
        level = 4;
        desc = "No operation issued due to the backend, load, TLB miss.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Load_Cache : public MetricBase {
public:
    Load_Cache() {
        name = "Load_Cache";
        domain = "Stalls";
        area = "BE/Mem";
        level = 4;
        desc = "No operation issued due to the backend, load, cache miss.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Store_Bound : public MetricBase {
public:
    Store_Bound() {
        name = "Store_Bound";
        domain = "Stalls";
        area = "BE/Mem";
        level = 3;
        desc = "No operation issued due to the backend, store.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Store_TLB : public MetricBase {
public:
    Store_TLB() {
        name = "Store_TLB";
        domain = "Stalls";
        area = "BE/Mem";
        level = 4;
        desc = "No operation issued due to the backend, store, TLB miss.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Store_Buffer : public MetricBase {
public:
    Store_Buffer() {
        name = "Store_Buffer";
        domain = "Stalls";
        area = "BE/Mem";
        level = 4;
        desc = "No operation issued due to the backend, store, STB full.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Core_Bound : public MetricBase {
public:
    Core_Bound() {
        name = "Core_Bound";
        domain = "Slots";
        area = "BE/Core";
        level = 2;
        desc = "This metric represents fraction of slots where Core non- "
               "memory issues were of a bottleneck.  Shortage in hardware "
               "compute resources; or dependencies in software's instructions "
               "are both categorized under Core Bound.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Interlock_Bound : public MetricBase {
public:
    Interlock_Bound() {
        name = "Interlock_Bound";
        domain = "Stalls";
        area = "BE/Core";
        level = 3;
        desc = "No operation issued due to the backend interlock.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Interlock_AGU : public MetricBase {
public:
    Interlock_AGU() {
        name = "Interlock_AGU";
        domain = "Stalls";
        area = "BE/Core";
        level = 4;
        desc = "No operation issued due to the backend, interlock, AGU(address "
               "generate unit).";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Interlock_FPU : public MetricBase {
public:
    Interlock_FPU() {
        name = "Interlock_FPU";
        domain = "Stalls";
        area = "BE/Core";
        level = 4;
        desc = "No operation issued due to the backend, interlock, FPU(float "
               "point unit).";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Core_Bound_Others : public MetricBase {
public:
    Core_Bound_Others() {
        name = "Core_Bound_Others";
        domain = "Stalls";
        area = "BE/Core";
        level = 4;
        desc = "the rest undetectable core bound stalls.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Retiring : public MetricBase {
public:
    Retiring() {
        name = "Retiring";
        domain = "Slots";
        area = "RET";
        level = 1;
        desc = "Instruction architecturally executed, called retired. Note "
               "that a "
               "high Retiring value does not necessary "
               "mean there is no room for more performance. For example, simd "
               "optimize for scalar implement.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class LD_Retiring : public MetricBase {
public:
    LD_Retiring() {
        name = "LD_Retiring";
        domain = "Slots";
        area = "RET";
        level = 2;
        desc = "Operation speculatively executed, load.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class ST_Retiring : public MetricBase {
public:
    ST_Retiring() {
        name = "ST_Retiring";
        domain = "Slots";
        area = "RET";
        level = 2;
        desc = "Operation speculatively executed, store.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class DP_Retiring : public MetricBase {
public:
    DP_Retiring() {
        name = "DP_Retiring";
        domain = "Slots";
        area = "RET";
        level = 2;
        desc = "Operation speculatively executed, integer data processing.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class ASE_Retiring : public MetricBase {
public:
    ASE_Retiring() {
        name = "ASE_Retiring";
        domain = "Slots";
        area = "RET";
        level = 2;
        desc = "Operation speculatively executed, advanced SIMD instruction.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class VFP_Retiring : public MetricBase {
public:
    VFP_Retiring() {
        name = "VFP_Retiring";
        domain = "Slots";
        area = "RET";
        level = 2;
        desc = "Operation speculatively executed, floating-point instruction.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class PC_Write_Retiring : public MetricBase {
public:
    PC_Write_Retiring() {
        name = "PC_Write_Retiring";
        domain = "Slots";
        area = "RET";
        level = 2;
        desc = "Operation speculatively executed, software change of the PC. "
               "This event counts retired branch instructions.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class BR_IMMED_Retiring : public MetricBase {
public:
    BR_IMMED_Retiring() {
        name = "BR_IMMED_Retiring";
        domain = "Slots";
        area = "RET";
        level = 3;
        desc = "Instruction architecturally executed, immediate branch. This "
               "event counts all branches decoded as immediate branches, taken "
               "or not, and popped from the branch monitor. This excludes "
               "exception entries, debug entries, and CCFAIL branches.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class BR_RETURN_Retiring : public MetricBase {
public:
    BR_RETURN_Retiring() {
        name = "BR_RETURN_Retiring";
        domain = "Slots";
        area = "RET";
        level = 3;
        desc = "Instruction architecturally executed, condition code check "
               "pass, procedure return.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class BR_INDIRECT_Retiring : public MetricBase {
public:
    BR_INDIRECT_Retiring() {
        name = "BR_INDIRECT_Retiring";
        domain = "Slots";
        area = "RET";
        level = 3;
        desc = "Branch speculatively executed, indirect branch.This event "
               "counts retired indirect branch instructions.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_IPC : public MetricBase {
public:
    Metric_IPC() {
        name = "Metric_IPC";
        domain = "Metric";
        area = "Info.Thread";
        desc = "Instructions Per Cycle (per Logical Processor)";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_UPI : public MetricBase {
public:
    Metric_UPI() {
        name = "Metric_UPI";
        domain = "Metric";
        area = "Info.Thread";
        desc = "Uops Per Instruction";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_CPI : public MetricBase {
public:
    Metric_CPI() {
        name = "Metric_CPI";
        domain = "Metric";
        area = "Info.Thread";
        desc = "Cycles Per Instruction (per Logical Processor)";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_CLKS : public MetricBase {
public:
    Metric_CLKS() {
        name = "Metric_CLKS";
        domain = "Count";
        area = "Info.Thread";
        desc = "Per-Logical Processor actual clocks when the Logical Processor "
               "is active.";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_SLOTS : public MetricBase {
public:
    Metric_SLOTS() {
        name = "Metric_SLOTS";
        domain = "Count";
        area = "Info.Thread";
        desc = "Total issue-pipeline slots (per-Physical Core till ICL; per- "
               "Logical Processor ICL onward)";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_CoreIPC : public MetricBase {
public:
    Metric_CoreIPC() {
        name = "Metric_CoreIPC";
        domain = "Core_Metric";
        area = "Info.Core";
        desc = "Instructions Per Cycle across hyper-threads (per physical "
               "core)";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_CORE_CLKS : public MetricBase {
public:
    Metric_CORE_CLKS() {
        name = "Metric_CORE_CLKS";
        domain = "Count";
        area = "Info.Core";
        desc = "Core actual clocks when any Logical Processor is active on the "
               "Physical Core";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_IpLoad : public MetricBase {
public:
    Metric_IpLoad() {
        name = "Metric_IpLoad";
        domain = "Inst_Metric";
        area = "Info.Inst_Mix";
        desc = "Instructions per Load (lower number means higher occurrence "
               "rate)";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_IpStore : public MetricBase {
public:
    Metric_IpStore() {
        name = "Metric_IpStore";
        domain = "Inst_Metric";
        area = "Info.Inst_Mix";
        desc = "Instructions per Store (lower number means higher occurrence "
               "rate)";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_IpBranch : public MetricBase {
public:
    Metric_IpBranch() {
        name = "Metric_IpBranch";
        domain = "Inst_Metric";
        area = "Info.Inst_Mix";
        desc = "Instructions per Branch (lower number means higher occurrence "
               "rate)";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_Instructions : public MetricBase {
public:
    Metric_Instructions() {
        name = "Metric_Instructions";
        domain = "Count";
        area = "Info.Inst_Mix";
        desc = "Total number of retired Instructions";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_IpMispredict : public MetricBase {
public:
    Metric_IpMispredict() {
        name = "Metric_IpMispredict";
        domain = "Inst_Metric";
        area = "Info.Bad_Spec";
        desc = "Number of Instructions per non-speculative Branch "
               "Misprediction (JEClear) (lower number means higher occurrence "
               "rate)";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L1MPKI : public MetricBase {
public:
    Metric_L1MPKI() {
        name = "Metric_L1MPKI";
        domain = "Metric";
        area = "Info.Memory";
        desc = "L1 cache true misses per kilo instruction for retired demand "
               "loads";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L2MPKI : public MetricBase {
public:
    Metric_L2MPKI() {
        name = "Metric_L2MPKI";
        domain = "Metric";
        area = "Info.Memory";
        desc = "L2 cache true misses per kilo instruction for retired demand "
               "loads";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L3MPKI : public MetricBase {
public:
    Metric_L3MPKI() {
        name = "Metric_L3MPKI";
        domain = "Metric";
        area = "Info.Memory";
        desc = "L3 cache true misses per kilo instruction for retired demand "
               "loads";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L1_BW_Use : public MetricBase {
public:
    Metric_L1_BW_Use() {
        name = "Metric_L1_BW_Use";
        domain = "Metric";
        area = "Info.System";
        desc = "L1 Bandwidth Use for reads and writes [GB "
               "/ sec]";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L2_BW_Use : public MetricBase {
public:
    Metric_L2_BW_Use() {
        name = "Metric_L2_BW_Use";
        domain = "Metric";
        area = "Info.System";
        desc = "L2 Bandwidth Use for reads and writes [GB "
               "/ sec]";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L3_BW_Use : public MetricBase {
public:
    Metric_L3_BW_Use() {
        name = "Metric_L3_BW_Use";
        domain = "Metric";
        area = "Info.System";
        desc = "L3 Bandwidth Use for reads and writes [GB "
               "/ sec]";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
// Note: the compute interface for Metric_DRAM_BW_Use is dependent on the soc
// support dsu pmu, we find dsu pmu on some mtk soc
class Metric_DRAM_BW_Use : public MetricBase {
public:
    Metric_DRAM_BW_Use() {
        name = "Metric_DRAM_BW_Use";
        domain = "GB/sec";
        area = "Info.System";
        desc = "Average external Memory Bandwidth Use for reads and writes [GB "
               "/ sec]";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_GFLOPs_Use : public MetricBase {
public:
    Metric_GFLOPs_Use() {
        name = "Metric_GFLOPs_Use";
        domain = "GFlop/sec";
        area = "Info.System";
        desc = "...";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_LD_Ratio : public MetricBase {
public:
    Metric_LD_Ratio() {
        name = "Metric_LD_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "load instruction ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_ST_Ratio : public MetricBase {
public:
    Metric_ST_Ratio() {
        name = "Metric_ST_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "store instruction ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_ASE_Ratio : public MetricBase {
public:
    Metric_ASE_Ratio() {
        name = "Metric_ASE_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "advanced SIMD instruction ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_VFP_Ratio : public MetricBase {
public:
    Metric_VFP_Ratio() {
        name = "Metric_VFP_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "floating point instruction ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_DP_Ratio : public MetricBase {
public:
    Metric_DP_Ratio() {
        name = "Metric_DP_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "integer data processing instruction ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_BR_IMMED_Ratio : public MetricBase {
public:
    Metric_BR_IMMED_Ratio() {
        name = "Metric_BR_IMMED_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "immediate branch instruction ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_BR_RETURN_Ratio : public MetricBase {
public:
    Metric_BR_RETURN_Ratio() {
        name = "Metric_BR_RETURN_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "return branch instruction ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_BR_INDIRECT_Ratio : public MetricBase {
public:
    Metric_BR_INDIRECT_Ratio() {
        name = "Metric_BR_INDIRECT_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "indirect branch instruction ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L1I_Miss_Ratio : public MetricBase {
public:
    Metric_L1I_Miss_Ratio() {
        name = "Metric_L1I_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L1I miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L1D_Miss_Ratio : public MetricBase {
public:
    Metric_L1D_Miss_Ratio() {
        name = "Metric_L1D_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L1D miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L1D_RD_Miss_Ratio : public MetricBase {
public:
    Metric_L1D_RD_Miss_Ratio() {
        name = "Metric_L1D_RD_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L1D read miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L1D_WR_Miss_Ratio : public MetricBase {
public:
    Metric_L1D_WR_Miss_Ratio() {
        name = "Metric_L1D_WR_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L1D write miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L2D_Miss_Ratio : public MetricBase {
public:
    Metric_L2D_Miss_Ratio() {
        name = "Metric_L2D_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L2D miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L2D_RD_Miss_Ratio : public MetricBase {
public:
    Metric_L2D_RD_Miss_Ratio() {
        name = "Metric_L2D_RD_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L2D read miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L2D_WR_Miss_Ratio : public MetricBase {
public:
    Metric_L2D_WR_Miss_Ratio() {
        name = "Metric_L2D_WR_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L2D write miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L3D_Miss_Ratio : public MetricBase {
public:
    Metric_L3D_Miss_Ratio() {
        name = "Metric_L3D_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L3D miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L3D_RD_Miss_Ratio : public MetricBase {
public:
    Metric_L3D_RD_Miss_Ratio() {
        name = "Metric_L3D_RD_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L3D read miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L3D_WR_Miss_Ratio : public MetricBase {
public:
    Metric_L3D_WR_Miss_Ratio() {
        name = "Metric_L3D_WR_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L3D write miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_BR_Mispred_Ratio : public MetricBase {
public:
    Metric_BR_Mispred_Ratio() {
        name = "Metric_BR_Mispred_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "branch misprediction ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L1I_TLB_Miss_Ratio : public MetricBase {
public:
    Metric_L1I_TLB_Miss_Ratio() {
        name = "Metric_L1I_TLB_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L1I TLB miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L1D_TLB_Miss_Ratio : public MetricBase {
public:
    Metric_L1D_TLB_Miss_Ratio() {
        name = "Metric_L1D_TLB_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L1D TLB miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_L2_TLB_Miss_Ratio : public MetricBase {
public:
    Metric_L2_TLB_Miss_Ratio() {
        name = "Metric_L2_TLB_Miss_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "L2 TLB miss ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_ITLB_Table_Walk_Ratio : public MetricBase {
public:
    Metric_ITLB_Table_Walk_Ratio() {
        name = "Metric_ITLB_Table_Walk_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "I-side page table walk ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_DTLB_Table_Walk_Ratio : public MetricBase {
public:
    Metric_DTLB_Table_Walk_Ratio() {
        name = "Metric_DTLB_Table_Walk_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "D-side page table walk ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_Unaligned_LDST_Ratio : public MetricBase {
public:
    Metric_Unaligned_LDST_Ratio() {
        name = "Metric_Unaligned_LDST_Ratio";
        domain = "...";
        area = "Info.System";
        desc = "unaligned load and store ratio";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_FPU_Util : public MetricBase {
public:
    Metric_FPU_Util() {
        name = "Metric_FPU_Util";
        domain = "...";
        area = "Info.System";
        desc = "Metric_FPU_Util";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_Load_Port_Util : public MetricBase {
public:
    Metric_Load_Port_Util() {
        name = "Metric_Load_Port_Util";
        domain = "...";
        area = "Info.System";
        desc = "Metric_Load_Port_Util";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};
class Metric_Store_Port_Util : public MetricBase {
public:
    Metric_Store_Port_Util() {
        name = "Metric_Store_Port_Util";
        domain = "...";
        area = "Info.System";
        desc = "Metric_Store_Port_Util";
        parent = nullptr;
        func_compute = (void*)&compute;
    }
    static float compute(FEV EV);
};

float Frontend_Bound::compute(FEV EV) {
    val = Pipeline_Width * EV({"STALL_FRONTEND", 0x23, 4, 0}, 1) /
          SLOTS(EV, 1, thresh);
    thresh = (val > 0.15);
    return val;
}
float Fetch_Latency::compute(FEV EV) {
    val = Pipeline_Width * Frontend_Latency_Cycles(EV, 2, thresh) /
          SLOTS(EV, 2, thresh);
    thresh = (val > 0.10) && Frontend_Bound::thresh;
    return val;
}
float ICache_Misses::compute(FEV EV) {
    val = EV({"STALL_FRONTEND_CACHE", 0xE1, 4, 0}, 3) / CLKS(EV, 3, thresh);
    thresh = (val > 0.05) && Fetch_Latency::thresh;
    return val;
}
float ITLB_Misses::compute(FEV EV) {
    val = EV({"STALL_FRONTEND_TLB", 0xE2, 4, 0}, 3) / CLKS(EV, 3, thresh);
    thresh = (val > 0.05) && Fetch_Latency::thresh;
    return val;
}
float Predecode_Error::compute(FEV EV) {
    val = EV({"STALL_FRONTEND_PDERR", 0xE3, 4, 0}, 3) / CLKS(EV, 3, thresh);
    thresh = (val > 0.05) && Fetch_Latency::thresh;
    return val;
}
float Fetch_Bandwidth::compute(FEV EV) {
    val = Frontend_Bound::compute(EV) - Fetch_Latency::compute(EV);
    thresh = (val > 0.1) && Frontend_Bound::thresh && HighIPC(EV, 2, thresh);
    return val;
}
float Bad_Speculation::compute(FEV EV) {
    val = 1 - (Frontend_Bound::compute(EV) + Backend_Bound::compute(EV) +
               Retiring::compute(EV));
    thresh = (val > 0.15);
    return val;
}
float Branch_Mispredicts::compute(FEV EV) {
    val = EV({"BR_MIS_PRED_RETIRED", 0x22, 4, 0}, 2);
    thresh = (val > 0.1) && Bad_Speculation::thresh;
    return val;
}
float Backend_Bound::compute(FEV EV) {
    val = Pipeline_Width * EV({"STALL_BACKEND", 0x24, 4, 0}, 1) /
          SLOTS(EV, 1, thresh);
    thresh = (val > 0.2);
    return val;
}
float Memory_Bound::compute(FEV EV) {
    val = (EV({"STALL_BACKEND_LD", 0xE7, 4, 0}, 2) +
           EV({"STALL_BACKEND_ST", 0xE8, 4, 0}, 2)) /
          CLKS(EV, 2, thresh);
    thresh = (val > 0.2) && Backend_Bound::thresh;
    return val;
}
float Load_Bound::compute(FEV EV) {
    val = EV({"STALL_BACKEND_LD", 0xE7, 4, 0}, 3) / CLKS(EV, 3, thresh);
    thresh = (val > 0.1) && Memory_Bound::thresh;
    return val;
}
float Load_DTLB::compute(FEV EV) {
    val = EV({"STALL_BACKEND_LD_TLB", 0xEA, 4, 0}, 4) / CLKS(EV, 4, thresh);
    thresh = (val > 0.1) && Load_Bound::thresh;
    return val;
}
float Load_Cache::compute(FEV EV) {
    val = EV({"STALL_BACKEND_LD_CACHE", 0xE9, 4, 0}, 4) / CLKS(EV, 4, thresh);
    thresh = (val > 0.1) && Load_Bound::thresh;
    return val;
}
float Store_Bound::compute(FEV EV) {
    val = EV({"STALL_BACKEND_ST", 0xE8, 4, 0}, 3) / CLKS(EV, 3, thresh);
    thresh = (val > 0.1) && Memory_Bound::thresh;
    return val;
}
float Store_TLB::compute(FEV EV) {
    val = EV({"STALL_BACKEND_ST_TLB", 0xEC, 4, 0}, 4) / CLKS(EV, 4, thresh);
    thresh = (val > 0.1) && Store_Bound::thresh;
    return val;
}
float Store_Buffer::compute(FEV EV) {
    val = EV({"STALL_BACKEND_ST_STB", 0xEB, 4, 0}, 4) / CLKS(EV, 4, thresh);
    thresh = (val > 0.1) && Store_Bound::thresh;
    return val;
}
float Core_Bound::compute(FEV EV) {
    val = Backend_Bound::compute(EV) - Memory_Bound::compute(EV);
    thresh = (val > 0.1) && Backend_Bound::thresh;
    return val;
}
float Interlock_Bound::compute(FEV EV) {
    val = EV({"STALL_BACKEND_ILOCK", 0xE4, 4, 0}, 3) / CLKS(EV, 3, thresh);
    thresh = (val > 0.1) && Memory_Bound::thresh;
    return val;
}
float Interlock_AGU::compute(FEV EV) {
    val = EV({"STALL_BACKEND_ILOCK_AGU", 0xE5, 4, 0}, 4) / CLKS(EV, 4, thresh);
    thresh = (val > 0.1) && Memory_Bound::thresh;
    return val;
}
float Interlock_FPU::compute(FEV EV) {
    val = EV({"STALL_BACKEND_ILOCK_FPU", 0xE6, 4, 0}, 4) / CLKS(EV, 4, thresh);
    thresh = (val > 0.1) && Memory_Bound::thresh;
    return val;
}
float Core_Bound_Others::compute(FEV EV) {
    val = Core_Bound::compute(EV) - Interlock_Bound::compute(EV);
    thresh = (val > 0.1) && Memory_Bound::thresh;
    return val;
}
float Retiring::compute(FEV EV) {
    val = Retired_Slots(EV, 1, thresh) / SLOTS(EV, 1, thresh);
    thresh = true;
    return val;
}
float LD_Retiring::compute(FEV EV) {
    val = EV({"LD_SPEC", 0x70, 4, 0}, 2) / SLOTS(EV, 1, thresh);
    return val;
}
float ST_Retiring::compute(FEV EV) {
    val = EV({"ST_SPEC", 0x71, 4, 0}, 2) / SLOTS(EV, 1, thresh);
    return val;
}
float DP_Retiring::compute(FEV EV) {
    val = EV({"DP_SPEC", 0x73, 4, 0}, 2) / SLOTS(EV, 1, thresh);
    return val;
}
float ASE_Retiring::compute(FEV EV) {
    val = EV({"ASE_SPEC", 0x74, 4, 0}, 2) / SLOTS(EV, 1, thresh);
    return val;
}
float VFP_Retiring::compute(FEV EV) {
    val = EV({"VFP_SPEC", 0x75, 4, 0}, 2) / SLOTS(EV, 1, thresh);
    return val;
}
float PC_Write_Retiring::compute(FEV EV) {
    val = EV({"PC_WRITE_SPEC", 0x76, 4, 0}, 2) / SLOTS(EV, 1, thresh);
    return val;
}
float BR_IMMED_Retiring::compute(FEV EV) {
    val = EV({"BR_IMMED_SPEC", 0x78, 4, 0}, 3) / SLOTS(EV, 1, thresh);
    return val;
}
float BR_RETURN_Retiring::compute(FEV EV) {
    val = EV({"BR_RETURN_SPEC", 0x79, 4, 0}, 3) / SLOTS(EV, 1, thresh);
    return val;
}
float BR_INDIRECT_Retiring::compute(FEV EV) {
    val = EV({"BR_INDIRECT_SPEC", 0x7a, 4, 0}, 3) / SLOTS(EV, 1, thresh);
    return val;
}
float Metric_IPC::compute(FEV EV) {
    val = IPC(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_UPI::compute(FEV EV) {
    val = UPI(EV, 0, thresh);
    thresh = (val > 1.05);
    return val;
}
float Metric_CPI::compute(FEV EV) {
    val = CPI(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_CLKS::compute(FEV EV) {
    val = CLKS(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_SLOTS::compute(FEV EV) {
    val = SLOTS(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_CoreIPC::compute(FEV EV) {
    val = CoreIPC(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_CORE_CLKS::compute(FEV EV) {
    val = CORE_CLKS(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_IpLoad::compute(FEV EV) {
    val = IpLoad(EV, 0, thresh);
    thresh = (val < 3);
    return val;
}
float Metric_IpStore::compute(FEV EV) {
    val = IpStore(EV, 0, thresh);
    thresh = (val < 8);
    return val;
}
float Metric_IpBranch::compute(FEV EV) {
    val = IpBranch(EV, 0, thresh);
    thresh = (val < 8);
    return val;
}
float Metric_Instructions::compute(FEV EV) {
    val = Instructions(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_IpMispredict::compute(FEV EV) {
    val = IpMispredict(EV, 0, thresh);
    thresh = (val < 200);
    return val;
}
float Metric_L1MPKI::compute(FEV EV) {
    val = L1MPKI(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L2MPKI::compute(FEV EV) {
    val = L2MPKI(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L3MPKI::compute(FEV EV) {
    val = L3MPKI(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L1_BW_Use::compute(FEV EV) {
    val = L1_BW_Use(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L2_BW_Use::compute(FEV EV) {
    val = L2_BW_Use(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L3_BW_Use::compute(FEV EV) {
    val = L3_BW_Use(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_DRAM_BW_Use::compute(FEV EV) {
    val = DRAM_BW_Use(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_GFLOPs_Use::compute(FEV EV) {
    val = GFLOPs_Use(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_LD_Ratio::compute(FEV EV) {
    val = LD_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_ST_Ratio::compute(FEV EV) {
    val = ST_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_ASE_Ratio::compute(FEV EV) {
    val = ASE_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_VFP_Ratio::compute(FEV EV) {
    val = VFP_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_DP_Ratio::compute(FEV EV) {
    val = DP_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_BR_IMMED_Ratio::compute(FEV EV) {
    val = BR_IMMED_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_BR_RETURN_Ratio::compute(FEV EV) {
    val = BR_RETURN_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_BR_INDIRECT_Ratio::compute(FEV EV) {
    val = BR_INDIRECT_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L1I_Miss_Ratio::compute(FEV EV) {
    val = L1I_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L1D_Miss_Ratio::compute(FEV EV) {
    val = L1D_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L1D_RD_Miss_Ratio::compute(FEV EV) {
    val = L1D_RD_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L1D_WR_Miss_Ratio::compute(FEV EV) {
    val = L1D_WR_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L2D_Miss_Ratio::compute(FEV EV) {
    val = L2D_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L2D_RD_Miss_Ratio::compute(FEV EV) {
    val = L2D_RD_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L2D_WR_Miss_Ratio::compute(FEV EV) {
    val = L2D_WR_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L3D_Miss_Ratio::compute(FEV EV) {
    val = L3D_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L3D_RD_Miss_Ratio::compute(FEV EV) {
    val = L3D_RD_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L3D_WR_Miss_Ratio::compute(FEV EV) {
    val = L3D_WR_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_BR_Mispred_Ratio::compute(FEV EV) {
    val = BR_Mispred_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L1I_TLB_Miss_Ratio::compute(FEV EV) {
    val = L1I_TLB_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L1D_TLB_Miss_Ratio::compute(FEV EV) {
    val = L1D_TLB_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_L2_TLB_Miss_Ratio::compute(FEV EV) {
    val = L2_TLB_Miss_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_ITLB_Table_Walk_Ratio::compute(FEV EV) {
    val = ITLB_Table_Walk_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_DTLB_Table_Walk_Ratio::compute(FEV EV) {
    val = DTLB_Table_Walk_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_Unaligned_LDST_Ratio::compute(FEV EV) {
    val = Unaligned_LDST_Ratio(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_FPU_Util::compute(FEV EV) {
    val = FPU_Util(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_Load_Port_Util::compute(FEV EV) {
    val = Load_Port_Util(EV, 0, thresh);
    thresh = true;
    return val;
}
float Metric_Store_Port_Util::compute(FEV EV) {
    val = Store_Port_Util(EV, 0, thresh);
    thresh = true;
    return val;
}
float Frontend_Latency_Cycles(FEV EV, int level, bool& thresh) {
    auto func = [&]() -> float {
        return std::min<float>(EV({"CPU_CYCLES", 0x11, 4, 0}, level),
                               EV({"STALL_FRONTEND", 0x23, 4, 0}, level));
    };
    return g_ev_process(func, level);
}
float HighIPC(FEV EV, int level, bool& thresh) {
    float val = IPC(EV, level, thresh) / Pipeline_Width;
    return val;
}
float Retired_Slots(FEV EV, int level, bool& thresh) {
    return EV({"INST_RETIRED", 0x08, 4, 0}, level);
}
float IPC(FEV EV, int level, bool& thresh) {
    return EV({"INST_RETIRED", 0x08, 4, 0}, level) / CLKS(EV, level, thresh);
}
float UPI(FEV EV, int level, bool& thresh) {
    float val = Retired_Slots(EV, level, thresh) /
                EV({"INST_RETIRED", 0x08, 4, 0}, level);
    thresh = (val > 1.05);
    return val;
}
float CPI(FEV EV, int level, bool& thresh) {
    return 1 / IPC(EV, level, thresh);
}
float CLKS(FEV EV, int level, bool& thresh) {
    return EV({"CPU_CYCLES", 0x11, 4, 0}, level);
}
float SLOTS(FEV EV, int level, bool& thresh) {
    return Pipeline_Width * CORE_CLKS(EV, level, thresh);
}
float CoreIPC(FEV EV, int level, bool& thresh) {
    return EV({"INST_RETIRED", 0x08, 4, 0}, level) /
           CORE_CLKS(EV, level, thresh);
}
float CORE_CLKS(FEV EV, int level, bool& thresh) {
    return CLKS(EV, level, thresh);
}
float IpLoad(FEV EV, int level, bool& thresh) {
    float val = EV({"INST_RETIRED", 0x08, 4, 0}, level) /
                EV({"LD_RETIRED", 0x06, 4, 0}, level);
    thresh = (val < 3);
    return val;
}
float IpStore(FEV EV, int level, bool& thresh) {
    float val = EV({"INST_RETIRED", 0x08, 4, 0}, level) /
                EV({"ST_RETIRED", 0x07, 4, 0}, level);
    thresh = (val < 8);
    return val;
}
float IpBranch(FEV EV, int level, bool& thresh) {
    float val = EV({"INST_RETIRED", 0x08, 4, 0}, level) /
                EV({"BR_RETIRED", 0x21, 4, 0}, level);
    thresh = (val < 8);
    return val;
}
float Instructions(FEV EV, int level, bool& thresh) {
    return EV({"INST_RETIRED", 0x08, 4, 0}, level);
}
float IpMispredict(FEV EV, int level, bool& thresh) {
    float val = EV({"INST_RETIRED", 0x08, 4, 0}, level) /
                EV({"BR_MIS_PRED_RETIRED", 0x22, 4, 0}, level);
    thresh = (val < 200);
    return val;
}
float L1MPKI(FEV EV, int level, bool& thresh) {
    return 1000 * EV({"L1D_CACHE_REFILL", 0x03, 4, 0}, level) /
           EV({"INST_RETIRED", 0x08, 4, 0}, level);
}
float L2MPKI(FEV EV, int level, bool& thresh) {
    return 1000 * EV({"L2D_CACHE_REFILL", 0x17, 4, 0}, level) /
           EV({"INST_RETIRED", 0x08, 4, 0}, level);
}
float L3MPKI(FEV EV, int level, bool& thresh) {
    return 1000 * EV({"L3D_CACHE_REFILL", 0x2A, 4, 0}, level) /
           EV({"INST_RETIRED", 0x08, 4, 0}, level);
}
// L1_BW_Use is not precisely, because don't know the accurate granularity for
// L1D access
float L1_BW_Use(FEV EV, int level, bool& thresh) {
    return (LS_rd_bitwidth * EV({"MEM_ACCESS_RD", 0x66, 4, 0}, level) +
            LS_wr_bitwidth * EV({"MEM_ACCESS_WR", 0x67, 4, 0}, level)) /
           OneBillion / Time(EV, level, thresh);
}
float L2_BW_Use(FEV EV, int level, bool& thresh) {
    return (Cacheline_size *
            (EV({"L1D_CACHE_REFILL", 0x03, 4, 0}, level) +
             EV({"L1D_CACHE_WB", 0x15, 4, 0}, level) +
             EV({"L1D_CACHE_REFILL_PREFETCH", 0xC2, 4, 0}, level))) /
           OneBillion / Time(EV, level, thresh);
}
// Note: L2D_CACHE_REFILL_PREFETCH on a55 is not count when the soc is
// configured with per-core L2 cache
float L3_BW_Use(FEV EV, int level, bool& thresh) {
    return (Cacheline_size * (EV({"L2D_CACHE_REFILL", 0x17, 4, 0}, level) +
                              EV({"L2D_CACHE_WB", 0x18, 4, 0}, level))) /
           OneBillion / Time(EV, level, thresh);
}
#if 0  // another way to measure ddr bandwidth
float DRAM_BW_Use(FEV EV, int level, bool& thresh) {
    return (ACE_bus_bitwidth *
            EV({"DSU_BUS_ACCESS", 0x19, 7, 0, true}, level)) /
           OneBillion / Time(EV, level, thresh);
}
#endif
float DRAM_BW_Use(FEV EV, int level, bool& thresh) {
    return (Cacheline_size *
            (EV({"DSU_L3D_CACHE_REFILL", 0x002A, 7, 0, true}, level) +
             EV({"DSU_L3D_CACHE_WB", 0x002C, 7, 0, true}, level))) /
           OneBillion / Time(EV, level, thresh);
}
float GFLOPs_Use(FEV EV, int level, bool& thresh) {
    return (op_nums_per_ase_inst * EV({"ASE_SPEC", 0x74, 4, 0}, level) +
            EV({"VFP_SPEC", 0x75, 4, 0}, level)) /
           OneBillion / Time(EV, level, thresh);
}
float LD_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"LD_SPEC", 0x70, 4, 0}, level) /
           EV({"INST_SPEC", 0x1b, 4, 0}, level);
}
float ST_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"ST_SPEC", 0x71, 4, 0}, level) /
           EV({"INST_SPEC", 0x1b, 4, 0}, level);
}
float ASE_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"ASE_SPEC", 0x74, 4, 0}, level) /
           EV({"INST_SPEC", 0x1b, 4, 0}, level);
}
float VFP_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"VFP_SPEC", 0x75, 4, 0}, level) /
           EV({"INST_SPEC", 0x1b, 4, 0}, level);
}
float DP_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"DP_SPEC", 0x73, 4, 0}, level) /
           EV({"INST_SPEC", 0x1b, 4, 0}, level);
}
float BR_IMMED_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"BR_IMMED_SPEC", 0x78, 4, 0}, level) /
           EV({"INST_SPEC", 0x1b, 4, 0}, level);
}
float BR_RETURN_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"BR_RETURN_SPEC", 0x79, 4, 0}, level) /
           EV({"INST_SPEC", 0x1b, 4, 0}, level);
}
float BR_INDIRECT_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"BR_INDIRECT_SPEC", 0x7a, 4, 0}, level) /
           EV({"INST_SPEC", 0x1b, 4, 0}, level);
}
float L1I_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L1I_CACHE_REFILL", 0x01, 4, 0}, level) /
           EV({"L1I_CACHE", 0x14, 4, 0}, level);
}
float L1D_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L1D_CACHE_REFILL", 0x03, 4, 0}, level) /
           EV({"L1D_CACHE", 0x04, 4, 0}, level);
}
float L1D_RD_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L1D_CACHE_REFILL_RD", 0x42, 4, 0}, level) /
           EV({"L1D_CACHE_RD", 0x40, 4, 0}, level);
}
float L1D_WR_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L1D_CACHE_REFILL_WR", 0x43, 4, 0}, level) /
           EV({"L1D_CACHE_WR", 0x41, 4, 0}, level);
}
float L2D_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L2D_CACHE_REFILL", 0x17, 4, 0}, level) /
           EV({"L2D_CACHE", 0x16, 4, 0}, level);
}
float L2D_RD_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L2D_CACHE_REFILL_RD", 0x52, 4, 0}, level) /
           EV({"L2D_CACHE_RD", 0x50, 4, 0}, level);
}
float L2D_WR_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L2D_CACHE_REFILL_WR", 0x53, 4, 0}, level) /
           EV({"L1D_CACHE_WR", 0x51, 4, 0}, level);
}
float L3D_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L3D_CACHE_REFILL", 0x2a, 4, 0}, level) /
           EV({"L3D_CACHE", 0x2b, 4, 0}, level);
}
float L3D_RD_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L3D_CACHE_REFILL_RD", 0xa2, 4, 0}, level) /
           EV({"L3D_CACHE_RD", 0xa0, 4, 0}, level);
}
float L3D_WR_Miss_Ratio(FEV EV, int level, bool& thresh) {
    mperf_log_warn(
            "L3D_CACHE_REFILL_WR and L3D_CACHE_WR are not defined in a55 pmu.");
    return 0;
}
float BR_Mispred_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"BR_MIS_PRED_RETIRED", 0x22, 4, 0}, level) /
           EV({"BR_RETIRED", 0x21, 4, 0}, level);
}
float L1D_TLB_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L1D_TLB_REFILL", 0x05, 4, 0}, level) /
           EV({"L1D_TLB", 0x25, 4, 0}, level);
}
float L1I_TLB_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L1I_TLB_REFILL", 0x02, 4, 0}, level) /
           EV({"L1I_TLB", 0x26, 4, 0}, level);
}
float L2_TLB_Miss_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"L2D_TLB_REFILL", 0x2d, 4, 0}, level) /
           EV({"L2D_TLB", 0x2f, 4, 0}, level);
}
float DTLB_Table_Walk_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"DTLB_WALK", 0x34, 4, 0}, level) /
           EV({"L1D_TLB", 0x25, 4, 0}, level);
}
float ITLB_Table_Walk_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"ITLB_WALK", 0x35, 4, 0}, level) /
           EV({"L1I_TLB", 0x26, 4, 0}, level);
}
float Unaligned_LDST_Ratio(FEV EV, int level, bool& thresh) {
    return EV({"UNALIGNED_LDST_RETIRED", 0x0f, 4, 0}, level) /
           EV({"LDST_SPEC", 0x72, 4, 0}, level);
}
float FPU_Util(FEV EV, int level, bool& thresh) {
    return EV({"ASE_SPEC", 0x74, 4, 0}, level) /
           EV({"CPU_CYCLES", 0x11, 4, 0}, level);
}
float Load_Port_Util(FEV EV, int level, bool& thresh) {
    return EV({"LD_SPEC", 0x70, 4, 0}, level) /
           EV({"CPU_CYCLES", 0x11, 4, 0}, level);
}
float Store_Port_Util(FEV EV, int level, bool& thresh) {
    return EV({"ST_SPEC", 0x71, 4, 0}, level) /
           EV({"CPU_CYCLES", 0x11, 4, 0}, level);
}
float Time(FEV EV, int level, bool& thresh) {
    return EV({"time_interval", 0x00, 0, 0}, level) * 1e-3;
}
}  // namespace

A55SetUpImpl::A55SetUpImpl() {
    m_vmtc_core.push_back(std::make_pair("Frontend_Bound",
                                         (MetricBase*)(new Frontend_Bound())));
    m_vmtc_core.push_back(std::make_pair("Fetch_Latency",
                                         (MetricBase*)(new Fetch_Latency())));
    m_vmtc_core.push_back(std::make_pair("ICache_Misses",
                                         (MetricBase*)(new ICache_Misses())));
    m_vmtc_core.push_back(
            std::make_pair("ITLB_Misses", (MetricBase*)(new ITLB_Misses())));
    m_vmtc_core.push_back(std::make_pair("Predecode_Error",
                                         (MetricBase*)(new Predecode_Error())));
    m_vmtc_core.push_back(std::make_pair("Fetch_Bandwidth",
                                         (MetricBase*)(new Fetch_Bandwidth())));
    m_vmtc_core.push_back(std::make_pair("Bad_Speculation",
                                         (MetricBase*)(new Bad_Speculation())));
    m_vmtc_core.push_back(std::make_pair(
            "Branch_Mispredicts", (MetricBase*)(new Branch_Mispredicts())));
    m_vmtc_core.push_back(std::make_pair("Backend_Bound",
                                         (MetricBase*)(new Backend_Bound())));
    m_vmtc_core.push_back(
            std::make_pair("Memory_Bound", (MetricBase*)(new Memory_Bound())));
    m_vmtc_core.push_back(
            std::make_pair("Load_Bound", (MetricBase*)(new Load_Bound())));
    m_vmtc_core.push_back(
            std::make_pair("Load_DTLB", (MetricBase*)(new Load_DTLB())));
    m_vmtc_core.push_back(
            std::make_pair("Load_Cache", (MetricBase*)(new Load_Cache())));
    m_vmtc_core.push_back(
            std::make_pair("Store_Bound", (MetricBase*)(new Store_Bound())));
    m_vmtc_core.push_back(
            std::make_pair("Store_TLB", (MetricBase*)(new Store_TLB())));
    m_vmtc_core.push_back(
            std::make_pair("Store_Buffer", (MetricBase*)(new Store_Buffer())));
    m_vmtc_core.push_back(
            std::make_pair("Core_Bound", (MetricBase*)(new Core_Bound())));
    m_vmtc_core.push_back(std::make_pair("Interlock_Bound",
                                         (MetricBase*)(new Interlock_Bound())));
    m_vmtc_core.push_back(std::make_pair("Interlock_AGU",
                                         (MetricBase*)(new Interlock_AGU())));
    m_vmtc_core.push_back(std::make_pair("Interlock_FPU",
                                         (MetricBase*)(new Interlock_FPU())));
    m_vmtc_core.push_back(std::make_pair(
            "Core_Bound_Others", (MetricBase*)(new Core_Bound_Others())));
    m_vmtc_core.push_back(
            std::make_pair("Retiring", (MetricBase*)(new Retiring())));
    m_vmtc_core.push_back(
            std::make_pair("LD_Retiring", (MetricBase*)(new LD_Retiring())));
    m_vmtc_core.push_back(
            std::make_pair("ST_Retiring", (MetricBase*)(new ST_Retiring())));
    m_vmtc_core.push_back(
            std::make_pair("DP_Retiring", (MetricBase*)(new DP_Retiring())));
    m_vmtc_core.push_back(
            std::make_pair("ASE_Retiring", (MetricBase*)(new ASE_Retiring())));
    m_vmtc_core.push_back(
            std::make_pair("VFP_Retiring", (MetricBase*)(new VFP_Retiring())));
    m_vmtc_core.push_back(std::make_pair(
            "PC_Write_Retiring", (MetricBase*)(new PC_Write_Retiring())));
    m_vmtc_core.push_back(std::make_pair(
            "BR_IMMED_Retiring", (MetricBase*)(new BR_IMMED_Retiring())));
    m_vmtc_core.push_back(std::make_pair(
            "BR_RETURN_Retiring", (MetricBase*)(new BR_RETURN_Retiring())));
    m_vmtc_extra.push_back(std::make_pair(
            "BR_INDIRECT_Retiring", (MetricBase*)(new BR_INDIRECT_Retiring())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_UPI", (MetricBase*)(new Metric_UPI())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_CPI", (MetricBase*)(new Metric_CPI())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_CLKS", (MetricBase*)(new Metric_CLKS())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_SLOTS", (MetricBase*)(new Metric_SLOTS())));
    m_vmtc_extra.push_back(std::make_pair("Metric_CoreIPC",
                                          (MetricBase*)(new Metric_CoreIPC())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_CORE_CLKS", (MetricBase*)(new Metric_CORE_CLKS())));
    m_vmtc_extra.push_back(std::make_pair("Metric_IpLoad",
                                          (MetricBase*)(new Metric_IpLoad())));
    m_vmtc_extra.push_back(std::make_pair("Metric_IpStore",
                                          (MetricBase*)(new Metric_IpStore())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_IpBranch", (MetricBase*)(new Metric_IpBranch())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_Instructions", (MetricBase*)(new Metric_Instructions())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_IpMispredict", (MetricBase*)(new Metric_IpMispredict())));
    m_vmtc_extra.push_back(std::make_pair("Metric_L1MPKI",
                                          (MetricBase*)(new Metric_L1MPKI())));
    m_vmtc_extra.push_back(std::make_pair("Metric_L2MPKI",
                                          (MetricBase*)(new Metric_L2MPKI())));
    m_vmtc_extra.push_back(std::make_pair("Metric_L3MPKI",
                                          (MetricBase*)(new Metric_L3MPKI())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_L1_BW_Use", (MetricBase*)(new Metric_L1_BW_Use())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_L2_BW_Use", (MetricBase*)(new Metric_L2_BW_Use())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_L3_BW_Use", (MetricBase*)(new Metric_L3_BW_Use())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_DRAM_BW_Use", (MetricBase*)(new Metric_DRAM_BW_Use())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_GFLOPs_Use", (MetricBase*)(new Metric_GFLOPs_Use())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_LD_Ratio", (MetricBase*)(new Metric_LD_Ratio())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_ST_Ratio", (MetricBase*)(new Metric_ST_Ratio())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_ASE_Ratio", (MetricBase*)(new Metric_ASE_Ratio())));
    m_vmtc_extra.push_back(std::make_pair(
            "Metric_DP_Ratio", (MetricBase*)(new Metric_DP_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_BR_IMMED_Ratio",
                           (MetricBase*)(new Metric_BR_IMMED_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_BR_RETURN_Ratio",
                           (MetricBase*)(new Metric_BR_RETURN_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_BR_INDIRECT_Ratio",
                           (MetricBase*)(new Metric_BR_INDIRECT_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L1I_Miss_Ratio",
                           (MetricBase*)(new Metric_L1I_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L1D_Miss_Ratio",
                           (MetricBase*)(new Metric_L1D_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L1D_RD_Miss_Ratio",
                           (MetricBase*)(new Metric_L1D_RD_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L1D_WR_Miss_Ratio",
                           (MetricBase*)(new Metric_L1D_WR_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L2D_Miss_Ratio",
                           (MetricBase*)(new Metric_L2D_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L2D_RD_Miss_Ratio",
                           (MetricBase*)(new Metric_L2D_RD_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L2D_WR_Miss_Ratio",
                           (MetricBase*)(new Metric_L2D_WR_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L3D_Miss_Ratio",
                           (MetricBase*)(new Metric_L3D_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L3D_RD_Miss_Ratio",
                           (MetricBase*)(new Metric_L3D_RD_Miss_Ratio())));
    // Note: Metric_L3D_WR_Miss_Ratio is not support in arm a55
    // m_vmtc_extra.push_back(std::make_pair(
    //        "Metric_L3D_WR_Miss_Ratio", (MetricBase*)(new
    //        Metric_L3D_WR_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_BR_Mispred_Ratio",
                           (MetricBase*)(new Metric_BR_Mispred_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L1I_TLB_Miss_Ratio",
                           (MetricBase*)(new Metric_L1I_TLB_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L1D_TLB_Miss_Ratio",
                           (MetricBase*)(new Metric_L1D_TLB_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_L2_TLB_Miss_Ratio",
                           (MetricBase*)(new Metric_L2_TLB_Miss_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_DTLB_Table_Walk_Ratio",
                           (MetricBase*)(new Metric_DTLB_Table_Walk_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_ITLB_Table_Walk_Ratio",
                           (MetricBase*)(new Metric_ITLB_Table_Walk_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_Unaligned_LDST_Ratio",
                           (MetricBase*)(new Metric_Unaligned_LDST_Ratio())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_FPU_Util",
                           (MetricBase*)(new Metric_FPU_Util())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_Load_Port_Util",
                           (MetricBase*)(new Metric_Load_Port_Util())));
    m_vmtc_extra.push_back(
            std::make_pair("Metric_Store_Port_Util",
                           (MetricBase*)(new Metric_Store_Port_Util())));
}
A55SetUpImpl::~A55SetUpImpl() {
    size_t cz = m_vmtc_core.size();
    for (size_t i = 0; i < cz; ++i) {
        if (m_vmtc_core[i].second) {
            delete m_vmtc_core[i].second;
        }
    }
    size_t ez = m_vmtc_extra.size();
    for (size_t i = 0; i < ez; ++i) {
        if (m_vmtc_extra[i].second) {
            delete m_vmtc_extra[i].second;
        }
    }
}

}  // namespace tma
}  // namespace mperf