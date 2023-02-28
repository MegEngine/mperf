/*
 * ---------------------------------------------------------------------------
 * \file include/mperf/cpu_info.h
 *
 * Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2022-2023 Megvii Inc. All rights
 * reserved.
 *
 * ---------------------------------------------------------------------------
 */
#pragma once
#include <stddef.h>
#include <stdio.h>
#include <string>

namespace mperf {
// print cpu supported features
std::string cpu_info_support_features();

// neon = armv7 neon or aarch64 asimd
int cpu_info_support_arm_neon();
// vfpv4 = armv7 fp16 + fma
int cpu_info_support_arm_vfpv4();
// asimdhp = aarch64 asimd half precision
int cpu_info_support_arm_asimdhp();
// asimddp = aarch64 asimd dot product
int cpu_info_support_arm_asimddp();

// avx = x86 avx
int cpu_info_support_x86_avx();
// xop = x86 xop
int cpu_info_support_x86_xop();
// fma = x86 fma
int cpu_info_support_x86_fma();
// avx2 = x86 avx2 + fma + f16c
int cpu_info_support_x86_avx2();
// avx_vnni = x86 avx vnni
int cpu_info_support_x86_avx_vnni();
// avx512 = x86 avx512f + avx512vl
int cpu_info_support_x86_avx512();
// avx512_vnni = x86 avx512 vnni
int cpu_info_support_x86_avx512_vnni();

// msa = mips mas
int cpu_info_support_mips_msa();
// mmi = loongson mmi
int cpu_info_support_loongson_mmi();

// v = riscv vector
int cpu_info_support_riscv_v();
// zfh = riscv half-precision float
int cpu_info_support_riscv_zfh();
// vlenb = riscv vector length in bytes
int cpu_info_cpu_riscv_vlenb();

// cpu info
int cpu_info_get_cpu_count();
int cpu_info_get_active_cpu_count(int powersave);
int cpu_info_get_little_cpu_count();
int cpu_info_get_middle_cpu_count();
int cpu_info_get_big_cpu_count();

#if defined __ANDROID__ || defined __linux__
int cpu_info_get_max_freq_khz(int cpuid);
#endif

// bind all threads on little clusters if powersave enabled
// affects HMP arch cpu like ARM big.LITTLE
// only implemented on android at the moment
// switching powersave is expensive and not thread-safe
// 0 = all cores enabled(default)
// 1 = only little clusters enabled
// 2 = only middle clusters enabled
// 3 = only big clusters enabled
// return 0 if success for setter function
int cpu_info_get_cpu_powersave();
int cpu_info_set_cpu_powersave(int powersave);

// need to flush denormals on Intel Chipset.
// Other architectures such as ARM can be added as needed.
// 0 = DAZ OFF, FTZ OFF
// 1 = DAZ ON , FTZ OFF
// 2 = DAZ OFF, FTZ ON
// 3 = DAZ ON,  FTZ ON
int cpu_info_get_flush_denormals();
int cpu_info_set_flush_denormals(int flush_denormals);

// CPU clock frequency(TSC), khz
uint64_t cpu_info_ref_freq(int dev_id = 0);
// cycles(TSC)
uint64_t cpu_info_ref_cycles(int dev_id = 0);

double cpu_thread_time_ms();
double cpu_process_time_ms();
}  // namespace mperf