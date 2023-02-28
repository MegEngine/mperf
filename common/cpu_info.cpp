// Part of the code in this module comes from
// ncnn(https://github.com/Tencent/ncnn/blob/18fbaebe68f167eca5b3e92774ba7c8d7337d9ee/src/cpu.cpp),
// and mperf has made some modifications.
/*
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https: *opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 * ---------------------------------------------------------------------------
 * \file common/cpu_info.cpp
 *
 * Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2022-2023 Megvii Inc. All rights
 * reserved.
 *
 * ---------------------------------------------------------------------------
 */

#include "mperf/cpu_info.h"
#include "mperf/cpu_affinity.h"
#include "mperf/cpu_march_probe.h"

#include <fcntl.h>
#include <limits.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <string>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || \
        defined(_M_X64)
#ifdef _MSC_VER
#include <immintrin.h>  // _xgetbv()
#include <intrin.h>     // __cpuid()
#endif
#if defined(__clang__) || defined(__GNUC__)
#include <cpuid.h>  // __get_cpuid() and __cpuid_count()
#endif
#endif

#if defined __ANDROID__ || defined __linux__
#if defined __ANDROID__
#include <dlfcn.h>
#endif
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__SSE3__)
#include <immintrin.h>
#endif

#if (defined _WIN32 && !(defined __MINGW32__))
#define WIN32_LEAN_AND_MEAN
#include <process.h>
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <assert.h>
#include <errno.h>
#include <map>
#include <vector>

#if (defined _WIN32 && !(defined __MINGW32__))
class ThreadLocalStorage {
public:
    ThreadLocalStorage() { key = TlsAlloc(); }
    ~ThreadLocalStorage() { TlsFree(key); }
    void set(void* value) { TlsSetValue(key, (LPVOID)value); }
    void* get() { return (void*)TlsGetValue(key); }

private:
    DWORD key;
};
#else
class ThreadLocalStorage {
public:
    ThreadLocalStorage() { pthread_key_create(&key, 0); }
    ~ThreadLocalStorage() { pthread_key_delete(key); }
    void set(void* value) { pthread_setspecific(key, value); }
    void* get() { return pthread_getspecific(key); }

private:
    pthread_key_t key;
};
#endif

#if defined __ANDROID__ || defined __linux__

#define AT_HWCAP 16
#define AT_HWCAP2 26

#if defined __ANDROID__
// Probe the system's C library for a 'getauxval' function and call it if
// it exits, or return 0 for failure. This function is available since API
// level 20.
//
// This code does *NOT* check for '__ANDROID_API__ >= 20' to support the
// edge case where some NDK developers use headers for a platform that is
// newer than the one really targetted by their application.
// This is typically done to use newer native APIs only when running on more
// recent Android versions, and requires careful symbol management.
//
// Note that getauxval() can't really be re-implemented here, because
// its implementation does not parse /proc/self/auxv. Instead it depends
// on values  that are passed by the kernel at process-init time to the
// C runtime initialization layer.
static unsigned int get_elf_hwcap_from_getauxval() {
    typedef unsigned long getauxval_func_t(unsigned long);

    dlerror();
    void* libc_handle = dlopen("libc.so", RTLD_NOW);
    if (!libc_handle) {
        printf("dlopen libc.so failed %s\n", dlerror());
        return 0;
    }

    unsigned int result = 0;
    getauxval_func_t* func = (getauxval_func_t*)dlsym(libc_handle, "getauxval");
    if (!func) {
        printf("dlsym getauxval failed\n");
    } else {
        // Note: getauxval() returns 0 on failure. Doesn't touch errno.
        result = (unsigned int)(*func)(AT_HWCAP);
    }
    dlclose(libc_handle);

    return result;
}
#endif  // defined __ANDROID__

// extract the ELF HW capabilities bitmap from /proc/self/auxv
static unsigned int get_elf_hwcap_from_proc_self_auxv() {
    FILE* fp = fopen("/proc/self/auxv", "rb");
    if (!fp) {
        printf("fopen /proc/self/auxv failed\n");
        return 0;
    }

#if __aarch64__ || __riscv_xlen == 64
    struct {
        uint64_t tag;
        uint64_t value;
    } entry;
#else
    struct {
        unsigned int tag;
        unsigned int value;
    } entry;

#endif

    unsigned int result = 0;
    while (!feof(fp)) {
        int nread = fread((char*)&entry, sizeof(entry), 1, fp);
        if (nread != 1)
            break;

        if (entry.tag == 0 && entry.value == 0)
            break;

        if (entry.tag == AT_HWCAP) {
            result = entry.value;
            break;
        }
    }

    fclose(fp);

    return result;
}

static unsigned int get_elf_hwcap() {
#if defined __ANDROID__
    unsigned int hwcap = get_elf_hwcap_from_getauxval();
    if (hwcap)
        return hwcap;
#endif

    return get_elf_hwcap_from_proc_self_auxv();
}

static unsigned int g_hwcaps = get_elf_hwcap();

#if __aarch64__
// from arch/arm64/include/uapi/asm/hwcap.h
#define HWCAP_ASIMD (1 << 1)
#define HWCAP_ASIMDHP (1 << 10)
#define HWCAP_ASIMDDP (1 << 20)
#else
// from arch/arm/include/uapi/asm/hwcap.h
#define HWCAP_NEON (1 << 12)
#define HWCAP_VFPv4 (1 << 16)
#endif

#if __mips__
// from arch/mips/include/uapi/asm/hwcap.h
#define HWCAP_MIPS_MSA (1 << 1)
#define HWCAP_LOONGSON_MMI (1 << 11)
#endif

#if __riscv
// from arch/riscv/include/uapi/asm/hwcap.h
#define COMPAT_HWCAP_ISA_F (1 << ('F' - 'A'))
#define COMPAT_HWCAP_ISA_V (1 << ('V' - 'A'))
#endif

#endif  // defined __ANDROID__ || defined __linux__

class CpuSet {
public:
    CpuSet();
    void enable(int cpu);
    void disable(int cpu);
    void disable_all();
    bool is_enabled(int cpu) const;
    int num_enabled() const;
    void print() const;

public:
#if defined __ANDROID__ || defined __linux__
    cpu_set_t cpu_set;
#endif
};

#if defined __ANDROID__ || defined __linux__
CpuSet::CpuSet() {
    disable_all();
}

void CpuSet::print() const {
    int len = 16;
    for (int i = 0; i < len; i++) {
        printf(" %lu ", cpu_set.__bits[i]);
    }
    printf("\n");
}

void CpuSet::enable(int cpu) {
    CPU_SET(cpu, &cpu_set);
}

void CpuSet::disable(int cpu) {
    CPU_CLR(cpu, &cpu_set);
}

void CpuSet::disable_all() {
    CPU_ZERO(&cpu_set);
}

bool CpuSet::is_enabled(int cpu) const {
    return CPU_ISSET(cpu, &cpu_set);
}

int CpuSet::num_enabled() const {
    int num_enabled = 0;
    for (int i = 0; i < (int)sizeof(cpu_set_t) * 8; i++) {
        if (is_enabled(i))
            num_enabled++;
    }

    return num_enabled;
}
#else
CpuSet::CpuSet() {}

void CpuSet::enable(int /* cpu */) {}

void CpuSet::disable(int /* cpu */) {}

void CpuSet::disable_all() {}

bool CpuSet::is_enabled(int /* cpu */) const {
    return true;
}

int CpuSet::num_enabled() const {
    return mperf::cpu_info_get_cpu_count();
}
#endif

int mperf::cpu_info_support_arm_neon() {
#if defined __ANDROID__ || defined __linux__
#if __aarch64__
    return g_hwcaps & HWCAP_ASIMD;
#else
    return g_hwcaps & HWCAP_NEON;
#endif
#else
    return 0;
#endif
}

int mperf::cpu_info_support_arm_vfpv4() {
#if defined __ANDROID__ || defined __linux__
#if __aarch64__
    // neon always enable fma and fp16
    return g_hwcaps & HWCAP_ASIMD;
#else
    return g_hwcaps & HWCAP_VFPv4;
#endif
#else
    return 0;
#endif
}

int mperf::cpu_info_support_arm_asimdhp() {
#if defined __ANDROID__ || defined __linux__
#if __aarch64__
    return g_hwcaps & HWCAP_ASIMDHP;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int mperf::cpu_info_support_arm_asimddp() {
#if defined __ANDROID__ || defined __linux__
#if __aarch64__
    return g_hwcaps & HWCAP_ASIMDDP;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || \
        defined(_M_X64)
static inline void x86_cpuid(int level, unsigned int out[4]) {
#if defined(_MSC_VER)
    __cpuid((int*)out, level);
#elif defined(__clang__) || defined(__GNUC__)
    __get_cpuid(level, out, out + 1, out + 2, out + 3);
#else
    printf("x86_cpuid is unknown for current compiler\n");
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
#endif
}

static inline void x86_cpuid_sublevel(int level, int sublevel,
                                      unsigned int out[4]) {
#if defined(_MSC_VER)
    __cpuidex((int*)out, level, sublevel);
#elif defined(__clang__) || defined(__GNUC__)
    __cpuid_count(level, sublevel, out[0], out[1], out[2], out[3]);
#else
    printf("x86_cpuid_sublevel is unknown for current compiler\n");
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
#endif
}

static inline int x86_get_xcr0() {
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 160040219)
    return _xgetbv(0);
#elif defined(__i386__) || defined(__x86_64__)
    int xcr0 = 0;
    asm(".byte 0x0f, 0x01, 0xd0" : "=a"(xcr0) : "c"(0) : "%edx");
    return xcr0;
#else
    printf("x86_get_xcr0 is unknown for current compiler\n");
    return 0xffffffff;  // assume it will work
#endif
}

static int cpu_info_get_support_x86_avx() {
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 1)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) ||
        !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    return 1;
}

static int cpu_info_get_support_x86_fma() {
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) ||
        !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    return cpu_info[2] & (1u << 12);
}

static int cpu_info_get_support_x86_xop() {
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0x80000000, cpu_info);

    if (cpu_info[0] < 0x80000001)
        return 0;

    x86_cpuid(0x80000001, cpu_info);

    return cpu_info[2] & (1u << 11);
}

static int cpu_info_get_support_x86_avx2() {
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) ||
        !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    x86_cpuid_sublevel(7, 0, cpu_info);
    return cpu_info[1] & (1u << 5);
}

static int cpu_info_get_support_x86_avx_vnni() {
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) ||
        !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    x86_cpuid_sublevel(7, 1, cpu_info);
    return cpu_info[0] & (1u << 4);
}

static int cpu_info_get_support_x86_avx512() {
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) ||
        !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    // check avx512 XSAVE enabled by kernel
    if ((x86_get_xcr0() & 0xe0) != 0xe0)
        return 0;

    x86_cpuid_sublevel(7, 0, cpu_info);
    return (cpu_info[1] & (1u << 16)) && (cpu_info[1] & (1u << 31));
}

static int cpu_info_get_support_x86_avx512_vnni() {
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) ||
        !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    // check avx512 XSAVE enabled by kernel
    if ((x86_get_xcr0() & 0xe0) != 0xe0)
        return 0;

    x86_cpuid_sublevel(7, 0, cpu_info);
    return cpu_info[2] & (1u << 11);
}

static int g_cpu_support_x86_avx = cpu_info_get_support_x86_avx();
static int g_cpu_support_x86_fma = cpu_info_get_support_x86_fma();
static int g_cpu_support_x86_xop = cpu_info_get_support_x86_xop();
static int g_cpu_support_x86_avx2 = cpu_info_get_support_x86_avx2();
static int g_cpu_support_x86_avx_vnni = cpu_info_get_support_x86_avx_vnni();
static int g_cpu_support_x86_avx512 = cpu_info_get_support_x86_avx512();
static int g_cpu_support_x86_avx512_vnni =
        cpu_info_get_support_x86_avx512_vnni();
#else   // defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) ||
        // defined(_M_X64)
static const int g_cpu_support_x86_avx = 0;
static const int g_cpu_support_x86_fma = 0;
static const int g_cpu_support_x86_xop = 0;
static const int g_cpu_support_x86_avx2 = 0;
static const int g_cpu_support_x86_avx_vnni = 0;
static const int g_cpu_support_x86_avx512 = 0;
static const int g_cpu_support_x86_avx512_vnni = 0;
#endif  // defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) ||
        // defined(_M_X64)

int mperf::cpu_info_support_x86_avx() {
    return g_cpu_support_x86_avx;
}

int mperf::cpu_info_support_x86_fma() {
    return g_cpu_support_x86_fma;
}

int mperf::cpu_info_support_x86_xop() {
    return g_cpu_support_x86_xop;
}

int mperf::cpu_info_support_x86_avx2() {
    return g_cpu_support_x86_avx2;
}

int mperf::cpu_info_support_x86_avx_vnni() {
    return g_cpu_support_x86_avx_vnni;
}

int mperf::cpu_info_support_x86_avx512() {
    return g_cpu_support_x86_avx512;
}

int mperf::cpu_info_support_x86_avx512_vnni() {
    return g_cpu_support_x86_avx512_vnni;
}

int mperf::cpu_info_support_mips_msa() {
#if defined __ANDROID__ || defined __linux__
#if __mips__
    return g_hwcaps & HWCAP_MIPS_MSA;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int mperf::cpu_info_support_loongson_mmi() {
#if defined __ANDROID__ || defined __linux__
#if __mips__
    return g_hwcaps & HWCAP_LOONGSON_MMI;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int mperf::cpu_info_support_riscv_v() {
#if defined __ANDROID__ || defined __linux__
#if __riscv
    return g_hwcaps & COMPAT_HWCAP_ISA_V;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int mperf::cpu_info_support_riscv_zfh() {
#if defined __ANDROID__ || defined __linux__
#if __riscv
    // v + f does not imply zfh, but how to discover zfh properly ?
    // upstream issue https://github.com/riscv/riscv-isa-manual/issues/414
    return g_hwcaps & COMPAT_HWCAP_ISA_V && g_hwcaps & COMPAT_HWCAP_ISA_F;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int mperf::cpu_info_cpu_riscv_vlenb() {
#if __riscv
    if (!mperf::cpu_info_support_riscv_v())
        return 0;

    int a = 0;
    asm volatile(
            ".word  0xc22026f3  \n"  // csrr  a3, vlenb
            "mv     %0, a3      \n"
            : "=r"(a)
            :
            : "memory", "a3");
    return a;
#else
    return 0;
#endif
}

static int cpu_info_get_cpucount() {
    int count = 0;
#if defined __ANDROID__ || defined __linux__
    // get cpu count from /proc/cpuinfo
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp)
        return 1;

    char line[1024];
    while (!feof(fp)) {
        char* s = fgets(line, 1024, fp);
        if (!s)
            break;

        if (memcmp(line, "processor", 9) == 0) {
            count++;
        }
    }

    fclose(fp);
#else
    count = 1;
#endif

    if (count < 1)
        count = 1;

    return count;
}

static int g_cpucount = cpu_info_get_cpucount();
static int g_little_core_num = 1;
static int g_middle_core_num = 1;
static int g_big_core_num = 1;

#if 1
int mperf::cpu_info_get_cpu_count() {
    return g_cpucount;
}
#else
void cpu_info_get_cpu_count() {
    size_t cpu_num = sysconf(_SC_NPROCESSORS_ONLN);
    printf("there are %zu cores\n", cpu_num);
}
#endif

int mperf::cpu_info_get_active_cpu_count(int powersave) {
    if (powersave == 0)
        return g_cpucount;

    if (powersave == 1)
        return g_little_core_num;

    if (powersave == 2)
        return g_middle_core_num;

    if (powersave == 3)
        return g_big_core_num;

    printf("powersave %d not supported\n", powersave);
    return g_cpucount;
}

#if defined __ANDROID__ || defined __linux__
int mperf::cpu_info_get_max_freq_khz(int cpuid) {
    // first try, for all possible cpu
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state",
            cpuid);

    FILE* fp = fopen(path, "rb");

    if (!fp) {
        // second try, for online cpu
        sprintf(path,
                "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
                cpuid);
        fp = fopen(path, "rb");

        if (fp) {
            int max_freq_khz = 0;
            while (!feof(fp)) {
                int freq_khz = 0;
                int nscan = fscanf(fp, "%d %*d", &freq_khz);
                if (nscan != 1)
                    break;

                if (freq_khz > max_freq_khz)
                    max_freq_khz = freq_khz;
            }

            fclose(fp);

            if (max_freq_khz != 0)
                return max_freq_khz;

            fp = NULL;
        }

        if (!fp) {
            // third try, for online cpu
            sprintf(path,
                    "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
                    cpuid);
            fp = fopen(path, "rb");

            if (!fp)
                return -1;

            int max_freq_khz = -1;
            int nscan = fscanf(fp, "%d", &max_freq_khz);
            if (nscan != 1) {
                printf("fscanf cpuinfo_max_freq error %d\n", nscan);
            }
            fclose(fp);

            return max_freq_khz;
        }
    }

    int max_freq_khz = 0;
    while (!feof(fp)) {
        int freq_khz = 0;
        int nscan = fscanf(fp, "%d %*d", &freq_khz);
        if (nscan != 1)
            break;

        if (freq_khz > max_freq_khz)
            max_freq_khz = freq_khz;
    }

    fclose(fp);

    return max_freq_khz;
}

#endif  // defined __ANDROID__ || defined __linux__

static int g_powersave = 0;

int mperf::cpu_info_get_cpu_powersave() {
    return g_powersave;
}

static CpuSet g_thread_affinity_mask_all;
static CpuSet g_thread_affinity_mask_little;
static CpuSet g_thread_affinity_mask_middle;
static CpuSet g_thread_affinity_mask_big;

static int setup_thread_affinity_masks() {
    // printf("### g_cpucount is %d\n", g_cpucount);
    g_thread_affinity_mask_all.disable_all();

#if defined __ANDROID__ || defined __linux__
    std::map<int, std::vector<int>> any_freq;
    for (int i = 0; i < g_cpucount; i++) {
        int max_freq_khz = mperf::cpu_info_get_max_freq_khz(i);
        // printf("core %d, freq is %d\n", i, max_freq_khz);
        if (i == 0) {
            any_freq.insert(
                    std::make_pair(max_freq_khz, std::vector<int>({i})));
        } else {
            auto iter = any_freq.find(max_freq_khz);
            if (iter != any_freq.end()) {
                iter->second.push_back(i);
            } else {
                any_freq.insert(
                        std::make_pair(max_freq_khz, std::vector<int>({i})));
            }
        }
    }
    auto size_any_freq = any_freq.size();
    assert(size_any_freq >= 1);
    if (size_any_freq == 3) {
        auto iter = any_freq.begin();
        for (size_t k = 0; k < iter->second.size(); k++) {
            // printf("little: core%d freq(%d kHz)\n", iter->second[k],
            //        iter->first);
            g_thread_affinity_mask_little.enable(iter->second[k]);
        }
        g_little_core_num = iter->second.size();
        iter++;
        for (size_t k = 0; k < iter->second.size(); k++) {
            // printf("middle: core%d freq(%d kHz)\n", iter->second[k],
            //        iter->first);
            g_thread_affinity_mask_middle.enable(iter->second[k]);
        }
        g_middle_core_num = iter->second.size();
        iter++;
        for (size_t k = 0; k < iter->second.size(); k++) {
            // printf("big: core%d freq(%d kHz)\n", iter->second[k],
            // iter->first);
            g_thread_affinity_mask_big.enable(iter->second[k]);
        }
        g_big_core_num = iter->second.size();
    } else if (size_any_freq == 2) {
        g_thread_affinity_mask_little.disable_all();
        auto iter = any_freq.begin();
        for (size_t k = 0; k < iter->second.size(); k++) {
            g_thread_affinity_mask_middle.enable(iter->second[k]);
        }
        g_middle_core_num = iter->second.size();
        iter++;
        for (size_t k = 0; k < iter->second.size(); k++) {
            g_thread_affinity_mask_big.enable(iter->second[k]);
        }
        g_big_core_num = iter->second.size();
    } else {
        g_thread_affinity_mask_little.disable_all();
        g_thread_affinity_mask_middle.disable_all();
        // g_thread_affinity_mask_big = g_thread_affinity_mask_all;
        auto iter = any_freq.begin();
        for (size_t k = 0; k < iter->second.size(); k++) {
            g_thread_affinity_mask_big.enable(iter->second[k]);
        }
        g_big_core_num = g_cpucount;
    }
#else
    g_thread_affinity_mask_little.disable_all();
    g_thread_affinity_mask_middle.disable_all();
    g_thread_affinity_mask_big = g_thread_affinity_mask_all;
#endif

    return 0;
}

static const CpuSet& get_cpu_thread_affinity_mask(int powersave) {
    setup_thread_affinity_masks();

    if (powersave == 0) {
        return g_thread_affinity_mask_all;
    }
    if (powersave == 1)
        return g_thread_affinity_mask_little;

    if (powersave == 2)
        return g_thread_affinity_mask_middle;

    if (powersave == 3)
        return g_thread_affinity_mask_big;

    printf("powersave %d not supported\n", powersave);

    // fallback to all cores anyway
    return g_thread_affinity_mask_all;
}

int mperf::cpu_info_get_little_cpu_count() {
    return get_cpu_thread_affinity_mask(1).num_enabled();
}

int mperf::cpu_info_get_middle_cpu_count() {
    return get_cpu_thread_affinity_mask(2).num_enabled();
}

int mperf::cpu_info_get_big_cpu_count() {
    int big_cpu_count = get_cpu_thread_affinity_mask(3).num_enabled();
    return big_cpu_count ? big_cpu_count : g_cpucount;
}

int mperf::cpu_info_set_cpu_powersave(int powersave) {
#if defined __ANDROID__ || defined __linux__
    if (powersave < 0 || powersave > 3) {
        printf("powersave %d not supported\n", powersave);
        return -1;
    }
    const CpuSet& thread_affinity_mask =
            get_cpu_thread_affinity_mask(powersave);

    int ret = set_cpu_thread_affinity(thread_affinity_mask.cpu_set);
    if (ret != 0)
        return ret;

    g_powersave = powersave;

    return 0;
#else
    return -1;
#endif
}

static ThreadLocalStorage tls_flush_denormals;

int mperf::cpu_info_get_flush_denormals() {
#if defined(__SSE3__)
    return (int)reinterpret_cast<size_t>(tls_flush_denormals.get());
#else
    return 0;
#endif
}

int mperf::cpu_info_set_flush_denormals(int flush_denormals) {
    if (flush_denormals < 0 || flush_denormals > 3) {
        printf("denormals_zero %d not supported\n", flush_denormals);
        return -1;
    }
#if defined(__SSE3__)
    if (flush_denormals == 0) {
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    } else if (flush_denormals == 1) {
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    } else if (flush_denormals == 2) {
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    } else if (flush_denormals == 3) {
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    }

    tls_flush_denormals.set(reinterpret_cast<void*>((size_t)flush_denormals));
    return 0;
#else
    return 0;
#endif
}

std::string mperf::cpu_info_support_features() {
    std::string result;
    if (mperf::cpu_info_support_arm_neon()) {
        result += " NEON";
    }
    if (mperf::cpu_info_support_arm_vfpv4()) {
        result += " VFPV4";
    }
    if (mperf::cpu_info_support_arm_asimdhp()) {
        result += " AARCH64_ASIMD_HALF_PRECISION";
    }
    if (mperf::cpu_info_support_arm_asimddp()) {
        result += " AARCH64_ASIMD_DOT_PRODUCT";
    }
    if (mperf::cpu_info_support_x86_avx()) {
        result += " AVX";
    }
    if (mperf::cpu_info_support_x86_avx2()) {
        result += " AVX2";
    }
    if (mperf::cpu_info_support_x86_avx_vnni()) {
        result += " AVX_VNNI";
    }
    if (mperf::cpu_info_support_x86_avx512()) {
        result += " AVX512";
    }
    if (mperf::cpu_info_support_x86_avx512_vnni()) {
        result += " AVX512_VNNI";
    }
    if (mperf::cpu_info_support_x86_xop()) {
        result += " XOP";
    }
    if (mperf::cpu_info_support_x86_fma()) {
        result += " FMA";
    }
    if (mperf::cpu_info_support_mips_msa()) {
        result += " MSA";
    }
    if (mperf::cpu_info_support_loongson_mmi()) {
        result += " LOONGSON_MMI";
    }
    if (mperf::cpu_info_support_riscv_v()) {
        result += " RISCV_V";
    }
    if (mperf::cpu_info_support_riscv_zfh()) {
        result += " RSICV_HALF_PRECISION";
    }
    if (mperf::cpu_info_cpu_riscv_vlenb()) {
        result += " RISCV_VLENB";
    }

    return result;
}

namespace {
uint64_t base_query_hz(int dev_id = 0) {
    uint64_t freq = 0;

    auto read_info_from_file = [](const char* file, uint64_t* value) -> bool {
        bool ret = false;
        int fd = open(file, O_RDONLY);
        if (fd != -1) {
            char line[1024];
            char* err;
            memset(line, '\0', sizeof(line));
            int len = read(fd, line, sizeof(line) - 1);
            if (len <= 0) {
                ret = false;
            } else {
                const long temp_value = strtol(line, &err, 10);
                if (line[0] != '\0' && (*err == '\n' || *err == '\0')) {
                    *value = temp_value;
                    ret = true;
                }
            }
            close(fd);
        }
        return ret;
    };

    char fname[256];
    snprintf(fname, sizeof(fname), "/sys/devices/system/cpu/cpu%d/tsc_freq_khz",
             dev_id);
    if (read_info_from_file(fname, &freq)) {
        return freq * 1e3;
    }

    char fname2[256];
    snprintf(fname2, sizeof(fname2),
             "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", dev_id);
    if (read_info_from_file(fname2, &freq)) {
        // freq khz
        return freq * 1e3;
    }

#if 0
    int cmp_freq = mperf::cpu_mhz();
    if (cmp_freq != -1) {
        return freq * 1e3;
    }
#endif

    return 1;
}
}  // namespace

uint64_t mperf::cpu_info_ref_freq(int dev_id) {
    (void)base_query_hz;
#if defined(__i386__) || defined(__x86_64__) || defined(__amd64__) || \
        defined(__riscv)
    return base_query_hz(dev_id);
#elif defined(__aarch64__)
    uint64_t freq;  // hz
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
#elif defined(__ARM_ARCH)
    // https://github.com/apritzel/pine64/blob/master/tools/test_timer.c
    // https://android.googlesource.com/kernel/msm.git/+/android-msm-mako-3.4-kitkat-mr2/arch/arm/kernel/arch_timer.c
    uint32_t freq;
    __asm__ volatile("mrc p15, 0, %0, c14, c0, 0\n" : "=r"(freq));
    return freq;
#endif
    return 1;
}

// code reference: google-benchmark/src/cycleclock.h
uint64_t mperf::cpu_info_ref_cycles(int dev_id) {
#if defined(__i386__)
    uint64_t ret;
    __asm__ volatile("rdtsc" : "=A"(ret));
    return ret;
#elif defined(__x86_64__) || defined(__amd64__)
    uint64_t low, high;
    __asm__ volatile("rdtscp" : "=a"(low), "=d"(high));
    return (high << 32) | low;
#elif defined(__aarch64__)
    // System timer of ARMv8 runs at a different frequency than the CPU's.
    // The frequency is fixed, typically in the range 1-50MHz.  It can be
    // read at CNTFRQ special register.  We assume the OS has set up
    // the virtual timer properly.
    uint64_t virtual_timer_value;
    asm volatile("mrs %0, cntvct_el0" : "=r"(virtual_timer_value));
    return virtual_timer_value;
#elif defined(__ARM_ARCH)
// V6 is the earliest arch that has a standard cyclecount
// Native Client validator doesn't allow MRC instructions.
#if (__ARM_ARCH >= 6)
    uint32_t pmccntr;
    uint32_t pmuseren;
    uint32_t pmcntenset;
    // Read the user mode perf monitor counter access permissions.
    asm volatile("mrc p15, 0, %0, c9, c14, 0" : "=r"(pmuseren));
    if (pmuseren & 1) {  // Allows reading perfmon counters for user mode code.
        asm volatile("mrc p15, 0, %0, c9, c12, 1" : "=r"(pmcntenset));
        if (pmcntenset & 0x80000000ul) {  // Is it counting?
            asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(pmccntr));
            // The counter is set up to count every 64th cycle
            return static_cast<int64_t>(pmccntr) << 6;
        }
    }
#endif
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#elif defined(__riscv)  // RISC-V
// Use RDCYCLE (and RDCYCLEH on riscv32)
#if __riscv_xlen == 32
    uint32_t cycles_lo, cycles_hi0, cycles_hi1;
    // This asm also includes the PowerPC overflow handling strategy, as above.
    // Implemented in assembly because Clang insisted on branching.
    asm volatile(
            "rdcycleh %0\n"
            "rdcycle %1\n"
            "rdcycleh %2\n"
            "sub %0, %0, %2\n"
            "seqz %0, %0\n"
            "sub %0, zero, %0\n"
            "and %1, %1, %0\n"
            : "=r"(cycles_hi0), "=r"(cycles_lo), "=r"(cycles_hi1));
    return (static_cast<uint64_t>(cycles_hi1) << 32) | cycles_lo;
#else
    uint64_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#endif
#else
#error You need to define cpu_ref_cycles for your OS and CPU
#endif
    return 1;
}

double mperf::cpu_thread_time_ms() {
    struct timespec ts;
    // equal to RDTSC
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0)
        return ts.tv_sec * 1e3 + (static_cast<double>(ts.tv_nsec) * 1e-6);

    return 0.0;
}

double mperf::cpu_process_time_ms() {
    struct timespec ts;
    // equal to RDTSC
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) == 0)
        return ts.tv_sec * 1e3 + (static_cast<double>(ts.tv_nsec) * 1e-6);
    return 0.0;
}
