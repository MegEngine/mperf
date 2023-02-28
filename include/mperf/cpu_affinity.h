/**
 * \file include/mperf/cpu_affinity.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#pragma once
#include <ctype.h>
#include <stdint.h>
#include <cstdlib>

#ifndef __USE_GNU
#define __USE_GNU
#endif

#ifndef __UCLIBC_LINUX_SPECIFIC__
#define __UCLIBC_LINUX_SPECIFIC__
#endif

#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>

#if defined __ANDROID__ || defined __linux__
#include <sched.h>
#if defined __ANDROID__
#include <dlfcn.h>
#endif
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Bind to a single core
int set_cpu_thread_affinity_spec_core(size_t dev_id);

int set_cpu_thread_affinity(cpu_set_t thread_affinity_mask);

// Bind to a set of cores
int set_cpu_thread_affinity_mask(size_t mask);

/*
 * Allocates a new set for ncpus and returns size in bytes and size in bits
 */
cpu_set_t* cpuset_alloc(int ncpus, size_t* setsize, size_t* nbits);

void cpuset_free(cpu_set_t* set);

/*
 * Number of bits in a CPU bitmask on current system
 */
int get_max_number_of_cpus(void);

/*
 * Parses string with list of CPU ranges.
 * Returns 0 on success.
 * Returns 1 on error.
 * Returns 2 if fail is set and a cpu number passed in the list doesn't fit
 * into the cpu_set. If fail is not set cpu numbers that do not fit are
 * ignored and 0 is returned instead.
 */
int cpulist_parse(const char* str, cpu_set_t* set, size_t setsize, int fail,
                  int* cpu_list);