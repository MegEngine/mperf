/**
 * \file common/cpu_affinity.cpp
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#include "mperf/cpu_affinity.h"
#include "mperf/utils.h"

// Bind to a specifical core
int set_cpu_thread_affinity_spec_core(size_t dev_id) {
    cpu_set_t cst;
    CPU_ZERO(&cst);
    CPU_SET(dev_id, &cst);
    if (sched_setaffinity(0, sizeof(cst), &cst)) {
        // intentional not to use mperf_throw, because the program can still run
        // under an failed affinity set
        mperf_log_error("set affinity of core:%zu failed: error: %s\n", dev_id,
                        strerror(errno));
        return 1;
    }
    mperf_log("set affinity of core:%zu success\n", dev_id);
    return 0;
}

int set_cpu_thread_affinity(cpu_set_t thread_affinity_mask) {
#if defined __ANDROID__ || defined __linux__
// set affinity for thread
#if defined(__BIONIC__)
    pid_t pid = gettid();
#else
    pid_t pid = syscall(SYS_gettid);
#endif
    printf("************  binding core mask : %lu  *****************\n",
           thread_affinity_mask.__bits[0]);
    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(cpu_set_t),
                             &thread_affinity_mask);
    if (syscallret) {
        printf("syscall(set affinity) error: %s\n", strerror(errno));
        return -1;
    }
    return 0;
#else
    (void)thread_affinity_mask;
    return -1;
#endif  // defined __ANDROID__ || defined __linux__
}

// Bind to a set of cores
int set_cpu_thread_affinity_mask(size_t mask) {
#if defined __ANDROID__ || defined __linux__
    cpu_set_t thread_affinity_mask;
    thread_affinity_mask.__bits[0] = mask;
    return set_cpu_thread_affinity(thread_affinity_mask);
#else
    (void)mask;
    return -1;
#endif
}

// code reference: util-linux/lib/cpuset.c
#define cpuset_nbits(setsize) (8 * (setsize))
/*
 * Allocates a new set for ncpus and returns size in bytes and size in bits
 */
cpu_set_t* cpuset_alloc(int ncpus, size_t* setsize, size_t* nbits) {
    cpu_set_t* set = CPU_ALLOC(ncpus);

    if (!set)
        return NULL;
    if (setsize)
        *setsize = CPU_ALLOC_SIZE(ncpus);
    if (nbits)
        *nbits = cpuset_nbits(CPU_ALLOC_SIZE(ncpus));
    return set;
}

void cpuset_free(cpu_set_t* set) {
    CPU_FREE(set);
}

/*
 * Number of bits in a CPU bitmask on current system
 */
int get_max_number_of_cpus(void) {
#ifdef SYS_sched_getaffinity
    int n, cpus = 2048;
    size_t setsize;
    cpu_set_t* set = cpuset_alloc(cpus, &setsize, NULL);

    if (!set)
        return -1; /* error */

    for (;;) {
        CPU_ZERO_S(setsize, set);

        /* the library version does not return size of cpumask_t */
        n = syscall(SYS_sched_getaffinity, 0, setsize, set);

        if (n < 0 && errno == EINVAL && cpus < 1024 * 1024) {
            cpuset_free(set);
            cpus *= 2;
            set = cpuset_alloc(cpus, &setsize, NULL);
            if (!set)
                return -1; /* error */
            continue;
        }
        cpuset_free(set);
        return n * 8;
    }
#endif
    return -1;
}

static const char* nexttoken(const char* q, int sep) {
    if (q)
        q = strchr(q, sep);
    if (q)
        q++;
    return q;
}

static int nextnumber(const char* str, char** end, unsigned int* result) {
    errno = 0;
    if (str == NULL || *str == '\0' || !isdigit(*str))
        return -EINVAL;
    *result = (unsigned int)strtoul(str, end, 10);
    if (errno)
        return -errno;
    if (str == *end)
        return -EINVAL;
    return 0;
}

/*
 * Parses string with list of CPU ranges.
 * Returns 0 on success.
 * Returns 1 on error.
 * Returns 2 if fail is set and a cpu number passed in the list doesn't fit
 * into the cpu_set. If fail is not set cpu numbers that do not fit are
 * ignored and 0 is returned instead.
 */
int cpulist_parse(const char* str, cpu_set_t* set, size_t setsize, int fail,
                  int* cpu_list) {
    size_t max = cpuset_nbits(setsize);
    const char *p, *q;
    char* end = NULL;
    int list_idx = 0;

    q = str;
    CPU_ZERO_S(setsize, set);

    while (p = q, q = nexttoken(q, ','), p) {
        unsigned int a; /* beginning of range */
        unsigned int b; /* end of range */
        unsigned int s; /* stride */
        const char *c1, *c2;

        if (nextnumber(p, &end, &a) != 0)
            return 1;
        b = a;
        cpu_list[list_idx++] = a;
        s = 1;
        p = end;

        c1 = nexttoken(p, '-');
        c2 = nexttoken(p, ',');

        if (c1 != NULL && (c2 == NULL || c1 < c2)) {
            if (nextnumber(c1, &end, &b) != 0)
                return 1;
            cpu_list[list_idx++] = b;

            c1 = end && *end ? nexttoken(end, ':') : NULL;

            if (c1 != NULL && (c2 == NULL || c1 < c2)) {
                if (nextnumber(c1, &end, &s) != 0)
                    return 1;
                cpu_list[list_idx++] = s;
                if (s == 0)
                    return 1;
            }
        }

        if (!(a <= b))
            return 1;
        while (a <= b) {
            if (fail && (a >= max))
                return 2;
            CPU_SET_S(a, setsize, set);
            a += s;
        }
    }

    if (end && *end)
        return 1;
    return 0;
}
