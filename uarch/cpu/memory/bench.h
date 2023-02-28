#pragma once

#include "mperf/cpu_affinity.h"
#include "mperf/cpu_march_probe.h"
#include "mperf/timer.h"

namespace mperf {

typedef void (*benchmp_f)(int iterations, void* cookie);

double benchmp_simple(benchmp_f initialize, benchmp_f benchmark,
                      benchmp_f cleanup, int enough, int parallel, int warmup,
                      int repetitions, void* cookie);

void keep_int(int result);
void keep_pointer(void* result);
void* valloc_internal(size_t size);
}  // namespace mperf