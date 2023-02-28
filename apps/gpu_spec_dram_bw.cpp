#include <stdint.h>
#include <stdio.h>
#include "mperf_build_config.h"

#if MPERF_WITH_OPENCL
#include "mperf/gpu_march_probe.h"
using namespace mperf;

int main() {
    printf("global memory bandwidth:%f\n", gpu_global_memory_bw());
    return 0;
}
#else
int main() {
    return 0;
}
#endif