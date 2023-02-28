#include <stdint.h>
#include <stdio.h>
#include "mperf_build_config.h"

#if MPERF_WITH_OPENCL
#include "mperf/gpu_march_probe.h"
using namespace mperf;

int main() {
    printf("gpu micro arch probe info:\n");
    printf("max_reg_num_per_thread: %d\n", gpu_max_reg_num_per_thread());
    printf("warp_size: %d\n", gpu_warp_size());
    printf("unified_cacheline_size: %d Bytes\n", gpu_unified_cacheline_size());
    printf("gpu_texture_cacheline_size: %d Bytes\n ",
           gpu_texture_cacheline_size());
    return 0;
}
#else
int main() {
    return 0;
}
#endif