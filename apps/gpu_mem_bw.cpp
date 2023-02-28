#include <stdint.h>
#include <stdio.h>
#include "mperf_build_config.h"

#if MPERF_WITH_OPENCL
#include "mperf/gpu_march_probe.h"
using namespace mperf;

int main() {
    printf("buffer bandwidth: %f GBPS\n", gpu_mem_bw());
    printf("texture cache bandwidth: %f GBPS\n", gpu_texture_cache_bw());
    printf("local memory bandwidth:%f GBPS\n", gpu_local_memory_bw());
    return 0;
}
#else
int main() {
    return 0;
}
#endif