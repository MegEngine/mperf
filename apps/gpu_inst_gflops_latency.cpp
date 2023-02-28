#include "mperf_build_config.h"
#include "utils/utils.h"

#if MPERF_WITH_OPENCL
#include "mperf/cpu_affinity.h"
#include "mperf/gpu_march_probe.h"

int main(int ac, char** av) {
    if (ac < 2) {
        fprintf(stderr, "sample usage:\n");
        fprintf(stderr, "./gpu_inst_gflops coreid\n");
        return -1;
    }
    int dev_id = atoi(av[1]);
    if (set_cpu_thread_affinity_spec_core(dev_id)) {
        return -1;
    }

    mperf::gpu_insts_gflops_latency();

    return 0;
}
#else
int main() {
    return 0;
}
#endif