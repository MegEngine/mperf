#include "mperf/cpu_affinity.h"
#include "mperf/cpu_march_probe.h"

int main(int ac, char** av) {
    if (ac < 2) {
        fprintf(stderr, "sample usage:\n");
        fprintf(stderr, "./cpu_inst_gflops coreid\n");
        return -1;
    }
    int dev_id = atoi(av[1]);
    if (set_cpu_thread_affinity_spec_core(dev_id)) {
        return -1;
    }

    mperf::cpu_insts_gflops_latency();

    return 0;
}