#include "mperf/cpu_affinity.h"
#include "mperf/cpu_march_probe.h"
#include "utils/utils.h"

int main(int ac, char** av) {
    int i, j, l;
    int parallel = 1;
    int warmup = 1;
    int repetitions = 10;
    int c;
    int len = 1;

    char* dev_id_list = NULL;
    std::string usage =
            "[-M <len>[K|M]] [-P <parallelism>] [-W "
            "<warmup>] [-N <repetitions>] [-C <core id>]\n";

    while ((c = getopt(ac, av, "M:P:W:N:C:")) != EOF) {
        switch (c) {
            case 'P': {
                parallel = atoi(optarg);
                if (parallel <= 0)
                    mperf_usage(ac, av, usage);
            } break;
            case 'M':
                len = bytes(optarg);
                break;
            case 'W':
                warmup = atoi(optarg);
                break;
            case 'N':
                repetitions = atoi(optarg);
                break;
            case 'C': {
                dev_id_list = optarg;
                cpu_set_t* new_set;
                size_t new_setsize;
                int ncpus = get_max_number_of_cpus();
                if (ncpus <= 0) {
                    printf("cannot determine NR_CPUS.\n");
                    return -1;
                }
                new_set = cpuset_alloc(ncpus, &new_setsize, NULL);
                if (!new_set) {
                    printf("cpuset_alloc failed.\n");
                    return -1;
                }
                int core_list[100];
                cpulist_parse(dev_id_list, new_set, new_setsize, 0, core_list);
                sched_setaffinity(0, new_setsize, new_set);
            } break;
            default:
                mperf_usage(ac, av, usage);
                break;
        }
    }

    return mperf::cpu_stream({parallel, warmup, repetitions}, len);
}