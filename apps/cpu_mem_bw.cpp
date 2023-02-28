/*
 * Usage: mperf_cpu_mem_bw [-P <parallelism>] [-W <warmup>] [-N <repetitions>]
 * size op op: srd swr scp fwr frd fcp bzero bcopy
 */
#include "mperf/cpu_affinity.h"
#include "mperf/cpu_march_probe.h"
#include "utils/utils.h"

int main(int ac, char** av) {
    int parallel = 1;
    int warmup = 1;
    int repetitions = 10;
    int dev_id_mask = 0;
    char* dev_id_list = NULL;
    size_t nbytes;
    int core_list[100];
    memset(core_list, -1, 100 * sizeof(int));

    // size is the actual amount of data (bytes, TYPE independent)
    std::string usage =
            "[-P <parallelism>] [-W <warmup>] [-N <repetitions>] [-C <core "
            "id1[,id2,...]>] [-M <core mask>] <size> what "
            "[conflict]\nwhat: srd swr scp fwr frd frdwr fcp bzero bcopy triad "
            "rnd_rd rnd_wr add1 add2 mla\n<size> "
            "must be larger than 512B";

    int c;
    while ((c = getopt(ac, av, "P:W:N:C:M:")) != EOF) {
        switch (c) {
            case 'P': {
                parallel = atoi(optarg);
                if (parallel <= 0)
                    mperf_usage(ac, av, usage);
            } break;
            case 'W': {
                warmup = atoi(optarg);
            } break;
            case 'N': {
                repetitions = atoi(optarg);
            } break;
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
                cpulist_parse(dev_id_list, new_set, new_setsize, 0, core_list);
                sched_setaffinity(0, new_setsize, new_set);
            } break;
            case 'M': {
                dev_id_mask = atoi(optarg);
                set_cpu_thread_affinity_mask(dev_id_mask);
            } break;
            default: { mperf_usage(ac, av, usage); } break;
        }
    }

    int aligned = 0;
    if (optind + 3 == ac) {  // conflict
        aligned = 1;
    } else if (optind + 2 != ac) {
        mperf_usage(ac, av, usage);
    }

    nbytes = bytes(av[optind]);
    if (nbytes < 512) { /* this is the number of bytes in the loop */
        mperf_usage(ac, av, usage);
    }

    char* mop = av[optind + 1];

    char c_str[200];
    int c_id = 0;
    for (int i = 0; i < 200; ++i) {
        if (core_list[i] == -1) {
            break;
        }
        sprintf(&c_str[c_id++], "%d", core_list[i]);
        c_str[c_id++] = '_';
    }

    mperf::cpu_mem_bw({parallel, warmup, repetitions}, aligned, nbytes, mop,
                      c_str);

    return 0;
}
