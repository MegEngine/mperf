#include <stdio.h>
#include <string>
#include "mperf/cpu_affinity.h"
#include "mperf/timer.h"
#include "mperf/xpmu/xpmu.h"
#include "mperf_build_config.h"

#if MPERF_WITH_PFM
namespace {
// NHWC
void transpose(const float* src, float* dst, int height, int width,
               int channel) {
    int dst_stride1 = channel * height;
    int dst_stride2 = channel;
    int src_stride1 = channel * width;
    int src_stride2 = channel;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channel; c++) {
                dst[w * dst_stride1 + h * dst_stride2 + c] =
                        src[h * src_stride1 + w * src_stride2 + c];
            }
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "sample usage:\n");
        fprintf(stderr,
                "./cpu_pmu_test core_id iter_num cpu_events(eg. "
                "CYCLES,INSTRUCTIONS)\n");
        return -1;
    }

    int height = 1500;
    int width = 2000;
    int channel = 1;
    int tsize = height * width * channel;
    float* src = new float[tsize];
    float* dst = new float[tsize];
    const int dev_id = atoi(argv[1]);
    const int iter_num = atoi(argv[2]);
    std::string cpu_events = argv[3];
    if (set_cpu_thread_affinity_spec_core(dev_id)) {
        return -1;
    }

    // warmup
    for (int i = 0; i < 5; i++) {
        mperf::Timer cv_t;
        transpose(src, dst, height, width, channel);
        printf("the warm iter %d, and time use %f\n", i, cv_t.get_msecs());
    }

    mperf::GpuCounterSet gpuset;
    mperf::XPMU xpmu(cpu_events, gpuset);
    // Start XPMU once at the beginning of the profiling session
    xpmu.run();

    for (int i = 0; i < iter_num; i++) {
        // Call sample() to sample counters with the frequency you need
        transpose(src, dst, height, width, channel);
        auto cpu_measurements = xpmu.sample().cpu;
        for (size_t j = 0; j < cpu_measurements->size(); j++) {
            auto iter = (*cpu_measurements)[j];
            printf("%s:%lu, ", iter.first.c_str(), iter.second);
        }
        printf("\n");
    }

    // At the end of the profiling session, stop XPMU
    xpmu.stop();

    // Note: avoid cpu to optimized to ignore the transpose operation calls
    printf("the dst[0] %f\n", dst[0]);
    delete[] src;
    delete[] dst;
    return 0;
}
#else
int main() {
    return 0;
}
#endif