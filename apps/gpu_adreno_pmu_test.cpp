#include <stdio.h>
#include "mperf_build_config.h"

#if MPERF_WITH_ADRENO
#include <sys/resource.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "mperf/opencl_driver.h"
#include "mperf/xpmu/xpmu.h"

#define PERF_ALIGN 64
using namespace mperf;

void execute(const mperf::GpuCounterSet& group_events, uint64_t PSIZE,
             float* buf, int global_size, int local_size, int iter_num) {
    std::string kernel_file = "./data/pmu_test_kernel.cl";

    auto env = OpenCLEnv::instance();

    auto load_program = [](const std::string& fname) -> std::string {
        std::ifstream stream(fname.c_str());
        if (!stream.is_open()) {
            std::cout << "Cannot open file: " << fname << std::endl;
            exit(1);
        }

        return std::string(std::istreambuf_iterator<char>(stream),
                           (std::istreambuf_iterator<char>()));
    };
    cl_program program =
            env.build_program_from_source(load_program(kernel_file));
    cl_int err = 0;
    cl_kernel kernel = clCreateKernel(program, "pmu_test_kernel", &err);

    uint64_t nsize = PSIZE;
    nsize = nsize & (~(PERF_ALIGN - 1));
    nsize = nsize / sizeof(float);

    int params[2];
    cl_mem d_params = env.malloc_buffer(CL_MEM_READ_WRITE, sizeof(params));
    cl_mem d_buf = env.malloc_buffer(CL_MEM_READ_WRITE, sizeof(float) * nsize);

    double value = -1.0;
    for (uint64_t i = 0; i < nsize; ++i) {
        buf[i] = value;
    }
    env.host_copy(buf, d_buf, sizeof(float) * nsize);

    int warm_iter = 5;
    clSetKernelArg(kernel, 0, sizeof(int), &nsize);
    clSetKernelArg(kernel, 1, sizeof(int), &warm_iter);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_params);

    size_t max_workgroup_size = env.get_kernel_max_work_group_size(kernel);
    if (global_size == 0) {
        global_size = nsize;
    }
    NDRange global(global_size);
    auto init_local = [&]() -> NDRange {
        if (local_size == 0)
            return NDRange();
        else
            return NDRange(local_size);
    };
    NDRange local = init_local();

    // warmup
    env.execute_kernel(kernel, global, local);

    mperf::GpuCounterSet gpuset = group_events;
    mperf::XPMU h(gpuset);

    clSetKernelArg(kernel, 1, sizeof(int), &iter_num);

    double kern_time;
    uint64_t kern_time_in_nano_seconds;
    h.run();
    kern_time = env.execute_kernel(kernel, global, local);

    kern_time_in_nano_seconds = kern_time * 1e3;
    // Note: you need call set_kern_time interface manually before sample,
    // when some counters need accurate kern_time, like GFLOPs, GBPs
    h.gpu_profiler()->set_kern_time(kern_time_in_nano_seconds);

    // Call sample() to sample counters with the frequency you need
    auto gpu_measurements = h.sample().gpu;
    for (size_t j = 0; j < gpu_measurements->size(); j++) {
        auto iter = (*gpu_measurements)[j];
        printf("%s:%f\n", iter.first.c_str(), iter.second.get<double>());
    }

    // At the end of the profiling session, stop XPMU
    h.stop();

    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

#define PERF_MEMORY_MAX 32 * 1024 * 1024  // 32M
int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "sample usage:\n");
        fprintf(stderr, "./cpu_pmu_test global_size local_size iter_num\n");
        return -1;
    }

    int global_size = atoi(argv[1]);
    int local_size = atoi(argv[2]);
    int iter_num = atoi(argv[3]);

    std::string gpu_group_and_events =
            "EVENT_GROUP_UCHE,UCHE_VBIF_READ_BEATS_SP,UCHE_VBIF_"
            "READ_BEATS_TP;EVENT_GROUP_SP,SP_CS_INSTRUCTIONS,SP_"
            "GM_STORE_INSTRUCTIONS;EVENT_GROUP_CUSTOM,GFLOPs,GBPs";

    uint64_t PSIZE = PERF_MEMORY_MAX;
    float* __restrict__ sglbuf = (float*)memalign(4096, PSIZE);

    execute(gpu_group_and_events, PSIZE, sglbuf, global_size, local_size,
            iter_num);

    free(sglbuf);

    return 0;
}
#else
int main() {
    return 0;
}
#endif