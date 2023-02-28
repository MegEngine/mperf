#include <stdio.h>
#include "mperf_build_config.h"

#if MPERF_WITH_MALI
#include <sys/resource.h>
#include <fstream>
#include <iostream>
#include <string>
#include "mperf/opencl_driver.h"
#include "mperf/timer.h"
#include "mperf/xpmu/xpmu.h"

#define PERF_MEMORY_MAX 32 * 1024 * 1024  // 32M
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
    for (int i = 0; i < 5; ++i) {
        env.execute_kernel(kernel, global, local);
    }

    mperf::GpuCounterSet gpuset = group_events;
    mperf::XPMU xpmu(gpuset);

    clSetKernelArg(kernel, 1, sizeof(int), &iter_num);

    double kern_time;
    uint64_t kern_time_in_nano_seconds;
    xpmu.run();
    kern_time = env.execute_kernel(kernel, global, local);

    kern_time_in_nano_seconds = kern_time * 1e3;
    // Note: you need call set_kern_time interface manually before sample,
    // when some counters need accurate kern_time, like GFLOPs, GBPs
    xpmu.gpu_profiler()->set_kern_time(kern_time_in_nano_seconds);

    auto gpu_measurements = xpmu.sample().gpu;

    for (auto iter = gpu_measurements->begin(); iter != gpu_measurements->end();
         iter++) {
        // printf("%d:%f\n", iter->first, iter->second.get<double>());
        printf("%s:%f\n",
               counter_enum_to_names.find(iter->first)->second.c_str(),
               iter->second.get<double>());
    }

    // At the end of the profiling session, stop XPMU
    xpmu.stop();

    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

using GpuCounter = mperf::GpuCounter;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "sample usage:\n");
        fprintf(stderr, "./cpu_pmu_test global_size local_size iter_num\n");
        return -1;
    }

    int global_size = atoi(argv[1]);
    int local_size = atoi(argv[2]);
    // No args, let the OpenCL runtime decide
    // global_size = 0;
    // local_size = 0;
    int iter_num = atoi(argv[3]);

    const mperf::GpuCounterSet user_counters{
            GpuCounter::GpuCycles,
            GpuCounter::VertexComputeCycles,
            GpuCounter::FragmentCycles,
            GpuCounter::TilerCycles,
            GpuCounter::CacheReadLookups,
            GpuCounter::CacheWriteLookups,
            GpuCounter::ExternalMemoryReadAccesses,
            GpuCounter::ExternalMemoryWriteAccesses,
            GpuCounter::ExternalMemoryReadStalls,
            GpuCounter::ExternalMemoryWriteStalls,
            GpuCounter::ExternalMemoryReadBytes,
            GpuCounter::ExternalMemoryWriteBytes,
            GpuCounter::Instructions,
            GpuCounter::ShaderTextureCycles,
            GpuCounter::CacheReadLookups,

            mperf::GpuCounter::ComputeWarps,
            mperf::GpuCounter::WarpRegSize64,
            mperf::GpuCounter::LoadStoreReadFull,
            mperf::GpuCounter::LoadStoreReadPartial,
            mperf::GpuCounter::LoadStoreWriteFull,
            mperf::GpuCounter::LoadStoreWritePartial,
            mperf::GpuCounter::LscRdBeats,
            mperf::GpuCounter::LscRdExtBeats,

            mperf::GpuCounter::AluUtil,
            mperf::GpuCounter::LoadStoreUtil,
            mperf::GpuCounter::PartialReadRatio,
            mperf::GpuCounter::PartialWriteRatio,
            mperf::GpuCounter::GFLOPs,
            mperf::GpuCounter::GBPs,
            mperf::GpuCounter::L2ReadMissRatio,
            mperf::GpuCounter::L2WriteMissRatio,
            mperf::GpuCounter::FullRegWarpRatio,
            mperf::GpuCounter::WarpDivergenceRatio,
    };

    uint64_t PSIZE = PERF_MEMORY_MAX;
    float* __restrict__ sglbuf = (float*)memalign(4096, PSIZE);

    execute(user_counters, PSIZE, sglbuf, global_size, local_size, iter_num);

    free(sglbuf);

    return 0;
}
#else
int main() {
    return 0;
}
#endif
