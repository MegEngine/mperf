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
             float* buf, float* w_buf, int global_size, int local_size,
             const char* kernel_name, int num) {
    std::string file_name = "./data/compare_vload4_vload16_test_kernel.cl";

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
    cl_program program = env.build_program_from_source(load_program(file_name));
    cl_int err = 0;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);

    uint64_t nsize = PSIZE;
    nsize = nsize & (~(PERF_ALIGN - 1));
    nsize = nsize / sizeof(float);

    cl_mem d_buf1 = env.malloc_buffer(CL_MEM_READ_WRITE, sizeof(float) * nsize);
    float value = 2.0;
    for (uint64_t i = 0; i < nsize; ++i) {
        buf[i] = value;
    }
    env.host_copy(buf, d_buf1, sizeof(float) * nsize);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_buf1);

    cl_mem d_buf2 = env.malloc_buffer(CL_MEM_READ_WRITE, sizeof(float) * nsize);
    env.host_copy(w_buf, d_buf2, sizeof(float) * nsize);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_buf2);

    // size_t max_workgroup_size = env.get_kernel_max_work_group_size(kernel1);
    NDRange global(global_size);

    auto init_local = [&]() -> NDRange {
        if (local_size == 0)
            return NDRange();
        else
            return NDRange(local_size);
    };
    NDRange local = init_local();

    // warmup
    for (int j = 0; j < 5; ++j) {
        env.execute_kernel(kernel, global, local);
    }

    mperf::GpuCounterSet gpuset = group_events;
    mperf::XPMU xpmu(gpuset);

    double kern_time;
    uint64_t kern_time_in_nano_seconds;

    printf("----------vload%d----------\n", num);
    xpmu.run();
    kern_time = env.execute_kernel(kernel, global, local);

    kern_time_in_nano_seconds = kern_time * 1e3;
    // Note: you need call set_kern_time interface manually before sample,
    // when some counters need accurate kern_time, like GFLOPs, GBPs
    xpmu.gpu_profiler()->set_kern_time(kern_time_in_nano_seconds);
    auto gpu_measurements = xpmu.sample().gpu;

    for (auto iter = gpu_measurements->begin(); iter != gpu_measurements->end();
         iter++) {
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
    int global_size1 = PERF_MEMORY_MAX / 16 / 4 / 4;
    int global_size2 = PERF_MEMORY_MAX / 16 / 4 / 16;
    int local_size = 256;

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
    float* __restrict__ w_sglbuf = (float*)memalign(4096, PSIZE);
    execute(user_counters, PSIZE, sglbuf, w_sglbuf, global_size1, local_size,
            "global_bandwidth_v4_local_offset", 4);
    execute(user_counters, PSIZE, sglbuf, w_sglbuf, global_size2, local_size,
            "global_bandwidth_v16_local_offset", 16);
    free(sglbuf);
    free(w_sglbuf);

    return 0;
}
#else
int main() {
    return 0;
}
#endif
