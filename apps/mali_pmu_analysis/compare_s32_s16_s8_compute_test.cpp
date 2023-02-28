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

using namespace mperf;

void execute(const mperf::GpuCounterSet& group_events, int* buf,
             int global_size, int local_size, int _A) {
    std::string kernel_file =
            "./data/compare_s32_s16_s8_compute_test_kernel.cl";

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

    cl_kernel kernel[3] = {clCreateKernel(program, "compute_s32", &err),
                           clCreateKernel(program, "compute_s16", &err),
                           clCreateKernel(program, "compute_s8", &err)};
    cl_mem d_buf;
    for (int i = 0; i < 3; i++) {
        if (i == 0) {
            d_buf = env.malloc_buffer(CL_MEM_READ_WRITE, sizeof(int));
            env.host_copy(buf, d_buf, sizeof(int));
            printf("-----------int-----------\n");
        } else if (i == 1) {
            d_buf = env.malloc_buffer(CL_MEM_READ_WRITE, sizeof(short));
            env.host_copy((short*)buf, d_buf, sizeof(short));
            printf("----------short-----------\n");
        } else {
            d_buf = env.malloc_buffer(CL_MEM_READ_WRITE, sizeof(char));
            env.host_copy((char*)buf, d_buf, sizeof(char));
            printf("-----------char-----------\n");
        }

        clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &d_buf);
        clSetKernelArg(kernel[i], 1, sizeof(int), &_A);

        size_t max_workgroup_size =
                env.get_kernel_max_work_group_size(kernel[i]);
        if (global_size == 0) {
            global_size = 1;
        }
        NDRange global(global_size);
        auto init_local = [&]() -> NDRange {
            if (local_size == 0)
                return NDRange();
            else
                return NDRange(local_size);
        };
        NDRange local = init_local();

        mperf::GpuCounterSet gpuset = group_events;
        mperf::XPMU xpmu(gpuset);
        for (int warmup_iter = 0; warmup_iter < 5; ++warmup_iter) {
            env.execute_kernel(kernel[i], global, local);
        }

        double kern_time;
        uint64_t kern_time_in_nano_seconds;
        xpmu.run();
        kern_time = env.execute_kernel(kernel[i], global, local);
        kern_time_in_nano_seconds = kern_time * 1e3;
        // Call sample() to sample counters with the frequency you need
        xpmu.gpu_profiler()->set_dtype_sz(size_t(4 / pow(2, i)));
        xpmu.gpu_profiler()->set_kern_time(kern_time_in_nano_seconds);

        auto gpu_measurements = xpmu.sample().gpu;
        for (auto iter = gpu_measurements->begin();
             iter != gpu_measurements->end(); iter++) {
            printf("%s:%f\n",
                   counter_enum_to_names.find(iter->first)->second.c_str(),
                   iter->second.get<double>());
        }

        // At the end of the profiling session, stop XPMU
        xpmu.stop();

        clReleaseKernel(kernel[i]);
    }
    clReleaseProgram(program);
}

using GpuCounter = mperf::GpuCounter;

int main(int argc, char* argv[]) {
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

    int global_size = 1024 * 1024;
    int local_size = 256;
    int _A = 15;
    int* __restrict__ sglbuf = (int*)memalign(32, sizeof(int));
    execute(user_counters, sglbuf, global_size, local_size, _A);

    return 0;
}
#else
int main() {
    return 0;
}
#endif
