#include <stdio.h>
#include "mperf_build_config.h"

#if MPERF_WITH_ADRENO
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

void execute_compute(const mperf::GpuCounterSet& group_events, float* buf,
                     int global_size, int local_size, float _A) {
    std::string kernel_file = "./data/compute_float4.cl";

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

    cl_kernel kernel = clCreateKernel(program, "compute", &err);
    cl_mem d_buf;

    d_buf = env.malloc_buffer(CL_MEM_READ_WRITE, sizeof(float));
    env.host_copy(buf, d_buf, sizeof(float));
    printf("-----------compute-----------\n");

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_buf);
    clSetKernelArg(kernel, 1, sizeof(float), &_A);

    size_t max_workgroup_size = env.get_kernel_max_work_group_size(kernel);

    NDRange global(global_size);
    auto init_local = [&]() -> NDRange {
        if (local_size == 0)
            return NDRange();
        else
            return NDRange(local_size);
    };
    NDRange local = init_local();

    for (int warmup_iter = 0; warmup_iter < 5; ++warmup_iter) {
        env.execute_kernel(kernel, global, local);
    }

    mperf::GpuCounterSet gpuset = group_events;
    mperf::XPMU xpmu(gpuset);
    double kern_time;
    uint64_t kern_time_in_nano_seconds;
    xpmu.run();
    kern_time = env.execute_kernel(kernel, global, local);
    kern_time_in_nano_seconds = kern_time * 1e3;
    // Call sample() to sample counters with the frequency you need
    xpmu.gpu_profiler()->set_kern_time(kern_time_in_nano_seconds);
    printf("kern_time_in_nano_seconds:%f\n", kern_time_in_nano_seconds * 1e-6);
    auto gpu_measurements = xpmu.sample().gpu;
    for (size_t j = 0; j < gpu_measurements->size(); j++) {
        auto iter = (*gpu_measurements)[j];
        printf("%s:%f\n", iter.first.c_str(), iter.second.get<double>());
    }

    // At the end of the profiling session, stop XPMU
    xpmu.stop();

    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

int main(int argc, char* argv[]) {
    int global_size = 32 * 1024 * 1024;
    int local_size = 256;
    float _A = 15;

    std::string gpu_group_and_events =
            "EVENT_GROUP_UCHE,"
            "UCHE_VBIF_READ_BEATS_SP,"
            "UCHE_READ_REQUESTS_SP,"
            "UCHE_VBIF_READ_BEATS_TP,"
            "UCHE_READ_REQUESTS_TP;"

            "EVENT_GROUP_SP,"
            "SP_BUSY_CYCLES,"
            "SP_NON_EXECUTION_CYCLES,"
            "SP_FS_STAGE_FULL_ALU_INSTRUCTIONS,"
            "SP_VS_STAGE_FULL_ALU_INSTRUCTIONS,"
            "SP_CS_INSTRUCTIONS,"
            "SP_ICL1_MISSES,"
            "SP_ICL1_REQUESTS,"
            "SP_LM_LOAD_INSTRUCTIONS,"
            "SP_LM_STORE_INSTRUCTIONS,"
            "SP_LM_ATOMICS,"
            "SP_GM_LOAD_INSTRUCTIONS,"
            "SP_GM_STORE_INSTRUCTIONS,"
            "SP_GM_ATOMICS,"
            "SP_STALL_CYCLES_VPC,"
            "SP_STALL_CYCLES_TP,"
            "SP_STALL_CYCLES_UCHE,"
            "SP_STALL_CYCLES_RB;"

            "EVENT_GROUP_TP,"
            "TP_BUSY_CYCLES,"
            "TP_L1_CACHELINE_MISSES,"
            "TP_L1_CACHELINE_REQUESTS,"
            "TP_BUSY_CYCLES,"
            "TP_STALL_CYCLES_UCHE,"
            "TP_LATENCY_CYCLES,"
            "TP_STARVE_CYCLES_SP,"
            "TP_STARVE_CYCLES_UCHE,"
            "TP_OUTPUT_PIXELS_POINT,"
            "TP_OUTPUT_PIXELS_BILINEA,"
            "TP_OUTPUT_PIXELS_MIP,"
            "TP_OUTPUT_PIXELS_ANISO,"
            "TP_OUTPUT_PIXELS_ZERO_LOD;"

            "EVENT_GROUP_CUSTOM,"
            "GFLOPs,"
            "GBPs,"
            "GPUCycles,"
            "ShaderComputeCycles,"
            "ShaderLoadStoreCycles,"
            "ShaderTextureCycles,"
            "AluUtil,"
            "LoadStoreUtil,"
            "TextureUtil,"
            "FullAluRatio,"
            "ShaderBusyRatio,"
            "ShaderStalledRatio,"
            "TexturePipesBusyRatio,"
            "TextureL1MissRatio,"
            "TextureL2ReadMissRatio,"
            "L2ReadMissRatio,"
            "InstructionCacheMissRatio,"
            "L1TextureMissPerPixel,";

    float* __restrict__ sglbuf = (float*)memalign(32, sizeof(float));
    execute_compute(gpu_group_and_events, sglbuf, global_size, local_size, _A);
    free(sglbuf);

    return 0;
}
#else
int main() {
    return 0;
}
#endif
