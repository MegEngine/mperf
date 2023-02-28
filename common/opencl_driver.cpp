/**
 * MegPeaK is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied.
 * ---------------------------------------------------------------------------
 * \file common/opencl_driver.cpp
 *
 * Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2022-2023 Megvii Inc. All rights
 * reserved.
 *
 * ---------------------------------------------------------------------------
 */

#include "mperf/opencl_driver.h"
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "mperf/utils.h"
#include "mperf_build_config.h"

using namespace mperf;
using cl_int1 = cl_int;

namespace {

// clang-format off
static std::unordered_map<std::string, const char*> PROGRAMS = {
        {"compute_dp_kernels",
#include "compute_dp_kernels.h"
        },
        {"compute_sp_kernels",
#include "compute_sp_kernels.h"
        },
        {"compute_hp_kernels",
#include "compute_hp_kernels.h"
        },
        {"compute_s32_kernels",
#include "compute_s32_kernels.h"
        },
        {"compute_s16_kernels",
#include "compute_s16_kernels.h"
        },
        {"compute_s8_kernels",
#include "compute_s8_kernels.h"
        },
        {"global_memory_bandwidth",
#include "global_bandwidth_kernels.h"
        },
        {"local_memory_bandwidth",
#include "local_memory_bandwidth.h"
        }
};
// clang-format on

std::string to_string(const std::vector<char> str) {
    std::string ret(str.begin(), str.end());
    return ret;
}

template <typename T>
void populate(std::vector<T>& arr) {
    for (size_t i = 0; i < arr.size(); i++) {
        arr[i] = i;
    }
}

int round_to_power2(int number) {
    return pow(2, floorf(::log2f(number)));
}

NDRange get_global_size_divdown(const NDRange& gs, const NDRange& ls) {
    NDRange ret = gs;
    for (size_t i = 0; i < gs.dimension(); i++) {
        size_t local_size = ls.get()[i];
        size_t origin = gs.get()[i];
        ret.get()[i] = origin / local_size * local_size;
    }
    return ret;
}

}  // namespace

const char* mperf::get_error_string(cl_int error) {
#define ERROR(_err) \
    case _err:      \
        return #_err;

    switch (error) {
        ERROR(CL_SUCCESS)
        ERROR(CL_DEVICE_NOT_FOUND)
        ERROR(CL_DEVICE_NOT_AVAILABLE)
        ERROR(CL_COMPILER_NOT_AVAILABLE)
        ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE)
        ERROR(CL_OUT_OF_RESOURCES)
        ERROR(CL_OUT_OF_HOST_MEMORY)
        ERROR(CL_PROFILING_INFO_NOT_AVAILABLE)
        ERROR(CL_MEM_COPY_OVERLAP)
        ERROR(CL_IMAGE_FORMAT_MISMATCH)
        ERROR(CL_IMAGE_FORMAT_NOT_SUPPORTED)
        ERROR(CL_BUILD_PROGRAM_FAILURE)
        ERROR(CL_MAP_FAILURE)
        ERROR(CL_MISALIGNED_SUB_BUFFER_OFFSET)
        ERROR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
        ERROR(CL_COMPILE_PROGRAM_FAILURE)
        ERROR(CL_LINKER_NOT_AVAILABLE)
        ERROR(CL_LINK_PROGRAM_FAILURE)
        ERROR(CL_DEVICE_PARTITION_FAILED)
        ERROR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
        ERROR(CL_INVALID_VALUE)
        ERROR(CL_INVALID_DEVICE_TYPE)
        ERROR(CL_INVALID_PLATFORM)
        ERROR(CL_INVALID_DEVICE)
        ERROR(CL_INVALID_CONTEXT)
        ERROR(CL_INVALID_QUEUE_PROPERTIES)
        ERROR(CL_INVALID_COMMAND_QUEUE)
        ERROR(CL_INVALID_HOST_PTR)
        ERROR(CL_INVALID_MEM_OBJECT)
        ERROR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        ERROR(CL_INVALID_IMAGE_SIZE)
        ERROR(CL_INVALID_SAMPLER)
        ERROR(CL_INVALID_BINARY)
        ERROR(CL_INVALID_BUILD_OPTIONS)
        ERROR(CL_INVALID_PROGRAM)
        ERROR(CL_INVALID_PROGRAM_EXECUTABLE)
        ERROR(CL_INVALID_KERNEL_NAME)
        ERROR(CL_INVALID_KERNEL_DEFINITION)
        ERROR(CL_INVALID_KERNEL)
        ERROR(CL_INVALID_ARG_INDEX)
        ERROR(CL_INVALID_ARG_VALUE)
        ERROR(CL_INVALID_ARG_SIZE)
        ERROR(CL_INVALID_KERNEL_ARGS)
        ERROR(CL_INVALID_WORK_DIMENSION)
        ERROR(CL_INVALID_WORK_GROUP_SIZE)
        ERROR(CL_INVALID_WORK_ITEM_SIZE)
        ERROR(CL_INVALID_GLOBAL_OFFSET)
        ERROR(CL_INVALID_EVENT_WAIT_LIST)
        ERROR(CL_INVALID_EVENT)
        ERROR(CL_INVALID_OPERATION)
        ERROR(CL_INVALID_GL_OBJECT)
        ERROR(CL_INVALID_BUFFER_SIZE)
        ERROR(CL_INVALID_MIP_LEVEL)
        ERROR(CL_INVALID_GLOBAL_WORK_SIZE)
        ERROR(CL_INVALID_PROPERTY)
        ERROR(CL_INVALID_IMAGE_DESCRIPTOR)
        ERROR(CL_INVALID_COMPILER_OPTIONS)
        ERROR(CL_INVALID_LINKER_OPTIONS)
        ERROR(CL_INVALID_DEVICE_PARTITION_COUNT)

        default:
            return "unknown error";
    }
}

////////////////////////////// NDRange /////////////////////////////////
size_t NDRange::operator[](size_t idx) {
    mperf_assert(idx < m_dimension, "invalid index: %zu expected < %zu", idx,
                 m_dimension);
    return m_size[idx];
}

bool NDRange::operator<(const NDRange& rhs) const {
    return std::tie(m_dimension, m_size[0], m_size[1], m_size[2]) <
           std::tie(rhs.m_dimension, rhs.m_size[0], rhs.m_size[1],
                    rhs.m_size[2]);
}

//////////////////////////////////// Runner //////////////////////////////////
float Runner::run_kernel(cl_kernel kern, const NDRange& global_size,
                         const NDRange& local_size, size_t iters) {
    mperf_assert(global_size.dimension() == local_size.dimension() &&
                         local_size.dimension() == 1,
                 "only support 1-dim ndrange, got: %zu",
                 local_size.dimension());
    auto gs = get_global_size_divdown(global_size, local_size);
    float used = 0;
    opencl_check(clEnqueueNDRangeKernel(
            m_env.m_queue, kern, global_size.dimension(), nullptr, gs.get(),
            local_size.get(), 0, nullptr, nullptr));
    clFinish(m_env.m_queue);

    cl_event ev;
    for (size_t i = 0; i < iters; i++) {
        opencl_check(clEnqueueNDRangeKernel(
                m_env.m_queue, kern, global_size.dimension(), nullptr, gs.get(),
                local_size.get(), 0, nullptr, &ev));
        opencl_check(clWaitForEvents(1, &ev));
        cl_ulong start, end;
        opencl_check(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                             sizeof(cl_ulong), &start,
                                             nullptr));
        opencl_check(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                             sizeof(cl_ulong), &end, nullptr));

        used += (end - start) / 1000.0;
    }
    opencl_check(clReleaseEvent(ev));
    return used / iters;
}

///////////////////////// LocalMemRunner /////////////////////////////////////
LocalMemRunner::LocalMemRunner(const OpenCLEnv& env) : Runner(env) {}

float LocalMemRunner::run(int) {
    // This assumes that local memory must be greater than 1K.
    constexpr static size_t nr_elems = 1024 * 1024;
    constexpr static size_t iters = 50;
    size_t local_size = m_env.m_dev_info.nthread_logic;

    std::vector<int> elems(nr_elems, 0);
    populate(elems);
    printf("Local memory bandwidth (GBPS)\n");

    cl_int err;
    cl_mem src = clCreateBuffer(m_env.m_ctx, CL_MEM_READ_WRITE,
                                nr_elems * sizeof(int), nullptr, &err);
    opencl_check(err);
    opencl_check(clEnqueueWriteBuffer(m_env.m_queue, src, CL_TRUE, 0,
                                      nr_elems * sizeof(int), elems.data(), 0,
                                      nullptr, nullptr));

    cl_program program =
            m_env.build_program_from_fname("local_memory_bandwidth");

    float ret = 0.0;
#define cb(i)                                                                  \
    auto kernel_v##i =                                                         \
            clCreateKernel(program, "krn_localmem_juggling_int" #i, &err);     \
    local_size = std::min<size_t>(                                             \
            local_size, m_env.get_kernel_max_work_group_size(kernel_v##i));    \
    opencl_check(err);                                                         \
    err = clSetKernelArg(kernel_v##i, 0, sizeof(cl_mem), &src);                \
    err |= clSetKernelArg(kernel_v##i, 1, local_size * sizeof(cl_int##i) * 6,  \
                          nullptr);                                            \
    opencl_check(err);                                                         \
    ret = std::max<float>(ret, run_step(kernel_v##i, nr_elems / i, local_size, \
                                        nr_elems, iters, "int" #i));           \
    clReleaseKernel(kernel_v##i);

    cb(1);
    cb(2);
    cb(4);
#undef cb
    clReleaseProgram(program);

    return ret;
}

float LocalMemRunner::run_step(cl_kernel kernel, size_t global_size,
                               size_t local_size, size_t nr_elems, size_t iters,
                               const std::string& tag) {
    float used = run_kernel(kernel, global_size, local_size, iters);

    constexpr size_t TOTAL_ITERATIONS = 16;
    float memory =
            (6LL + 4 * 5 * TOTAL_ITERATIONS + 6) * nr_elems * sizeof(int);
    float gbps = memory / used / 1e3;
    printf("    %s : %f\n", tag.c_str(), gbps);

    return gbps;
}

///////////////////////// GlobalMemRunner /////////////////////////////////////
GlobalMemRunner::GlobalMemRunner(const OpenCLEnv& env) : Runner(env) {}

float GlobalMemRunner::run(int) {
    constexpr static size_t FETCH_PER_WI = 16;
    constexpr static size_t iters = 50;
    int nr_elems = m_env.m_dev_info.mem_obj_max_alloc_size / sizeof(float) / 2;
    nr_elems = round_to_power2(nr_elems);

    std::vector<float> elems(nr_elems, 0);
    populate(elems);
    printf("Global memory bandwidth (GBPS)\n");
    size_t local_size = m_env.m_dev_info.nthread_logic;

    cl_int err;
    cl_mem src = clCreateBuffer(m_env.m_ctx, CL_MEM_READ_ONLY,
                                nr_elems * sizeof(float), nullptr, &err);
    opencl_check(err);
    cl_mem dst = clCreateBuffer(m_env.m_ctx, CL_MEM_WRITE_ONLY,
                                nr_elems * sizeof(float), nullptr, &err);
    opencl_check(err);
    opencl_check(clEnqueueWriteBuffer(m_env.m_queue, src, CL_TRUE, 0,
                                      nr_elems * sizeof(float), elems.data(), 0,
                                      nullptr, nullptr));

    cl_program program =
            m_env.build_program_from_fname("global_memory_bandwidth");

    float ret = 0.0;
#define cb(i)                                                                \
    auto kernel_v##i##_global = clCreateKernel(                              \
            program, "global_bandwidth_v" #i "_global_offset", &err);        \
    opencl_check(err);                                                       \
    local_size = std::min<size_t>(                                           \
            local_size,                                                      \
            m_env.get_kernel_max_work_group_size(kernel_v##i##_global));     \
    err = clSetKernelArg(kernel_v##i##_global, 0, sizeof(cl_mem), &src);     \
    err |= clSetKernelArg(kernel_v##i##_global, 1, sizeof(cl_mem), &dst);    \
    opencl_check(err);                                                       \
    auto kernel_v##i##_local = clCreateKernel(                               \
            program, "global_bandwidth_v" #i "_local_offset", &err);         \
    opencl_check(err);                                                       \
    local_size = std::min<size_t>(                                           \
            local_size,                                                      \
            m_env.get_kernel_max_work_group_size(kernel_v##i##_local));      \
    err = clSetKernelArg(kernel_v##i##_local, 0, sizeof(cl_mem), &src);      \
    err |= clSetKernelArg(kernel_v##i##_local, 1, sizeof(cl_mem), &dst);     \
    opencl_check(err);                                                       \
    ret = std::max<float>(                                                   \
            ret, run_step(kernel_v##i##_global, kernel_v##i##_local,         \
                          nr_elems / i / FETCH_PER_WI, local_size, nr_elems, \
                          iters, "float" #i));                               \
    clReleaseKernel(kernel_v##i##_global);                                   \
    clReleaseKernel(kernel_v##i##_local);

    cb(1);
    cb(2);
    cb(4);
    cb(8);
    cb(16);
#undef cb
    clReleaseProgram(program);

    return ret;
}

float GlobalMemRunner::run_step(cl_kernel kernel_go, cl_kernel kernel_lo,
                                size_t global_size, size_t local_size,
                                size_t nr_elems, size_t iters,
                                const std::string& tag) {
    float used_go = run_kernel(kernel_go, global_size, local_size, iters);
    float used_lo = run_kernel(kernel_lo, global_size, local_size, iters);

    float used = std::min<float>(used_lo, used_go);
    float gbps = static_cast<float>(nr_elems) * sizeof(float) / used / 1e3;
    printf("    %s : %f\n", tag.c_str(), gbps);

    return gbps;
}

///////////////////////// CompRunner /////////////////////////////////////
template <typename T>
CompRunner<T>::CompRunner(const OpenCLEnv& env) : Runner(env) {}

template <typename T>
float CompRunner<T>::run(int mod) {
    std::string src_tag, kernel_tag, tag;
    std::string ext_compile_options = "";
    if (std::is_same<T, cl_int>::value) {
        src_tag = "compute_s32_kernels";
        kernel_tag = "compute_s32_v";
        tag = "int";

        switch (mod) {
            case 0: {
                ext_compile_options = " -DISA_MAD24=1 ";
                tag += "(mad24)";
            } break;
            case 1: {
                tag += "(native mul/add)";
            }
            default:
                break;
        }
    } else if (std::is_same<T, cl_short>::value) {
        src_tag = "compute_s16_kernels";
        kernel_tag = "compute_s16_v";
        tag = "short";
    } else if (std::is_same<T, cl_char>::value) {
        src_tag = "compute_s8_kernels";
        kernel_tag = "compute_s8_v";
        tag = "char";
    } else if (std::is_same<T, cl_half>::value) {
        if (!m_env.m_dev_info.half_supported) {
            fprintf(stderr, "half not suppprted, skip\n");
            return 0.0;
        }
        src_tag = "compute_hp_kernels";
        kernel_tag = "compute_hp_v";
        tag = "half";
    } else if (std::is_same<T, cl_float>::value) {
        src_tag = "compute_sp_kernels";
        kernel_tag = "compute_sp_v";
        tag = "float";
    } else if (std::is_same<T, cl_double>::value) {
        if (!m_env.m_dev_info.double_supported) {
            fprintf(stderr, "double not suppprted, skip\n");
            return 0.0;
        }
        src_tag = "compute_dp_kernels";
        kernel_tag = "compute_dp_v";
        tag = "double";
    } else {
        fprintf(stderr, "unknown type, skip\n");
        return 0.0;
    }

    constexpr static size_t iters = 10;
    size_t global_size =
            m_env.m_dev_info.num_cus * 2048 * m_env.m_dev_info.nthread_logic;
    global_size = std::min<size_t>(global_size * sizeof(T),
                                   m_env.m_dev_info.mem_obj_max_alloc_size);
    global_size = round_to_power2(global_size);
    global_size = global_size / sizeof(T);
    size_t local_size = m_env.m_dev_info.nthread_logic;

    std::vector<T> elems(global_size, 0);
    printf("%s compute (GFLOPS)\n", tag.c_str());

    cl_int err;
    cl_mem dst = clCreateBuffer(m_env.m_ctx, CL_MEM_WRITE_ONLY,
                                global_size * sizeof(T), nullptr, &err);
    opencl_check(err);

    cl_program program =
            m_env.build_program_from_fname(src_tag, ext_compile_options);
#if MPERF_AARCH64 || MPERF_ARMV7
    //! As in android, the opencl driver may cause CL_INVALID_ARG_SIZE if we
    //! pass a as half in half compute, so here we make dtype of a float.
    //! we should not make a as float in half compute in k8.
    //! FIXME: I don't known why.
    float a = static_cast<float>(1.3f);
#else
    T a = static_cast<T>(1.3f);
#endif

    float ret = 0.0;
#define cb(i)                                                                 \
    auto kernel_v##i = clCreateKernel(                                        \
            program, (kernel_tag + std::to_string(i)).c_str(), &err);         \
    local_size = std::min<size_t>(                                            \
            local_size, m_env.get_kernel_max_work_group_size(kernel_v##i));   \
    opencl_check(err);                                                        \
    err = clSetKernelArg(kernel_v##i, 0, sizeof(cl_mem), &dst);               \
    err |= clSetKernelArg(kernel_v##i, 1, sizeof(a), &a);                     \
    opencl_check(err);                                                        \
    ret = std::max<float>(ret, run_step(kernel_v##i, global_size, local_size, \
                                        iters, tag + std::to_string(i)));     \
    clReleaseKernel(kernel_v##i);

    cb(1);
    cb(2);
    cb(4);
    cb(8);
    cb(16);
#undef cb
    clReleaseProgram(program);

    return ret;
}

template <typename T>
float CompRunner<T>::run_step(cl_kernel kernel, size_t global_size,
                              size_t local_size, size_t iters,
                              const std::string& tag) {
    constexpr static size_t work_per_wi = 4096;
    float used = run_kernel(kernel, global_size, local_size, iters);
    float gflops = static_cast<float>(global_size) * work_per_wi / used / 1e3;
    printf("    %s : %f\n", tag.c_str(), gflops);

    return gflops;
}

////////////////////////////// DeviceInfo //////////////////////////////
namespace mperf {
template <typename T, cl_device_info info>
struct DeviceInfoGetter {
    static T get(cl_device_id dev_id);
};

template <typename T, cl_device_info info>
struct DeviceInfoGetter<T[], info> {
    static std::vector<T> get(cl_device_id dev_id) {
        size_t res_size = 0;
        opencl_check(clGetDeviceInfo(dev_id, info, 0, nullptr, &res_size));
        std::vector<T> res(res_size, T{0});
        opencl_check(clGetDeviceInfo(dev_id, info, res_size, &res[0], nullptr));
        return res;
    }
};

#define cb(_type)                                                           \
    template <cl_device_info info>                                          \
    struct DeviceInfoGetter<_type, info> {                                  \
        static _type get(cl_device_id dev_id) {                             \
            _type res = 0;                                                  \
            opencl_check(clGetDeviceInfo(dev_id, info, sizeof(_type), &res, \
                                         nullptr));                         \
            return res;                                                     \
        }                                                                   \
    };
cb(cl_uint) cb(cl_ulong)
#undef cb

}  // namespace mperf

////////////////////////////// OpenCLEnv //////////////////////////////
OpenCLEnv::OpenCLEnv() {
    cl_uint nr_platforms = 0;

    opencl_check(clGetPlatformIDs(0, nullptr, &nr_platforms));
    mperf_assert(nr_platforms > 0, "No opencl platform: %u", nr_platforms);

    std::vector<cl_platform_id> platform_ids(nr_platforms);
    opencl_check(clGetPlatformIDs(nr_platforms, platform_ids.data(), nullptr));

    cl_uint nr_devices = 0;
    std::vector<cl_device_id> device_ids;
    for (auto pid : platform_ids) {
        opencl_check(clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 0, nullptr,
                                    &nr_devices));
        if (!nr_devices) {
            continue;
        }

        device_ids.resize(nr_devices);
        opencl_check(clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, nr_devices,
                                    device_ids.data(), nullptr));
        m_platform = pid;
        break;
    }

    m_dev_id = device_ids[0];

    size_t property_pos = 0;
    cl_context_properties context_properties[32];

    auto add_prop = [&](cl_context_properties key, cl_context_properties val) {
        context_properties[property_pos++] = key;
        context_properties[property_pos++] = val;
        mperf_assert(
                property_pos < sizeof(context_properties) /
                                       sizeof(context_properties[0]),
                "invalid property_pos(%zu) > max_prop_size(%zu)", property_pos,
                sizeof(context_properties) / sizeof(context_properties[0]));
    };

    add_prop(CL_CONTEXT_PLATFORM, (cl_context_properties)m_platform);

    m_dev_info.extensions = to_string(
            mperf::DeviceInfoGetter<char[], CL_DEVICE_EXTENSIONS>::get(
                    m_dev_id));
    if ((m_dev_info.extensions.find("cl_qcom_perf_hint") !=
         std::string::npos)) {
        add_prop(CL_CONTEXT_PERF_HINT_QCOM, CL_PERF_HINT_HIGH_QCOM);
        add_prop(CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM);
    }

    context_properties[property_pos++] = 0;

    cl_int err;
    m_ctx = clCreateContext(context_properties, 1, &m_dev_id, nullptr, nullptr,
                            &err);
    opencl_check(err);

    m_queue = clCreateCommandQueue(m_ctx, m_dev_id, CL_QUEUE_PROFILING_ENABLE,
                                   &err);
    opencl_check(err);

    // init device info
    m_dev_info.num_cus =
            mperf::DeviceInfoGetter<cl_uint, CL_DEVICE_MAX_COMPUTE_UNITS>::get(
                    m_dev_id);
    m_dev_info.device_name = to_string(
            mperf::DeviceInfoGetter<char[], CL_DEVICE_NAME>::get(m_dev_id));
    m_dev_info.driver_version = to_string(
            mperf::DeviceInfoGetter<char[], CL_DRIVER_VERSION>::get(m_dev_id));
    m_dev_info.extensions = to_string(
            mperf::DeviceInfoGetter<char[], CL_DEVICE_EXTENSIONS>::get(
                    m_dev_id));

    opencl_check(clGetDeviceInfo(m_dev_id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                 sizeof(m_dev_info.max_wg_size_per_dim),
                                 m_dev_info.max_wg_size_per_dim, nullptr));
    m_dev_info.nthread_logic = static_cast<size_t>(
            mperf::DeviceInfoGetter<
                    cl_ulong, CL_DEVICE_MAX_WORK_GROUP_SIZE>::get(m_dev_id));
    m_dev_info.nthread_logic = std::max<size_t>(
            mperf::DeviceInfoGetter<
                    size_t[], CL_DEVICE_MAX_WORK_ITEM_SIZES>::get(m_dev_id)[0],
            256);
#if 0  // TODO(hc) check it.
    // FIXME limit max-workgroup size for qualcomm platform to 128
    // Kernel launch fails for workgroup size 256(CL_DEVICE_MAX_WORK_ITEM_SIZES)
    std::string vendor = to_string(
            mperf::DeviceInfoGetter<char[], CL_DEVICE_VENDOR>::get(m_dev_id));
    if ((vendor.find("QUALCOMM") != std::string::npos) ||
        (vendor.find("qualcomm") != std::string::npos)) {
        m_dev_info.nthread_logic =
                std::min<size_t>(m_dev_info.nthread_logic, 128);
    }

    if ((vendor.find("NVIDIA") != std::string::npos) ||
        (vendor.find("nvidia") != std::string::npos)) {
        m_dev_info.nthread_logic =
                std::min<size_t>(m_dev_info.nthread_logic, 256);
    }
#endif

    m_dev_info.global_mem_size_max = static_cast<size_t>(
            mperf::DeviceInfoGetter<cl_ulong, CL_DEVICE_GLOBAL_MEM_SIZE>::get(
                    m_dev_id));
    m_dev_info.mem_obj_max_alloc_size = static_cast<size_t>(
            mperf::DeviceInfoGetter<
                    cl_ulong, CL_DEVICE_MAX_MEM_ALLOC_SIZE>::get(m_dev_id));
    m_dev_info.l2_cacheline_size = static_cast<size_t>(
            mperf::DeviceInfoGetter<
                    cl_uint,
                    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>::get(m_dev_id));
    m_dev_info.l2_cache_size = static_cast<size_t>(
            mperf::DeviceInfoGetter<
                    cl_ulong, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>::get(m_dev_id));

    m_dev_info.max_clock_freq = static_cast<int>(
            mperf::DeviceInfoGetter<
                    cl_uint, CL_DEVICE_MAX_CLOCK_FREQUENCY>::get(m_dev_id));

    // set extensions
    m_dev_info.double_supported = false;
    m_dev_info.half_supported = false;
    if ((m_dev_info.extensions.find("cl_khr_fp16") != std::string::npos)) {
        m_dev_info.half_supported = true;
    }

    if ((m_dev_info.extensions.find("cl_khr_fp64") != std::string::npos) ||
        (m_dev_info.extensions.find("cl_amd_fp64") != std::string::npos)) {
        m_dev_info.double_supported = true;
    }

    size_t res_size = 0;
    m_dev_info.has_page_size =
            CL_SUCCESS == clGetDeviceInfo(m_dev_id, CL_DEVICE_PAGE_SIZE_QCOM, 0,
                                          nullptr, &res_size);
    if (m_dev_info.has_page_size) {
        m_dev_info.page_size = static_cast<size_t>(
                mperf::DeviceInfoGetter<
                        cl_ulong, CL_DEVICE_PAGE_SIZE_QCOM>::get(m_dev_id));
    }

    m_dev_info.support_img =
            CL_SUCCESS == clGetDeviceInfo(m_dev_id, CL_DEVICE_IMAGE_SUPPORT, 0,
                                          nullptr, &res_size);
    if (m_dev_info.support_img) {
        m_dev_info.img_width_max = static_cast<size_t>(
                mperf::DeviceInfoGetter<
                        cl_ulong, CL_DEVICE_IMAGE2D_MAX_WIDTH>::get(m_dev_id));
        m_dev_info.img_height_max = static_cast<size_t>(
                mperf::DeviceInfoGetter<
                        cl_ulong, CL_DEVICE_IMAGE2D_MAX_HEIGHT>::get(m_dev_id));
    }
}

cl_mem OpenCLEnv::malloc_buffer(cl_mem_flags mem_flags, size_t size_in_bytes) {
    cl_int err;
    cl_mem mem_obj =
            clCreateBuffer(m_ctx, mem_flags, size_in_bytes, nullptr, &err);
    opencl_check(err);
    return mem_obj;
}

cl_int OpenCLEnv::host_copy(void* src, cl_mem& dst, size_t len) const {
    cl_int err;
    auto dst_ptr =
            clEnqueueMapBuffer(m_queue, dst, true, CL_MAP_READ | CL_MAP_WRITE,
                               0, len, 0, nullptr, nullptr, &err);
    opencl_check(err);
    memcpy(dst_ptr, src, len);
    return CL_SUCCESS;
}

void OpenCLEnv::free_buffer(cl_mem mem) {
    opencl_check(clReleaseMemObject(mem));
}

cl_mem OpenCLEnv::create_image_1d(cl_mem_flags flags, ImageFormat format,
                                  size_t width, size_t row_pitch,
                                  void* host_ptr) {
    cl_int err;

    cl_image_desc desc = {
            CL_MEM_OBJECT_IMAGE1D, width, 0, 0, 0, 0, 0, 0, 0, {0}};
    cl_mem mem_obj =
            clCreateImage(m_ctx, flags, &format, &desc, host_ptr, &err);
    opencl_check(err);
    return mem_obj;
}

cl_mem OpenCLEnv::create_image_2d(cl_mem_flags flags, ImageFormat format,
                                  size_t width, size_t height, size_t row_pitch,
                                  void* host_ptr) {
    cl_int err;

    cl_image_desc desc = {CL_MEM_OBJECT_IMAGE2D,
                          width,
                          height,
                          0,
                          0,  // depth, array size (unused)
                          row_pitch,
                          0,
                          0,
                          0,
                          {0}};
    cl_mem mem_obj =
            clCreateImage(m_ctx, flags, &format, &desc, host_ptr, &err);
    opencl_check(err);
    return mem_obj;
}

MapBuffer OpenCLEnv::map_buf(const cl_mem& buf, size_t offset, size_t size,
                             cl_event* event) {
    cl_int err;
    auto rv = (float*)clEnqueueMapBuffer(
            m_queue, buf, true, CL_MAP_READ | CL_MAP_WRITE, offset, size, 0,
            nullptr, (cl_event*)event, &err);
    opencl_check(err);
    return MapBuffer{rv, size};
}

void OpenCLEnv::unmap_buf(const cl_mem& buf, MapBuffer& mapped,
                          const std::vector<cl_event>* events, cl_event event) {
    cl_event tmp;

    clEnqueueUnmapMemObject(m_queue, buf, mapped,
                            (events != nullptr) ? (cl_uint)events->size() : 0,
                            (events != nullptr && events->size() > 0)
                                    ? (cl_event*)&events->front()
                                    : nullptr,
                            (event != nullptr) ? &tmp : nullptr);
    mapped = {};
}

MapImage OpenCLEnv::map_img_1d(const cl_mem& img, size_t width,
                               const std::vector<cl_event>* events,
                               cl_event* event) {
    cl_int err;
    cl_event tmp;
    std::array<size_t, 3> origin{};
    std::array<size_t, 3> region{width, 1, 1};
    size_t row_pitch;
    size_t slice_pitch;
    void* result = clEnqueueMapImage(
            m_queue, img, true, CL_MAP_READ | CL_MAP_WRITE, origin.data(),
            region.data(), &row_pitch, &slice_pitch,
            (events != NULL) ? (cl_uint)events->size() : 0,
            (events != NULL && events->size() > 0) ? (cl_event*)&events->front()
                                                   : NULL,
            (event != NULL) ? &tmp : NULL, &err);
    opencl_check(err);
    return MapImage{result, width, 1, 1, row_pitch, slice_pitch};
}

void OpenCLEnv::unmap_img_1d(const cl_mem& img, MapImage& mapped,
                             const std::vector<cl_event>* events,
                             cl_event* event) {
    cl_event tmp;
    clEnqueueUnmapMemObject(m_queue, img, mapped,
                            (events != NULL) ? (cl_uint)events->size() : 0,
                            (events != NULL && events->size() > 0)
                                    ? (cl_event*)&events->front()
                                    : NULL,
                            (event != NULL) ? &tmp : NULL);
}

cl_program OpenCLEnv::build_program_from_fname(
        const std::string& source_name, std::string ext_compile_options) const {
    const char* program_buffer = PROGRAMS[source_name];
    size_t program_size = strlen(program_buffer);

    cl_int err;
    cl_program program = clCreateProgramWithSource(
            m_ctx, 1, (const char**)&program_buffer, &program_size, &err);
    opencl_check(err);

    std::string compile_options =
            "-cl-fast-relaxed-math -cl-mad-enable " + ext_compile_options;
    if (m_dev_info.extensions.find("cl_qcom_perf_hint") != std::string::npos) {
        compile_options += " -D V1_CASE=1";
    } else {
        printf("`V1_CASE=0` use on mali device, and you will got an "
               "incorrect result on vec1 interger compute testcase\n");
        compile_options += " -D V1_CASE=0";
    }

    err = clBuildProgram(program, 1, &m_dev_id, compile_options.c_str(),
                         nullptr, nullptr);

    if (err != CL_SUCCESS) {
        size_t str_len = 0;
        std::string build_log;
        clGetProgramBuildInfo(program, m_dev_id, CL_PROGRAM_BUILD_LOG, 0,
                              nullptr, &str_len);
        build_log.resize(str_len);
        clGetProgramBuildInfo(program, m_dev_id, CL_PROGRAM_BUILD_LOG, str_len,
                              &build_log[0], &str_len);
        mperf_assert(0, "clBuildProgram error: src=%s opt=%s errno=%s\n%s",
                     source_name.c_str(), compile_options.c_str(),
                     build_log.c_str(), mperf::get_error_string(err));
    }
    opencl_check(err);

    return program;
}

cl_program OpenCLEnv::build_program_from_source(
        const std::string& source, std::string ext_compile_options) const {
    const char* program_buffer = source.c_str();
    size_t program_size = source.size();

    cl_int err;
    cl_program program = clCreateProgramWithSource(
            m_ctx, 1, (const char**)&program_buffer, &program_size, &err);
    opencl_check(err);

    std::string compile_options =
            "-cl-fast-relaxed-math -cl-mad-enable " + ext_compile_options;
    err = clBuildProgram(program, 1, &m_dev_id, compile_options.c_str(),
                         nullptr, nullptr);

    if (err != CL_SUCCESS) {
        size_t str_len = 0;
        std::string build_log;
        clGetProgramBuildInfo(program, m_dev_id, CL_PROGRAM_BUILD_LOG, 0,
                              nullptr, &str_len);
        build_log.resize(str_len);
        clGetProgramBuildInfo(program, m_dev_id, CL_PROGRAM_BUILD_LOG, str_len,
                              &build_log[0], &str_len);
        mperf_assert(0, "clBuildProgram error: opt=%s errno=%s\n%s",
                     compile_options.c_str(), build_log.c_str(),
                     mperf::get_error_string(err));
    }
    opencl_check(err);

    return program;
}

void OpenCLEnv::print_device_info() const {
    printf("Device: %s\n", m_dev_info.device_name.c_str());
    printf("Extensions: {%s}\n", m_dev_info.extensions.c_str());
    printf("Driver version: %s\n", m_dev_info.driver_version.c_str());
    printf("Number of Compute units: %u\n", m_dev_info.num_cus);
    printf("Clock frequency: %u MHz\n", m_dev_info.max_clock_freq);
    if (m_dev_info.has_page_size) {
        printf("Device page size:%zu Bytes\n", m_dev_info.page_size);
    }
    printf("global memory max size: %f GB\n",
           m_dev_info.global_mem_size_max / 1024.0 / 1024 / 1024);
    printf("Maximum size of memory object allocation in bytes:%f GB\n",
           m_dev_info.mem_obj_max_alloc_size / 1024.0 / 1024 / 1024);
    printf("unified L2 cacheline size: %d Bytes\n",
           m_dev_info.l2_cacheline_size);
    printf("unified L2 cache size: %.2f KB\n",
           m_dev_info.l2_cache_size / 1024.0);
    if (m_dev_info.support_img) {
        printf("Image2D max width:%zu\n", m_dev_info.img_width_max);
        printf("Image2D max height:%zu\n", m_dev_info.img_height_max);
    } else {
        printf("Image is not supported\n");
    }
    printf("number of logical threads in each CU/SP:%zu\n",
           m_dev_info.nthread_logic);
    printf("max work group size:total(after correction)(%zu), "
           "per_dim(raw)(%zu,%zu,%zu)\n",
           m_dev_info.nthread_logic, m_dev_info.max_wg_size_per_dim[0],
           m_dev_info.max_wg_size_per_dim[1],
           m_dev_info.max_wg_size_per_dim[2]);
}

float OpenCLEnv::local_memory_bw() const {
    LocalMemRunner lmr(*this);
    return lmr.run();
}

float OpenCLEnv::global_memory_bw() const {
    GlobalMemRunner gmr(*this);
    return gmr.run();
}

float OpenCLEnv::insts_gflops_latency() const {
    float fp32_ret = 0.0;
    CompRunner<cl_int> s32_runner(*this);
    s32_runner.run(0);  // mad24
    s32_runner.run(1);  // naive mul/add
    CompRunner<cl_short> s16_runner(*this);
    s16_runner.run();
    CompRunner<cl_char> s8_runner(*this);
    s8_runner.run();
    CompRunner<cl_float> float_runner(*this);
    fp32_ret = float_runner.run();
    CompRunner<cl_half> half_runner(*this);
    half_runner.run();
    CompRunner<cl_double> double_runner(*this);
    double_runner.run();
    return fp32_ret;
}

// Find the minimal number of iterations that a kernel can run up to
// `min_time_us` microseconds.
void OpenCLEnv::ensure_min_niter(double min_time_us, int& niter,
                                 std::function<double()> run) {
    const int DEFAULT_NITER = 100;
    niter = DEFAULT_NITER;
    for (int i = 0; i < 100; ++i) {
        double t = run();
        if (t > min_time_us * 0.99) {
            mperf_log("found minimal niter=%d to take %f us", niter,
                      min_time_us);
            return;
        }
        mperf_log_debug("niter=%d doesn't run long enough (%f us <= %f us",
                        niter, t, min_time_us);
        niter = int(niter * min_time_us / t);
    }
    printf("unable to find a minimal iteration number, is your code "
           "aggresively optimized by the compiler?");
}

float OpenCLEnv::bench_kernel(cl_kernel kern, const NDRange& global_size,
                              const NDRange& local_size, size_t iters) const {
    mperf_assert(global_size.dimension() == local_size.dimension());
    float used = 0;
    opencl_check(clEnqueueNDRangeKernel(m_queue, kern, global_size.dimension(),
                                        nullptr, global_size.get(),
                                        local_size.get(), 0, nullptr, nullptr));
    clFinish(m_queue);

    cl_event ev;
    for (size_t i = 0; i < iters; i++) {
        opencl_check(clEnqueueNDRangeKernel(
                m_queue, kern, global_size.dimension(), nullptr,
                global_size.get(), local_size.get(), 0, nullptr, &ev));
        opencl_check(clWaitForEvents(1, &ev));
        cl_ulong start, end;
        opencl_check(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                             sizeof(cl_ulong), &start,
                                             nullptr));
        opencl_check(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                             sizeof(cl_ulong), &end, nullptr));

        used += (end - start) / 1000.0;
    }
    opencl_check(clReleaseEvent(ev));
    return used / iters;
}

float OpenCLEnv::execute_kernel(cl_kernel kern, const NDRange& global_size,
                                const NDRange& local_size) const {
    cl_event ev;
    opencl_check(clEnqueueNDRangeKernel(m_queue, kern, global_size.dimension(),
                                        nullptr, global_size.get(),
                                        local_size.get(), 0, nullptr, &ev));
    opencl_check(clWaitForEvents(1, &ev));
    cl_ulong start, end;
    opencl_check(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &start, nullptr));
    opencl_check(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &end, nullptr));

    return (end - start) / 1000.0;
}

size_t OpenCLEnv::get_kernel_max_work_group_size(cl_kernel kern) const {
    size_t lws = 0;
    opencl_check(clGetKernelWorkGroupInfo(kern, m_dev_id,
                                          CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof(size_t), &lws, nullptr));
    return lws;
}

// vim: syntax=cpp.doxygen
