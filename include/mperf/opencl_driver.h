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
 * \file include/mperf/opencl_driver.h
 *
 * Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2022-2023 Megvii Inc. All rights
 * reserved.
 *
 * ---------------------------------------------------------------------------
 */

#pragma once

#include <MCL/cl.h>
#include <MCL/cl_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define opencl_check(_expr)                                           \
    do {                                                              \
        cl_int _err = (_expr);                                        \
        if (_err != CL_SUCCESS) {                                     \
            fprintf(stderr, "opencl error %d: %s (%s at %s:%s:%d)\n", \
                    int(_err), mperf::get_error_string(_err), #_expr, \
                    __FILE__, __func__, __LINE__);                    \
            exit(1);                                                  \
        }                                                             \
    } while (0)

namespace mperf {
const char* get_error_string(cl_int error);

/**
 * \brief N-dimensional index space, where N is one, two or three.
 */
class NDRange {
private:
    size_t m_size[3];
    size_t m_dimension;

public:
    //! \brief resulting range has zero dimensions.
    NDRange() : m_dimension(0) {
        m_size[0] = 0;
        m_size[1] = 0;
        m_size[2] = 0;
    }

    //! \brief Constructs one-dimensional range.
    NDRange(size_t size0) : m_dimension(1) {
        m_size[0] = size0;
        m_size[1] = 1;
        m_size[2] = 1;
    }

    //! \brief Constructs two-dimensional range.
    NDRange(size_t size0, size_t size1) : m_dimension(2) {
        m_size[0] = size0;
        m_size[1] = size1;
        m_size[2] = 1;
    }

    //! \brief Constructs three-dimensional range.
    NDRange(size_t size0, size_t size1, size_t size2) : m_dimension(3) {
        m_size[0] = size0;
        m_size[1] = size1;
        m_size[2] = size2;
    }

    bool operator<(const NDRange& rhs) const;

    //! \brief Queries the number of dimensions in the range.
    size_t dimension() const { return m_dimension; }

    //! \brief Returns the size of the object in bytes based on the
    // runtime number of dimensions
    size_t size() const { return m_dimension * sizeof(size_t); }

    size_t* get() { return m_dimension ? m_size : nullptr; }

    size_t operator[](size_t idx);
    size_t operator[](size_t idx) const {
        return const_cast<NDRange*>(this)->operator[](idx);
    };

    const size_t* get() const { return const_cast<NDRange*>(this)->get(); }

    size_t total_size() const { return m_size[0] * m_size[1] * m_size[2]; }
};

struct DeviceInfo {
    std::string device_name;
    std::string driver_version;
    std::string extensions;

    int num_cus;
    size_t max_wg_size_per_dim[3];
    size_t nthread_logic;  // alias max_wg_size_total

    size_t mem_obj_max_alloc_size;
    size_t global_mem_size_max;
    int l2_cacheline_size;
    size_t l2_cache_size;

    int max_clock_freq;  // MHz

    bool half_supported;
    bool double_supported;

    bool has_page_size;
    size_t page_size;  // Bytes

    bool support_img;
    size_t img_width_max;
    size_t img_height_max;
};

class OpenCLEnv;
class Runner {
public:
    Runner(const OpenCLEnv& env) : m_env{env} {};
    virtual float run(int mod = -1) = 0;
    virtual ~Runner() = default;

    float run_kernel(cl_kernel kern, const NDRange& global_size,
                     const NDRange& local_size, size_t iters);
    const OpenCLEnv& m_env;
};

class LocalMemRunner : public Runner {
public:
    LocalMemRunner(const OpenCLEnv& env);
    float run(int mod = -1) override;

protected:
    float run_step(cl_kernel kernel, size_t global_size, size_t local_size,
                   size_t nr_elems, size_t iters, const std::string& tag);
};

class GlobalMemRunner : public Runner {
public:
    GlobalMemRunner(const OpenCLEnv& env);
    float run(int mod = -1) override;

protected:
    float run_step(cl_kernel kernel_go, cl_kernel kernel_lo, size_t global_size,
                   size_t local_size, size_t nr_elems, size_t iters,
                   const std::string& tag);
};

template <typename T>
class CompRunner : public Runner {
public:
    CompRunner(const OpenCLEnv& env);
    float run(int mod = -1) override;

protected:
    float run_step(cl_kernel kernel, size_t global_size, size_t local_size,
                   size_t iters, const std::string& tag);
};

/*************************************
 * * cl_qcom_perf_hint extension *
 * *************************************/
#define CL_PERF_HINT_NONE_QCOM 0
typedef cl_uint cl_perf_hint;
#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2
/*cl_perf_hint*/
#define CL_PERF_HINT_HIGH_QCOM 0x40C3
#define CL_PERF_HINT_NORMAL_QCOM 0x40C4
#define CL_PERF_HINT_LOW_QCOM 0x40C5
/*************************************
 * * cl_qcom_priority_hint extension *
 * *************************************/
#define CL_PRIORITY_HINT_NONE_QCOM 0
typedef cl_uint cl_priority_hint;
#define CL_CONTEXT_PRIORITY_HINT_QCOM 0x40C9
/*cl_priority_hint*/
#define CL_PRIORITY_HINT_HIGH_QCOM 0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM 0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM 0x40CC

/*! \stuct ImageFormat
 *  \brief Adds constructors and member functions for cl_image_format.
 *
 *  \see cl_image_format
 */
struct ImageFormat : public cl_image_format {
    //! \brief Default constructor - performs no initialization.
    ImageFormat() {}

    //! \brief Initializing constructor.
    ImageFormat(cl_channel_order order, cl_channel_type type) {
        image_channel_order = order;
        image_channel_data_type = type;
    }

    //! \brief Assignment operator.
    ImageFormat& operator=(const ImageFormat& rhs) {
        if (this != &rhs) {
            this->image_channel_data_type = rhs.image_channel_data_type;
            this->image_channel_order = rhs.image_channel_order;
        }
        return *this;
    }
};

struct MapImage {
    void* data;
    size_t width;
    size_t height;
    size_t depth;
    size_t row_pitch;
    size_t slice_pitch;

    operator void*() const { return data; }
};

struct MapBuffer {
    void* data;
    size_t size;

    operator void*() const { return data; }
};

class OpenCLEnv {
public:
    static OpenCLEnv& instance() {
        static OpenCLEnv inst;
        return inst;
    }
    ~OpenCLEnv() {
        clReleaseCommandQueue(m_queue);
        clReleaseDevice(m_dev_id);
        clReleaseContext(m_ctx);
    }

    cl_mem malloc_buffer(cl_mem_flags mem_flags, size_t size_in_bytes);
    void free_buffer(cl_mem mem);
    cl_int host_copy(void* src, cl_mem& dst, size_t len) const;

    cl_mem create_image_1d(cl_mem_flags flags, ImageFormat format, size_t width,
                           size_t row_pitch = 0, void* host_ptr = nullptr);
    cl_mem create_image_2d(cl_mem_flags flags, ImageFormat format, size_t width,
                           size_t height, size_t row_pitch = 0,
                           void* host_ptr = nullptr);
    MapBuffer map_buf(const cl_mem& buf, size_t offset, size_t size,
                      cl_event* event = nullptr);
    void unmap_buf(const cl_mem& buf, MapBuffer& mapped,
                   const std::vector<cl_event>* events = nullptr,
                   cl_event event = nullptr);
    MapImage map_img_1d(const cl_mem& img, size_t width,
                        const std::vector<cl_event>* events = nullptr,
                        cl_event* event = nullptr);
    void unmap_img_1d(const cl_mem& img, MapImage& mapped,
                      const std::vector<cl_event>* events = nullptr,
                      cl_event* event = nullptr);

    cl_program build_program_from_fname(
            const std::string& source_name,
            std::string ext_compile_options = "") const;
    cl_program build_program_from_source(
            const std::string& source_name,
            std::string ext_compile_options = "") const;
    void print_device_info() const;
    float local_memory_bw() const;
    float global_memory_bw() const;
    float insts_gflops_latency() const;
    void ensure_min_niter(double min_time_us, int& niter,
                          std::function<double()> run);
    float bench_kernel(cl_kernel kern, const NDRange& global_size,
                       const NDRange& local_size, size_t iters) const;
    float execute_kernel(cl_kernel kern, const NDRange& global_size,
                         const NDRange& local_size) const;

    DeviceInfo m_dev_info;

    size_t get_kernel_max_work_group_size(cl_kernel kern) const;

private:
    cl_device_id m_dev_id;
    cl_context m_ctx;
    cl_platform_id m_platform;
    cl_command_queue m_queue;

    friend class Runner;
    friend class LocalMemRunner;
    friend class GlobalMemRunner;
    friend class CompRunner<cl_int>;
    friend class CompRunner<cl_short>;
    friend class CompRunner<cl_char>;
    friend class CompRunner<cl_half>;
    friend class CompRunner<cl_double>;
    friend class CompRunner<cl_float>;

    OpenCLEnv();
};

}  // namespace mperf

// vim: syntax=cpp.doxygen
