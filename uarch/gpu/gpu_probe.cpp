// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
/**
 * ---------------------------------------------------------------------------
 * \file uarch/gpu/gpu_probe.cpp
 *
 * Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2022-2023 Megvii Inc. All rights
 * reserved.
 *
 * ---------------------------------------------------------------------------
 */
#include "mperf/timer.h"
#include "utils.hpp"

namespace mperf {
//************************** kernels ***************************/

// This aspect tests the number of registers owned by each thread. On some
// architectures, like Arm Mali, the workgroup size can be larger if each
// thread is using a smaller number of registers, so the report will be the
// maximal number of threads at which number of registers being available.
//
// In the experiment, an array of 32b words, here floats, is allocated, the
// registers will hold a number of data, and finally be wriiten
// back to memory to prevent optimization. When the number of register used
// exceeds the physical register file size, i.e. register spill, some or all
// the registers will be fallbacked to memory, and kernel latency is
// significantly increased.
//
// TODO: (penguinliong) It can get negative delta time many times. Very likely
// it can be the kernel exited too early when too many registers are
// allocated. Need a way to know the upper limit in advance, or it's just a
// driver bug.
int reg_count(OpenCLEnv& env) {
    // This number should be big enough to contain the number of registers
    // reserved for control flow, but not too small to ignore register count
    // boundaries in first three iterations.
    const double COMPENSATE = 0.01;
    const double THRESHOLD = 10;
    const int NREG_MIN = 1;
    const int NREG_MAX = 512;
    const int NREG_STEP = 1;
    const int NGRP_MIN = 1;
    const int NGRP_MAX = 64;
    const int NGRP_STEP = 1;

    int NITER = 1;

    cl_mem out_buf = env.malloc_buffer(0, sizeof(float));
    printf("nthread, ngrp, nreg, niter, time(us)\n");

    auto bench = [&](int nthread, int ngrp, int nreg) {
        std::string reg_declr = "";
        std::string reg_comp = "";
        std::string reg_reduce = "";
        for (int i = 0; i < nreg; ++i) {
            reg_declr += utils::format("float reg_data", i,
                                       " = "
                                       "(float)niter + ",
                                       i, ";\n");
        }
        for (int i = 0; i < nreg; ++i) {
            int last_i = i == 0 ? nreg - 1 : i - 1;
            reg_comp +=
                    utils::format("reg_data", i, " *= reg_data", last_i, ";\n");
        }
        for (int i = 0; i < nreg; ++i) {
            reg_reduce +=
                    utils::format("out_buf[", i, " * i] = reg_data", i, ";\n");
        }

        auto src =
                utils::format(R"(
        __kernel void reg_count(
          __global float* out_buf,
          __private const int niter
        ) {
          )",
                              reg_declr, R"(
          int i = 0;
          for (; i < niter; ++i) {
          )", reg_comp, R"(
          }
          i = i >> 31;
          )", reg_reduce, R"(
        }
      )");
        // log::debug(src);
        cl_int err = 0;
        cl_program program = env.build_program_from_source(src);
        cl_kernel kernel = clCreateKernel(program, "reg_count", &err);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_int), &NITER);
        // Make sure all SMs are used. It's good for accuracy.
        NDRange global(nthread, ngrp, 1);
        NDRange local(nthread, 1, 1);
        double time = env.bench_kernel(kernel, global, local, 10);
        printf("%10d,%10d,%10d,%10d,%7.3f\n", nthread, ngrp, nreg, NITER, time);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return time;
    };

    env.ensure_min_niter(1000, NITER, [&]() { return bench(1, 1, NREG_MIN); });

    // Step 1. Probe for the most number of registers we can use by activating
    // different numbers of threads to try to use up all the register resources.
    // In case of register spill the kernel latency would surge.
    //
    // The group size is kept 1 to minimize the impact of scheduling.
    int nreg_max;
    {
        mperf_log(
                "testing register availability when only 1 thread is "
                "dispatched");

        DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
        int nreg = NREG_MIN;
        for (; nreg <= NREG_MAX; nreg += NREG_STEP) {
            double time = bench(1, 1, nreg);
            mperf_log_debug("testing nreg=%d, time=%f", nreg, time);

            if (dj.push(time)) {
                nreg -= NREG_STEP;
                mperf_log("%d registers are available at most", nreg);
                nreg_max = nreg;
                break;
            }
        }
        if (nreg >= NREG_MAX) {
            mperf_log_warn("unable to conclude a maximal register count");
            nreg_max = NREG_STEP;
        } else {
            printf("RegCount: %d\n", nreg_max);
        }
    }

    // Step 2: Knowing the maximal number of registers available to a single
    // thread, we wanna check if the allocation is pooled. A pool of register is
    // shared by all threads concurrently running, which means the more register
    // we use per thread, the less number of concurrent threads we can have in a
    // warp. Then we would have to consider the degradation of parallelism due
    // to register shortage.
    //
    // In this implementation we measure at which number of workgroups full-
    // occupation and half-occupation of registers are having a jump in latency.
    // If the registers are pooled, the full-occupation case should only support
    // half the number of threads of the half-occupation case. Otherwise, the
    // full-occupation case jump at the same point as half-occupation, the
    // registers should be dedicated to each physical thread.
    auto find_ngrp_by_nreg = [&](int nreg) -> int {
        DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
        for (auto ngrp = NGRP_MIN; ngrp <= NGRP_MAX; ngrp += NGRP_STEP) {
            auto time = bench(1, ngrp, nreg);
            mperf_log_debug("testing occupation (nreg= %d); ngrp=%d, time=%f\n",
                            nreg, ngrp, time);

            if (dj.push(time)) {
                ngrp -= NGRP_STEP;
                mperf_log(
                        "using %d, registers can have %d concurrent "
                        "single-thread workgroups",
                        nreg, ngrp);
                return ngrp;
            }
        }
        mperf_log_warn(
                "unable to conclude a maximum number of concurrent "
                "single-thread workgroups when %d registers are occupied",
                nreg);
        return 1;
    };

    int ngrp_full = find_ngrp_by_nreg(nreg_max);
    int ngrp_half = find_ngrp_by_nreg(nreg_max / 2);
    printf("FullRegConcurWorkgroupCount:%d\n", ngrp_full);
    printf("HalfRegConcurWorkgroupCount:%d\n", ngrp_half);

    std::string reg_ty;
    if (ngrp_full * 1.5 < ngrp_half) {
        mperf_log("all physical threads in an sm share %d registers", nreg_max);
        printf("RegType: Pooled\n");
    } else {
        mperf_log("each physical thread has %d registers", nreg_max);
        printf("RegType: Dedicated\n");
    }

    return nreg_max;
}

// This aspect tests the cacheline size of the buffer data pathway. A
// cacheline is the basic unit of storage in a cache hierarchy, and the
// cacheline size of the top-level cache system directly determines the
// optimal alignment of memory access. No matter how much data the kernel
// would fetch from lower hierarchy, the entire cacheline will be fetched in.
// If we are not reading all the data in a cacheline, a portion of memory
// bandwidth will be wasted, which is an undesired outcome.
//
// In this experiment all logically concurrent threads read from the memory
// hierarchy with a varying stride. The stride is initially small and many
// accesses will hit the cachelines densely; but as the stride increases there
// will be only a single float taken from each cacheline, which leads to
// serious cache flush and increased latency.
int buf_cacheline_size(OpenCLEnv& env) {
    // CHECK(zxb): check the differ between this buf and global mem
    const int NTHREAD_LOGIC = env.m_dev_info.nthread_logic;
    const int BUF_CACHE_SIZE = env.m_dev_info.l2_cache_size;

    const double COMPENSATE = 0.01;
    const double THRESHOLD = 10;

    const int PITCH = BUF_CACHE_SIZE * 2 / NTHREAD_LOGIC;
    const int BUF_SIZE = PITCH * NTHREAD_LOGIC;
    const int MAX_STRIDE = PITCH / 2;

    int NITER;
    printf("nthread, stride(byte), pitch(byte), niter, time(us)\n");

    const char* src = R"(
      __kernel void buf_cacheline_size(
        __global const float* src,
        __global float* dst,
        __private const int niter,
        __private const int stride,
        __private const int pitch
      ) {
        float c = 0;
        for (int i = 0; i < niter; ++i) {
          const int zero = i >> 31;
          c += src[zero + stride * 0 + pitch * get_global_id(0)];
          c += src[zero + stride * 1 + pitch * get_global_id(0)];
        }
        dst[0] = c;
      }
    )";
    cl_int err = 0;
    cl_program program = env.build_program_from_source(src);
    cl_kernel kernel = clCreateKernel(program, "buf_cacheline_size", &err);

    cl_mem in_buf = env.malloc_buffer(0, BUF_SIZE);
    cl_mem out_buf = env.malloc_buffer(0, sizeof(float));

    auto bench = [&](int stride) {
        NDRange global(NTHREAD_LOGIC, 1, 1);
        NDRange local(NTHREAD_LOGIC, 1, 1);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel, 2, sizeof(cl_int), &NITER);
        int pstride = (int)(stride / sizeof(float));
        clSetKernelArg(kernel, 3, sizeof(cl_int), &pstride);
        int ppitch = (int)(PITCH / sizeof(float));
        clSetKernelArg(kernel, 4, sizeof(cl_int), &ppitch);
        auto time = env.bench_kernel(kernel, global, local, 10);
        printf("%10d,%10d,%10d,%10d,%7.3f\n", NTHREAD_LOGIC, stride, PITCH,
               NITER, time);
        return time;
    };

    env.ensure_min_niter(1000, NITER, [&]() { return bench(sizeof(float)); });

    int cacheline_size = 1;
    mperf_log("testing buffer cacheline size");
    DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
    int stride = sizeof(float);
    for (; stride <= MAX_STRIDE; stride += sizeof(float)) {
        double time = bench(stride);
        mperf_log_debug("testing stride=%d, time=%f", stride, time);

        if (dj.push(time)) {
            cacheline_size = stride;
            mperf_log("top level buffer cacheline size is %d B",
                      cacheline_size);
            break;
        }
    }
    if (stride >= MAX_STRIDE) {
        mperf_log_warn("unable to conclude a top level buffer cacheline size");
    } else {
        printf("BufTopLevelCachelineSize:%d\n", cacheline_size);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return cacheline_size;
}

// In the experiment, we assume L1 cacheline size is small enough that
// several threads reading float4s can exceed it. In both direction, along the
// width and the height, each logically concurrent thread in an SM reads a
// float4. Such memory access should be satisfied by a single cache fetch, but
// if the cache is not large enough to contain all requested data, multiple
// fetches will significantly increase access latency. A large iteration
// number is used to magnify the latency.
int img_cacheline_size(OpenCLEnv& env) {
    printf("nthread, dim (x/y), niter, time (us)\n");

    const int NTHREAD_LOGIC = env.m_dev_info.nthread_logic;
    const int MAX_IMG_WIDTH = env.m_dev_info.img_width_max;
    const int MAX_IMG_HEIGHT = env.m_dev_info.img_height_max;
    const int PX_SIZE = 4 * sizeof(float);

    const double COMPENSATE = 0.01;
    const double THRESHOLD = 10;

    int NITER;

    const char* src = R"(
      __constant sampler_t SAMPLER =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
      __kernel void img_cacheline_size_x(
        __read_only image2d_t src,
        __global float* dst,
        __private const int niter
      ) {
        float4 c = 0;
        for (int j = 0; j < niter; ++j) {
          int zero = j >> 31;
          c += read_imagef(src, SAMPLER, (int2)(get_global_id(0), zero));
        }
        dst[0] = c.x * c.y * c.z * c.w;
      }
      __kernel void img_cacheline_size_y(
        __read_only image2d_t src,
        __global float* dst,
        __private const int niter
      ) {
        float4 c = 0;
        for (int j = 0; j < niter; ++j) {
          int zero = j >> 31;
          c += read_imagef(src, SAMPLER, (int2)(zero, get_global_id(0)));
        }
        dst[0] = c.x * c.y * c.z * c.w;
      }
    )";
    cl_int err = 0;
    cl_program program = env.build_program_from_source(src);
    cl_kernel kernels[] = {
            clCreateKernel(program, "img_cacheline_size_x", &err),
            clCreateKernel(program, "img_cacheline_size_y", &err),
    };

    ImageFormat img_fmt(CL_RGBA, CL_FLOAT);
    cl_mem in_img =
            env.create_image_2d(0, img_fmt, MAX_IMG_WIDTH, MAX_IMG_HEIGHT);
    cl_mem out_buf = env.malloc_buffer(0, sizeof(float));

    auto bench = [&](int nthread, int dim) {
        mperf_assert(dim < 2, "invalid image dimension");
        auto& kernel = kernels[dim];

        NDRange global(nthread, 1, 1);
        NDRange local(nthread, 1, 1);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_img);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel, 2, sizeof(cl_int), &NITER);
        auto time = env.bench_kernel(kernel, global, local, 200);
        printf("%10d,%10d,%10d,%7.3f\n", nthread, dim, NITER, time);
        return time;
    };

    const char* param_name_by_dim[]{
            "ImgMinTimeConcurThreadCountX",
            "ImgMinTimeConcurThreadCountY",
    };

    int concur_nthread_by_dim[2];
    for (int dim = 0; dim < 2; ++dim) {
        const int IMG_EDGE = dim == 0 ? MAX_IMG_WIDTH : MAX_IMG_HEIGHT;
        const int IMG_OTHER_EDGE = dim == 1 ? MAX_IMG_WIDTH : MAX_IMG_HEIGHT;
        const int MAX_NTHREAD = std::min(NTHREAD_LOGIC, IMG_OTHER_EDGE);

        int& concur_nthread = concur_nthread_by_dim[dim];

        mperf_log("testing image cacheline size along dim=%d", dim);

        env.ensure_min_niter(1000, NITER, [&]() { return bench(1, dim); });

        DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
        int nthread = 1;
        for (; nthread <= MAX_NTHREAD; ++nthread) {
            double time = bench(nthread, dim);
            mperf_log_debug("testing nthread=%d, time=%f", nthread, time);

            if (dj.push(time)) {
                concur_nthread = nthread - 1;
                mperf_log(
                        "can concurrently access %d px with minimal cost along "
                        "dim=%d",
                        concur_nthread, dim);
                break;
            }
        }
        if (nthread >= MAX_NTHREAD) {
            mperf_log_warn(
                    "unable to conclude a top level image cacheline size");

        } else {
            concur_nthread_by_dim[dim] = concur_nthread;
            printf("%s: %d\n", param_name_by_dim[dim], concur_nthread);
        }
    }

    const int concur_nthread_x = concur_nthread_by_dim[0];
    const int concur_nthread_y = concur_nthread_by_dim[1];

    int cacheline_size = PX_SIZE *
                         std::max(concur_nthread_x, concur_nthread_y) /
                         std::min(concur_nthread_x, concur_nthread_y);
    printf("ImgCachelineSize: %d\n", cacheline_size);

    std::string cacheline_dim =
            concur_nthread_x >= concur_nthread_y ? "X" : "Y";
    printf("ImgCachelineDim: %s\n", cacheline_dim.c_str());

    clReleaseKernel(kernels[0]);
    clReleaseKernel(kernels[1]);
    clReleaseMemObject(in_img);
    clReleaseProgram(program);
    return cacheline_size;
}

double img_bandwidth(OpenCLEnv& env) {
    printf("access range (byte), time (us), bandwidth (gbps)\n");

    const int MAX_IMAGE_WIDTH = env.m_dev_info.img_width_max;
    const int NTHREAD_LOGIC = env.m_dev_info.nthread_logic;
    const int NUM_CU = env.m_dev_info.num_cus;

    // Size configs in bytes. These settings should be adjusted by hand.
    const int VEC_WIDTH = 4;
    const size_t VEC_SIZE = VEC_WIDTH * sizeof(float);
    const size_t RANGE = 128 * 1024 * 1024;
    const int NFLUSH = 4;
    const int NUNROLL = 16;
    const int NITER = 4;
    const int NREAD_PER_THREAD = NUNROLL * NITER;

    auto bench = [&](size_t access_size) {
        const size_t CACHE_SIZE = access_size;

        const size_t NVEC = RANGE / VEC_SIZE;
        const size_t NVEC_CACHE = CACHE_SIZE / VEC_SIZE;

        const int nthread_total = NVEC / NREAD_PER_THREAD;
        const int local_x = NTHREAD_LOGIC;
        const int global_x =
                (nthread_total / local_x * local_x) * NUM_CU * NFLUSH;
        // log::debug("local_x=", local_x, "; global_x=", global_x);

        auto src = utils::format(
                R"(
        __constant sampler_t SAMPLER =
          CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
        __kernel void img_bandwidth(
          __read_only image1d_t A,
          __global float4 *B,
          __private const int niter,
          __private const int addr_mask
        ) {
          float4 sum = 0;
          int offset = (get_group_id(0) * )",
                local_x * NREAD_PER_THREAD,
                R"( + get_local_id(0)) & addr_mask;

          for (int i = 0; i < niter; ++i)
          {)",
                [&]() {
                    std::stringstream ss;
                    for (int i = 0; i < NUNROLL; ++i) {
                        ss << "sum *= read_imagef(A, SAMPLER, offset); "
                              "offset = (offset + "
                           << local_x << ") & addr_mask;\n";
                    }
                    return ss.str();
                }(),
                R"(
          }
          B[get_local_id(0)] = sum;
        })");
        // log::debug(src);
        cl_int err = 0;
        cl_program program = env.build_program_from_source(src);
        cl_kernel kernel = clCreateKernel(program, "img_bandwidth", &err);

        ImageFormat img_fmt(CL_RGBA, CL_FLOAT);
        cl_mem in_buf = env.create_image_1d(
                0, img_fmt,
                std::min<size_t>(NVEC, env.m_dev_info.img_width_max));
        cl_mem out_buf = env.malloc_buffer(0, VEC_SIZE * NTHREAD_LOGIC);

        NDRange global(global_x, 1, 1);
        NDRange local(local_x, 1, 1);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel, 2, sizeof(cl_int), &NITER);
        int tmp = NVEC_CACHE - 1;
        clSetKernelArg(kernel, 3, sizeof(cl_int), &tmp);
        auto time = env.bench_kernel(kernel, global, local, 10);
        const size_t SIZE_TRANS = global_x * NREAD_PER_THREAD * VEC_SIZE;
        auto gbps = SIZE_TRANS * 1e-3 / time;
        mperf_log_debug(
                "image bandwidth accessing %zu B unique data is %f gbps(%f us)",
                access_size, gbps, time);
        printf("%10zu,%7.3f,%7.3f\n", access_size, time, gbps);

        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(in_buf);
        return gbps;
    };

    MaxStats<double> max_bandwidth;
    MinStats<double> min_bandwidth;
    for (size_t access_size = VEC_SIZE; access_size < RANGE; access_size *= 2) {
        double gbps = bench(access_size);
        max_bandwidth.push(gbps);
        min_bandwidth.push(gbps);
    }

    printf("MaxBandwidth: %f\n", (double)max_bandwidth);
    printf("MinBandwidth: %f\n", (double)min_bandwidth);

    mperf_log("discovered image read bandwidth min=%f; max=%f",
              (double)min_bandwidth, (double)max_bandwidth);
    return (double)max_bandwidth;
}

double buf_bandwidth(OpenCLEnv& env) {
    printf("access range (byte), time (us), bandwidth (gbps)\n");

    const int NTHREAD_LOGIC = env.m_dev_info.nthread_logic;
    const int NUM_CU = env.m_dev_info.num_cus;

    // Size configs in bytes. These settings should be adjusted by hand.
    const size_t RANGE = 128 * 1024 * 1024;
    const int VEC_WIDTH = 4;
    const int NFLUSH = 4;
    const int NUNROLL = 16;
    const int NITER = 4;
    const int NREAD_PER_THREAD = NUNROLL * NITER;

    const size_t VEC_SIZE = VEC_WIDTH * sizeof(float);
    auto bench = [&](size_t access_size) {
        const size_t CACHE_SIZE = access_size;

        const size_t NVEC = RANGE / VEC_SIZE;
        const size_t NVEC_CACHE = CACHE_SIZE / VEC_SIZE;

        const int nthread_total = NVEC / NREAD_PER_THREAD;
        const int local_x = NTHREAD_LOGIC;
        const int global_x =
                (nthread_total / local_x * local_x) * NUM_CU * NFLUSH;
        // log::debug("local_x=", local_x, "; global_x=", global_x);

        auto src = utils::format(
                R"(
        __kernel void buf_bandwidth(
          __global float4 *A,
          __global float4 *B,
          __private const int niter,
          __private const int addr_mask
        ) {
          float4 sum = 0;
          int offset = (get_group_id(0) * )",
                local_x * NREAD_PER_THREAD,
                R"( + get_local_id(0)) & addr_mask;

          for (int i = 0; i < niter; ++i)
          {)",
                [&]() {
                    std::stringstream ss;
                    for (int i = 0; i < NUNROLL; ++i) {
                        ss << "sum *= A[offset]; offset = (offset + " << local_x
                           << ") & addr_mask;\n";
                    }
                    return ss.str();
                }(),
                R"(
          }
          B[get_local_id(0)] = sum;
        })");
        // log::debug(src);
        cl_int err = 0;
        cl_program program = env.build_program_from_source(src);
        cl_kernel kernel = clCreateKernel(program, "buf_bandwidth", &err);

        cl_mem in_buf = env.malloc_buffer(0, RANGE);
        cl_mem out_buf = env.malloc_buffer(0, VEC_SIZE * NTHREAD_LOGIC);

        NDRange global(global_x, 1, 1);
        NDRange local(local_x, 1, 1);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel, 2, sizeof(cl_int), &NITER);
        int tmp = (NVEC_CACHE - 1);
        clSetKernelArg(kernel, 3, sizeof(cl_int), &tmp);
        auto time = env.bench_kernel(kernel, global, local, 10);
        const size_t SIZE_TRANS = global_x * NREAD_PER_THREAD * VEC_SIZE;
        auto gbps = SIZE_TRANS * 1e-3 / time;
        mperf_log_debug(
                "buffer bandwidth accessing %zu B unique data is %f gbps(%f "
                "us)",
                access_size, gbps, time);
        printf("%10zu,%7.3f,%7.3f\n", access_size, time, gbps);

        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return gbps;
    };

    MaxStats<double> max_bandwidth{};
    MinStats<double> min_bandwidth{};
    for (size_t access_size = VEC_SIZE; access_size < RANGE; access_size *= 2) {
        double gbps = bench(access_size);
        max_bandwidth.push(gbps);
        min_bandwidth.push(gbps);
    }

    printf("MaxBandwidth: %f\n", (double)max_bandwidth);
    printf("MinBandwidth: %f\n", (double)min_bandwidth);

    mperf_log("discovered buffer read bandwidth min=%f; max=%f",
              (double)min_bandwidth, (double)max_bandwidth);
    return (double)min_bandwidth;
}

// This aspect tests the warping of the SMs. A warp is an atomic schedule unit
// where all threads in a warp can be executed in parallel. An GPU SM usually
// consumes more threads than it physically can so that it can hide the
// latency of expensive operations like memory accesses by interleaved
// execution.
//
// The warping mechanism is, however, hard to conclude with a single method
// because of two kinds of scheduling behavior the driver could have:
//
// 1. It waits for more threads than it physically have. (ARM Mali)
// 2. It runs more threads than it logically should. (Qualcomm Adreno)
//
// In Case 1, the SM waits until all the works and dummy works to finish, for
// a uniform control flow limited by a shared program counter. Even if a
// number of threads are not effective workloads, the SM cannot exit early and
// still have to spend time executing the dummy works.
//
// In Case 2, the SM acquires a large number of threads and the threads are
// distributed to physical threads in warps. When the incoming workload has
// too few threads. The driver might decides to pack multiple works together
// and dispatch them at once.
//
// We devised two different micro-benchmarks to reveal the true parallelism
// capacity of the underlying platform against the interference of these
// situations.
int warp_size(OpenCLEnv& env) {
    const int NTHREAD_LOGIC = env.m_dev_info.nthread_logic;

    // Method A: Let all the threads in a warp to race and atomically fetch-add
    // a counter, then store the counter values to the output buffer as
    // scheduling order of these threads. If all the order numbers followings a
    // pattern charactorized by the scheduler hardware, then the threads are
    // likely executing within a warp. Threads in different warps are not
    // managed by the same scheduler, so they would racing for a same ID out of
    // order, unaware of each other.
    //
    // This method helps us identify warp sizes when the SM sub-divides its ALUs
    // into independent groups, like the three execution engines in a Mali G76
    // core. It helps warp-probing in Case 1 because it doesn't depend on kernel
    // timing, so the extra wait time doesn't lead to inaccuracy.
    {
        printf("nthread, sum\n");

        int nthread_warp_a = 1;
        auto src = R"(
          __kernel void warp_size(__global int* output) {
            __local int local_counter;
            local_counter = 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            int i = atomic_inc(&local_counter);
            barrier(CLK_LOCAL_MEM_FENCE);
            output[get_global_id(0)] = i;
          }
        )";
        cl_int err = 0;
        cl_program program = env.build_program_from_source(src);
        cl_kernel kernel = clCreateKernel(program, "warp_size", &err);

        auto bench = [&](int nthread) {
            NDRange global(nthread, 1, 1);
            NDRange local(nthread, 1, 1);

            auto size = NTHREAD_LOGIC * sizeof(int32_t);
            cl_mem out_buf = env.malloc_buffer(0, size);

            clSetKernelArg(kernel, 0, sizeof(cl_mem), &out_buf);
            env.bench_kernel(kernel, global, local, 1);

            int32_t sum;
            auto mapped = env.map_buf(out_buf, 0, size);
            {
                int32_t* data = (int32_t*)(void*)mapped;

                // for (auto j = 0; j < nthread; ++j) {
                //     printf("%d ", data[j]);
                // }
                // printf("\n");

                int32_t last = -1;
                auto j = 0;
                for (; j < nthread; ++j) {
                    if (last >= data[j]) {
                        break;
                    }
                    last = data[j];
                }
                sum = j;
            }
            env.unmap_buf(out_buf, mapped);
            printf("%10d,%7.3d\n", nthread, sum);
            return sum;
        };

        // TODO: (penguinliong) Improve this warp size inference by stats of
        // all sequential ascend of order of multiple executions.
        int i = 1;
        for (; i <= NTHREAD_LOGIC; ++i) {
            int sum = bench(i);
            if (sum != i) {
                nthread_warp_a = i - 1;
                break;
            }
        }
        if (i > NTHREAD_LOGIC) {
            mperf_log_warn("unable to conclude a warp size by method a");
        }
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        printf("WarpThreadCount: %d\n", nthread_warp_a);
    }
    // Method B: Time the latency of a pure computation kernel, with different
    // number of logical threads in a workgroup. When the logical thread count
    // exceeds the physical capability of the GPU, kernel latency jumps slightly
    // to indicate higher scheduling cost.
    //
    // This timing-based method helps us identify warp sizes in Case 2, when
    // threads of multiple warps are managed by the same scheduler at the same
    // time.
    //
    // The division in the kernel helps to reduce the impact of latency hiding
    // by warping because integral division is an examplary multi-cycle
    // instruction that can hardly be optimized.
    int nthread_warp_b = 1;
    {
        printf("nthread, time (us)\n");

        const int PRIME = 3;
        const double COMPENSATE = 0.01;
        const double THRESHOLD = 10;

        int NITER;

        auto src = R"(
        __kernel void warp_size2(
          __global float* src,
          __global int* dst,
          const int niter,
          const int prime_number
        ) {
          int drain = 0;
          for (int j = 0; j < niter; ++j) {
            drain += j / prime_number;
            barrier(0);
          }
          dst[get_local_id(0)] = drain;
        }
      )";
        cl_int err = 0;
        cl_program program = env.build_program_from_source(src);
        cl_kernel kernel = clCreateKernel(program, "warp_size2", &err);

        auto size = NTHREAD_LOGIC * sizeof(float);
        cl_mem src_buf = env.malloc_buffer(0, size);
        cl_mem dst_buf = env.malloc_buffer(0, size);

        auto bench = [&](int nthread) {
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buf);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_buf);
            clSetKernelArg(kernel, 2, sizeof(cl_int), &NITER);
            clSetKernelArg(kernel, 3, sizeof(cl_int), &PRIME);

            NDRange global(nthread, 1024, 1);
            NDRange local(nthread, 1, 1);

            double time = env.bench_kernel(kernel, global, local, 10);
            printf("%10d,%7.3f\n", nthread, time);
            return time;
        };

        env.ensure_min_niter(1000, NITER, [&]() { return bench(1); });

        DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
        auto nthread = 1;
        for (; nthread <= NTHREAD_LOGIC; ++nthread) {
            double time = bench(nthread);
            mperf_log_debug("nthread=%d(%f us)", nthread, time);
            if (dj.push(time)) {
                nthread_warp_b = nthread - 1;
                printf("WarpThreadCount: %d\n", nthread_warp_b);
                break;
            }
        }
        if (nthread >= NTHREAD_LOGIC) {
            mperf_log_warn("unable to conclude a warp size by method b");
        }

        mperf_log("discovered the warp size being %d by method b",
                  nthread_warp_b);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return nthread_warp_b;
    }
}

int img_cache_hierarchy_pchase(OpenCLEnv& env) {
    printf("access range (byte), stride (byte), niter, time (us)\n");

    const int MAX_IMG_WIDTH = env.m_dev_info.img_width_max;
    const int NCOMP = 4;
    const int PX_SIZE = NCOMP * sizeof(int32_t);

    static_assert(NCOMP == 1 || NCOMP == 2 || NCOMP == 4,
                  "image component count must be 1, 2 or 4");

    const int MAX_DATA_SIZE = MAX_IMG_WIDTH * PX_SIZE;
    const int NPX = MAX_DATA_SIZE / PX_SIZE;

    const int MAX_LV = 4;

    // Compensate less because p-chase is much more sensitive than other tests.
    double COMPENSATE = 0.01;
    const double THRESHOLD = 10;

    int NITER;

    cl_mem dst_buf = env.malloc_buffer(0, sizeof(int32_t));

    auto src = utils::format(R"(
      __constant sampler_t SAMPLER =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
      __kernel void img_cache_hierarchy_pchase(
        __read_only image1d_t src,
        __global int* dst,
        const int niter
      ) {
        int idx = 0;
        for (int i = 0; i < niter; ++i) {
          idx = read_imagei(src, SAMPLER, idx).x;
        }
        *dst = idx;
      }
    )");
    cl_int err = 0;
    cl_program program = env.build_program_from_source(src);
    cl_kernel kernel =
            clCreateKernel(program, "img_cache_hierarchy_pchase", &err);

    ImageFormat img_fmt(channel_order_by_ncomp(NCOMP), CL_SIGNED_INT32);
    cl_mem src_img = env.create_image_1d(0, img_fmt, NPX);

    auto bench = [&](int ndata, int stride) {
        auto mapped = env.map_img_1d(src_img, NPX);
        int32_t* idx_buf = (int32_t*)(void*)mapped;
        // The loop ends at `ndata` because we don't expect to read more than
        // this amount.
        for (int i = 0; i < ndata; ++i) {
            idx_buf[i * NCOMP] = (i + stride) % ndata;
        }
        env.unmap_img_1d(src_img, mapped);

        NDRange global(1, 1, 1);
        NDRange local(1, 1, 1);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_img);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_buf);
        clSetKernelArg(kernel, 2, sizeof(cl_int), &NITER);

        double time = env.bench_kernel(kernel, global, local, 10);
        mperf_log_debug("range=%d B; stride=%d; time=%f us", ndata * PX_SIZE,
                        stride * PX_SIZE, time);
        printf("%10d,%10d,%10d,%7.3f\n", ndata * PX_SIZE, stride * PX_SIZE,
               NITER, time);
        return time;
    };

    env.ensure_min_niter(1000, NITER, [&]() { return bench(1, 1); });
    // Make sure every part of the memory is accessed at least once.
    NITER = std::max(NPX * 2, NITER);
    COMPENSATE *= 1000 / bench(1, 1);

    int ndata_cacheline;
    int ndata_cache;

    ndata_cacheline = 1;
    ndata_cache = 1;

    for (int lv = 1; lv <= MAX_LV; ++lv) {
        // Step 1: Find the size of cache.
        std::string cache_name = "CachePixelCountLv" + std::to_string(lv);
        {
            DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
            int ndata = ndata_cache + ndata_cacheline;
            for (; ndata <= NPX; ndata += ndata_cacheline) {
                double time = bench(ndata, ndata_cacheline);
                if (dj.push(time)) {
                    ndata_cache = ndata - ndata_cacheline;
                    mperf_log("found cache size %s",
                              pretty_data_size(ndata_cache * PX_SIZE).c_str());
                    printf("%s: %d\n", cache_name.c_str(),
                           ndata_cache * PX_SIZE);
                    break;
                }
            }
            if (ndata >= NPX) {
                ndata_cache = NPX;
                mperf_log("found allocation boundary %s",
                          pretty_data_size(ndata_cache * PX_SIZE).c_str());
                break;
            }
        }

        // Step 2: Find the size of cacheline. It's possible the test might fail
        // because of noisy benchmark results when execution time is too long.
        std::string cacheline_name =
                "CachelinePixelCountLv" + std::to_string(lv);
        {
            DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
            int stride = ndata_cacheline;
            for (; stride < ndata_cache; ++stride) {
                double time = bench(ndata_cache + ndata_cacheline, stride);
                if (dj.push(time)) {
                    ndata_cacheline = stride - 1;
                    mperf_log("found cacheline size %s",
                              pretty_data_size(ndata_cacheline * PX_SIZE)
                                      .c_str());
                    printf("%s: %d\n", cacheline_name.c_str(),
                           ndata_cacheline * PX_SIZE);
                    break;
                }
            }
        }
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(src_img);
    return ndata_cache * PX_SIZE;
}

int buf_cache_hierarchy_pchase(OpenCLEnv& env) {
    printf("range (byte), stride (byte), niter, time (us)\n");

    const size_t MAX_BUF_SIZE = 512 * 1024;
    const int NCOMP = 4;
    const int VEC_SIZE = NCOMP * sizeof(int32_t);

    static_assert(
            NCOMP == 1 || NCOMP == 2 || NCOMP == 4 || NCOMP == 8 || NCOMP == 16,
            "buffer vector component count must be 1, 2, 4, 8 or 16");

    const int MAX_DATA_SIZE = MAX_BUF_SIZE * VEC_SIZE;
    const int NVEC = MAX_BUF_SIZE / VEC_SIZE;

    const int MAX_LV = 4;

    // Compensate less because p-chase is much more sensitive than other tests.
    double COMPENSATE = 0.01;
    const double THRESHOLD = 10;

    int NITER;

    cl_mem dst_buf = env.malloc_buffer(0, sizeof(int32_t));

    auto src = utils::format(R"(
      __constant sampler_t SAMPLER =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
      __kernel void buf_cache_hierarchy_pchase(
        __global )",
                             vec_name_by_ncomp("int", NCOMP), R"(* src,
        __global int* dst,
        const int niter
      ) {
        int idx = 0;
        for (int i = 0; i < niter; ++i) {
          idx = src[idx].x;
        }
        *dst = idx;
      }
    )");
    cl_int err = 0;
    cl_program program = env.build_program_from_source(src);
    cl_kernel kernel =
            clCreateKernel(program, "buf_cache_hierarchy_pchase", &err);

    cl_mem src_buf = env.malloc_buffer(0, NVEC * VEC_SIZE);

    auto bench = [&](int ndata, int stride) {
        auto mapped = env.map_buf(src_buf, 0, NVEC * VEC_SIZE);
        int32_t* idx_buf = (int32_t*)(void*)mapped;
        for (int i = 0; i < ndata; ++i) {
            idx_buf[i * NCOMP] = (i + stride) % ndata;
        }
        env.unmap_buf(src_buf, mapped);

        NDRange global(1, 1, 1);
        NDRange local(1, 1, 1);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_buf);
        clSetKernelArg(kernel, 2, sizeof(cl_int), &NITER);

        double time = env.bench_kernel(kernel, global, local, 10);
        mperf_log_debug("range=%d B; stride=%d B; time=%f us", ndata * VEC_SIZE,
                        stride * VEC_SIZE, time);
        printf("%10d,%10d,%10d,%7.3f\n", ndata * VEC_SIZE, stride * VEC_SIZE,
               NITER, time);
        return time;
    };

    env.ensure_min_niter(1000, NITER, [&]() { return bench(1, 1); });
    // Make sure every part of the memory is accessed at least once.
    NITER = std::max(NVEC * 2, NITER);
    COMPENSATE *= 1000 / bench(1, 1);

    int ndata_cacheline;
    int ndata_cache;

    ndata_cacheline = 1;
    ndata_cache = 1;

    for (int lv = 1; lv <= MAX_LV; ++lv) {
        // Step 1: Find the size of cache.
        std::string cache_name = "CacheVectorCountLv" + std::to_string(lv);
        {
            DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
            int ndata = ndata_cache + ndata_cacheline;
            for (; ndata <= NVEC; ndata += ndata_cacheline) {
                double time = bench(ndata, ndata_cacheline);
                if (dj.push(time)) {
                    ndata_cache = ndata - ndata_cacheline;
                    mperf_log("found cache size %s",
                              pretty_data_size(ndata_cache * VEC_SIZE).c_str());
                    printf("%s: %d\n", cache_name.c_str(),
                           ndata_cache * VEC_SIZE);
                    break;
                }
            }
            if (ndata >= NVEC) {
                ndata_cache = NVEC;
                mperf_log("found allocation boundary %s",
                          pretty_data_size(ndata_cache * VEC_SIZE).c_str());
                break;
            }
        }

        // Step 2: Find the size of cacheline. It's possible the test might fail
        // because of noisy benchmark results when execution time is too long
        std::string cacheline_name =
                "CachelineVectorCountLv" + std::to_string(lv);
        {
            DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
            int stride = ndata_cacheline;
            for (; stride < ndata_cache; ++stride) {
                double time = bench(ndata_cache + ndata_cacheline, stride);
                if (dj.push(time)) {
                    ndata_cacheline = stride - 1;
                    mperf_log("found cacheline size %s",
                              pretty_data_size(ndata_cacheline * VEC_SIZE)
                                      .c_str());
                    printf("%s: %d\n", cacheline_name.c_str(),
                           ndata_cacheline * VEC_SIZE);
                    break;
                }
            }
        }
    }
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return ndata_cache * VEC_SIZE;
}

int gpu_max_reg_num_per_thread() {
    return reg_count(OpenCLEnv::instance());
}
int gpu_warp_size() {
    return warp_size(OpenCLEnv::instance());
}

double gpu_mem_bw() {
    return buf_bandwidth(OpenCLEnv::instance());
}
int gpu_unified_cache_latency() {
    // TBD.
    return 0;
}
int gpu_unified_cacheline_size() {
    return buf_cacheline_size(OpenCLEnv::instance());
}
int gpu_unified_cache_hierarchy_pchase() {
    return buf_cache_hierarchy_pchase(OpenCLEnv::instance());
}

double gpu_texture_cache_bw() {
    return img_bandwidth(OpenCLEnv::instance());
}
int gpu_texture_cacheline_size() {
    return img_cacheline_size(OpenCLEnv::instance());
}
int gpu_texture_cache_hierachy_pchase() {
    return img_cache_hierarchy_pchase(OpenCLEnv::instance());
}

float gpu_insts_gflops_latency() {
    auto env = OpenCLEnv::instance();
    env.print_device_info();
    return env.insts_gflops_latency();
}

float gpu_local_memory_bw() {
    auto env = OpenCLEnv::instance();
    // env.print_device_info();
    return env.local_memory_bw();
}

float gpu_global_memory_bw() {
    auto env = OpenCLEnv::instance();
    // env.print_device_info();
    return env.global_memory_bw();
}
}  // namespace mperf