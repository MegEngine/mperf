package(default_visibility = ["//brain/mperf:__pkg__"])

cc_library(
    name = "opencl-stub",
    srcs = [
        "src/libopencl.cpp",
        "src/libopencl-wrap.h",
    ],
    hdrs = [
        "include/MCL/cl.h",
        "include/MCL/cl_ext.h",
        "include/MCL/cl_gl.h",
        "include/MCL/cl_gl_ext.h",
        "include/MCL/cl_platform.h",
        "include/MCL/opencl.h",
    ],
    includes = ["include"],
)
