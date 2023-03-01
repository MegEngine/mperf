# mperf
<p align="center">
  <img width="502" height="206" src="logo.svg">
</p>

[Release Notes](CHANGELOG.md) | [Roadmap](ROADMAP.md) | [Apps](apps) | [中文](README_CH.md)

mperf is a modular micro-benchmark/toolkit for kernel performance analysis.

## Features.
* Investigate the basic micro-architectural parameters(uarch) of the target CPU/GPU.
* Draw graph of hierarchical roofline model, used to evaluate performance.
* Collect CPU/GPU PMU events data.
* Analyze CPU/GPU PMU events data([TMA Methodology](https://www.intel.com/content/www/us/en/develop/documentation/vtune-cookbook/top/methodologies/top-down-microarchitecture-analysis-method.html#top-down-microarchitecture-analysis-method_GUID-FA8F07A1-3590-4A91-864D-CE96456F84D7) and customized metrics), used to identify performance bottlenecks.
* OpenCL Linter, used to guide manual OpenCL kernel optimization[TBD].
* C++ Project
* support platform: ARM CPUs, Mali GPUs, Adreno 6xx GPUs
* Lightweight and embeddable library
* The iOS platform is not yet fully functional.

## Installation
mperf support CMake build system and require CMake version upper than 3.15.2, you can compile the mperf follow the step:
* clone or download the project
    ```bash
    git clone https://github.com/MegEngine/mperf.git
    git submodule update --init --recursive
    ```
* choose a test platform
    - if you will test arm processor in android OS
        - a [ndk](https://developer.android.com/ndk) is required
            - download the NDK and extract to the host machine
            - set the `NDK_ROOT` env to the path of extracted NDK directory
    - if you will test x86 processor in linux OS
        - a gcc or clang compiler should find by cmake through `PATH` env
* if your target test OS is android，run the `android_build.sh` to build it
    * print the usage about android_build.sh
        ```bash
        ./android_build.sh -h
        ```
    * build for armv7 cpu
        ```bash
        ./android_build.sh -m armeabi-v7a
        ```
    * build for arm64 cpu
        ```bash
        ./android_build.sh -m arm64-v8a
        ```
    * build with mali mobile gpu
        ```bash
        ./android_build.sh -g mali [arm64-v8a, armeabi-v7a]
        ```
    * build with adreno mobile gpu
        ```bash
        ./android_build.sh -g adreno [arm64-v8a, armeabi-v7a]
        ```
    * build with pfm
        ```bash
        ./android_build.sh -p [arm64-v8a, armeabi-v7a]
        ```
    * build in debug mode
        ```bash
        ./android_build.sh -d [arm64-v8a, armeabi-v7a]
        ```
* if you target test OS is linux，if you want to enable pfm add `-DMPERF_ENABLE_PFM=ON` to cmake command
    ```bash
    cmake -S . -B "build-x86" -DMPERF_ENABLE_PFM=ON
    cmake --build "build-x86" --config Release 
    ```
* after build, some executable files are stored in mperf build_dir/apps directory. And you can install the mperf to your system path or your custom install directory
    ```bash
    cmake --build <mperf_build_dir> --target install 
    ```
* and now, you can use `find_package` command to import the installed mperf, and use like
    ```bash
    # set(mperf_DIR /path/to/your/installed/mperfConfig.cmake) # uncomment it if cannot find mperfConfig.cmake
    find_package(mperf REQUIRED)
    target_link_libraries(your_target PRIVATE mperf::mperf_interface_include)
    target_link_libraries(your_target PRIVATE "/path/to/your/installed/libmperf.a")
    ```
* alternatively, `add_subdirectory(mperf)` will incorporate the library directly in to your's CMake project.

## Usage
* basic usage for mperf xpmu module:
    ```bash
    mperf::CpuCounterSet cpuset = "CYCLES,INSTRUCTIONS,...";
    mperf::XPMU xpmu(cpuset);
    xpmu.run();
    
    ... // add your function to be measured

    xpmu.sample();
    xpmu.stop();
    ```
    please see [cpu_pmu](apps/cpu_pmu_transpose.cpp) / [mali_pmu](apps/gpu_mali_pmu_test.cpp) / [adreno_pmu](apps/gpu_adreno_pmu_test.cpp) for more details.
* basic usage for mperf tma module:
    ```bash
    mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::A55);
    mpf_tma.init(
            {"Frontend_Bound", "Bad_Speculation", "Backend_Bound", "Retiring", ...});
    size_t gn = mpf_tma.group_num();
    for (size_t i = 0; i < gn; ++i) {
        mpf_tma.start(i);
        for (size_t j = 0; j < iter_num; ++j) {
            ... // add your function to be measured
        }
        mpf_tma.sample_and_stop(iter_num);
    }
    mpf_tma.deinit();
    ```
    please see [arm_cpu_tma](apps/cpu_tma_transpose.cpp) for more details.

## Source Directory Structure
* `apps` Various user examples, please see [apps doc](./apps/README.md) for more details.
* `eca` A module for collecting and analyzing PMU events data(Including TMA analysis).
* `uarch` A set of low-level micro-benchmarks to investigate the basic micro-architectural parameters(uarch) of the target CPU/GPU.
* `doc` Some documents about roofline and tma usage, please see [index](doc/index.md) for the list.
* `cmake` Some cmake relative files.
* `common` Some common helper functions.
* `third_party` Some dependent libraries.
* `linter` OpenCL Linter [TBD].

## Tutorial
* A tutorial about how to optimize matmul to achieve peak performance on ARM A55 core, which will illustrate the basic logic of how to use mperf help your optimization job, please reference [optimize the matmul with the help of mperf](doc/how_to_optimize_matmul/借助mperf进行矩阵乘法极致优化.md). 

## License
mperf is licensed under the [Apache-2.0](LICENSE) license.
