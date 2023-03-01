# mperf
<p align="center">
  <img width="502" height="206" src="logo.svg">
</p>

[Release Notes](CHANGELOG.md) | [Roadmap](ROADMAP.md) | [Apps](apps) | [English](README.md)

mperf是一个微架构层次的算子性能调优工具箱，主要面向移动/嵌入式平台的CPU/GPU核心，目标是“为构建一个更接近闭环的算子调优反馈回路”提供系列基础工具。

## 功能及特点
* 测试微架构层次的各类常用性能分析参数(性能峰值/带宽/延迟等)
* 绘制 Hierarchical Roofline，用于评估优化水平
* 提供 CPU/GPU PMU 数据获取能力
* 提供 PMU Metrics/[TMA](https://www.intel.com/content/www/us/en/develop/documentation/vtune-cookbook/top/methodologies/top-down-microarchitecture-analysis-method.html#top-down-microarchitecture-analysis-method_GUID-FA8F07A1-3590-4A91-864D-CE96456F84D7) 分析能力，用于分析性能瓶颈
* 提供 OpenCL Linter方案，用于指导OpenCL算子优化(后续版本提供)
* C++工程
* 轻量级的、可嵌入的API级别的库
* 目前支持架构：ARM CPUs, Mali GPUs, Adreno 6xx GPUs, 可扩展更多架构
* 暂不能完整支持iOS系统

## 安装
mperf 支持 CMake 编译，要求 CMake 版本不低于 3.15.2，可以遵照以下步骤进行编译：
* 下载 repo 代码：
    ```bash
    git clone https://github.com/MegEngine/mperf.git
    ```
* 请选择一个试验平台
    - 如果你是在一个安卓手机平台进行试验
        - 则需要先安装一个 [ndk](https://developer.android.com/ndk) 环境
            - 选择一个版本的 ndk 下载后解压到 host 设备上
            - 设置环境变量 `NDK_ROOT` 为解压后的 ndk 目录
    - 如果你是在一个 Linux 系统下的 x86 平台上进行试验
        - 则需要保证可以在环境变量 `PATH` 中找到gcc或这clang编译器
* 如果是安卓平台，可以执行 `android_build.sh` 脚本帮助构建mperf
    * 查询 `android_build.sh` 脚本的使用样例
        ```bash
        ./android_build.sh -h
        ```
    * 构建支持 armv7 cpu 的 mperf
        ```bash
        ./android_build.sh -m armeabi-v7a
        ```
    * 构建支持 arm aarch64 cpu 的 mperf
        ```bash
        ./android_build.sh -m arm64-v8a
        ```
    * 构建支持 mali gpu 的 mperf
        ```bash
        ./android_build.sh -g mali [arm64-v8a, armeabi-v7a]
        ```
    * 构建支持 adreno gpu 的 mperf
        ```bash
        ./android_build.sh -g adreno [arm64-v8a, armeabi-v7a]
        ```
    * 构建支持 pfm 的 mperf
        ```bash
        ./android_build.sh -p [arm64-v8a, armeabi-v7a]
        ```
    * 构建 debug 版本的 mperf
        ```bash
        ./android_build.sh -d [arm64-v8a, armeabi-v7a]
        ```
* 如果你的试验平台是 Linux X86，你可以开启 pfm 支持进行构建：
    ```bash
    cmake -S . -B "build-x86" -DMPERF_ENABLE_PFM=ON
    cmake --build "build-x86" --config Release
    ```
* 构建完成之后，一些可执行文件会被转存到 build_dir/apps 目录下。同时，你也可以将编译好的mperf安装到系统目录或者自定义的安装目录下
    ```bash
    cmake --build <mperf_build_dir> --target install
    ```
* 至此，你可以使用 CMake `find_package` 命令导入安装好的mperf，操作如下：
    ```bash
    # set(mperf_DIR /path/to/your/installed/mperfConfig.cmake) # uncomment it if cannot find mperfConfig.cmake
    find_package(mperf REQUIRED)
    target_link_libraries(your_target PRIVATE mperf::mperf_interface_include)
    target_link_libraries(your_target PRIVATE "/path/to/your/installed/libmperf.a")
    ```
* 当然，你也可以直接在你的项目中源码集成  mperf，并用 `add_subdirectory(mperf)` 命令引入对 mperf 的编译依赖。

## 使用样例
* mperf xpmu 模块的基础使用样例：
    ```bash
    mperf::CpuCounterSet cpuset = "CYCLES,INSTRUCTIONS,...";
    mperf::XPMU xpmu(cpuset);
    xpmu.run();
    
    ... // add your function to be measured

    xpmu.sample();
    xpmu.stop();
    ```
    更多详细信息请参考 [cpu_pmu](apps/cpu_pmu_transpose.cpp) / [mali_pmu](apps/gpu_mali_pmu_test.cpp) / [adreno_pmu](apps/gpu_adreno_pmu_test.cpp)。
* mperf tma 模块的基础使用样例：
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
    更多详细信息请参考 [arm_cpu_tma](apps/cpu_tma_transpose.cpp)。

## 源码目录结构
* `apps` 各种使用样例，详细信息请查看[apps 文档](./apps/README.md)。
* `eca` 搜集和分析 PMU 数据(包括 TMA 分析)。
* `uarch` 一系列基础 micro-benchmark 用于测试CPU/GPU微架构层次的各种常用性能参数。
* `doc` 一些关于 mperf 的 roofline 和 tma 功能如何使用的文档，具体列表请查看 [文档索引](doc/index.md)。
* `cmake` cmake相关的文件。
* `common` 一些公共的基础组件。
* `third_party` mperf 对外依赖的库。 
* `linter` OpenCL Linter [TBD]。

## Tutorial
* 一个关于如何在 ARM A55 核心上将矩阵乘优化到极致性能的教程，从中阐述了如何借助 mperf 进行性能优化工作的基本逻辑，请参见[optimize the matmul with the help of mperf](doc/how_to_optimize_matmul/借助mperf进行矩阵乘法极致优化.md)。

## 开源许可
mperf 使用 [Apache-2.0](LICENSE) License。
