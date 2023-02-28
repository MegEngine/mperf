This file will introduce the step to get the roofline date through mperf and plot the roofline graph.
# get the data for roofline
## get the data for the platform
### arm cpu
* build mperf for arm64 cpu
    ```bash
    ./android_build.sh -m arm64-v8a
    ```

    #### get peak memory bandwidth of the cpu
    * copy the executable file `apps/cpu_spec_dram_bw` to your android phone and run it, you will get the GBPs for the cpu

    #### get peak compute bandwidth of the cpu
    * copy the executable file `apps/cpu_inst_gflops_latency` to your android phone and run it, you will get the GFLOPs for the cpu

### mobile gpu
* build mperf for the adreno or mali
    ```bash
    ./android_build.sh -m arm64-v8a -g [adreno | mali]
    ```
    #### get peak memory bandwidth of the gpu
    * copy the executable file `apps/gpu_spec_dram_bw` to your android phone and run it, you will get the GBPs for the gpu

    #### get peak compute bandwidth of the gpu
    * copy the executable file `apps/gpu_inst_gflops_latency` to your android phone and run it, you will get the GFLOPs for the gpu

## get the data for your function call
### arm cpu
* enable the following metrics for mperf tma in [cpu_tma_transpose](../apps/cpu_tma_transpose.cpp#L891)
    ```bash
    mpf_tma.init(
            {"Metric_GFLOPs_Use", "Metric_DRAM_BW_Use", "Metric_L3_BW_Use", "Metric_L2_BW_Use"});
    ```
    recompile, copy to your phone, and run it, you will get the memory and compute bandwidth for your function call. And in this case, the function is an neon transpose implement. 
### mobile gpu

##### mali gpu
* recompile mperf with mali gpu enabled, and copy the executable file `apps/gpu_mali_pmu_test` to your android phone and run it, you will get the GBPs and GFLOPs for your custom gpu kernel call on mali gpu.
##### adreno gpu
* recompile mperf with adreno gpu enabled, and copy the executable file `apps/gpu_adreno_pmu_test` to your android phone and run it, you will get the GBPs and GFLOPs for your custom gpu kernel call on adreno gpu.
# plot roofline
* now, you get all the data need to plot roofline, please edit these values into a roofline_data.txt file like [roofline_data.txt](../apps/scripts/roofline/roofline_data.txt), and you may need edit the params about axes coordinate range at [plot_roofline.py](../apps/scripts/roofline/plot_roofline.py#L65) according to values of the peak memory and compute bandwidth on your platform, and finally, plot it
    ```
    python3 plot_roofline.py ./roofline_data.txt
    ```
    see the [roofline_readme](../apps/scripts/roofline/README.md) for more details.

