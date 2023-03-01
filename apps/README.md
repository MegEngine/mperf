# apps, there are variouse user examples.

## scripts
* `scripts/bench_cpu_mem_bw` script to run `cpu_mem_bw` and process the results of `cpu_mem_bw`
* `scripts/roofline` script and document to draw graph of roofline model. 

## basic testcases
* `cpu_info_test.cpp` get cpu information(Eg. number of big-core/freq)
* `cpu_inst_gflops_latency.cpp` measure instruction throughput/latency
* `cpu_mem_bw.cpp` measure CPU hierarchical memory bandwidths/latency of micro-kernels
* `cpu_stream.cpp` mperf version of John McCalpin's STREAM benchmark
* `cpu_spec_dram_bw.cpp` measure dram bandwidth
* `cpu_pmu_transpose.cpp` collect data of cpu pmu events
* `cpu_tma_transpose.cpp` ARM TMA example
* `gpu_march_probe.cpp` get gpu micro-arch parameters(number of register/warp size/Cache Line size)
* `gpu_spec_dram_bw.cpp` measure GPU DRAM Bandwidth
* `gpu_mem_bw.cpp` measure Bandwidth of GPU multi-level caches
* `gpu_adreno_pmu_test.cpp` collect data of Adreno GPU pmu events
* `gpu_mali_pmu_test.cpp` collect data of Mali GPU pmu events
* `gpu_inst_gflops_latency.cpp` measure gpu/OpenCL instruction throughput/latency

## arm cpu pmu analysis cases
* `cpu_pmu_analysis/` 
* store some study cases on arm cpu platform, keep adding.

## mali pmu analysis cases
* `mali_pmu_analysis/` 
* store some study cases on mali gpu platform, keep adding.

## adreno pmu analysis cases
* `adreno_pmu_analysis/` 
* store some study cases on adreno gpu platform, keep adding.
