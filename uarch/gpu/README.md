## uarch/gpu
Part of the code in this module comes from [ArchProbe](https://github.com/microsoft/ArchProbe) and [MegPeak](https://github.com/MegEngine/MegPeak.git), and mperf has made some modifications.

### Features
* `gpu_max_reg_num_per_thread` gets the number of registers
* `gpu_warp_size` gets the number of registers in each warp

* `gpu_mem_bw` buffer bandwidth(multi-level cache, similar to cpu_mem_bw)(method1)
* `gpu_local_memory_bw` special local memory bandwidth(method2)
* `gpu_global_memory_bw` special global memory bandwidth(method2)
* `gpu_unified_cache_latency` L2 cache latency
* `gpu_unified_cacheline_size` L2 cache line size
* `gpu_unified_cache_hierarchy_pchase`
  
* `gpu_texture_cache_bw` texture cache bandwidth
* `gpu_texture_cacheline_size` texture cache line size
* `gpu_texture_cache_hierachy_pchase` 
