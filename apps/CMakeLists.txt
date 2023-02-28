
macro(compile_test name)
    add_executable(${name} "${name}.cpp" utils/utils.cpp)
    target_link_libraries(${name} mperf_interface_include)
endmacro(compile_test)

message(STATUS "compile cpu basic tests")
compile_test(cpu_info_test)
compile_test(cpu_inst_gflops_latency)

compile_test(cpu_mem_bw)
compile_test(cpu_spec_dram_bw)

compile_test(cpu_pmu_transpose)
compile_test(cpu_tma_transpose)

if(ANDROID)
message(STATUS "compile cpu analysis cases")
macro(compile_cpu_analysis_cases dir_name binary_name prefix)
    add_executable("${prefix}${binary_name}" "${PROJECT_SOURCE_DIR}/apps/${dir_name}/${binary_name}.cpp" utils/utils.cpp)
    target_link_libraries("${prefix}${binary_name}" mperf_interface_include)
endmacro(compile_cpu_analysis_cases)

compile_cpu_analysis_cases(cpu_pmu_analysis fmla arm_cpu_)
compile_cpu_analysis_cases(cpu_pmu_analysis prefetch arm_cpu_)
compile_cpu_analysis_cases(cpu_pmu_analysis optimize_ldq arm_cpu_)
compile_cpu_analysis_cases(cpu_pmu_analysis matmul_naive arm_cpu_)
compile_cpu_analysis_cases(cpu_pmu_analysis matmul_unroll arm_cpu_)
compile_cpu_analysis_cases(cpu_pmu_analysis matmul_block arm_cpu_)
compile_cpu_analysis_cases(cpu_pmu_analysis matmul_pack arm_cpu_)
compile_cpu_analysis_cases(cpu_pmu_analysis matmul_asm arm_cpu_)

endif()

if(ANDROID)
message(STATUS "compile gpu basic tests")
compile_test(gpu_march_probe)
compile_test(gpu_inst_gflops_latency)

compile_test(gpu_adreno_pmu_test)
compile_test(gpu_mali_pmu_test)

compile_test(gpu_mem_bw)
compile_test(gpu_spec_dram_bw)

message(STATUS "compile gpu analysis cases")
macro(compile_gpu_analysis_cases dir_name binary_name prefix)
    add_executable("${prefix}${binary_name}" "${PROJECT_SOURCE_DIR}/apps/${dir_name}/${binary_name}.cpp" utils/utils.cpp)
    target_link_libraries("${prefix}${binary_name}" mperf_interface_include)
endmacro(compile_gpu_analysis_cases)

if(MPERF_ENABLE_MALI)
compile_gpu_analysis_cases(mali_pmu_analysis compare_s32_s16_s8_compute_test mali_)
compile_gpu_analysis_cases(mali_pmu_analysis compare_vload4_vload16_test mali_)
endif()

if(MPERF_ENABLE_ADRENO)
compile_gpu_analysis_cases(adreno_pmu_analysis compute adreno_)
compile_gpu_analysis_cases(adreno_pmu_analysis load adreno_)
endif()
endif()