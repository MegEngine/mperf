cmake_minimum_required(VERSION 3.15.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

project(mperf LANGUAGES C CXX ASM VERSION 0.1)
set(CMAKE_CXX_STANDARD 14)

option(MPERF_ENABLE_OPENCL "build mperf with opencl." OFF)
option(MPERF_ENABLE_ADRENO "build mperf with adreno." OFF)
option(MPERF_ENABLE_MALI "build mperf with mali." OFF)
option(MPERF_ENABLE_PFM "build mperf with pfm." OFF)
option(MPERF_ENABLE_INSTALL "enable installation of mperf." ON)

if(CMAKE_TOOLCHAIN_FILE)
    message(STATUS "cross compile mperf.")
    if(ANDROID_TOOLCHAIN_ROOT)
        if(NOT "${ANDROID_ARCH_NAME}" STREQUAL "")
            set(ANDROID_ARCH ${ANDROID_ARCH_NAME})
        endif()
        if(${ANDROID_ARCH} STREQUAL "arm")
            set(MPERF_ARCH "armv7")
        elseif(${ANDROID_ARCH} STREQUAL "arm64")
            set(MPERF_ARCH "aarch64")
        else()
            message(FATAL_ERROR "DO NOT SUPPORT ANDROID ARCH NOW")
        endif()
    elseif(NOT "${ARM_CROSS_BUILD_ARCH}" STREQUAL "")
        set(MPERF_ARCH ${ARM_CROSS_BUILD_ARCH})
    else()
        message(FATAL_ERROR "Unknown cross-compiling settings.")
    endif()
else()
    message("CMAKE SYSTEM PROCESSOR :${CMAKE_SYSTEM_PROCESSOR}.")
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
        set(MPERF_ARCH "x86")
    else()
        set(MPERF_ARCH "aarch64")
    endif()
endif()

message(STATUS "CONFIG MPERF_ARCH TO ${MPERF_ARCH}")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 -g ${CMAKE_CXX_FLAGS}")
else()
    message(STATUS "build in Release mode")
    set(CMAKE_C_FLAGS "-Ofast ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-Ofast ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function ${CMAKE_CXX_FLAGS}")

if(${MPERF_ARCH} STREQUAL "aarch64")
    set(MPERF_AARCH64 ON)
    set(CMAKE_CXX_FLAGS " -Wno-deprecated-register ${CMAKE_CXX_FLAGS}")
elseif(${MPERF_ARCH} STREQUAL "armv7")
    set(MPERF_ARMV7 ON)
    set(CMAKE_CXX_FLAGS " -Wno-deprecated-register ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS}")
elseif(${MPERF_ARCH} STREQUAL "x86")
    set(MPERF_X86 ON)
    set(CMAKE_CXX_FLAGS " -Wno-unused-but-set-variable ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS " -msse4.2 -mavx2 -mfma ${CMAKE_CXX_FLAGS}")
else()
    message(FATAL_ERROR "Unknown MPERF_ARCH ${MPERF_ARCH}.")
endif()

file(GLOB_RECURSE SOURCES common/cpu_info.cpp common/timer.cpp common/utils.cpp common/cpu_affinity.cpp uarch/cpu/*.cpp)
if(MPERF_ENABLE_OPENCL)
  file(GLOB_RECURSE SOURCES_ common/opencl_driver.cpp uarch/gpu/*.cpp)
  list(APPEND SOURCES ${SOURCES_})
endif()

add_library(mperf ${SOURCES})

if(MPERF_ENABLE_OPENCL)
    message(STATUS "mperf build with opencl.")
    set(MPERF_WITH_OPENCL ON)
    add_subdirectory(third_party/opencl-stub)
    target_include_directories(mperf
                               PUBLIC
                               $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/opencl-stub/include/>
                               $<INSTALL_INTERFACE:include>)
    target_link_libraries(mperf PUBLIC OpenCL)
endif()
if(MPERF_ENABLE_ADRENO)
    message(STATUS "mperf build with adreno.")
    set(MPERF_WITH_ADRENO ON)
endif()
if(MPERF_ENABLE_MALI)
    message(STATUS "mperf build with mali.")
    set(MPERF_WITH_MALI ON)
endif()
if(MPERF_ENABLE_LOGGING)
    message(STATUS "mperf build with logging.")
    set(MPERF_WITH_LOGGING ON)
endif()
if(MPERF_ENABLE_PFM)
    message(STATUS "mperf build with pfm.")
    set(MPERF_WITH_PFM ON)
endif()

add_subdirectory(eca/xpmu)
if(MPERF_ENABLE_PFM)
    add_subdirectory(third_party/libpfm4)
    target_include_directories(mperf_xpmu PRIVATE ${PROJECT_SOURCE_DIR}/third_party/libpfm4/include/)
    target_link_libraries(mperf_xpmu PUBLIC pfm)
endif()

add_subdirectory(eca/tma)
target_link_libraries(mperf_tma PRIVATE mperf_xpmu)
target_include_directories(mperf
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
                           $<INSTALL_INTERFACE:include>)
target_include_directories(mperf PRIVATE ${PROJECT_SOURCE_DIR}/uarch/gpu/kernels/)

target_link_libraries(mperf PUBLIC mperf_xpmu)
target_link_libraries(mperf PUBLIC mperf_tma)

add_subdirectory(apps)

if(ANDROID)
    target_link_libraries(mperf PRIVATE log)
endif()

if(${MPERF_ARCH} STREQUAL "x86")
    find_package(Threads REQUIRED)
    target_link_libraries(mperf PRIVATE Threads::Threads)
endif()

# generate the build config
set(generate_config_header ${PROJECT_BINARY_DIR}/mperf_build_config.h)
message(STATUS "the generate_config_header is ${generate_config_header}")
configure_file("${PROJECT_SOURCE_DIR}/cmake/mperf_build_config.h.in" ${generate_config_header})
target_include_directories(mperf
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
                           $<INSTALL_INTERFACE:include>)

# about install
message(STATUS "install mperf")
message(STATUS "CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")
set(generated_dir "${PROJECT_BINARY_DIR}")

set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")

set(namespace "${PROJECT_NAME}::")

configure_file("${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in" "${project_config}" @ONLY)

set(targets_to_export mperf mperf_xpmu mperf_tma)
if(MPERF_ENABLE_OPENCL)
    list(APPEND targets_to_export OpenCL)
endif()

if(MPERF_ENABLE_PFM)
    list(APPEND targets_to_export pfm)
endif()

if(${MPERF_ARCH} STREQUAL aarch64 AND MPERF_ENABLE_PFM)
    list(APPEND targets_to_export pfm_aarch64)
elseif(${MPERF_ARCH} STREQUAL armv7 AND MPERF_ENABLE_PFM)
    list(APPEND targets_to_export pfm_armv7)
elseif(${MPERF_ARCH} STREQUAL x86 AND MPERF_ENABLE_PFM)
    list(APPEND targets_to_export pfm_x86)
endif()

if(MPERF_ENABLE_INSTALL)
    install(
        TARGETS ${targets_to_export}
        EXPORT ${targets_export_name}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/include/mperf"
        DESTINATION include
        FILES_MATCHING PATTERN "*.*h")

    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/include/mperf/xpmu"
        DESTINATION include/mperf
        FILES_MATCHING PATTERN "*.*h")

    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/include/mperf/tma"
        DESTINATION include/mperf
        FILES_MATCHING PATTERN "*.*h")

    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/third_party/opencl-stub/include/MCL"
        DESTINATION include/
        FILES_MATCHING PATTERN "*.*h")

    install(
        FILES "${generate_config_header}"
        DESTINATION include/)
        
    install(
        FILES "${project_config}"
        DESTINATION "lib/cmake/${PROJECT_NAME}")

    install(
        EXPORT "${targets_export_name}"
        NAMESPACE "${namespace}"
        DESTINATION "lib/cmake/${PROJECT_NAME}")
endif()