file(GLOB_RECURSE SOURCES tma.cpp arch_ratios/arch_ratios.cpp)

if(ANDROID)
    file(GLOB_RECURSE SOURCES_ arch_ratios/a5*_ratios.cpp)
    list(APPEND SOURCES ${SOURCES_})
endif()

if(${MPERF_ARCH} STREQUAL x86)
    file(GLOB_RECURSE SOURCES_ arch_ratios/hsx_server_ratios.cpp)
    list(APPEND SOURCES ${SOURCES_})
endif()

add_library(mperf_tma ${SOURCES})
target_include_directories(mperf_tma
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
                           $<INSTALL_INTERFACE:include>)

