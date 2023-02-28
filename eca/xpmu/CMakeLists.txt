

file(GLOB_RECURSE SOURCES xpmu.cpp vendor/cpu/*.cpp ${PROJECT_SOURCE_DIR}/common/utils.cpp)
if(MPERF_ENABLE_MALI)
  file(GLOB_RECURSE SOURCES_ vendor/mali/*.cpp)
  list(APPEND SOURCES ${SOURCES_})
endif()

if(MPERF_ENABLE_ADRENO)
  file(GLOB_RECURSE SOURCES_ vendor/adreno/*.cpp)
  list(APPEND SOURCES ${SOURCES_})
endif()

add_library(mperf_xpmu ${SOURCES})
target_include_directories(mperf_xpmu
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
                           $<INSTALL_INTERFACE:include>)
target_include_directories(mperf_xpmu
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
                           $<INSTALL_INTERFACE:include>)
