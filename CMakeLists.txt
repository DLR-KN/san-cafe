cmake_minimum_required (VERSION 2.6)
project (CAFE)
enable_testing()

set(CMAKE_CXX_FLAGS "-D_FORCE_INLINES ${CMAKE_CXX_FLAGS}")

set(NVCC_FLAGS_EXTRA "${NVCC_FLAGS_EXTRA} -D_FORCE_INLINES")

find_package(CUDA QUIET REQUIRED)
# Add Subdirectories for achitecture recognition and utility code
add_subdirectory(archs)
add_subdirectory(util)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_52,code=sm_52 -Xptxas -v -D_FORCE_INLINES)

list(APPEND cafe_gpu_lib_srcs
     ${CMAKE_CURRENT_SOURCE_DIR}/lib/pfb.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/lib/stream_to_streams.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/lib/filterbank_execute.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/lib/arb_resampler.cu)

list(APPEND cafe_host_lib_srcs
     ${CMAKE_CURRENT_SOURCE_DIR}/lib/cafe_cuda_helper.cc
     ${CMAKE_CURRENT_SOURCE_DIR}/lib/pfb_channelizer.cc
     ${CMAKE_CURRENT_SOURCE_DIR}/lib/pfb_arb_resampler.cc)
     # ${CMAKE_CURRENT_SOURCE_DIR}/lib/pfb_synthesizer.cc

SET(CAFE_TEST_DIR "${CMAKE_SOURCE_DIR}/testing/testfiles/")## Configure the Debug Transmitter Experiment
configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/testing/include/cafe/testing/qa_pfb_resampler.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/testing/include/cafe/testing/qa_pfb_resampler.h"
)

list(APPEND cafe_test_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/testing/src/test_san_cafe.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/testing/src/qa_pfb_resampler.cc)

include_directories(
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/
                    ${CMAKE_CURRENT_BINARY_DIR}/include/
                    ${CMAKE_CURRENT_SOURCE_DIR}/archs/
                    ${CMAKE_CURRENT_SOURCE_DIR}/testing/include/cafe/
                    ${CMAKE_BINARY_DIR}/testing/include/cafe/
                    ${CMAKE_CURRENT_SOURCE_DIR}/util/include/cafe/
)

# Build the GPU Device Library with c++98
set(CMAKE_CXX_FLAGS "-O3 -Wall -g -std=c++98")
cuda_add_library(cafe_gpu SHARED ${cafe_gpu_lib_srcs})

# Build the Host Library with C++14
set(CMAKE_CXX_FLAGS "-O3 -Wall -g -std=c++14")
cuda_add_library(cafe_host SHARED ${cafe_host_lib_srcs})

# Add test executable
add_executable(test_san_cafe ${cafe_test_srcs})

# Link gpu lib to host lib
target_link_libraries(cafe_host
                      ${CUDA_CUFFT_LIBRARIES}
                      cafe_gpu)

# link test executable to releveant libraries
target_link_libraries(test_san_cafe
                      ${CUDA_CUFFT_LIBRARIES}
                      cafe_host
                      cafe_gpu
                      cafe_util)

# Add test target
add_test(test_cafe test_san_cafe)
