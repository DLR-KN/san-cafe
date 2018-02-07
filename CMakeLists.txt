set(CMAKE_CXX_FLAGS "-D_FORCE_INLINES ${CMAKE_CXX_FLAGS}")

set(NVCC_FLAGS_EXTRA "${NVCC_FLAGS_EXTRA} -D_FORCE_INLINES")

find_package(CUDA QUIET REQUIRED)

add_subdirectory(archs)

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

include_directories(
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/
                    ${CMAKE_CURRENT_BINARY_DIR}/include/
                    ${CMAKE_CURRENT_SOURCE_DIR}/archs/
)

# Build the GPU Device Library with c++98
set(CMAKE_CXX_FLAGS "-O3 -Wall -g -std=c++98")
cuda_add_library(cafe_gpu SHARED ${cafe_gpu_lib_srcs})

# Build the Host Library with C++14
set(CMAKE_CXX_FLAGS "-O3 -Wall -g -std=c++14")
cuda_add_library(cafe_host SHARED ${cafe_host_lib_srcs})

target_link_libraries(cafe_host
                      cafe_gpu)
