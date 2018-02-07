include_directories(
  ${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++14")

list(APPEND util_srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/writeSamples.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/readSamples.cc)

add_library(cafe_util SHARED ${util_srcs})

