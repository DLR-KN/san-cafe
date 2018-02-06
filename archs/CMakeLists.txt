enable_testing()


list(APPEND archs_sources
     ${CMAKE_CURRENT_SOURCE_DIR}/arch_detection.cc
     ${CMAKE_CURRENT_SOURCE_DIR}/archs.cc)

add_library(pacs_archs SHARED ${archs_sources})

list(APPEND test_archs_sources
      ${CMAKE_CURRENT_SOURCE_DIR}/arch_detection.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/test_archs.cc)

add_executable(test_arch ${test_archs_sources})


include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/../../testing/include")

ADD_DEFINITIONS("-Wall -g -std=c++11")

#add_custom_target(test_archs COMMAND test_archs DEPENDS test_archs)

add_test(archs test_archs)
