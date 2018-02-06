cmake_minimum_required (VERSION 2.6)
project (GAU)

# The version number.
set (GAU_VERSION_MAJOR 0)
set (GAU_VERSION_MINOR 9)

file(GLOB header_files
      "include/*.h")

install(FILES ${header_files} DESTINATION include)

