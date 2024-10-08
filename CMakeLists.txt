cmake_minimum_required(VERSION 3.27)

project(omvac VERSION 0.0.0)

set(EXECUTABLE_OUTPUT_PATH bin)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(cmake/version.h.in version.h)

add_subdirectory(omvac)

install(TARGETS omvac DESTINATION bin)

