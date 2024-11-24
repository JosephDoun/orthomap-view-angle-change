cmake_minimum_required(VERSION 3.27)

# TODO get version dynamically from file.
project(mappov VERSION 0.0.0)

set(EXECUTABLE_OUTPUT_PATH bin)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(cmake/version.h.in version.h)

add_subdirectory(mappov)

install(TARGETS mappov DESTINATION bin)

