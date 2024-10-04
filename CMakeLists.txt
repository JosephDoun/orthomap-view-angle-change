cmake_minimum_required(VERSION 3.27)

project(omvac VERSION 0.0.0)

add_executable(omvac ${PROJECT_NAME}/main.cpp)

set(EXECUTABLE_OUTPUT_PATH bin)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(cmake/version.h.in version.h)

# include build directory.
target_include_directories(omvac PUBLIC "${PROJECT_BINARY_DIR}")

install(TARGETS omvac DESTINATION bin)

