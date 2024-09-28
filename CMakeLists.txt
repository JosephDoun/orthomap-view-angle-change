cmake_minimum_required(VERSION 3.27)

project(ovac VERSION 0.0.1-alpha)
add_executable(Tutorial ovac)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(config.h.in config.h)

target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )


