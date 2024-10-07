cmake_minimum_required(VERSION 3.27)

project(omvac VERSION 0.0.0)

set(EXECUTABLE_OUTPUT_PATH bin)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(cmake/version.h.in version.h)

add_subdirectory(omvac)

find_library(GDAL gdal)
find_library(OBLAS openblaso)
find_path(GDAL_HEADERS gdal/gdal_priv.h)

message("${PROJECT_NAME}")
message("${PROJECT_BINARY_DIR}")
message("${OBLAS}")
message("${GDAL_HEADERS}")

install(TARGETS omvac DESTINATION bin)

