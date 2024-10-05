cmake_minimum_required(VERSION 3.27)

project(omvac VERSION 0.0.0)

file(GLOB ADDITIONAL_SOURCES "${PROJECT_NAME}/*/*.cpp")
file(GLOB ADD_HEADERS "${PROJECT_NAME}/*.*.h")

add_executable(omvac ${PROJECT_NAME}/main.cpp)

set(EXECUTABLE_OUTPUT_PATH bin)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(cmake/version.h.in version.h)

# include build directory.
target_include_directories(omvac
	PUBLIC
	"${PROJECT_BINARY_DIR}"
	"${PROJECT_NAME}/io")

# additional sources for target.
target_sources(omvac PUBLIC ${ADDITIONAL_SOURCES})


install(TARGETS omvac DESTINATION bin)

