
add_executable(mappov main.cpp)

# Location of cmake generated config files.
target_include_directories(mappov PUBLIC ${PROJECT_BINARY_DIR})

add_subdirectory(data)
add_subdirectory(memory)
add_subdirectory(geometry)
add_subdirectory(errors)
add_subdirectory(config)

find_library(GDAL gdal)

target_link_libraries(mappov
	${GDAL}
	data
	memory
	# geometry
	# errors
	config
)


