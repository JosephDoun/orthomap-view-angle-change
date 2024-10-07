
add_executable(omvac main.cpp)

target_include_directories(omvac PUBLIC ${PROJECT_BINARY_DIR})

add_subdirectory(io)
add_subdirectory(memory)
add_subdirectory(geometry)
add_subdirectory(errors)
add_subdirectory(config)

target_link_libraries(omvac
	${GDAL}
	io
#	memory
#	geometry
#	errors
#	config
)


