
add_library(io "")

# find_path(GDAL_HEADERS gdal/gdal_priv.h)

target_sources(io
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/read.cpp
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/read.h
)

target_include_directories(io PUBLIC ${CMAKE_CURRENT_LIST_DIR})

