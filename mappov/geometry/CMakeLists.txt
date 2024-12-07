add_library(geometry "")

target_sources(geometry
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/transformations.cpp
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/transformations.h
)

target_include_directories(geometry PUBLIC ${CMAKE_CURRENT_LIST_DIR})