
add_library(data "")

target_sources(data
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/read.cpp
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/write.cpp
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/read.h
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/write.h
)

target_include_directories(data PUBLIC ${CMAKE_CURRENT_LIST_DIR})
