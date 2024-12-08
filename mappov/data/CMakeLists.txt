
add_library(data "")

target_sources(data
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/read.cpp
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/read.h
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/write.cpp
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/write.h
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/utils.h
)

target_include_directories(data PUBLIC ${CMAKE_CURRENT_LIST_DIR})
