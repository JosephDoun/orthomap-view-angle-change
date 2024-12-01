
add_library(filesystem "")

target_sources(filesystem
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/read.cpp
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/write.cpp
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/read.h
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/write.h
)

target_include_directories(filesystem PUBLIC ${CMAKE_CURRENT_LIST_DIR})
