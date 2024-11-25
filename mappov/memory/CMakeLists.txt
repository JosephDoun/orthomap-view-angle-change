add_library(memory "")

target_sources(memory
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/memory.cpp
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/memory.h
)

target_include_directories(memory PUBLIC ${CMAKE_CURRENT_LIST_DIR})
