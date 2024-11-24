add_library(config "")

target_sources(config
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/args.cpp
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/args.h
)

target_include_directories(config PUBLIC ${CMAKE_CURRENT_LIST_DIR})
