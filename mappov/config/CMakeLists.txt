add_library(config "")

# Current directory.
target_include_directories(config PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Location of cmake generated config files.
target_include_directories(config PUBLIC ${PROJECT_BINARY_DIR})

target_sources(config
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/args.cpp
	PUBLIC ${CMAKE_CURRENT_LIST_DIR}/args.h
)
