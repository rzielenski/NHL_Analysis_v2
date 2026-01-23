cmake_minimum_required(VERSION 3.10)

project(nhl_transformer VERSION 0.1)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

find_package(CURL REQUIRED)
add_executable(GetData src/get_pbp_data.cpp)

target_link_libraries(GetData PRIVATE 
    nlohmann_json::nlohmann_json
	CURL::libcurl	
)

add_executable(TensorTest 
    src/tensor_test.cpp
    src/tensor.cpp
)
target_include_directories(TensorTest PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
find_library(TCMALLOC_LIBRARIES NAMES tcmalloc_minimal)
target_link_libraries(TensorTest PRIVATE 
    ${TCMALLOC_LIBRARIES}
)

