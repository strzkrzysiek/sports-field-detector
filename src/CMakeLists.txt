cmake_minimum_required(VERSION 3.12)

project(court-lines)

if(MSVC)
  message(FATAL_ERROR "MSVC is not supported")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -pedantic -Werror)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(app)
add_subdirectory(line_model_detection)
