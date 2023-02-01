cmake_minimum_required(VERSION 3.12)

project(court-lines)

if(MSVC)
    message(FATAL_ERROR "MSVC is not supported")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -pedantic -Werror)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(glog REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(app)
