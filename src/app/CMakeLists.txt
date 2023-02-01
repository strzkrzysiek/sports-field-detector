find_package(Boost REQUIRED COMPONENTS program_options REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV 4 COMPONENTS highgui REQUIRED)

add_executable(court_lines_app
        main.cpp)

target_include_directories(court_lines_app SYSTEM
    PUBLIC ${OpenCV_INCLUDE_DIRS}
) 

target_link_libraries(court_lines_app
    Boost::program_options
    glog::glog
    ${OpenCV_LIBS}
)
