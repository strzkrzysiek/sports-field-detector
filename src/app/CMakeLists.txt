# Copyright 2023 Krzysztof Wrobel

find_package(Boost COMPONENTS program_options REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV 4 COMPONENTS highgui REQUIRED)

add_executable(sports_field_detector_app
  app_utils.cpp
  main.cpp)

target_include_directories(sports_field_detector_app SYSTEM
  PUBLIC ${OpenCV_INCLUDE_DIRS}
) 

target_link_libraries(sports_field_detector_app
  Boost::program_options
  glog::glog
  ${OpenCV_LIBS}

  line_model_detection
)
