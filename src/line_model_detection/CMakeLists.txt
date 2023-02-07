# Copyright 2023 Krzysztof Wrobel

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(glog REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Ceres REQUIRED)

add_library(line_model_detection
  direct_model_alignment.cpp
  drawing_routines.cpp
  homography_estimator.cpp
  line_detector.cpp
  line_model.cpp
  line_model_defs.cpp
  line_model_detector.cpp
  line_optimization_problem.cpp
  line_pixel_extractor.cpp
)

target_include_directories(line_model_detection SYSTEM
  PUBLIC ${OpenCV_INCLUDE_DIRS}
) 

target_link_libraries(line_model_detection
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PRIVATE Ceres::ceres
  PRIVATE glog::glog
)