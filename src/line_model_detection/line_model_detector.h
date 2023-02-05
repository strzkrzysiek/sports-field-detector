// Copyright 2023 Krzysztof Wrobel

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "line_model_detection/line_model.h"
#include "line_model_detection/types.h"

namespace hawkeye {

class LineModelDetector {
public:
  struct Result {
    std::optional<Mat3> model2camera_image_homography;
    cv::Mat visualization;
    std::map<std::string, cv::Mat> debug_images;
  };

  explicit LineModelDetector(const LineModel& line_model)
      : line_model_(line_model) {}

  Result detect(const cv::Mat& image, const Mat3& camera_matrix, bool generate_debug_images = false);

private:
  const LineModel& line_model_;
};


} // namespace hawkeye
