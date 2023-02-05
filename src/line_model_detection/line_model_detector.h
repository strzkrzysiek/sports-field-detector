#pragma once

#include <map>
#include <memory>
#include <string>

#include "line_model_detection/line_model.h"
#include "line_model_detection/types.h"

namespace hawkeye {

class LineModelDetector {
public:
  explicit LineModelDetector(const LineModel& line_model)
      : line_model_(line_model) {}

  std::map<std::string, cv::Mat> detect(const cv::Mat& image);

private:
  const LineModel& line_model_;
};


} // namespace hawkeye
