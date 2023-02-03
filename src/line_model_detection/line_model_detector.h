#pragma once

#include <map>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

namespace hawkeye {

class LineModelDetector {
public:
  std::map<std::string, cv::Mat> detect(const cv::Mat& image);
};


} // namespace hawkeye
