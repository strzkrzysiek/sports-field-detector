#pragma once

#include <memory>

#include <opencv2/core/mat.hpp>

namespace hawkeye {

class LineModelDetector {
public:
  cv::Mat detect(const cv::Mat& image);
};


} // namespace hawkeye
