#pragma once

#include <string>

#include "line_model_detection/types.h"

namespace hawkeye {

cv::Mat readImage(const std::string& imgpath, uint width, uint height);

Mat3 fakeCameraMatrix(const cv::Size& img_size);

} // namespace hawkeye