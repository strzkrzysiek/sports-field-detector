// Copyright 2023 Krzysztof Wrobel

#pragma once

#include <string>

#include "line_model_detection/line_model.h"
#include "line_model_detection/types.h"

namespace hawkeye {

cv::Mat readImage(const std::string& imgpath, uint width, uint height);
Mat3 fakeCameraMatrix(const cv::Size& img_size);
void generateOutputCsvFile(const std::string& outpath,
                           const Mat3& model2cam_img,
                           const cv::Size& img_size,
                           const LineModel& line_model);

} // namespace hawkeye