// Copyright 2023 Krzysztof Wrobel

#pragma once

#include "line_model_detection/line_detector.h"
#include "line_model_detection/line_model.h"
#include "line_model_detection/types.h"

namespace hawkeye {

void drawInfiniteLine(cv::Mat& canvas,
                      const Vec3& line_in_image,
                      const cv::Scalar& color);

void drawDetectedLines(cv::Mat& canvas,
                       const std::vector<DetectedLine>& detected_lines,
                       const Mat3& camera_matrix);

void drawModelLines(cv::Mat& canvas,
                    const LineModel& line_model,
                    const Mat3& model2camera,
                    const Mat3& camera_matrix,
                    const cv::Scalar& color);

void projectLineModelImage(cv::Mat& canvas,
                           const LineModel& line_model,
                           const Mat3& model2camera,
                           const Mat3& camera_matrix);

} // namespace hawkeye
