#include "line_model_detection/line_detector.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

namespace hawkeye {

// LineDetector ///////////////////////////////////////////////////////////////

LineDetector& LineDetector::addStep(std::unique_ptr<Step>&& step) {
  steps_.emplace_back(std::move(step));

  return *this;
}

LineDetector::Result LineDetector::detect(const LinePixelExtractor::Result& lpe_result) const {
  CHECK_GT(steps_.size(), 0) << "No LineDetector steps defined.";

  LOG(INFO) << "Detecting lines " << steps_.size() << " steps.";
  
  Result ld_result;

  for (const auto& step : steps_) {
    (*step)(lpe_result, ld_result, camera_matrix_);
  }

  return ld_result;
}

// HoughTransform /////////////////////////////////////////////////////////////

void HoughTransform::operator()(const LinePixelExtractor::Result& lpe_result,
                                LineDetector::Result& ld_result,
                                const Mat3& camera_matrix) const {
  LOG(INFO) << "HoughTransform (rho: " << rho_resolution_ << ", theta_deg: " << rad2deg(theta_resolution_) << ", thr: " << votes_threshold_
            << ", n: " << n_strongest_ << ")";
  CHECK(ld_result.lines.empty()) << "HoughTransform should be the first step of the detection algorithm.";

  std::vector<cv::Vec2f> lines_polar;
  cv::HoughLines(lpe_result.line_pixel_image,
                 lines_polar,
                 rho_resolution_,
                 theta_resolution_,
                 votes_threshold_);

  auto end_it = (n_strongest_ < 0 || lines_polar.end() - lines_polar.begin() < n_strongest_)
              ? lines_polar.end()
              : lines_polar.begin() + n_strongest_;

  std::transform(lines_polar.begin(),
                 end_it,
                 std::back_inserter(ld_result.lines),
                 [&camera_matrix](const cv::Vec2f& polar_coords) {
                   // polar_coords: (rho, theta)

                   Vec3 line_in_image(std::cos(polar_coords[1]),
                                      std::sin(polar_coords[1]),
                                      -polar_coords[0]);
                   
                   DetectedLine detected_line;
                   detected_line.line_in_camera = (camera_matrix.transpose() * line_in_image).normalized();

                   DLOG(INFO) << "Line: [ rho: " << polar_coords[0] << ", theta_deg: " << rad2deg(polar_coords[1]) << "] => "
                              << "[" << detected_line.line_in_camera.transpose() << "]";

                   return detected_line;
                 });
  LOG(INFO) << "HoughTransform - detected " << ld_result.lines.size() << " lines.";
}

// HoughTransformProb /////////////////////////////////////////////////////////////

void HoughTransformProb::operator()(const LinePixelExtractor::Result& lpe_result,
                                    LineDetector::Result& ld_result,
                                    const Mat3& camera_matrix) const {
  LOG(INFO) << "HoughTransformProb (rho: " << rho_resolution_ << ", theta_deg: " << rad2deg(theta_resolution_) << ", thr: " << votes_threshold_
            << ", min_line: " << min_line_length_ << ", max_gap: " << max_line_gap_ << ", n: " << n_strongest_ << ")";
  CHECK(ld_result.lines.empty()) << "HoughTransformProb should be the first step of the detection algorithm";

  std::vector<cv::Vec4i> detected_line_ends;
  cv::HoughLinesP(lpe_result.line_pixel_image,
                  detected_line_ends,
                  rho_resolution_,
                  theta_resolution_,
                  votes_threshold_,
                  min_line_length_,
                  max_line_gap_);

  auto end_it = (n_strongest_ < 0 || detected_line_ends.end() - detected_line_ends.begin() < n_strongest_)
              ? detected_line_ends.end()
              : detected_line_ends.begin() + n_strongest_;

  std::transform(detected_line_ends.begin(),
                 end_it,
                 std::back_inserter(ld_result.lines),
                 [&camera_matrix](const cv::Vec4i& line) {
                   // line: (x0, y0, x1, y1)

                   Vec3 pt0_in_image(line[0], line[1], 1.0);
                   Vec3 pt1_in_image(line[2], line[3], 1.0);
                   Vec3 line_in_image = pt0_in_image.cross(pt1_in_image);
                   
                   DetectedLine detected_line;
                   detected_line.line_in_camera = (camera_matrix.transpose() * line_in_image).normalized();

                   DLOG(INFO) << "Line: [" << line[0] << ", " << line[1] << "] - [" << line[2] << ", " << line[3] << "] => "
                              << "[" << detected_line.line_in_camera.transpose() << "]";

                   return detected_line;
                 });

  LOG(INFO) << "HoughTransformProb - detected " << ld_result.lines.size() << " lines.";
}

// NonMaximalSuppression //////////////////////////////////////////////////////

void NonMaximalSuppression::operator()(const LinePixelExtractor::Result& /* lpe_result */,
                                       LineDetector::Result& ld_result,
                                       const Mat3& /* camera_matrix */) const {
  LOG(INFO) << "NonMaximalSuppression (dist_deg: " << rad2deg(allowed_distance_) << ", propagate: " << propagate_suppressed_ << ")";

  auto& lines = ld_result.lines;

  //std::vector<bool> selected_lines(lines.size(), true);
  for (uint i = 0; i < lines.size(); i++) {
    if (!propagate_suppressed_ && lines[i].group == DetectedLine::Group::ToBeRemoved)
      continue;
    
    for (uint j = i + 1; j < lines.size(); j++) {
      if (lines[j].group == DetectedLine::Group::ToBeRemoved)
        continue;

      Scalar cos_dist = lines[i].line_in_camera.dot(lines[j].line_in_camera);
      Scalar sin_dist = std::sqrt(1.0 - cos_dist * cos_dist);

      if (sin_dist > allowed_distance_)
        continue;

      lines[j].group = DetectedLine::Group::ToBeRemoved;
      DLOG(INFO) << "Suppressed line: " << j;
    }
  }

  uint original_size = lines.size();
  std::erase_if(lines, [](auto& line) { return line.group == DetectedLine::Group::ToBeRemoved; });

  LOG(INFO) << "Suppressed " << original_size - lines.size() << " lines.";
}

// LineOptimizer //////////////////////////////////////////////////////////////

// IdealPointClassifier ///////////////////////////////////////////////////////

} // hawkeye
