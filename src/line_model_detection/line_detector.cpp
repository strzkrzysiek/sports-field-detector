#include "line_model_detection/line_detector.h"

#include <algorithm>
#include <cmath>
#include <set>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

#include "line_model_detection/line_optimization_problem.h"

namespace hawkeye {

// LineDetector ///////////////////////////////////////////////////////////////

const char* DetectedLine::group2str(Group group) {
  switch (group) {
  case Group::Undefined:
    return "Undefined";
  case Group::A:
    return "A";
  case Group::B:
    return "B";
  case Group::ToBeRemoved:
    return "ToBeRemoved";
  }
}

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

  for (uint i = 0; i < ld_result.lines.size(); i++) {
    switch (ld_result.lines[i].group) {
    case DetectedLine::Group::A:
      ld_result.a_group_ids.push_back(i);
      break;

    case DetectedLine::Group::B:
      ld_result.b_group_ids.push_back(i);
      break;

    default:
      break;
    }
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

void LineOptimizer::operator()(const LinePixelExtractor::Result& lpe_result,
                               LineDetector::Result& ld_result,
                               const Mat3& /* camera_matrix */) const {
  LOG(INFO) << "LineOptimizer (outlier_thr_deg: " << rad2deg(outlier_threshold_) << ")";

  LineOptimizationProblem problem(lpe_result.line_pixels_in_camera, outlier_threshold_);
  
  for (auto& line : ld_result.lines) {
    Vec3 optimized_line_in_camera = problem.optimize(line.line_in_camera);

    LOG(INFO) << "Optimized line: [" << line.line_in_camera.transpose() << "] => "
              << "[" << optimized_line_in_camera.transpose() << "]";
    line.line_in_camera = optimized_line_in_camera;
  }
}

// IdealPointClassifier ///////////////////////////////////////////////////////


void IdealPointClassifier::operator()(const LinePixelExtractor::Result& /* lpe_result */,
                                      LineDetector::Result& ld_result,
                                      const Mat3& /* camera_matrix */) const {
  LOG(INFO) << "IdealPointClassifier (dist_deg: " << rad2deg(allowed_distance_) << ")";

  findParallelGroup(ld_result.lines, DetectedLine::Group::A);
  findParallelGroup(ld_result.lines, DetectedLine::Group::B);
}

void IdealPointClassifier::findParallelGroup(std::vector<DetectedLine>& lines, DetectedLine::Group group) const {
  // Find parallel lines by finding an ideal point where the most of lines intersect
  
  struct Intersection {
    uint line0_id;
    uint line1_id;
    Vec3 pt_in_camera;

    uint closest_intersections_cnt;
    std::set<uint> intersecting_lines;
  };

  // Take lines that don't belong to any group yet
  std::vector<uint> remaining_line_ids;
  for (uint i = 0; i < lines.size(); i++) {
    if (lines[i].group == DetectedLine::Group::Undefined)
      remaining_line_ids.push_back(i);
  }

  LOG(INFO) << "Determining group " << DetectedLine::group2str(group) << " among " << remaining_line_ids.size() << " line candidates.";

  std::vector<Intersection> intersections;

  // Generate all possible line pairs and calculate their intersection points
  for (uint i = 0; i < remaining_line_ids.size(); i++) {
    for (uint j = i + 1; j < remaining_line_ids.size(); j++) {
      Intersection intersection;
      intersection.line0_id = i;
      intersection.line1_id = j;

      intersection.pt_in_camera = (lines[i].line_in_camera.cross(lines[j].line_in_camera)).normalized();

      intersections.push_back(intersection);
    }
  }

  // For each intersection:
  // * calculate the distances to other intersections
  // * count those that are nearby (closer than the allowed distance)
  // * define the set of the lines that intersect in this point
  for (uint i = 0; i < intersections.size(); i++) {
    auto& ideal_point_candidate = intersections[i];
    ideal_point_candidate.closest_intersections_cnt = 0;

    for (uint j = 0; j < intersections.size(); j++) {
      Scalar cos_dist = intersections[i].pt_in_camera.dot(intersections[j].pt_in_camera);
      Scalar sin_dist = std::sqrt(1 - cos_dist * cos_dist);

      if (sin_dist < allowed_distance_) {
        ideal_point_candidate.closest_intersections_cnt++;

        ideal_point_candidate.intersecting_lines.insert(intersections[j].line0_id);
        ideal_point_candidate.intersecting_lines.insert(intersections[j].line1_id);
      }
    }
  }

  // Find an intersection with the most other nearby intersections
  auto ideal_point_it = std::max_element(intersections.begin(),
                                         intersections.end(),
                                         [](auto& a, auto& b) {
                                           return a.closest_intersections_cnt < b.closest_intersections_cnt;
                                         });

  // If the strongest intersection clique has only one element, we have been unable to define the ideal point.
  // Consider all remaining lines to be in the group.
  if (ideal_point_it == intersections.end() || ideal_point_it->closest_intersections_cnt <= 1) {
    for (uint id : remaining_line_ids) {
      lines[id].group = group;
    }

    LOG(INFO) << "Could not determine the ideal point. All remaining lines considered to be in group " << DetectedLine::group2str(group);
    return;
  }

  // Otherwise, consider the lines intersecting in the "strongest" ideal point to be parallel
  // and assign them to the group.
  for (uint id : ideal_point_it->intersecting_lines) {
    lines[id].group = group;
  }

  LOG(INFO) << "Found ideal point: [" << ideal_point_it->pt_in_camera.transpose() << "]";
  LOG(INFO) << ideal_point_it->intersecting_lines.size() << " lines added to the group " << DetectedLine::group2str(group);
}

} // hawkeye
