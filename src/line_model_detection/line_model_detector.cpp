#include "line_model_detection/line_model_detector.h"

#include <cmath>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

#include "line_model_detection/line_pixel_extractor.h"
#include "line_model_detection/line_detector.h"

namespace hawkeye {

cv::Mat visualizeDetectedLines(const cv::Mat& image,
                               const LineDetector::Result& ld_result,
                               const Mat3& camera_matrix) {
  cv::Mat canvas;
  cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);

  Mat3 camera_matrix_invT = camera_matrix.inverse().transpose();

  Vec3 left_border_line(1., 0., 0.);
  Vec3 right_border_line(1., 0., -image.cols);
  Vec3 top_border_line(0., 1., 0.);
  Vec3 bottom_border_line(0., 1., -image.rows);

  for (const auto& detected_line : ld_result.lines) {
    const Vec3& line_in_camera = detected_line.line_in_camera;
    Vec3 line_in_image = camera_matrix_invT * line_in_camera;

    ImagePoint pt0, pt1;
    if (std::abs(line_in_image[0]) > std::abs(line_in_image[1])) { // line is more vertical
      Vec3 pt0_in_image = line_in_image.cross(top_border_line);
      Vec3 pt1_in_image = line_in_image.cross(bottom_border_line);

      pt0_in_image /= pt0_in_image[2];
      pt1_in_image /= pt1_in_image[2];

      pt0 = ImagePoint(pt0_in_image[0], pt0_in_image[1]);
      pt1 = ImagePoint(pt1_in_image[0], pt1_in_image[1]);
    } else {                                                       // line is more horizontal
      Vec3 pt0_in_image = line_in_image.cross(left_border_line);
      Vec3 pt1_in_image = line_in_image.cross(right_border_line);

      pt0_in_image /= pt0_in_image[2];
      pt1_in_image /= pt1_in_image[2];

      pt0 = ImagePoint(pt0_in_image[0], pt0_in_image[1]);
      pt1 = ImagePoint(pt1_in_image[0], pt1_in_image[1]);
    }

    cv::Scalar color;
    switch (detected_line.group) {
    case DetectedLine::Group::Undefined:   color = cv::Scalar(255,   0,   0); break;
    case DetectedLine::Group::A:           color = cv::Scalar(  0, 255,   0); break;
    case DetectedLine::Group::B:           color = cv::Scalar(  0,   0, 255); break;
    case DetectedLine::Group::ToBeRemoved: color = cv::Scalar(  0,   0,   0); break;
    }

    cv::line(canvas, cv::Point(pt0), cv::Point(pt1), color, 2, cv::LINE_AA);
  }

  return canvas;
}

std::map<std::string, cv::Mat> LineModelDetector::detect(const cv::Mat& image) {
  Scalar cx = image.cols / 2.0;
  Scalar cy = image.rows / 2.0;
  Scalar f = (cx + cy) * 2;
  Mat3 assumed_camera_matrix;
  assumed_camera_matrix << f,  0., cx,
                           0., f,  cy,
                           0., 0., 1.;
  
  uint assumed_line_width = 10;
  
  uint brightness_threshold = BrightPixelFilter::calculateBrightnessThreshold(0.05, 10, 220, image);
  uint darkness_threshold = brightness_threshold - 3;
  uint neighbor_distance = 2 * assumed_line_width;
  Scalar eigenval_threshold = 0.002;
  Scalar eigenval_ratio = 4.0;
  uint block_size = 20;
  uint aperture_size = 3;
  
  LinePixelExtractor::Result lpe_result = LinePixelExtractor(assumed_camera_matrix)
      .addFilter(std::make_unique<BrightPixelFilter>(brightness_threshold))
      .addFilter(std::make_unique<DarkNeighborhoodFilter>(darkness_threshold,
                                                          neighbor_distance))
      .addFilter(std::make_unique<LineFeatureFilter>(eigenval_threshold,
                                                     eigenval_ratio,
                                                     block_size,
                                                     aperture_size))
      .extract(image);

  Scalar rho_resolution = 3;
  Scalar theta_resolution = deg2rad(0.5);
  Scalar votes_threshold = std::min(image.cols, image.rows) / 2;
  uint min_line_length = std::min(image.cols, image.rows) / 5;
  uint max_line_gap = std::min(image.cols, image.rows) / 5;
  int n_strongest = 100;

  Scalar nms_distance_1 = deg2rad(2.0);
  bool nms_propagate_suppressed_1 = true;

  Scalar optimizer_outlier_threshold = deg2rad(0.3);
  
  Scalar nms_distance_2 = deg2rad(0.5);
  bool nms_propagate_suppressed_2 = false;

  Scalar ideal_point_dist = deg2rad(2.0);

  (void)min_line_length;
  (void)max_line_gap;
  (void)n_strongest;

  LineDetector::Result ld_result = LineDetector(assumed_camera_matrix)
      //.addStep(std::make_unique<HoughTransform>(rho_resolution, theta_resolution, votes_threshold, n_strongest))
      .addStep(std::make_unique<HoughTransformProb>(rho_resolution,
                                                    theta_resolution,
                                                    votes_threshold,
                                                    min_line_length,
                                                    max_line_gap))
      .addStep(std::make_unique<NonMaximalSuppression>(nms_distance_1, nms_propagate_suppressed_1))
      .addStep(std::make_unique<LineOptimizer>(optimizer_outlier_threshold))
      .addStep(std::make_unique<NonMaximalSuppression>(nms_distance_2, nms_propagate_suppressed_2))
      .addStep(std::make_unique<IdealPointClassifier>(ideal_point_dist))
      .detect(lpe_result);

  return {
    { "Line pixel image", lpe_result.line_pixel_image },
    { "Detected lines", visualizeDetectedLines(image, ld_result, assumed_camera_matrix) }
  };
}

} // namespace hawkeye
