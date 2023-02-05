#include "line_model_detection/line_model_detector.h"

#include <cmath>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

#include "line_model_detection/drawing_routines.h"
#include "line_model_detection/homography_estimator.h"
#include "line_model_detection/line_pixel_extractor.h"
#include "line_model_detection/line_detector.h"

namespace hawkeye {

cv::Mat visualizeDetectedLines(const cv::Mat& image,
                               const LineDetector::Result& ld_result,
                               const Mat3& camera_matrix) {
  cv::Mat canvas;
  cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);

  drawDetectedLines(canvas, ld_result.lines, camera_matrix);

  return canvas;
}

cv::Mat visualizeDetectedModel(const cv::Mat& image,
                               const LineModel& model,
                               const Mat3& model2camera,
                               const Mat3& camera_matrix) {
  cv::Mat canvas;
  cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);

  projectLineModelImage(canvas, model, model2camera, camera_matrix);

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

  // Line pixel extraction
  
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

  // Line detection

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

  // Homography estimation

  Scalar min_beta = 0.4;
  Scalar max_beta = 2.5;
  Scalar hit_award = 1.0;
  Scalar miss_penalty = 0.75;
  
  std::optional<Mat3> he_result = HomographyEstimator(assumed_camera_matrix, line_model_)
      .addTest(std::make_unique<IsotropicScalingTest>(min_beta, max_beta))
      .setScoring(std::make_unique<ModelAlignmentScoring>(hit_award, miss_penalty))
      .estimate(lpe_result, ld_result);

  if (!he_result) {
    LOG(ERROR) << "Model could not be detected!";
      return {
        // { "Line pixel image", lpe_result.line_pixel_image },
        { "Detected lines", visualizeDetectedLines(image, ld_result, assumed_camera_matrix) }
      };
  }
  
  return {
    // { "Line pixel image", lpe_result.line_pixel_image },
    { "Detected lines", visualizeDetectedLines(image, ld_result, assumed_camera_matrix) },
    { "Detected model", visualizeDetectedModel(image, line_model_, he_result.value(), assumed_camera_matrix) }
  };
}

} // namespace hawkeye
