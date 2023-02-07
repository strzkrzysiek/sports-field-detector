// Copyright 2023 Krzysztof Wrobel

#include "line_model_detection/line_model_detector.h"

#include <cmath>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

#include "line_model_detection/direct_model_alignment.h"
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

LineModelDetector::Result LineModelDetector::detect(const cv::Mat& image,
                                                    const Mat3& camera_matrix,
                                                    bool generate_debug_images) {  
  Result result;
  if (generate_debug_images) {
     result.debug_images["Model image"] = line_model_.getImage();
  }
  
  const uint assumed_line_width = 10;

  // Line pixel extraction ////////////////////////////////////////////////////
  
  const uint brightness_threshold = BrightPixelFilter::calculateBrightnessThreshold(0.05, 10, 220, image);
  const uint darkness_threshold = brightness_threshold - 3;
  const uint neighbor_distance = 2 * assumed_line_width;
  const Scalar eigenval_threshold = 0.002;
  const Scalar eigenval_ratio = 4.0;
  const uint block_size = 20;
  const uint aperture_size = 3;
  
  LinePixelExtractor::Result lpe_result = LinePixelExtractor(camera_matrix)
      .addFilter(std::make_unique<BrightPixelFilter>(brightness_threshold))
      .addFilter(std::make_unique<DarkNeighborhoodFilter>(darkness_threshold,
                                                          neighbor_distance))
      .addFilter(std::make_unique<LineFeatureFilter>(eigenval_threshold,
                                                     eigenval_ratio,
                                                     block_size,
                                                     aperture_size))
      .extract(image);

  if (generate_debug_images) {
    result.debug_images["Line pixel image"] = lpe_result.line_pixel_image;
  }

  // Line detection ///////////////////////////////////////////////////////////

  const Scalar rho_resolution = 3;
  const Scalar theta_resolution = deg2rad(0.5);
  const Scalar votes_threshold = std::min(image.cols, image.rows) / 2;
  const uint min_line_length = std::min(image.cols, image.rows) / 5;
  const uint max_line_gap = std::min(image.cols, image.rows) / 5;

  const Scalar nms_distance_1 = deg2rad(2.0);
  const bool nms_propagate_suppressed_1 = true;

  const Scalar optimizer_outlier_threshold = deg2rad(0.3);
  
  const Scalar nms_distance_2 = deg2rad(0.5);
  const bool nms_propagate_suppressed_2 = false;

  const Scalar ideal_point_dist = deg2rad(2.0);

  LineDetector::Result ld_result = LineDetector(camera_matrix)
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

  if (generate_debug_images) {
    result.debug_images["Detected lines"] = visualizeDetectedLines(image, ld_result, camera_matrix);
  }

  // Homography estimation ////////////////////////////////////////////////////

  const Scalar min_beta = 0.4;
  const Scalar max_beta = 2.5;
  const Scalar hit_award = 1.0;
  const Scalar miss_penalty = 0.75;
  
  std::optional<Mat3> he_result = HomographyEstimator(camera_matrix, line_model_)
      .addTest(std::make_unique<IsotropicScalingTest>(min_beta, max_beta))
      .setScoring(std::make_unique<ModelAlignmentScoring>(hit_award, miss_penalty))
      .estimate(lpe_result, ld_result);

  if (!he_result) {
    LOG(ERROR) << "Model could not be detected!";

    return result;
  }

  if (generate_debug_images) {
    result.debug_images["Coarse homography estimation"] = visualizeDetectedModel(image, line_model_, he_result.value(), camera_matrix);
  }

  // Direct model alignment ///////////////////////////////////////////////////

  Scalar blur_size = assumed_line_width;
  Mat3 model2camera = DirectModelAlignment(line_model_, image.size(), camera_matrix, blur_size)
      .align(he_result.value(), lpe_result.line_pixel_image);

  result.model2camera_image_homography = 
  result.model2camera_image_homography = camera_matrix * model2camera;
  result.visualization = visualizeDetectedModel(image, line_model_, model2camera, camera_matrix);
  
  LOG(INFO) << "Model successfully detected!";

  return result;
}

} // namespace hawkeye
