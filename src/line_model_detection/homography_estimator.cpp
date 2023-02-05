#include "line_model_detection/homography_estimator.h"

#include <cmath>
#include <limits>
#include <sstream>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/calib3d.hpp>

#include "line_model_detection/drawing_routines.h"

namespace hawkeye {

// HomographyEstimator ////////////////////////////////////////////////////////

HomographyEstimator& HomographyEstimator::addTest(std::unique_ptr<Test>&& test) {
  tests_.emplace_back(std::move(test));

  return *this;
}

HomographyEstimator& HomographyEstimator::setScoring(std::unique_ptr<Scoring>&& scoring) {
  scoring_ = std::move(scoring);

  return *this;
}

void printIds(const char* g, const std::vector<uint>& ids) {
  std::stringstream ss;
  for (auto id : ids) {
    ss << id << " ";
  }

  LOG(INFO) << g << ": " << ss.str();
}

std::optional<Mat3> HomographyEstimator::estimate(const LinePixelExtractor::Result& lpe_result,
                                                  const LineDetector::Result& ld_result) {
  CHECK_GT(tests_.size(), 0) << "No HomographyEstimator steps defined.";
  CHECK(scoring_) << "No HomographyEstimator scoring defined.";

  auto& a_ids = ld_result.a_group_ids;
  auto& b_ids = ld_result.b_group_ids;
  auto& x_ids = model_.getXGroupIds();
  auto& y_ids = model_.getYGroupIds();

  if (a_ids.size() < 2) {
    LOG(ERROR) << "Insufficient number of lines in group A";
    return {};
  }

  if (b_ids.size() < 2) {
    LOG(ERROR) << "Insufficient number of lines in group B";
    return {};
  }

  CHECK_GT(x_ids.size(), 2) << "Check model definition! Insufficient number of lines in group X.";
  CHECK_GT(y_ids.size(), 2) << "Check model definition! Insufficient number of lines in group Y.";

  LOG(INFO) << "Estimating homography with" << tests_.size() << " tests.";

  LOG(INFO) << "Detected lines: A(" << a_ids.size() << "), B(" << b_ids.size() << ")";
  LOG(INFO) << "Model lines: X(" << x_ids.size() << "), Y(" << y_ids.size() << ")";



  printIds("A", a_ids);
  printIds("B", b_ids);
  printIds("X", x_ids);
  printIds("Y", y_ids);

  auto a_pairs = generateLinePairs(a_ids);
  auto b_pairs = generateLinePairs(b_ids);
  auto x_pairs = generateLinePairs(x_ids);
  auto y_pairs = generateLinePairs(y_ids);

  uint combinations_cnt = 2 * a_pairs.size() * b_pairs.size() * x_pairs.size() * y_pairs.size();
  LOG(INFO) << "Verifying " << combinations_cnt << " matching possibilities.";

  std::vector<ImagePoint> detected_lines_as_points = getDetectedLinesAsPoints(ld_result);
  std::vector<ImagePoint> model_lines_as_points = getModelLinesAsPoints();
  
  std::vector<ImagePoint> src_pts(4);
  std::vector<ImagePoint> dst_pts_xa_yb(4);
  std::vector<ImagePoint> dst_pts_xb_ya(4);

  std::optional<Mat3> best_model2cam;
  Scalar best_score = std::numeric_limits<Scalar>::lowest();
  test_rejection_cnt_.assign(tests_.size(), 0);
  
  for (auto& a_pair : a_pairs) {
    dst_pts_xa_yb[0] = detected_lines_as_points[a_pair.first];
    dst_pts_xa_yb[1] = detected_lines_as_points[a_pair.second];

    dst_pts_xb_ya[2] = detected_lines_as_points[a_pair.first];
    dst_pts_xb_ya[3] = detected_lines_as_points[a_pair.second];

    for (auto& b_pair : b_pairs) {
      dst_pts_xa_yb[2] = detected_lines_as_points[b_pair.first];
      dst_pts_xa_yb[3] = detected_lines_as_points[b_pair.second];

      dst_pts_xb_ya[0] = detected_lines_as_points[b_pair.first];
      dst_pts_xb_ya[1] = detected_lines_as_points[b_pair.second];

      for (auto& x_pair : x_pairs) {
        src_pts[0] = model_lines_as_points[x_pair.first];
        src_pts[1] = model_lines_as_points[x_pair.second];

        for (auto& y_pair : y_pairs) {
          src_pts[2] = model_lines_as_points[y_pair.first];
          src_pts[3] = model_lines_as_points[y_pair.second];

          LOG(INFO) << "Test: A (" << a_pair.first << ", " << a_pair.second << "), "
                    << "B (" << b_pair.first << ", " << b_pair.second << "), "
                    << "X (" << x_pair.first << ", " << x_pair.second << "), "
                    << "Y (" << y_pair.first << ", " << y_pair.second << ")";

          cv::Matx33f H_xa_yb = cv::findHomography(src_pts, dst_pts_xa_yb, 0);
          cv::Matx33f H_xb_ya = cv::findHomography(src_pts, dst_pts_xb_ya, 0);

          std::vector<Mat3> model2cam_homographies = {
            toEigen<Scalar>(H_xa_yb).inverse().transpose(),
            toEigen<Scalar>(H_xb_ya).inverse().transpose(),
          };

          for (auto& model2cam : model2cam_homographies) {
            auto score_opt = validateHomography(model2cam, lpe_result, ld_result);

            if (score_opt && score_opt.value() > best_score) {
              // LOG(INFO) << "Test: A (" << a_pair.first << ", " << a_pair.second << "), "
              //           << "B (" << b_pair.first << ", " << b_pair.second << "), "
              //           << "X (" << x_pair.first << ", " << x_pair.second << "), "
              //           << "Y (" << y_pair.first << ", " << y_pair.second << ")";
              LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Best score: " << score_opt.value();
              best_score = score_opt.value();
              best_model2cam = model2cam;
            }
          }
        }
      }
    }
  }

  for (uint i = 0; i < test_rejection_cnt_.size(); i++) {
    LOG(INFO) << "Test " << i << " rejected " << test_rejection_cnt_[i] << " possibilities.";
  }

  return best_model2cam;
}

std::optional<Scalar> HomographyEstimator::validateHomography(const Mat3& model2cam,
                                                              const LinePixelExtractor::Result& lpe_result,
                                                              const LineDetector::Result& ld_result) {
  for (uint i = 0; i < tests_.size(); i++) {
    bool test_res = (*tests_[i])(lpe_result, ld_result, model2cam, camera_matrix_, model_);
    if (!test_res) {
      test_rejection_cnt_[i]++;
      return {};
    }
  }

  Scalar score = (*scoring_)(lpe_result, ld_result, model2cam, camera_matrix_, model_);

  return score;
}

std::vector<std::pair<uint, uint>> HomographyEstimator::generateLinePairs(const std::vector<uint>& line_ids) {
  std::vector<std::pair<uint, uint>> line_pairs;

  for (uint i = 0; i < line_ids.size(); i++) {
    for (uint j = i + 1; j < line_ids.size(); j++) {
      line_pairs.emplace_back(line_ids[i], line_ids[j]);
    }
  }

  return line_pairs;
}

std::vector<ImagePoint> HomographyEstimator::getDetectedLinesAsPoints(const LineDetector::Result& ld_result) {
  std::vector<ImagePoint> detected_lines_as_points;
  std::transform(ld_result.lines.begin(),
                 ld_result.lines.end(),
                 std::back_inserter(detected_lines_as_points),
                 [](auto& line) {
                   return ImagePoint(line.line_in_camera[0] / line.line_in_camera[2],
                                     line.line_in_camera[1] / line.line_in_camera[2]);
                 });
  return detected_lines_as_points;
}

std::vector<ImagePoint> HomographyEstimator::getModelLinesAsPoints() {
  std::vector<ImagePoint> model_lines_as_points;
  std::transform(model_.getLines().begin(),
                 model_.getLines().end(),
                 std::back_inserter(model_lines_as_points),
                 [](auto& line) {
                   return ImagePoint(line.line_in_model[0] / line.line_in_model[2],
                                     line.line_in_model[1] / line.line_in_model[2]);
                 });
  return model_lines_as_points;
}

// IsotropicScalingTest ///////////////////////////////////////////////////////

bool IsotropicScalingTest::operator()(const LinePixelExtractor::Result& /* lpe_result */,
                                      const LineDetector::Result& /* ld_result */,
                                      const Mat3& model2camera_homography,
                                      const Mat3& /* camera_matrix */,
                                      const LineModel& /* model */) const {
  // LOG(INFO) << "IsotropicScalingTest (min_beta: " << min_beta_ << ", max_beta: " << max_beta_ << ")";

  Scalar beta2 = model2camera_homography.col(1).squaredNorm() / model2camera_homography.col(0).squaredNorm();
  Scalar beta = std::sqrt(beta2);

  LOG(INFO) << "Calculated beta: " << beta;

  return (beta > min_beta_ && beta < max_beta_);
}

// ModelAlignmentScoring //////////////////////////////////////////////////////

Scalar ModelAlignmentScoring::operator()(const LinePixelExtractor::Result& lpe_result,
                                         const LineDetector::Result& /* ld_result */,
                                         const Mat3& model2camera_homography,
                                         const Mat3& camera_matrix,
                                         const LineModel& model) const {
  // LOG(INFO) << "ModelAlignmentScoring (hit_award: " << hit_award_ << ", miss_penalty: " << miss_penalty_ << ")";

  cv::Mat warped_model = cv::Mat::zeros(lpe_result.line_pixel_image.size(), CV_8U);
  drawModelLines(warped_model, model, model2camera_homography, camera_matrix, cv::Scalar(255));

  cv::Mat model_pixels_on_line_pixels = warped_model & lpe_result.dilated_line_pixel_image;
  cv::Mat model_pixels_off_line_pixels = warped_model & ~lpe_result.dilated_line_pixel_image;

  int hit_count = cv::countNonZero(model_pixels_on_line_pixels);
  int miss_count = cv::countNonZero(model_pixels_off_line_pixels);

  Scalar score = hit_award_ * hit_count - miss_penalty_ * miss_count;

  LOG(INFO) << "hits: " << hit_count << ", misses: " << miss_count << ", score: " << score;

  return score;
}

} // namespace hawkeye
