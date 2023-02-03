#include "line_model_detection/line_pixel_extractor.h"

#include <algorithm>
#include <utility>

#include <glog/logging.h>
#include <Eigen/LU>
#include <opencv2/imgproc.hpp>

namespace hawkeye {

LinePixelExtractor& LinePixelExtractor::addFilter(std::unique_ptr<Filter>&& filter) {
  filters_.emplace_back(std::move(filter));

  return *this;
}

LinePixelExtractor::Result LinePixelExtractor::extract(const cv::Mat& image, const Mat3& camera_matrix) {
  CHECK_GT(filters_.size(), 0) << "No LinePixelExtractor filters defined.";

  Result result;

  for (const auto& filter : filters_) {
    cv::Mat filtered_pixels = (*filter)(image);
    
    CHECK(filtered_pixels.size == image.size);
    CHECK(filtered_pixels.type() == CV_8U);

    if (result.line_pixel_image.empty()) {
      result.line_pixel_image = filtered_pixels;
    } else {
      result.line_pixel_image &= filtered_pixels;
    }
  }

  std::vector<cv::Point> pixel_coords;
  cv::findNonZero(result.line_pixel_image, pixel_coords);

  Mat3 camera_matrix_inv = camera_matrix.inverse();

  std::transform(pixel_coords.begin(),
                 pixel_coords.end(),
                 std::back_inserter(result.line_pixels_in_camera),
                 [&camera_matrix_inv](const cv::Point& pt) { return camera_matrix_inv * Vec3(pt.x, pt.y, 1.0); });

  return result;
}

cv::Mat BrightPixelFilter::operator()(const cv::Mat& image) const {
  LOG(INFO) << "BrightPixelFilter (thr: " << threshold_ << ")";

  cv::Mat brighter_pixels;
  cv::threshold(image, brighter_pixels, threshold_, 255, cv::THRESH_BINARY);

  return brighter_pixels;
}

uint BrightPixelFilter::calculateBrightnessThreshold(Scalar top_percentile,
                                                     uint min_value,
                                                     uint max_value,
                                                     const cv::Mat& image) {
  uint histogram[256] = {};
  const uint8_t* image_ptr = image.ptr<uint8_t>();
  uint image_size = image.total();
  for (uint i = 0; i < image_size; i++) {
    histogram[image_ptr[i]]++;
  }

  uint cumsum_threshold = static_cast<uint>(top_percentile * image_size);
  uint cumsum = 0;

  uint threshold = 256;
  do {
    cumsum += histogram[--threshold];
  } while (cumsum < cumsum_threshold);

  threshold = std::min(max_value, std::max(min_value, threshold));

  LOG(INFO) << "Calculated brightness threshold: " << threshold;

  return threshold;
}

cv::Mat DarkNeighborhoodFilter::operator()(const cv::Mat& image) const {
  LOG(INFO) << "DarkNeighborhoodFilter (thr: " << threshold_ << ", dist: " << distance_ << ")";

  cv::Mat darker_pixels;
  cv::threshold(image, darker_pixels, threshold_, 255, cv::THRESH_BINARY_INV);

  cv::Mat has_dark_horizontal_neighborhood(image.size(), CV_8U, 255);
  has_dark_horizontal_neighborhood.colRange(0, image.cols-distance_) &= darker_pixels.colRange(distance_, image.cols);
  has_dark_horizontal_neighborhood.colRange(distance_, image.cols) &= darker_pixels.colRange(0, image.cols-distance_);

  cv::Mat has_dark_vertical_neighborhood(image.size(), CV_8U, 255);
  has_dark_vertical_neighborhood.rowRange(0, image.rows-distance_) &= darker_pixels.rowRange(distance_, image.rows);
  has_dark_vertical_neighborhood.rowRange(distance_, image.rows) &= darker_pixels.rowRange(0, image.rows-distance_);

  cv::Mat has_dark_neighborhood = has_dark_horizontal_neighborhood | has_dark_vertical_neighborhood;

  return has_dark_neighborhood;
}

cv::Mat LineFeatureFilter::operator()(const cv::Mat& image) const {
  LOG(INFO) << "LineFeatureFilter (eigenval thr: " << eigenval_threshold_ << ", ratio: " << eigenval_ratio_
            << ", block: " << block_size_ << ", aperture: " << aperture_size_ << ")";

  cv::Mat eigen_vals_and_vecs;
  cv::cornerEigenValsAndVecs(image, eigen_vals_and_vecs, block_size_, aperture_size_);

  cv::Mat lambda1, lambda2;
  cv::extractChannel(eigen_vals_and_vecs, lambda1, 0);
  cv::extractChannel(eigen_vals_and_vecs, lambda2, 1);

  return (lambda1 > eigenval_threshold_) & (lambda1 > eigenval_ratio_ * lambda2);
}

} // namespace hawkeye
