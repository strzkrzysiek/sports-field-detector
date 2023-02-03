#pragma once

#include "line_model_detection/types.h"

#include <memory>
#include <vector>

namespace hawkeye {

class LinePixelExtractor {
public:
  struct Result {
    cv::Mat line_pixel_image;
    std::vector<Vec3> line_pixels_in_camera;
  };

  class Filter {
  public:
    virtual cv::Mat operator()(const cv::Mat& image) const = 0;
    virtual ~Filter() = default;
  };

  explicit LinePixelExtractor(const Mat3& camera_matrix)
      : camera_matrix_(camera_matrix) {}

  LinePixelExtractor& addFilter(std::unique_ptr<Filter>&& filter);
  Result extract(const cv::Mat& image) const;

private:
  const Mat3 camera_matrix_;
  std::vector<std::unique_ptr<Filter>> filters_;
};

class BrightPixelFilter : public LinePixelExtractor::Filter {
public:
  explicit BrightPixelFilter(uint threshold)
      : threshold_(threshold) {}

  cv::Mat operator()(const cv::Mat& image) const override;

  static uint calculateBrightnessThreshold(Scalar top_percentile,
                                           uint min_value,
                                           uint max_value,
                                           const cv::Mat& image);

private:
  const uint threshold_;
};

class DarkNeighborhoodFilter : public LinePixelExtractor::Filter {
public:
  explicit DarkNeighborhoodFilter(uint8_t threshold, uint distance)
      : threshold_(threshold),
        distance_(distance) {}

  cv::Mat operator()(const cv::Mat& image) const override;

private:
  const uint threshold_;
  const uint distance_;
};

class LineFeatureFilter : public LinePixelExtractor::Filter {
public:
  LineFeatureFilter(Scalar eigenval_threshold,
                    Scalar eigenval_ratio,
                    uint block_size,
                    uint aperture_size)
      : eigenval_threshold_(eigenval_threshold),
        eigenval_ratio_(eigenval_ratio),
        block_size_(block_size),
        aperture_size_(aperture_size) {}

  cv::Mat operator()(const cv::Mat& image) const override;

private:
  const Scalar eigenval_threshold_;
  const Scalar eigenval_ratio_;
  const uint block_size_;
  const uint aperture_size_;
};



} // namespace hawkeye
