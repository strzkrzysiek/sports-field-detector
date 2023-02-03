#include "line_model_detection/line_model_detector.h"

#include "line_model_detection/line_pixel_extractor.h"

namespace hawkeye {

cv::Mat LineModelDetector::detect(const cv::Mat& image) {
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
  
  LinePixelExtractor::Result result = LinePixelExtractor()
      .addFilter(std::make_unique<BrightPixelFilter>(brightness_threshold))
      .addFilter(std::make_unique<DarkNeighborhoodFilter>(darkness_threshold, neighbor_distance))
      .addFilter(std::make_unique<LineFeatureFilter>(eigenval_threshold, eigenval_ratio, block_size, aperture_size))
      .extract(image, assumed_camera_matrix);

  return result.line_pixel_image;
}

} // namespace hawkeye
