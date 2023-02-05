#include "app/app_utlis.h"

#include <fstream>

#include <glog/logging.h>

namespace hawkeye {

cv::Mat readImage(const std::string& imgpath, uint width, uint height) {
  LOG(INFO) << "Opening image: " << imgpath;

  std::ifstream ifs(imgpath, std::ios::in | std::ios::binary);
  if (!ifs) {
    LOG(ERROR) << "Could not open the file for reading: " << imgpath;
    return cv::Mat();
  }

  ifs.seekg(0, ifs.end);
  uint file_size = ifs.tellg();
  ifs.seekg(0, ifs.beg);

  if (file_size != width * height) {
    LOG(ERROR) << "Incorrect file size: " << file_size << ", expected: " << width * height;
    return cv::Mat();
  }

  cv::Mat image(height, width, CV_8U);
  ifs.read(image.ptr<char>(), width * height);

  ifs.close();

  return image;
}

Mat3 fakeCameraMatrix(const cv::Size& img_size) {
  Scalar cx = img_size.width / 2.0;
  Scalar cy = img_size.height / 2.0;
  Scalar f = (cx + cy) * 2;
  Mat3 fake_camera_matrix;
  fake_camera_matrix << f,  0., cx,
                            0., f,  cy,
                            0., 0., 1.;

  return fake_camera_matrix;
}

} // namespace hawkeye
