// Copyright 2023 Krzysztof Wrobel

#include "app/app_utlis.h"

#include <fstream>

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

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

void generateOutputCsvFile(const std::string& outpath,
                           const Mat3& model2cam_img,
                           const cv::Size& img_size,
                           const LineModel& line_model) {
  std::ofstream ofs(outpath, std::ios::out);
  if (!ofs) {
    LOG(ERROR) << "Could not open the file for writing: " << outpath;
    return;
  }

  const uint parts = 4;

  // CSV header line
  ofs << "name,";
  for (uint i = 0; i <= parts; i++) {
    ofs << "x" << i << ",y" << i << (i == parts ? "\n" : ",");
  }

  // For each line, dump its name and sample points along it in the image coordinates
  for (auto& line : line_model.getLines()) {
    std::vector<ImagePoint> pts_model = { line.pt0, line.pt1 };

    cv::Point2f pt0 = toOpenCV<float>(Vec2((model2cam_img * line.pt0_in_model).head<2>()));
    cv::Point2f pt1 = toOpenCV<float>(Vec2((model2cam_img * line.pt1_in_model).head<2>()));

    LOG(INFO) << line.name;
    LOG(INFO) << pt0 << " " << pt1;

    cv::Point pt0_i = pt0;
    cv::Point pt1_i = pt1;

    LOG(INFO) << " " << pt0_i << " " << pt1_i;

    LOG(INFO) << "S: " << img_size;

    if (!cv::clipLine(img_size, pt0_i, pt1_i)) {
      continue;
    }

    pt0 = pt0_i;
    pt1 = pt1_i;

    ofs << line.name << ",";

    for (uint i = 0; i <= parts; i++) {
      cv::Point pt = pt0 + (pt1 - pt0) * static_cast<float>(i) / static_cast<float>(parts);
      ofs << pt.x << "," << pt.y << (i == parts ? "\n" : ",");
    }
  }

  ofs.close();

  LOG(INFO) << "Successfully dumped line info into the CSV file: " << outpath;
}

} // namespace hawkeye
