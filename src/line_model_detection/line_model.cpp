#include "line_model_detection/line_model.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

namespace hawkeye {

LineModel::LineModel(unsigned width, unsigned height, unsigned line_width)
    : model_size_(width, height),
      line_width_(line_width),
      model_image_(cv::Mat::zeros(model_size_, CV_8U)) {
  CHECK_GE(line_width, 1);
  LOG(INFO) << "Defining a line model: [" << width << ", " << height << "]";
}

void LineModel::addLine(const std::string& name,
                        const ImagePoint& pt0,
                        const ImagePoint& pt1) {
  CHECK(pt0[0] == pt1[0] || pt0[1] == pt1[1]) << "Only axis-aligned lines supported by the model";

  ModelLine line;

  line.name = name;
  line.group = (pt0[1] == pt1[1]) ? ModelLine::Group::X : ModelLine::Group::Y;
  
  if (pt0[0] <= pt1[0] && pt0[1] <= pt1[1]) {
    line.pt0 = pt0;
    line.pt1 = pt1;
  } else {
    line.pt0 = pt1;
    line.pt1 = pt0;
  }

  line.pt0_in_model << line.pt0[0], line.pt0[1], 1.0;
  line.pt1_in_model << line.pt1[0], line.pt1[1], 1.0;

  line.line_in_model = line.pt0_in_model.cross(line.pt1_in_model).normalized();

  drawLine(line);

  LOG(INFO) << "Line definition (" << name << "):\n"
            << "Start: [" << line.pt0_in_model.head<2>().transpose() << "]\n"
            << "End: [" << line.pt1_in_model.head<2>().transpose() << "]\n"
            << "Line: [" << line.line_in_model.transpose() << "]";
}


void LineModel::drawLine(const ModelLine& line) {
  // cv.rectangle(canvas, tuple((pt0 - 0.5).astype(int)), tuple((pt1 + 0.5).astype(int)), color)

  Scalar offset = static_cast<Scalar>(line_width_ - 1) / 2.0;
  
  cv::Point top_right(line.pt0[0] - offset, line.pt0[1] - offset);
  cv::Point bottom_left(line.pt1[0] + offset, line.pt1[1] + offset);
  
  cv::rectangle(model_image_, top_right, bottom_left, 255);
}

} // namespace hawkeye
