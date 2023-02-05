#include "line_model_detection/line_model.h"

#include <algorithm>
#include <utility>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

namespace hawkeye {

LineModel::LineModel(uint width, uint height, uint line_width)
    : model_size_(width, height),
      line_width_(line_width),
      model_image_(cv::Mat::zeros(model_size_, CV_8U)),
      committed(false) {
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

  if ((line.group == ModelLine::Group::X && line.line_in_model[1] < 0)
      or
      (line.group == ModelLine::Group::Y && line.line_in_model[0] < 0)) {
    line.line_in_model *= -1.0;
  }

  lines_.push_back(std::move(line));
}

void LineModel::commit() {
  std::sort(lines_.begin(),
            lines_.end(),
            [](auto& a, auto& b) {
              if (a.group == b.group)
                return a.line_in_model[2] < b.line_in_model[2];

              if (a.group == ModelLine::Group::X)
                return true;
              else
                return false;
            });

  for (uint i = 0; i < lines_.size(); i++) {
    auto& line = lines_[i];

    switch (line.group) {
    case ModelLine::Group::X: x_group_ids_.push_back(i); break;
    case ModelLine::Group::Y: y_group_ids_.push_back(i); break;
    }

    LOG(INFO) << "Line definition (" << (line.group == ModelLine::Group::X ? "X" : "Y") << " " << line.name << ", "
              << "S: [" << line.pt0_in_model.head<2>().transpose() << "], "
              << "E: [" << line.pt1_in_model.head<2>().transpose() << "], "
              << "L: [" << line.line_in_model.transpose() << "])";
    drawLine(line);
  }

  committed = true;
}

const ImageSize& LineModel::getModelSize() const {
  CHECK(committed);
  return model_size_;
}

uint LineModel::getLineWidth() const {
  CHECK(committed);
  return line_width_;
}

const std::vector<ModelLine>& LineModel::getLines() const {
  CHECK(committed);
  return lines_;
}

const cv::Mat& LineModel::getImage() const {
  CHECK(committed);
  return model_image_;
}

const std::vector<uint>& LineModel::getXGroupIds() const {
  CHECK(committed);
  return x_group_ids_;
}

const std::vector<uint>& LineModel::getYGroupIds() const {
  CHECK(committed);
  return y_group_ids_;
}

void LineModel::drawLine(const ModelLine& line) {
  Scalar offset = static_cast<Scalar>(line_width_ - 1) / 2.0;
  
  cv::Point top_right(line.pt0[0] - offset, line.pt0[1] - offset);
  cv::Point bottom_left(line.pt1[0] + offset, line.pt1[1] + offset);
  
  cv::rectangle(model_image_, top_right, bottom_left, 255);
}

} // namespace hawkeye
