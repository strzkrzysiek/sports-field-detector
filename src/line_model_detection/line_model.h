#pragma once

#include <string>
#include <vector>

#include "line_model_detection/types.h"

namespace hawkeye {

struct ModelLine {
  enum class Group { X, Y };
  
  std::string name;
  Group group;

  ImagePoint pt0;
  ImagePoint pt1;

  Vec3 pt0_in_model;
  Vec3 pt1_in_model;

  Vec3 line_in_model;
};

class LineModel {
public:
  LineModel(unsigned width, unsigned height, unsigned line_width);
  void addLine(const std::string& name,
               const ImagePoint& pt0,
               const ImagePoint& pt1);

  const ImageSize& getModelSize() const { return model_size_; }
  unsigned getLineWidth() const { return line_width_; }
  const std::vector<ModelLine> getLines() const { return lines_; }
  const cv::Mat& getImage() const { return model_image_; }

private:
  void drawLine(const ModelLine& line);

  ImageSize model_size_;
  unsigned line_width_;
  std::vector<ModelLine> lines_;
  cv::Mat model_image_;
};

} // namespace hawkeye
