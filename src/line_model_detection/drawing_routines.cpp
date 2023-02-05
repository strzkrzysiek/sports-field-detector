// Copyright 2023 Krzysztof Wrobel

#include "line_model_detection/drawing_routines.h"

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

namespace hawkeye {

void drawInfiniteLine(cv::Mat& canvas,
                      const Vec3& line_in_image,
                      const cv::Scalar& color) {
  Vec3 left_border_line(1., 0., 0.);
  Vec3 right_border_line(1., 0., -canvas.cols);
  Vec3 top_border_line(0., 1., 0.);
  Vec3 bottom_border_line(0., 1., -canvas.rows);

  ImagePoint pt0, pt1;
  if (std::abs(line_in_image[0]) > std::abs(line_in_image[1])) { // line is more vertical
    Vec3 pt0_in_image = line_in_image.cross(top_border_line);
    Vec3 pt1_in_image = line_in_image.cross(bottom_border_line);

    pt0_in_image /= pt0_in_image[2];
    pt1_in_image /= pt1_in_image[2];

    pt0 = ImagePoint(pt0_in_image[0], pt0_in_image[1]);
    pt1 = ImagePoint(pt1_in_image[0], pt1_in_image[1]);
  } else {                                                       // line is more horizontal
    Vec3 pt0_in_image = line_in_image.cross(left_border_line);
    Vec3 pt1_in_image = line_in_image.cross(right_border_line);

    pt0_in_image /= pt0_in_image[2];
    pt1_in_image /= pt1_in_image[2];

    pt0 = ImagePoint(pt0_in_image[0], pt0_in_image[1]);
    pt1 = ImagePoint(pt1_in_image[0], pt1_in_image[1]);
  }

  cv::line(canvas, cv::Point(pt0), cv::Point(pt1), color, 2, cv::LINE_AA);
}

void drawDetectedLines(cv::Mat& canvas,
                       const std::vector<DetectedLine>& detected_lines,
                       const Mat3& camera_matrix) {
  Mat3 camera_matrix_invT = camera_matrix.inverse().transpose();

  for (const auto& detected_line : detected_lines) {
    const Vec3& line_in_camera = detected_line.line_in_camera;
    Vec3 line_in_image = camera_matrix_invT * line_in_camera;

    cv::Scalar color;
    switch (detected_line.group) {
    case DetectedLine::Group::Undefined:   color = cv::Scalar(255,   0,   0); break;
    case DetectedLine::Group::A:           color = cv::Scalar(  0, 255,   0); break;
    case DetectedLine::Group::B:           color = cv::Scalar(  0,   0, 255); break;
    case DetectedLine::Group::ToBeRemoved: color = cv::Scalar(  0,   0,   0); break;
    }

    drawInfiniteLine(canvas, line_in_image, color);
  }
}

void drawModelLines(cv::Mat& canvas,
                    const LineModel& line_model,
                    const Mat3& model2camera,
                    const Mat3& camera_matrix,
                    const cv::Scalar& color) {
  Mat3 model2image = camera_matrix * model2camera;

  for (auto& line : line_model.getLines()) {
    Vec3 pt0_in_image = model2image * line.pt0_in_model;
    Vec3 pt1_in_image = model2image * line.pt1_in_model;

    ImagePoint pt0(pt0_in_image[0] / pt0_in_image[2], pt0_in_image[1] / pt0_in_image[2]);
    ImagePoint pt1(pt1_in_image[0] / pt1_in_image[2], pt1_in_image[1] / pt1_in_image[2]);
    
    cv::line(canvas, cv::Point(pt0), cv::Point(pt1), color, 2, cv::LINE_8);
  }
}

void projectLineModelImage(cv::Mat& canvas,
                           const LineModel& line_model,
                           const Mat3& model2camera,
                           const Mat3& camera_matrix) {
  const cv::Mat& model_img = line_model.getImage();
  cv::Matx33f homography_cv = toOpenCV<float>(Mat3(camera_matrix * model2camera));

  cv::Mat warped_model_img;
  cv::warpPerspective(model_img, warped_model_img, homography_cv, canvas.size(), cv::INTER_LINEAR);

  for (int row = 0; row < canvas.rows; row++) {
    cv::Vec3b* canvas_ptr = canvas.ptr<cv::Vec3b>(row);
    uchar* model_ptr = warped_model_img.ptr<uchar>(row);

    for (int col = 0; col < canvas.cols; col++) {
      cv::Vec3b& canvas_pix = canvas_ptr[col];
      uchar& model_pix = model_ptr[col];
      cv::Vec3b red_model_pix(0, 0, model_pix);
      float alpha = model_pix / 255.0;

      cv::Vec3f pix = canvas_pix * (1.0 - alpha) + red_model_pix * alpha;
      canvas_pix = cv::Vec3b(pix);
    }
  }
}

} // namespace hawkeye
