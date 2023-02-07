// Copyright 2023 Krzysztof Wrobel

#pragma once

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

#include "line_model_detection/line_model.h"
#include "line_model_detection/types.h"

namespace hawkeye {

class DirectModelAlignment {
public:
  using Grid = ceres::Grid2D<Scalar, 1>;
  using Interpolator = ceres::BiCubicInterpolator<Grid>;

  DirectModelAlignment(const LineModel& model,
                       const cv::Size& camera_image_size,
                       const Mat3& camera_mat,
                       Scalar blur_size);

  Mat3 align(const Mat3& model2camera, const cv::Mat& pixel_image);

private:
  static ceres::Problem::Options createProblemOptions();
  static ceres::Solver::Options createSolverOptions();

  const LineModel& model_;
  cv::Size camera_image_size_;
  cv::Mat grid_image_;
  cv::Mat camera_subimage_;
  Grid grid_2d_;
  Interpolator interpolator_;
  Mat3 optimized_model2camera_image_hom_;

  Mat3 camera_matrix_;
  Scalar blur_size_;

  ceres::Problem::Options problem_options_;
  ceres::Solver::Options solver_options_;
  ceres::Problem problem_;
  ceres::SubsetManifold subset_manifold_;
};

} // namesapce hawkeye
