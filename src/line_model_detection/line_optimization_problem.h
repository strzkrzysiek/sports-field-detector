#pragma once

#include <vector>

#include <ceres/ceres.h>

#include "line_model_detection/types.h"

namespace hawkeye {

class LineOptimizationProblem {
public:
  LineOptimizationProblem(const std::vector<Vec3>& line_pixels_in_camera, Scalar outlier_threshold);

  Vec3 optimize(const Vec3& line_in_camera);

private:
  static ceres::Problem::Options createProblemOptions();
  static ceres::Solver::Options createSolverOptions();

  const std::vector<Vec3>& line_pixels_in_camera_;
  Vec3 optimized_line_in_camera_;

  ceres::Problem::Options problem_options_;
  ceres::Solver::Options solver_options_;
  ceres::Problem problem_;
  ceres::SphereManifold<3> sphere_manifold_;
  ceres::TukeyLoss tukey_loss_function_;
};

} // namespace hawkeye
