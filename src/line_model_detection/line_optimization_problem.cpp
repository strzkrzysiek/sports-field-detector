#include "line_model_detection/line_optimization_problem.h"

#include <Eigen/Geometry>
#include <glog/logging.h>

namespace hawkeye {

struct PixelDistanceError {
  explicit PixelDistanceError(const Vec3& pixel_in_camera)
      : pixel_in_camera(pixel_in_camera) {}

  template <class T>
  bool operator()(const T* const raw_line_in_camera, T* raw_residual) const {
    Eigen::Map<const Vec3T<T>> line_in_camera(raw_line_in_camera);
    T& residual = *raw_residual;

    residual = pixel_in_camera.cast<T>().dot(line_in_camera);

    return true;
  }

  static ceres::CostFunction* create(const Vec3& pixel_in_camera) {
    return new ceres::AutoDiffCostFunction<PixelDistanceError, 1, 3>(new PixelDistanceError(pixel_in_camera));
  }

  const Vec3& pixel_in_camera;
};

LineOptimizationProblem::LineOptimizationProblem(const std::vector<Vec3>& line_pixels_in_camera, Scalar outlier_threshold)
    : line_pixels_in_camera_(line_pixels_in_camera),
      problem_options_(createProblemOptions()),
      solver_options_(createSolverOptions()),
      problem_(problem_options_),
      tukey_loss_function_(outlier_threshold) {
  problem_.AddParameterBlock(optimized_line_in_camera_.data(), 3, &sphere_manifold_);

  for (const Vec3& pixel_in_camera : line_pixels_in_camera_) {
    ceres::CostFunction* cost_function = PixelDistanceError::create(pixel_in_camera);
    problem_.AddResidualBlock(cost_function, &tukey_loss_function_, optimized_line_in_camera_.data());
  }
}

Vec3 LineOptimizationProblem::optimize(const Vec3& line_in_camera) {
  optimized_line_in_camera_ = line_in_camera;
  
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options_, &problem_, &summary);

  LOG(INFO) << summary.BriefReport();
  DLOG(INFO) << summary.FullReport();

  return optimized_line_in_camera_;
}

ceres::Problem::Options LineOptimizationProblem::createProblemOptions() {
  ceres::Problem::Options options;
  options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

  return options;
}

ceres::Solver::Options LineOptimizationProblem::createSolverOptions() {
  ceres::Solver::Options options;
  options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
  options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;

  return options;
}

} // namespace hawkeye
