// Copyright 2023 Krzysztof Wrobel

#include "line_model_detection/direct_model_alignment.h"

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>

namespace hawkeye {

struct ModelPixelError {
  ModelPixelError(uint model_col,
                  uint model_row,
                  Scalar model_val,
                  const cv::Size& camera_image_size,
                  const DirectModelAlignment::Interpolator& interpolator)
      : pt_in_model(model_col, model_row, 1.0),
        model_val(model_val),
        camera_image_size(camera_image_size),
        interpolator(interpolator) {}

  template <class T>
  bool operator()(const T* const raw_model2cam, T* raw_residual) const {
    Eigen::Map<const Mat3T<T>> model2cam(raw_model2cam);
    T& residual = *raw_residual;

    Vec3T<T> pt_in_camera = (model2cam * pt_in_model.cast<T>());
    pt_in_camera /= pt_in_camera[2];
    
    T camera_pixel_value;
    interpolator.Evaluate(pt_in_camera[1], pt_in_camera[0], &camera_pixel_value);

    residual = T(model_val) - camera_pixel_value;

    return true;
  }

  static ceres::CostFunction* create(uint model_col,
                                     uint model_row,
                                     Scalar model_val,
                                     const cv::Size& camera_image_size,
                                     const DirectModelAlignment::Interpolator& interpolator) {
    return new ceres::AutoDiffCostFunction<ModelPixelError, 1, 9>(
      new ModelPixelError(model_col, model_row, model_val, camera_image_size, interpolator));
  }

  Vec3 pt_in_model;
  Scalar model_val;
  const cv::Size& camera_image_size;
  const DirectModelAlignment::Interpolator& interpolator;
};

DirectModelAlignment::DirectModelAlignment(const LineModel& model,
                                           const cv::Size& camera_image_size,
                                           const Mat3& camera_matrix,
                                           Scalar blur_size)
    : model_(model),
      camera_image_size_(camera_image_size),
      grid_image_(cv::Mat::zeros(camera_image_size.height + 2, camera_image_size.width + 2, CV_64F)),
      camera_subimage_(grid_image_(cv::Range(1, camera_image_size.height + 1),
                                   cv::Range(1, camera_image_size.width + 1))),
      grid_2d_(grid_image_.ptr<Scalar>(),
               -1, camera_image_size.height + 1,
               -1, camera_image_size.width + 1),
      interpolator_(grid_2d_),
      camera_matrix_(camera_matrix),
      blur_size_(blur_size),
      problem_options_(createProblemOptions()),
      solver_options_(createSolverOptions()),
      problem_(problem_options_),
      subset_manifold_(9, { 8 }) {
  LOG(INFO) << "DirectModelAlignment (blur: " << blur_size << ")";

  problem_.AddParameterBlock(optimized_model2camera_image_hom_.data(), 9, &subset_manifold_);
  
  for (int row = 0; row < model_.getImage().rows; row++) {
    const uchar* row_ptr = model_.getImage().ptr<uchar>(row);
    for (int col = 0; col < model_.getImage().cols; col++) {
      ceres::CostFunction* cost_function =
          ModelPixelError::create(col, row, row_ptr[col] / 255.0, camera_image_size_, interpolator_);
      problem_.AddResidualBlock(cost_function, nullptr, optimized_model2camera_image_hom_.data());
    }
  }
}

Mat3 DirectModelAlignment::align(const Mat3& model2camera, const cv::Mat& pixel_image) {
  CHECK(camera_subimage_.size == pixel_image.size);

  cv::Mat pixel_image_scaled;
  pixel_image.convertTo(pixel_image_scaled, camera_subimage_.type(), 1./255.);

  int ksize = static_cast<int>(std::floor(blur_size_ / 2)) * 2 + 1;
  cv::GaussianBlur(pixel_image_scaled, camera_subimage_, {ksize, ksize}, 0.);
  cv::normalize(camera_subimage_, camera_subimage_, 1.0, 0.0, cv::NORM_INF);

  optimized_model2camera_image_hom_ = camera_matrix_ * model2camera;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options_, &problem_, &summary);

  LOG(INFO) << summary.FullReport();

  return camera_matrix_.inverse() * optimized_model2camera_image_hom_;
}

ceres::Problem::Options DirectModelAlignment::createProblemOptions() {
  ceres::Problem::Options options;
  options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

  return options;
}

ceres::Solver::Options DirectModelAlignment::createSolverOptions() {
  ceres::Solver::Options options;
  options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
  options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;

  return options;
}

} // namespace hawkeye
