// Copyright 2023 Krzysztof Wrobel

#pragma once

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace hawkeye {

// Types

using Scalar = double;

using ImagePoint = cv::Vec2f;
using ImageSize = cv::Size;

template <class T>
using Vec2T = Eigen::Matrix<T, 2, 1>;
using Vec2 = Vec2T<Scalar>;

template <class T>
using Vec3T = Eigen::Matrix<T, 3, 1>;
using Vec3 = Vec3T<Scalar>;

template <class T>
using Mat3T = Eigen::Matrix<T, 3, 3>;
using Mat3 = Mat3T<Scalar>;


// Type conversions

template <class OutputType, class InputType, int Rows, int Cols>
inline Eigen::Matrix<OutputType, Rows, Cols> toEigen(const cv::Matx<InputType, Rows, Cols>& mat_cv) {
  Eigen::Matrix<InputType, Rows, Cols> mat_eigen;
  cv::cv2eigen(mat_cv, mat_eigen);
  return mat_eigen.template cast<OutputType>();
}

template <class OutputType, class InputType>
inline Eigen::Matrix<OutputType, 3, 1> toEigen(const cv::Point3_<InputType>& point_cv) {
  Eigen::Matrix<InputType, 3, 1> mat_eigen(point_cv.x, point_cv.y, point_cv.z);
  return mat_eigen.template cast<OutputType>();
}

template <class OutputType, class InputType>
inline Eigen::Matrix<OutputType, 2, 1> toEigen(const cv::Point_<InputType>& point_cv) {
  Eigen::Matrix<InputType, 2, 1> mat_eigen(point_cv.x, point_cv.y);
  return mat_eigen.template cast<OutputType>();
}

namespace details {

template <class OutputType, int Rows, int Cols>
struct RT_toOpenCV {
    using return_type = cv::Matx<OutputType, Rows, Cols>;
};

template <class OutputType, int Rows>
struct RT_toOpenCV<OutputType, Rows, 1> {
    using return_type = cv::Vec<OutputType, Rows>;
};

} // namespace details

template <class OutputType, class InputType, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
inline typename details::RT_toOpenCV<OutputType, Rows,Cols>::return_type toOpenCV(const Eigen::Matrix<InputType, Rows, Cols, Options, MaxRows, MaxCols>& mat_eigen) {
  Eigen::Matrix<OutputType, Rows, Cols> mat_eigen_output_type = mat_eigen.template cast<OutputType>();
  cv::Matx<OutputType, Rows, Cols> mat_cv;
  typename details::RT_toOpenCV<OutputType, Rows,Cols>::return_type result;
  cv::eigen2cv(mat_eigen_output_type, result);
  return result;
}

// Conversions

inline Scalar deg2rad(Scalar degrees) {
  return degrees * CV_PI / 180.0;
}

inline Scalar rad2deg(Scalar radians) {
  return radians * 180.0 / CV_PI;
}

} // hawkeye
