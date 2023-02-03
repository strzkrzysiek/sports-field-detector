#pragma once

#include <memory>
#include <vector>

#include "line_model_detection/line_pixel_extractor.h"
#include "line_model_detection/types.h"


namespace hawkeye {

struct DetectedLine {
  enum class Group { Undefined, A, B, ToBeRemoved };

  Group group = Group::Undefined;
  Vec3 line_in_camera;
};

class LineDetector {
public:
  struct Result {
    std::vector<DetectedLine> lines;
  };

  class Step {
  public:
    virtual void operator()(const LinePixelExtractor::Result& lpe_result,
                            Result& ld_result,
                            const Mat3& camera_matrix) const = 0;
    virtual ~Step() = default;
  };

  explicit LineDetector(const Mat3& camera_matrix)
      : camera_matrix_(camera_matrix) {}

  LineDetector& addStep(std::unique_ptr<Step>&& step);
  Result detect(const LinePixelExtractor::Result& lpe_result) const;

private:
  const Mat3 camera_matrix_;
  std::vector<std::unique_ptr<Step>> steps_;
};

class HoughTransform : public LineDetector::Step {
public:
  HoughTransform(Scalar rho_resolution,
                 Scalar theta_resolution,
                 uint votes_threshold,
                 int n_strongest = -1)
      : rho_resolution_(rho_resolution),
        theta_resolution_(theta_resolution),
        votes_threshold_(votes_threshold),
        n_strongest_(n_strongest) {}

  void operator()(const LinePixelExtractor::Result& lpe_result,
                  LineDetector::Result& ld_result,
                  const Mat3& camera_matrix) const override;

private:
  Scalar rho_resolution_;
  Scalar theta_resolution_;
  uint votes_threshold_;
  int n_strongest_;
};

class HoughTransformProb : public LineDetector::Step {
public:
  HoughTransformProb(Scalar rho_resolution,
                     Scalar theta_resolution,
                     uint votes_threshold,
                     uint min_line_length,
                     uint max_line_gap,
                     int n_strongest = -1)
      : rho_resolution_(rho_resolution),
        theta_resolution_(theta_resolution),
        votes_threshold_(votes_threshold),
        min_line_length_(min_line_length),
        max_line_gap_(max_line_gap),
        n_strongest_(n_strongest) {}

  void operator()(const LinePixelExtractor::Result& lpe_result,
                  LineDetector::Result& ld_result,
                  const Mat3& camera_matrix) const override;

private:
  Scalar rho_resolution_;
  Scalar theta_resolution_;
  uint votes_threshold_;
  uint min_line_length_;
  uint max_line_gap_;
  int n_strongest_;
};

class NonMaximalSuppression : public LineDetector::Step {
public:
  NonMaximalSuppression(Scalar allowed_distance, bool propagate_suppressed)
      : allowed_distance_(allowed_distance),
        propagate_suppressed_(propagate_suppressed) {}

  void operator()(const LinePixelExtractor::Result& lpe_result,
                  LineDetector::Result& ld_result,
                  const Mat3& camera_matrix) const override;

private:
  Scalar allowed_distance_;
  bool propagate_suppressed_;
};

class LineOptimizer : public LineDetector::Step {
public:
  explicit LineOptimizer(Scalar outlier_threshold)
      : outlier_threshold_(outlier_threshold) {}

  void operator()(const LinePixelExtractor::Result& lpe_result,
                  LineDetector::Result& ld_result,
                  const Mat3& camera_matrix) const override;

private:
  Scalar outlier_threshold_;
};

class IdealPointClassifier : public LineDetector::Step {
public:
  explicit IdealPointClassifier(Scalar allowed_distance)
      : allowed_distance_(allowed_distance) {}

  void operator()(const LinePixelExtractor::Result& lpe_result,
                  LineDetector::Result& ld_result,
                  const Mat3& camera_matrix) const override;

private:
  Scalar allowed_distance_;
};

} // hawkeye
