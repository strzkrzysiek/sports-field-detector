// Copyright 2023 Krzysztof Wrobel

#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "line_model_detection/line_detector.h"
#include "line_model_detection/line_model.h"
#include "line_model_detection/line_pixel_extractor.h"
#include "line_model_detection/types.h"

namespace hawkeye {

class HomographyEstimator {
public:
  class Test {
  public:
    virtual bool operator()(const LinePixelExtractor::Result& lpe_result,
                            const LineDetector::Result& ld_result,
                            const Mat3& model2camera_homography,
                            const Mat3& camera_matrix,
                            const LineModel& model) const = 0;
    virtual ~Test() = default;
  };

  class Scoring {
  public:
    virtual Scalar operator()(const LinePixelExtractor::Result& lpe_result,
                              const LineDetector::Result& ld_result,
                              const Mat3& model2camera_homography,
                              const Mat3& camera_matrix,
                              const LineModel& model) const = 0;
    virtual ~Scoring() = default;
  };

  HomographyEstimator(const Mat3& camera_matrix, const LineModel& model)
      : camera_matrix_(camera_matrix),
        model_(model) {}

  HomographyEstimator& addTest(std::unique_ptr<Test>&& test);
  HomographyEstimator& setScoring(std::unique_ptr<Scoring>&& scoring);
  std::optional<Mat3> estimate(const LinePixelExtractor::Result& lpe_result,
                               const LineDetector::Result& ld_result);

private:
  std::optional<Scalar> validateHomography(const Mat3& model2cam,
                                           const LinePixelExtractor::Result& lpe_result,
                                           const LineDetector::Result& ld_result);
  std::vector<std::pair<uint, uint>> generateLinePairs(const std::vector<uint>& line_ids);
  std::vector<ImagePoint> getDetectedLinesAsPoints(const LineDetector::Result& ld_result);
  std::vector<ImagePoint> getModelLinesAsPoints();

  const Mat3 camera_matrix_;
  const LineModel model_;
  std::vector<std::unique_ptr<Test>> tests_;
  std::unique_ptr<Scoring> scoring_;

  std::vector<uint> test_rejection_cnt_;
};

class IsotropicScalingTest : public HomographyEstimator::Test {
public:
  IsotropicScalingTest(Scalar min_beta, Scalar max_beta)
      : min_beta_(min_beta),
        max_beta_(max_beta) {}

  bool operator()(const LinePixelExtractor::Result& lpe_result,
                  const LineDetector::Result& ld_result,
                  const Mat3& model2camera_homography,
                  const Mat3& camera_matrix,
                  const LineModel& model) const override;

private:
  Scalar min_beta_;
  Scalar max_beta_;
};

class ModelAlignmentScoring : public HomographyEstimator::Scoring {
public:
  ModelAlignmentScoring(Scalar hit_award, Scalar miss_penalty)
      : hit_award_(hit_award),
        miss_penalty_(miss_penalty) {}

  Scalar operator()(const LinePixelExtractor::Result& lpe_result,
                    const LineDetector::Result& ld_result,
                    const Mat3& model2camera_homography,
                    const Mat3& camera_matrix,
                    const LineModel& model) const override;

private:
  Scalar hit_award_;
  Scalar miss_penalty_;
};

} // namespace hawkeye
