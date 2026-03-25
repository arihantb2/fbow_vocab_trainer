#ifndef FBOW_TRAINER_FEATURES_H_
#define FBOW_TRAINER_FEATURES_H_

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <cstddef>

#include "config/fbow_trainer_config.h"

cv::Ptr<cv::ORB> makeOrb(const OrbConfig& cfg);

cv::Ptr<cv::SIFT> makeSift(const SiftConfig& cfg);

// Returns the effective per-image descriptor cap given both per-image and total budgets.
// Distributes maxTotalDescriptors evenly across numImages so memory is bounded upfront.
// Returns 0 if neither cap is set (no limit).
int computeEffectiveMaxPerImage(const TrainerConfig& cfg, size_t numImages);

cv::Mat clipRows(const cv::Mat& descriptors, int rowsToKeep);

#endif  // FBOW_TRAINER_FEATURES_H_
