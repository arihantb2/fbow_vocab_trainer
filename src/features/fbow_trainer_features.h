#ifndef FBOW_TRAINER_FEATURES_H_
#define FBOW_TRAINER_FEATURES_H_

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <string>

#include "config/fbow_trainer_config.h"

// Load image as 8-bit grayscale, handling 16-bit Bayer BGGR (bayer_bggr16) input.
cv::Mat loadGray(const std::string& path);

// Create a CLAHE instance from config, or null if CLAHE is disabled.
// Call once before a processing loop and reuse across images.
cv::Ptr<cv::CLAHE> makeClahe(const ImagePrepConfig& cfg);

// Apply CLAHE (if non-null) then resize (if scale != 1.0) to an 8-bit grayscale image.
cv::Mat prepareImage(const cv::Mat& gray, const cv::Ptr<cv::CLAHE>& clahe, double scale);

cv::Ptr<cv::ORB> makeOrb(const OrbConfig& cfg);

cv::Ptr<cv::BRISK> makeBrisk(const BriskConfig& cfg);

// Returns the effective per-image descriptor cap given both per-image and total budgets.
// Distributes maxTotalDescriptors evenly across numImages so memory is bounded upfront.
// Returns 0 if neither cap is set (no limit).
int computeEffectiveMaxPerImage(const TrainerConfig& cfg, size_t numImages);

cv::Mat clipRows(const cv::Mat& descriptors, int rowsToKeep);

#endif  // FBOW_TRAINER_FEATURES_H_
