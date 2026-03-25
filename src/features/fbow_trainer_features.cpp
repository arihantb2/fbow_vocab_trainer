#include "features/fbow_trainer_features.h"

#include <algorithm>

cv::Ptr<cv::ORB> makeOrb(const OrbConfig& cfg)
{
    return cv::ORB::create(cfg.nfeatures, cfg.scaleFactor, cfg.nlevels, cfg.edgeThreshold, cfg.firstLevel, cfg.wtaK,
                           static_cast<cv::ORB::ScoreType>(cfg.scoreType), cfg.patchSize, cfg.fastThreshold);
}

cv::Ptr<cv::SIFT> makeSift(const SiftConfig& cfg)
{
    return cv::SIFT::create(cfg.nfeatures, cfg.nOctaveLayers, cfg.contrastThreshold, cfg.edgeThreshold, cfg.sigma);
}

int computeEffectiveMaxPerImage(const TrainerConfig& cfg, size_t numImages)
{
    const int fromTotal = (cfg.maxTotalDescriptors > 0 && numImages > 0)
                              ? std::max(1, cfg.maxTotalDescriptors / static_cast<int>(numImages))
                              : 0;

    if (cfg.maxFeaturesPerImage > 0 && fromTotal > 0)
    {
        return std::min(cfg.maxFeaturesPerImage, fromTotal);
    }
    return std::max(cfg.maxFeaturesPerImage, fromTotal);
}

cv::Mat clipRows(const cv::Mat& descriptors, int rowsToKeep)
{
    if (rowsToKeep <= 0 || rowsToKeep >= descriptors.rows)
    {
        if (!descriptors.isContinuous())
        {
            return descriptors.clone();
        }
        return descriptors;
    }
    return descriptors.rowRange(0, rowsToKeep).clone();
}
