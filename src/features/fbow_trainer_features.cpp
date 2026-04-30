#include "features/fbow_trainer_features.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

cv::Mat loadGray(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty())
    {
        return img;
    }
    if (img.type() == CV_16UC1)
    {
        // 16-bit Bayer BGGR — demosaic to BGR then convert to grayscale
        cv::Mat bgr16;
        cv::cvtColor(img, bgr16, cv::COLOR_BayerBG2BGR);
        cv::Mat bgr8;
        bgr16.convertTo(bgr8, CV_8UC3, 1.0 / 256.0);
        cv::Mat gray;
        cv::cvtColor(bgr8, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    if (img.type() == CV_16UC3)
    {
        cv::Mat bgr8;
        img.convertTo(bgr8, CV_8UC3, 1.0 / 256.0);
        cv::Mat gray;
        cv::cvtColor(bgr8, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    if (img.channels() == 1)
    {
        return img;
    }
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Ptr<cv::CLAHE> makeClahe(const ImagePrepConfig& cfg)
{
    if (!cfg.claheEnabled)
    {
        return cv::Ptr<cv::CLAHE>();
    }
    return cv::createCLAHE(cfg.claheClipLimit, cv::Size(cfg.claheTileGridSize, cfg.claheTileGridSize));
}

cv::Mat prepareImage(const cv::Mat& gray, const cv::Ptr<cv::CLAHE>& clahe, double scale)
{
    cv::Mat out = gray;
    if (clahe)
    {
        clahe->apply(out, out);
    }
    if (scale != 1.0)
    {
        cv::resize(out, out, cv::Size(), scale, scale);
    }
    return out;
}

cv::Ptr<cv::ORB> makeOrb(const OrbConfig& cfg)
{
    return cv::ORB::create(cfg.nfeatures, cfg.scaleFactor, cfg.nlevels, cfg.edgeThreshold, cfg.firstLevel, cfg.wtaK,
                           static_cast<cv::ORB::ScoreType>(cfg.scoreType), cfg.patchSize, cfg.fastThreshold);
}

cv::Ptr<cv::BRISK> makeBrisk(const BriskConfig& cfg)
{
    return cv::BRISK::create(cfg.thresh, cfg.octaves, cfg.patternScale);
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
