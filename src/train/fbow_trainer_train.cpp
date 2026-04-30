#include "train/fbow_trainer_train.h"

#include "features/fbow_trainer_features.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "fbow.h"
#include "vocabulary_creator.h"

void trainOrbVocab(const std::vector<boost::filesystem::path>& images, const AppConfig& cfg, fbow::Vocabulary* voc,
                   size_t* descriptorsUsed, size_t* imagesUsed)
{
    *descriptorsUsed = 0;
    *imagesUsed = 0;

    const int effectiveMaxPerImage = computeEffectiveMaxPerImage(cfg.trainer, images.size());
    const size_t maxTotal = cfg.trainer.maxTotalDescriptors > 0 ? static_cast<size_t>(cfg.trainer.maxTotalDescriptors)
                                                                : std::numeric_limits<size_t>::max();

    std::cout << "Training ORB vocabulary..." << std::endl;
    std::cout << "  k: " << cfg.trainer.k << ", L: " << cfg.trainer.L << std::endl;
    std::cout << "  nthreads: " << cfg.trainer.nthreads << ", max_iters: " << cfg.trainer.maxIters << std::endl;
    std::cout << "  images: " << images.size() << std::endl;
    if (effectiveMaxPerImage > 0)
    {
        std::cout << "  features_per_image: " << effectiveMaxPerImage << std::endl;
    }
    else
    {
        std::cout << "  features_per_image: unlimited" << std::endl;
    }

    if (effectiveMaxPerImage > 0 || cfg.trainer.maxTotalDescriptors > 0)
    {
        const size_t upperBound =
            (effectiveMaxPerImage > 0 && cfg.trainer.maxTotalDescriptors > 0)
                ? std::min(static_cast<size_t>(effectiveMaxPerImage) * images.size(),
                           static_cast<size_t>(cfg.trainer.maxTotalDescriptors))
                : (effectiveMaxPerImage > 0 ? static_cast<size_t>(effectiveMaxPerImage) * images.size()
                                            : static_cast<size_t>(cfg.trainer.maxTotalDescriptors));
        std::cout << "  total_features_upper_bound: " << upperBound << std::endl;
    }
    else
    {
        std::cout << "  total_features_upper_bound: unlimited" << std::endl;
    }

    std::vector<cv::Mat> features;
    features.reserve(std::min(images.size(), maxTotal > 0 ? maxTotal : images.size()));

    cv::Ptr<cv::ORB> extractor = makeOrb(cfg.orb);
    cv::Ptr<cv::CLAHE> clahe = makeClahe(cfg.imagePrep);
    const size_t reportEvery = std::max<size_t>(1, images.size() / 5);

    std::vector<cv::KeyPoint> keypoints;
    for (size_t i = 0; i < images.size() && *descriptorsUsed < maxTotal; ++i)
    {
        const cv::Mat gray = prepareImage(loadGray(images[i].string()), clahe, cfg.imagePrep.scale);
        if (gray.empty())
        {
            continue;
        }

        keypoints.clear();
        cv::Mat descriptors;
        extractor->detectAndCompute(gray, cv::Mat(), keypoints, descriptors);
        if (descriptors.empty())
        {
            continue;
        }

        if (i % reportEvery == 0 || i + 1 == images.size())
        {
            std::cout << "  feature_progress: " << (i + 1) << "/" << images.size() << std::endl;
        }

        if (descriptors.type() != CV_8UC1)
        {
            throw std::runtime_error("ORB descriptors must be CV_8UC1");
        }
        if (descriptors.cols != 32)
        {
            throw std::runtime_error("Unexpected ORB descriptor dimension (expected 32)");
        }

        const int rows = effectiveMaxPerImage > 0 ? std::min(descriptors.rows, effectiveMaxPerImage) : descriptors.rows;
        if (rows <= 0)
        {
            continue;
        }

        cv::Mat used = descriptors;
        if (rows != descriptors.rows)
        {
            used = descriptors.rowRange(0, rows).clone();
        }
        else if (!used.isContinuous())
        {
            used = used.clone();
        }

        *descriptorsUsed += static_cast<size_t>(used.rows);
        features.push_back(used);
        *imagesUsed = features.size();
    }

    if (features.empty())
    {
        throw std::runtime_error("No ORB features extracted from dataset");
    }

    fbow::VocabularyCreator creator;
    fbow::VocabularyCreator::Params params(cfg.trainer.k, cfg.trainer.L, cfg.trainer.nthreads, cfg.trainer.maxIters);
    creator.create(*voc, features, cfg.featureType, params);
}

void trainBriskVocab(const std::vector<boost::filesystem::path>& images, const AppConfig& cfg, fbow::Vocabulary* voc,
                     size_t* descriptorsUsed, size_t* imagesUsed)
{
    *descriptorsUsed = 0;
    *imagesUsed = 0;

    const int effectiveMaxPerImage = computeEffectiveMaxPerImage(cfg.trainer, images.size());
    const size_t maxTotal = cfg.trainer.maxTotalDescriptors > 0 ? static_cast<size_t>(cfg.trainer.maxTotalDescriptors)
                                                                : std::numeric_limits<size_t>::max();

    std::cout << "Training BRISK vocabulary..." << std::endl;
    std::cout << "  k: " << cfg.trainer.k << ", L: " << cfg.trainer.L << std::endl;
    std::cout << "  nthreads: " << cfg.trainer.nthreads << ", max_iters: " << cfg.trainer.maxIters << std::endl;
    std::cout << "  images: " << images.size() << std::endl;
    if (effectiveMaxPerImage > 0)
    {
        std::cout << "  features_per_image: " << effectiveMaxPerImage << std::endl;
    }
    else
    {
        std::cout << "  features_per_image: unlimited" << std::endl;
    }

    if (effectiveMaxPerImage > 0 || cfg.trainer.maxTotalDescriptors > 0)
    {
        const size_t upperBound =
            (effectiveMaxPerImage > 0 && cfg.trainer.maxTotalDescriptors > 0)
                ? std::min(static_cast<size_t>(effectiveMaxPerImage) * images.size(),
                           static_cast<size_t>(cfg.trainer.maxTotalDescriptors))
                : (effectiveMaxPerImage > 0 ? static_cast<size_t>(effectiveMaxPerImage) * images.size()
                                            : static_cast<size_t>(cfg.trainer.maxTotalDescriptors));
        std::cout << "  total_features_upper_bound: " << upperBound << std::endl;
    }
    else
    {
        std::cout << "  total_features_upper_bound: unlimited" << std::endl;
    }

    std::vector<cv::Mat> features;
    features.reserve(std::min(images.size(), maxTotal > 0 ? maxTotal : images.size()));

    cv::Ptr<cv::BRISK> extractor = makeBrisk(cfg.brisk);
    cv::Ptr<cv::CLAHE> clahe = makeClahe(cfg.imagePrep);
    const size_t reportEvery = std::max<size_t>(1, images.size() / 5);

    std::vector<cv::KeyPoint> keypoints;
    for (size_t i = 0; i < images.size() && *descriptorsUsed < maxTotal; ++i)
    {
        const cv::Mat gray = prepareImage(loadGray(images[i].string()), clahe, cfg.imagePrep.scale);
        if (gray.empty())
        {
            continue;
        }

        keypoints.clear();
        cv::Mat descriptors;
        extractor->detectAndCompute(gray, cv::Mat(), keypoints, descriptors);
        if (descriptors.empty())
        {
            continue;
        }

        if (i % reportEvery == 0 || i + 1 == images.size())
        {
            std::cout << "  feature_progress: " << (i + 1) << "/" << images.size() << std::endl;
        }

        if (descriptors.type() != CV_8UC1)
        {
            throw std::runtime_error("BRISK descriptors must be CV_8UC1");
        }
        if (descriptors.cols != 64)
        {
            throw std::runtime_error("Unexpected BRISK descriptor dimension (expected 64)");
        }

        const int rows = effectiveMaxPerImage > 0 ? std::min(descriptors.rows, effectiveMaxPerImage) : descriptors.rows;
        if (rows <= 0)
        {
            continue;
        }

        cv::Mat used = descriptors;
        if (rows != descriptors.rows)
        {
            used = descriptors.rowRange(0, rows).clone();
        }
        else if (!used.isContinuous())
        {
            used = used.clone();
        }

        *descriptorsUsed += static_cast<size_t>(used.rows);
        features.push_back(used);
        *imagesUsed = features.size();
    }

    if (features.empty())
    {
        throw std::runtime_error("No BRISK features extracted from dataset");
    }

    fbow::VocabularyCreator creator;
    fbow::VocabularyCreator::Params params(cfg.trainer.k, cfg.trainer.L, cfg.trainer.nthreads, cfg.trainer.maxIters);
    creator.create(*voc, features, cfg.featureType, params);
}
