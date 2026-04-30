#include "vocabulary_test/fbow_trainer_test.h"

#include "features/fbow_trainer_features.h"

#include <boost/filesystem.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fbow.h"

namespace
{

std::vector<boost::filesystem::path> selectTestImages(const std::vector<boost::filesystem::path>& images,
                                                      const TestConfig& cfg)
{
    if (!cfg.enabled || cfg.maxImages <= 0)
    {
        return {};
    }

    std::vector<boost::filesystem::path> out;
    const size_t n = std::min(static_cast<size_t>(cfg.maxImages), images.size());
    out.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        out.push_back(images[i]);
    }
    return out;
}

void writeTestResultsCsv(const std::string& outputFile,
                         const std::vector<std::tuple<std::string, int, uint32_t, float>>& rows)
{
    std::cout << "Writing test CSV: " << outputFile << " (rows=" << rows.size() << ")" << std::endl;
    const boost::filesystem::path outPath(outputFile);
    const boost::filesystem::path parent = outPath.parent_path();
    if (!parent.empty() && !boost::filesystem::exists(parent))
    {
        boost::filesystem::create_directories(parent);
    }

    std::ofstream of(outputFile.c_str(), std::ios::out | std::ios::trunc);
    if (!of.is_open())
    {
        throw std::runtime_error("Failed to open test output file: " + outputFile);
    }

    of << "image_path,rank,word_id,weight\n";
    for (size_t i = 0; i < rows.size(); ++i)
    {
        of << std::get<0>(rows[i]) << "," << std::get<1>(rows[i]) << "," << std::get<2>(rows[i]) << ","
           << std::get<3>(rows[i]) << "\n";
    }
}

std::vector<std::pair<uint32_t, float>> topWords(const fbow::fBow& bow, int k)
{
    std::vector<std::pair<uint32_t, float>> words;
    words.reserve(bow.size());
    for (fbow::fBow::const_iterator it = bow.begin(); it != bow.end(); ++it)
    {
        words.push_back(std::make_pair(it->first, static_cast<float>(it->second)));
    }

    std::sort(words.begin(), words.end(), [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b)
              { return a.second > b.second; });

    if (k <= 0 || static_cast<size_t>(k) >= words.size())
    {
        return words;
    }
    return std::vector<std::pair<uint32_t, float>>(words.begin(), words.begin() + static_cast<size_t>(k));
}

}  // namespace

void runVocabularyTest(fbow::Vocabulary& voc, const std::vector<boost::filesystem::path>& images, const AppConfig& cfg)
{
    const std::vector<boost::filesystem::path> testImages = selectTestImages(images, cfg.test);
    if (testImages.empty())
    {
        return;
    }

    std::cout << "Running vocabulary test..." << std::endl;
    std::cout << "  test_images: " << testImages.size() << std::endl;
    std::cout << "  top_k: " << cfg.test.topK << std::endl;

    const int maxPerImage =
        cfg.test.maxFeaturesPerImage > 0 ? cfg.test.maxFeaturesPerImage : cfg.trainer.maxFeaturesPerImage;
    std::vector<std::tuple<std::string, int, uint32_t, float>> rows;
    rows.reserve(testImages.size() * static_cast<size_t>(cfg.test.topK));

    if (cfg.featureType == "orb")
    {
        cv::Ptr<cv::ORB> extractor = makeOrb(cfg.orb);
        cv::Ptr<cv::CLAHE> clahe = makeClahe(cfg.imagePrep);
        const size_t reportEvery = std::max<size_t>(1, testImages.size() / 5);
        std::vector<cv::KeyPoint> keypoints;
        for (size_t i = 0; i < testImages.size(); ++i)
        {
            const cv::Mat gray = prepareImage(loadGray(testImages[i].string()), clahe, cfg.imagePrep.scale);
            if (gray.empty())
            {
                continue;
            }

            if (i % reportEvery == 0 || i + 1 == testImages.size())
            {
                std::cout << "  test_progress: " << (i + 1) << "/" << testImages.size() << std::endl;
            }

            keypoints.clear();
            cv::Mat descriptors;
            extractor->detectAndCompute(gray, cv::Mat(), keypoints, descriptors);
            if (descriptors.empty())
            {
                continue;
            }

            if (descriptors.type() != CV_8UC1 || descriptors.cols != 32)
            {
                continue;
            }

            const int rowsToKeep = maxPerImage > 0 ? std::min(descriptors.rows, maxPerImage) : descriptors.rows;
            const cv::Mat used = clipRows(descriptors, rowsToKeep);
            const fbow::fBow bow = voc.transform(used);
            const std::vector<std::pair<uint32_t, float>> words = topWords(bow, cfg.test.topK);

            for (size_t w = 0; w < words.size(); ++w)
            {
                rows.push_back(
                    std::make_tuple(testImages[i].string(), static_cast<int>(w), words[w].first, words[w].second));
            }
        }
    }
    else
    {
        cv::Ptr<cv::BRISK> extractor = makeBrisk(cfg.brisk);
        cv::Ptr<cv::CLAHE> clahe = makeClahe(cfg.imagePrep);
        const size_t reportEvery = std::max<size_t>(1, testImages.size() / 5);
        std::vector<cv::KeyPoint> keypoints;
        for (size_t i = 0; i < testImages.size(); ++i)
        {
            const cv::Mat gray = prepareImage(loadGray(testImages[i].string()), clahe, cfg.imagePrep.scale);
            if (gray.empty())
            {
                continue;
            }

            if (i % reportEvery == 0 || i + 1 == testImages.size())
            {
                std::cout << "  test_progress: " << (i + 1) << "/" << testImages.size() << std::endl;
            }

            keypoints.clear();
            cv::Mat descriptors;
            extractor->detectAndCompute(gray, cv::Mat(), keypoints, descriptors);
            if (descriptors.empty())
            {
                continue;
            }

            if (descriptors.type() != CV_8UC1 || descriptors.cols != 64)
            {
                continue;
            }

            const int rowsToKeep = maxPerImage > 0 ? std::min(descriptors.rows, maxPerImage) : descriptors.rows;
            const cv::Mat used = clipRows(descriptors, rowsToKeep);
            const fbow::fBow bow = voc.transform(used);
            const std::vector<std::pair<uint32_t, float>> words = topWords(bow, cfg.test.topK);

            for (size_t w = 0; w < words.size(); ++w)
            {
                rows.push_back(
                    std::make_tuple(testImages[i].string(), static_cast<int>(w), words[w].first, words[w].second));
            }
        }
    }

    if (!cfg.test.outputFile.empty() && !rows.empty())
    {
        writeTestResultsCsv(cfg.test.outputFile, rows);
    }
}
