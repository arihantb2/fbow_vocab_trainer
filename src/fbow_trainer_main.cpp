#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

#include "fbow.h"
#include "vocabulary_creator.h"

namespace
{

std::string toLower(const std::string& s)
{
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

template <typename T>
T getRequired(const YAML::Node& node, const std::string& key)
{
    if (!node[key])
    {
        throw std::runtime_error("Missing required key: " + key);
    }
    return node[key].as<T>();
}

std::vector<std::string> normalizeExtensions(const std::vector<std::string>& in)
{
    std::vector<std::string> out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        std::string ext = toLower(in[i]);
        if (!ext.empty() && ext[0] != '.')
        {
            ext = "." + ext;
        }
        out.push_back(ext);
    }
    return out;
}

struct DatasetConfig
{
    std::vector<std::string> imagesDirs;
    bool recursive = true;
    int maxImages = 0;
    std::vector<std::string> extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"};
};

struct OrbConfig
{
    int nfeatures = 2000;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int wtaK = 2;
    int scoreType = 0;
    int patchSize = 31;
    int fastThreshold = 20;
};

struct SiftConfig
{
    int nfeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10.0;
    double sigma = 1.6;
};

struct TrainerConfig
{
    uint32_t k = 10;
    int L = 6;
    uint32_t nthreads = 1;
    int maxIters = -2; // fbow default if -2

    int maxFeaturesPerImage = 0;
    int maxTotalDescriptors = 0;  // 0 = unlimited; caps aggregate descriptor count across all images
};

struct OutputConfig
{
    std::string vocabPath;
};

struct TestConfig
{
    bool enabled = true;
    int maxImages = 5;
    int topK = 10;
    int maxFeaturesPerImage = 0;  // if 0, fall back to trainer.maxFeaturesPerImage
    std::string outputFile;        // if empty, default to <vocabPath>.test.csv
};

struct AppConfig
{
    std::string featureType; // "orb" or "sift"
    DatasetConfig dataset;
    OrbConfig orb;
    SiftConfig sift;
    TrainerConfig trainer;
    OutputConfig output;
    TestConfig test;
};

AppConfig loadConfig(const std::string& yamlPath)
{
    const YAML::Node root = YAML::LoadFile(yamlPath);

    AppConfig cfg;
    cfg.featureType = toLower(getRequired<std::string>(root["feature"], "type"));
    if (cfg.featureType != "orb" && cfg.featureType != "sift")
    {
        throw std::runtime_error("feature.type must be 'orb' or 'sift'");
    }

    const YAML::Node ds = getRequired<YAML::Node>(root, "dataset");
    // Support images_dirs (list) or images_dir (single string, backward compat).
    if (ds["images_dirs"])
    {
        cfg.dataset.imagesDirs = ds["images_dirs"].as<std::vector<std::string> >();
        if (cfg.dataset.imagesDirs.empty())
        {
            throw std::runtime_error("dataset.images_dirs must not be empty");
        }
    }
    else if (ds["images_dir"])
    {
        cfg.dataset.imagesDirs.push_back(ds["images_dir"].as<std::string>());
    }
    else
    {
        throw std::runtime_error("Missing required key: dataset.images_dir or dataset.images_dirs");
    }
    if (ds["recursive"])
    {
        cfg.dataset.recursive = ds["recursive"].as<bool>();
    }
    if (ds["max_images"])
    {
        cfg.dataset.maxImages = ds["max_images"].as<int>();
    }
    if (ds["extensions"])
    {
        cfg.dataset.extensions = ds["extensions"].as<std::vector<std::string> >();
    }

    const YAML::Node trainer = getRequired<YAML::Node>(root, "trainer");
    cfg.trainer.k = getRequired<uint32_t>(trainer, "k");
    cfg.trainer.L = getRequired<int>(trainer, "L");
    if (trainer["nthreads"])
    {
        cfg.trainer.nthreads = trainer["nthreads"].as<uint32_t>();
    }
    if (trainer["max_iters"])
    {
        cfg.trainer.maxIters = trainer["max_iters"].as<int>();
    }
    if (trainer["max_features_per_image"])
    {
        cfg.trainer.maxFeaturesPerImage = trainer["max_features_per_image"].as<int>();
    }
    if (trainer["max_total_descriptors"])
    {
        cfg.trainer.maxTotalDescriptors = trainer["max_total_descriptors"].as<int>();
    }

    const YAML::Node output = getRequired<YAML::Node>(root, "output");
    cfg.output.vocabPath = getRequired<std::string>(output, "vocab_path");

    const YAML::Node testNode = root["test"];
    if (testNode)
    {
        if (testNode["enabled"])
        {
            cfg.test.enabled = testNode["enabled"].as<bool>();
        }
        if (testNode["max_images"])
        {
            cfg.test.maxImages = testNode["max_images"].as<int>();
        }
        if (testNode["top_k"])
        {
            cfg.test.topK = testNode["top_k"].as<int>();
        }
        if (testNode["max_features_per_image"])
        {
            cfg.test.maxFeaturesPerImage = testNode["max_features_per_image"].as<int>();
        }
        if (testNode["output_file"])
        {
            cfg.test.outputFile = testNode["output_file"].as<std::string>();
        }
    }

    const YAML::Node orb = root["orb"];
    if (orb)
    {
        if (orb["nfeatures"])
            cfg.orb.nfeatures = orb["nfeatures"].as<int>();
        if (orb["scale_factor"])
            cfg.orb.scaleFactor = orb["scale_factor"].as<float>();
        if (orb["nlevels"])
            cfg.orb.nlevels = orb["nlevels"].as<int>();
        if (orb["edge_threshold"])
            cfg.orb.edgeThreshold = orb["edge_threshold"].as<int>();
        if (orb["first_level"])
            cfg.orb.firstLevel = orb["first_level"].as<int>();
        if (orb["wta_k"])
            cfg.orb.wtaK = orb["wta_k"].as<int>();
        if (orb["score_type"])
            cfg.orb.scoreType = orb["score_type"].as<int>();
        if (orb["patch_size"])
            cfg.orb.patchSize = orb["patch_size"].as<int>();
        if (orb["fast_threshold"])
            cfg.orb.fastThreshold = orb["fast_threshold"].as<int>();
    }

    const YAML::Node sift = root["sift"];
    if (sift)
    {
        if (sift["nfeatures"])
            cfg.sift.nfeatures = sift["nfeatures"].as<int>();
        if (sift["n_octave_layers"])
            cfg.sift.nOctaveLayers = sift["n_octave_layers"].as<int>();
        if (sift["contrast_threshold"])
            cfg.sift.contrastThreshold = sift["contrast_threshold"].as<double>();
        if (sift["edge_threshold"])
            cfg.sift.edgeThreshold = sift["edge_threshold"].as<double>();
        if (sift["sigma"])
            cfg.sift.sigma = sift["sigma"].as<double>();
    }

    return cfg;
}

std::vector<boost::filesystem::path> collectImages(const DatasetConfig& cfg)
{
    const std::vector<std::string> allowed = normalizeExtensions(cfg.extensions);

    std::vector<boost::filesystem::path> images;
    for (size_t d = 0; d < cfg.imagesDirs.size(); ++d)
    {
        const boost::filesystem::path root(cfg.imagesDirs[d]);
        if (!boost::filesystem::exists(root))
        {
            throw std::runtime_error("images directory does not exist: " + cfg.imagesDirs[d]);
        }
        if (!boost::filesystem::is_directory(root))
        {
            throw std::runtime_error("images path is not a directory: " + cfg.imagesDirs[d]);
        }

        if (cfg.recursive)
        {
            for (boost::filesystem::recursive_directory_iterator it(root), end; it != end; ++it)
            {
                if (!boost::filesystem::is_regular_file(*it))
                {
                    continue;
                }
                const std::string ext = toLower(it->path().extension().string());
                if (std::find(allowed.begin(), allowed.end(), ext) != allowed.end())
                {
                    images.push_back(it->path());
                }
            }
        }
        else
        {
            for (boost::filesystem::directory_iterator it(root), end; it != end; ++it)
            {
                if (!boost::filesystem::is_regular_file(*it))
                {
                    continue;
                }
                const std::string ext = toLower(it->path().extension().string());
                if (std::find(allowed.begin(), allowed.end(), ext) != allowed.end())
                {
                    images.push_back(it->path());
                }
            }
        }
    }

    std::sort(images.begin(), images.end());
    // Remove duplicates that may arise if directories overlap.
    images.erase(std::unique(images.begin(), images.end()), images.end());

    if (cfg.maxImages > 0 && static_cast<int>(images.size()) > cfg.maxImages)
    {
        images.resize(static_cast<size_t>(cfg.maxImages));
    }
    return images;
}

void ensureParentDirectory(const std::string& filePath)
{
    const boost::filesystem::path p(filePath);
    const boost::filesystem::path parent = p.parent_path();
    if (!parent.empty() && !boost::filesystem::exists(parent))
    {
        boost::filesystem::create_directories(parent);
    }
}

cv::Ptr<cv::ORB> makeOrb(const OrbConfig& cfg);
cv::Ptr<cv::SIFT> makeSift(const SiftConfig& cfg);

// Returns the effective per-image descriptor cap given both per-image and total budgets.
// Distributes maxTotalDescriptors evenly across numImages so memory is bounded upfront.
// Returns 0 if neither cap is set (no limit).
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

void writeTestResultsCsv(const std::string& outputFile,
                           const std::vector<std::tuple<std::string, int, uint32_t, float> >& rows)
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
        of << std::get<0>(rows[i]) << "," << std::get<1>(rows[i]) << "," << std::get<2>(rows[i]) << "," << std::get<3>(rows[i]) << "\n";
    }
}

std::vector<std::pair<uint32_t, float> > topWords(const fbow::fBow& bow, int k)
{
    std::vector<std::pair<uint32_t, float> > words;
    words.reserve(bow.size());
    for (fbow::fBow::const_iterator it = bow.begin(); it != bow.end(); ++it)
    {
        words.push_back(std::make_pair(it->first, static_cast<float>(it->second)));
    }

    std::sort(words.begin(), words.end(), [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
        return a.second > b.second;
    });

    if (k <= 0 || static_cast<size_t>(k) >= words.size())
    {
        return words;
    }
    return std::vector<std::pair<uint32_t, float> >(words.begin(), words.begin() + static_cast<size_t>(k));
}

void runVocabularyTest(fbow::Vocabulary& voc, const std::vector<boost::filesystem::path>& images,
                        const AppConfig& cfg)
{
    const std::vector<boost::filesystem::path> testImages = selectTestImages(images, cfg.test);
    if (testImages.empty())
    {
        return;
    }

    std::cout << "Running vocabulary test..." << std::endl;
    std::cout << "  test_images: " << testImages.size() << std::endl;
    std::cout << "  top_k: " << cfg.test.topK << std::endl;

    const int maxPerImage = cfg.test.maxFeaturesPerImage > 0 ? cfg.test.maxFeaturesPerImage : cfg.trainer.maxFeaturesPerImage;
    std::vector<std::tuple<std::string, int, uint32_t, float> > rows;
    rows.reserve(testImages.size() * static_cast<size_t>(cfg.test.topK));

    if (cfg.featureType == "orb")
    {
        cv::Ptr<cv::ORB> extractor = makeOrb(cfg.orb);
        const size_t reportEvery = std::max<size_t>(1, testImages.size() / 5);
        for (size_t i = 0; i < testImages.size(); ++i)
        {
            const cv::Mat gray = cv::imread(testImages[i].string(), cv::IMREAD_GRAYSCALE);
            if (gray.empty())
            {
                continue;
            }

            if (i % reportEvery == 0 || i + 1 == testImages.size())
            {
                std::cout << "  test_progress: " << (i + 1) << "/" << testImages.size() << std::endl;
            }

            std::vector<cv::KeyPoint> keypoints;
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
            const std::vector<std::pair<uint32_t, float> > words = topWords(bow, cfg.test.topK);

            for (size_t w = 0; w < words.size(); ++w)
            {
                rows.push_back(std::make_tuple(testImages[i].string(), static_cast<int>(w), words[w].first, words[w].second));
            }
        }
    }
    else
    {
        cv::Ptr<cv::SIFT> extractor = makeSift(cfg.sift);
        const size_t reportEvery = std::max<size_t>(1, testImages.size() / 5);
        for (size_t i = 0; i < testImages.size(); ++i)
        {
            const cv::Mat gray = cv::imread(testImages[i].string(), cv::IMREAD_GRAYSCALE);
            if (gray.empty())
            {
                continue;
            }

            if (i % reportEvery == 0 || i + 1 == testImages.size())
            {
                std::cout << "  test_progress: " << (i + 1) << "/" << testImages.size() << std::endl;
            }

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            extractor->detectAndCompute(gray, cv::Mat(), keypoints, descriptors);
            if (descriptors.empty())
            {
                continue;
            }

            if (descriptors.type() != CV_32FC1 || descriptors.cols != 128)
            {
                continue;
            }

            const int rowsToKeep = maxPerImage > 0 ? std::min(descriptors.rows, maxPerImage) : descriptors.rows;
            const cv::Mat used = clipRows(descriptors, rowsToKeep);
            const fbow::fBow bow = voc.transform(used);
            const std::vector<std::pair<uint32_t, float> > words = topWords(bow, cfg.test.topK);

            for (size_t w = 0; w < words.size(); ++w)
            {
                rows.push_back(std::make_tuple(testImages[i].string(), static_cast<int>(w), words[w].first, words[w].second));
            }
        }
    }

    if (!cfg.test.outputFile.empty() && !rows.empty())
    {
        writeTestResultsCsv(cfg.test.outputFile, rows);
    }
}

cv::Ptr<cv::ORB> makeOrb(const OrbConfig& cfg)
{
    return cv::ORB::create(cfg.nfeatures, cfg.scaleFactor, cfg.nlevels, cfg.edgeThreshold, cfg.firstLevel,
                           cfg.wtaK, static_cast<cv::ORB::ScoreType>(cfg.scoreType), cfg.patchSize,
                           cfg.fastThreshold);
}

cv::Ptr<cv::SIFT> makeSift(const SiftConfig& cfg)
{
    return cv::SIFT::create(cfg.nfeatures, cfg.nOctaveLayers, cfg.contrastThreshold, cfg.edgeThreshold,
                             cfg.sigma);
}

void trainOrbVocab(const std::vector<boost::filesystem::path>& images, const AppConfig& cfg,
                    fbow::Vocabulary* voc, size_t* descriptorsUsed, size_t* imagesUsed)
{
    *descriptorsUsed = 0;
    *imagesUsed = 0;

    const int effectiveMaxPerImage = computeEffectiveMaxPerImage(cfg.trainer, images.size());
    const size_t maxTotal = cfg.trainer.maxTotalDescriptors > 0
                                ? static_cast<size_t>(cfg.trainer.maxTotalDescriptors)
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
        const size_t upperBound = (effectiveMaxPerImage > 0 && cfg.trainer.maxTotalDescriptors > 0)
                                      ? std::min(static_cast<size_t>(effectiveMaxPerImage) * images.size(),
                                                 static_cast<size_t>(cfg.trainer.maxTotalDescriptors))
                                      : (effectiveMaxPerImage > 0
                                             ? static_cast<size_t>(effectiveMaxPerImage) * images.size()
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
    const size_t reportEvery = std::max<size_t>(1, images.size() / 5);

    for (size_t i = 0; i < images.size() && *descriptorsUsed < maxTotal; ++i)
    {
        const cv::Mat gray = cv::imread(images[i].string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty())
        {
            continue;
        }

        std::vector<cv::KeyPoint> keypoints;
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

        const int rows = effectiveMaxPerImage > 0
                               ? std::min(descriptors.rows, effectiveMaxPerImage)
                               : descriptors.rows;
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

void trainSiftVocab(const std::vector<boost::filesystem::path>& images, const AppConfig& cfg,
                    fbow::Vocabulary* voc, size_t* descriptorsUsed, size_t* imagesUsed)
{
    *descriptorsUsed = 0;
    *imagesUsed = 0;

    const int effectiveMaxPerImage = computeEffectiveMaxPerImage(cfg.trainer, images.size());
    const size_t maxTotal = cfg.trainer.maxTotalDescriptors > 0
                                ? static_cast<size_t>(cfg.trainer.maxTotalDescriptors)
                                : std::numeric_limits<size_t>::max();

    std::cout << "Training SIFT vocabulary..." << std::endl;
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
        const size_t upperBound = (effectiveMaxPerImage > 0 && cfg.trainer.maxTotalDescriptors > 0)
                                      ? std::min(static_cast<size_t>(effectiveMaxPerImage) * images.size(),
                                                 static_cast<size_t>(cfg.trainer.maxTotalDescriptors))
                                      : (effectiveMaxPerImage > 0
                                             ? static_cast<size_t>(effectiveMaxPerImage) * images.size()
                                             : static_cast<size_t>(cfg.trainer.maxTotalDescriptors));
        std::cout << "  total_features_upper_bound: " << upperBound << std::endl;
    }
    else
    {
        std::cout << "  total_features_upper_bound: unlimited" << std::endl;
    }

    std::vector<cv::Mat> features;
    features.reserve(std::min(images.size(), maxTotal > 0 ? maxTotal : images.size()));

    cv::Ptr<cv::SIFT> extractor = makeSift(cfg.sift);
    const size_t reportEvery = std::max<size_t>(1, images.size() / 5);

    for (size_t i = 0; i < images.size() && *descriptorsUsed < maxTotal; ++i)
    {
        const cv::Mat gray = cv::imread(images[i].string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty())
        {
            continue;
        }

        std::vector<cv::KeyPoint> keypoints;
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

        if (descriptors.type() != CV_32FC1)
        {
            throw std::runtime_error("SIFT descriptors must be CV_32FC1");
        }
        if (descriptors.cols != 128)
        {
            throw std::runtime_error("Unexpected SIFT descriptor dimension (expected 128)");
        }

        const int rows = effectiveMaxPerImage > 0
                               ? std::min(descriptors.rows, effectiveMaxPerImage)
                               : descriptors.rows;
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
        throw std::runtime_error("No SIFT features extracted from dataset");
    }

    fbow::VocabularyCreator creator;
    fbow::VocabularyCreator::Params params(cfg.trainer.k, cfg.trainer.L, cfg.trainer.nthreads, cfg.trainer.maxIters);
    creator.create(*voc, features, cfg.featureType, params);
}

void printUsage()
{
    std::cout << "Usage: fbow_vocab_trainer --config-file <path_to_yaml>" << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 3 || std::string(argv[1]) != "--config-file")
        {
            printUsage();
            return 1;
        }

        const std::string configPath = argv[2];
        std::cout << "Loading config: " << configPath << std::endl;
        AppConfig cfg = loadConfig(configPath);

        std::cout << "Collecting images from " << cfg.dataset.imagesDirs.size() << " director"
                  << (cfg.dataset.imagesDirs.size() == 1 ? "y" : "ies") << ":" << std::endl;
        for (size_t d = 0; d < cfg.dataset.imagesDirs.size(); ++d)
        {
            std::cout << "  " << cfg.dataset.imagesDirs[d] << std::endl;
        }
        const std::vector<boost::filesystem::path> images = collectImages(cfg.dataset);
        if (images.empty())
        {
            throw std::runtime_error("No images found in dataset.images_dir");
        }
        std::cout << "Found images: " << images.size() << std::endl;

        ensureParentDirectory(cfg.output.vocabPath);

        size_t descriptorsUsed = 0;
        size_t imagesUsed = 0;
        fbow::Vocabulary voc;
        if (cfg.featureType == "orb")
        {
            trainOrbVocab(images, cfg, &voc, &descriptorsUsed, &imagesUsed);
        }
        else
        {
            trainSiftVocab(images, cfg, &voc, &descriptorsUsed, &imagesUsed);
        }

        std::cout << "Saving vocabulary to: " << cfg.output.vocabPath << std::endl;
        voc.saveToFile(cfg.output.vocabPath);

        // Validate that the written vocab can be reloaded and used.
        if (cfg.test.enabled)
        {
            if (cfg.test.outputFile.empty())
            {
                cfg.test.outputFile = cfg.output.vocabPath + ".test.csv";
            }

            std::cout << "Reloading vocabulary for test..." << std::endl;
            fbow::Vocabulary reloaded;
            reloaded.readFromFile(cfg.output.vocabPath);
            runVocabularyTest(reloaded, images, cfg);
        }

        std::cout << "feature_type: " << cfg.featureType << std::endl;
        std::cout << "images_scanned: " << images.size() << std::endl;
        std::cout << "images_used: " << imagesUsed << std::endl;
        std::cout << "descriptors_used: " << descriptorsUsed << std::endl;
        std::cout << "output_vocab: " << cfg.output.vocabPath << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "fbow_vocab_trainer failed: " << e.what() << std::endl;
        return 1;
    }
}

