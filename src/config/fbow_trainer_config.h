#ifndef FBOW_TRAINER_CONFIG_H_
#define FBOW_TRAINER_CONFIG_H_

#include <cstdint>
#include <string>
#include <vector>

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
    int maxIters = -2;  // fbow default if -2

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
    std::string outputFile;       // if empty, default to <vocabPath>.test.csv
};

struct AppConfig
{
    std::string featureType;  // "orb" or "sift"
    DatasetConfig dataset;
    OrbConfig orb;
    SiftConfig sift;
    TrainerConfig trainer;
    OutputConfig output;
    TestConfig test;
};

AppConfig loadConfig(const std::string& yamlPath);

#endif  // FBOW_TRAINER_CONFIG_H_
