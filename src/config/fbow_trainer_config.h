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

struct BriskConfig
{
    int thresh = 30;
    int octaves = 3;
    float patternScale = 1.0f;
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

struct ImagePrepConfig
{
    bool claheEnabled = false;
    double claheClipLimit = 2.0;
    int claheTileGridSize = 8;  // tiles are square (NxN)
    double scale = 1.0;         // resize factor applied after CLAHE; 1.0 = no resize
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
    std::string featureType;  // "orb" or "brisk"
    DatasetConfig dataset;
    ImagePrepConfig imagePrep;
    OrbConfig orb;
    BriskConfig brisk;
    TrainerConfig trainer;
    OutputConfig output;
    TestConfig test;
};

AppConfig loadConfig(const std::string& yamlPath);

#endif  // FBOW_TRAINER_CONFIG_H_
