#include "config/fbow_trainer_config.h"
#include "dataset/fbow_trainer_dataset.h"
#include "train/fbow_trainer_train.h"
#include "vocabulary_test/fbow_trainer_test.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "fbow.h"

namespace
{

void printUsage()
{
    std::cout << "Usage: fbow_vocab_trainer --config-file <path_to_yaml>" << std::endl;
}

}  // namespace

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
            trainBriskVocab(images, cfg, &voc, &descriptorsUsed, &imagesUsed);
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
