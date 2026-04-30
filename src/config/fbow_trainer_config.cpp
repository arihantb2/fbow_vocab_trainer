#include "config/fbow_trainer_config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

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

}  // namespace

AppConfig loadConfig(const std::string& yamlPath)
{
    const YAML::Node root = YAML::LoadFile(yamlPath);

    AppConfig cfg;
    cfg.featureType = toLower(getRequired<std::string>(root["feature"], "type"));
    if (cfg.featureType != "orb" && cfg.featureType != "brisk")
    {
        throw std::runtime_error("feature.type must be 'orb' or 'brisk'");
    }

    const YAML::Node ds = getRequired<YAML::Node>(root, "dataset");
    // Support images_dirs (list) or images_dir (single string, backward compat).
    if (ds["images_dirs"])
    {
        cfg.dataset.imagesDirs = ds["images_dirs"].as<std::vector<std::string>>();
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
        cfg.dataset.extensions = ds["extensions"].as<std::vector<std::string>>();
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

    const YAML::Node imagePrep = root["image_prep"];
    if (imagePrep)
    {
        if (imagePrep["scale"])
        {
            cfg.imagePrep.scale = imagePrep["scale"].as<double>();
            if (cfg.imagePrep.scale <= 0.0)
            {
                throw std::runtime_error("image_prep.scale must be > 0");
            }
        }
        const YAML::Node clahe = imagePrep["clahe"];
        if (clahe)
        {
            if (clahe["enabled"])
            {
                cfg.imagePrep.claheEnabled = clahe["enabled"].as<bool>();
            }
            if (clahe["clip_limit"])
            {
                cfg.imagePrep.claheClipLimit = clahe["clip_limit"].as<double>();
            }
            if (clahe["tile_grid_size"])
            {
                cfg.imagePrep.claheTileGridSize = clahe["tile_grid_size"].as<int>();
            }
        }
    }

    const YAML::Node orb = root["orb"];
    if (orb)
    {
        if (orb["nfeatures"])
        {
            cfg.orb.nfeatures = orb["nfeatures"].as<int>();
        }
        if (orb["scale_factor"])
        {
            cfg.orb.scaleFactor = orb["scale_factor"].as<float>();
        }
        if (orb["nlevels"])
        {
            cfg.orb.nlevels = orb["nlevels"].as<int>();
        }
        if (orb["edge_threshold"])
        {
            cfg.orb.edgeThreshold = orb["edge_threshold"].as<int>();
        }
        if (orb["first_level"])
        {
            cfg.orb.firstLevel = orb["first_level"].as<int>();
        }
        if (orb["wta_k"])
        {
            cfg.orb.wtaK = orb["wta_k"].as<int>();
        }
        if (orb["score_type"])
        {
            cfg.orb.scoreType = orb["score_type"].as<int>();
        }
        if (orb["patch_size"])
        {
            cfg.orb.patchSize = orb["patch_size"].as<int>();
        }
        if (orb["fast_threshold"])
        {
            cfg.orb.fastThreshold = orb["fast_threshold"].as<int>();
        }
    }

    const YAML::Node brisk = root["brisk"];
    if (brisk)
    {
        if (brisk["thresh"])
        {
            cfg.brisk.thresh = brisk["thresh"].as<int>();
        }
        if (brisk["octaves"])
        {
            cfg.brisk.octaves = brisk["octaves"].as<int>();
        }
        if (brisk["pattern_scale"])
        {
            cfg.brisk.patternScale = brisk["pattern_scale"].as<float>();
        }
    }

    return cfg;
}
