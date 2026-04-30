#include "config/fbow_trainer_config.h"

#include <gtest/gtest.h>

#include <boost/filesystem.hpp>

#include <fstream>
#include <stdexcept>
#include <string>

namespace
{

boost::filesystem::path makeTempYaml(const std::string& contents)
{
    const boost::filesystem::path dir =
        boost::filesystem::temp_directory_path() / boost::filesystem::unique_path("fbow_trainer_cfg-%%%%-%%%%");
    boost::filesystem::create_directories(dir);
    const boost::filesystem::path yamlPath = dir / "trainer.yaml";
    std::ofstream out(yamlPath.string().c_str(), std::ios::out | std::ios::trunc);
    out << contents;
    out.close();
    return yamlPath;
}

}  // namespace

TEST(LoadConfig, MinimalOrbParsesRequiredFields)
{
    const std::string yaml = R"(
feature:
  type: ORB
dataset:
  images_dir: /tmp/fbow_trainer_unit_test_images
trainer:
  k: 10
  L: 6
output:
  vocab_path: /tmp/fbow_trainer_unit_test_vocab.bin
)";
    const boost::filesystem::path path = makeTempYaml(yaml);
    const AppConfig cfg = loadConfig(path.string());
    EXPECT_EQ(cfg.featureType, "orb");
    ASSERT_EQ(cfg.dataset.imagesDirs.size(), 1u);
    EXPECT_EQ(cfg.dataset.imagesDirs[0], "/tmp/fbow_trainer_unit_test_images");
    EXPECT_EQ(cfg.trainer.k, 10u);
    EXPECT_EQ(cfg.trainer.L, 6);
    EXPECT_EQ(cfg.output.vocabPath, "/tmp/fbow_trainer_unit_test_vocab.bin");
    EXPECT_TRUE(cfg.dataset.recursive);
}

TEST(LoadConfig, ImagesDirsList)
{
    const std::string yaml = R"(
feature:
  type: brisk
dataset:
  images_dirs:
    - /a
    - /b
trainer:
  k: 5
  L: 4
output:
  vocab_path: /out.bin
)";
    const boost::filesystem::path path = makeTempYaml(yaml);
    const AppConfig cfg = loadConfig(path.string());
    EXPECT_EQ(cfg.featureType, "brisk");
    ASSERT_EQ(cfg.dataset.imagesDirs.size(), 2u);
    EXPECT_EQ(cfg.dataset.imagesDirs[0], "/a");
    EXPECT_EQ(cfg.dataset.imagesDirs[1], "/b");
    EXPECT_EQ(cfg.trainer.k, 5u);
}

TEST(LoadConfig, TrainerOptionalFields)
{
    const std::string yaml = R"(
feature:
  type: orb
dataset:
  images_dir: /img
trainer:
  k: 10
  L: 6
  nthreads: 4
  max_iters: 10
  max_features_per_image: 100
  max_total_descriptors: 1000
output:
  vocab_path: /v.bin
)";
    const boost::filesystem::path path = makeTempYaml(yaml);
    const AppConfig cfg = loadConfig(path.string());
    EXPECT_EQ(cfg.trainer.nthreads, 4u);
    EXPECT_EQ(cfg.trainer.maxIters, 10);
    EXPECT_EQ(cfg.trainer.maxFeaturesPerImage, 100);
    EXPECT_EQ(cfg.trainer.maxTotalDescriptors, 1000);
}

TEST(LoadConfig, InvalidFeatureTypeThrows)
{
    const std::string yaml = R"(
feature:
  type: akaze
dataset:
  images_dir: /img
trainer:
  k: 10
  L: 6
output:
  vocab_path: /v.bin
)";
    const boost::filesystem::path path = makeTempYaml(yaml);
    EXPECT_THROW(loadConfig(path.string()), std::runtime_error);
}

TEST(LoadConfig, MissingDatasetThrows)
{
    const std::string yaml = R"(
feature:
  type: orb
trainer:
  k: 10
  L: 6
output:
  vocab_path: /v.bin
)";
    const boost::filesystem::path path = makeTempYaml(yaml);
    EXPECT_THROW(loadConfig(path.string()), std::runtime_error);
}
