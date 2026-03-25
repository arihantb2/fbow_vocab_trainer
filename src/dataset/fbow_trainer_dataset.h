#ifndef FBOW_TRAINER_DATASET_H_
#define FBOW_TRAINER_DATASET_H_

#include <boost/filesystem.hpp>

#include <string>
#include <vector>

#include "config/fbow_trainer_config.h"

std::vector<boost::filesystem::path> collectImages(const DatasetConfig& cfg);

void ensureParentDirectory(const std::string& filePath);

#endif  // FBOW_TRAINER_DATASET_H_
