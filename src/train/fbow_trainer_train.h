#ifndef FBOW_TRAINER_TRAIN_H_
#define FBOW_TRAINER_TRAIN_H_

#include <boost/filesystem.hpp>

#include <cstddef>
#include <vector>

#include "config/fbow_trainer_config.h"

namespace fbow
{
class Vocabulary;
}

void trainOrbVocab(const std::vector<boost::filesystem::path>& images, const AppConfig& cfg, fbow::Vocabulary* voc,
                   size_t* descriptorsUsed, size_t* imagesUsed);

void trainBriskVocab(const std::vector<boost::filesystem::path>& images, const AppConfig& cfg, fbow::Vocabulary* voc,
                     size_t* descriptorsUsed, size_t* imagesUsed);

#endif  // FBOW_TRAINER_TRAIN_H_
