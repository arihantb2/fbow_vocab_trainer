#ifndef FBOW_TRAINER_TEST_H_
#define FBOW_TRAINER_TEST_H_

#include <boost/filesystem.hpp>

#include <vector>

#include "config/fbow_trainer_config.h"

namespace fbow
{
class Vocabulary;
}

void runVocabularyTest(fbow::Vocabulary& voc, const std::vector<boost::filesystem::path>& images, const AppConfig& cfg);

#endif  // FBOW_TRAINER_TEST_H_
