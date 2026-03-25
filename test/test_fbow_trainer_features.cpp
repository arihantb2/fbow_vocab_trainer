#include "features/fbow_trainer_features.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

TEST(ComputeEffectiveMaxPerImage, NoCapsReturnsZero)
{
    TrainerConfig cfg;
    cfg.maxFeaturesPerImage = 0;
    cfg.maxTotalDescriptors = 0;
    EXPECT_EQ(computeEffectiveMaxPerImage(cfg, 10), 0);
}

TEST(ComputeEffectiveMaxPerImage, OnlyPerImageCap)
{
    TrainerConfig cfg;
    cfg.maxFeaturesPerImage = 200;
    cfg.maxTotalDescriptors = 0;
    EXPECT_EQ(computeEffectiveMaxPerImage(cfg, 5), 200);
}

TEST(ComputeEffectiveMaxPerImage, OnlyTotalCapEvenSplit)
{
    TrainerConfig cfg;
    cfg.maxFeaturesPerImage = 0;
    cfg.maxTotalDescriptors = 1000;
    EXPECT_EQ(computeEffectiveMaxPerImage(cfg, 10), 100);
}

TEST(ComputeEffectiveMaxPerImage, OnlyTotalCapIntegerDivision)
{
    TrainerConfig cfg;
    cfg.maxFeaturesPerImage = 0;
    cfg.maxTotalDescriptors = 1000;
    EXPECT_EQ(computeEffectiveMaxPerImage(cfg, 3), 333);
}

TEST(ComputeEffectiveMaxPerImage, BothCapsTakesMinimum)
{
    TrainerConfig cfg;
    cfg.maxFeaturesPerImage = 50;
    cfg.maxTotalDescriptors = 1000;
    EXPECT_EQ(computeEffectiveMaxPerImage(cfg, 10), 50);
}

TEST(ComputeEffectiveMaxPerImage, BothCapsTotalIsTighter)
{
    TrainerConfig cfg;
    cfg.maxFeaturesPerImage = 500;
    cfg.maxTotalDescriptors = 100;
    EXPECT_EQ(computeEffectiveMaxPerImage(cfg, 10), 10);
}

TEST(ComputeEffectiveMaxPerImage, ZeroImagesIgnoresTotalCapForFromTotal)
{
    TrainerConfig cfg;
    cfg.maxFeaturesPerImage = 0;
    cfg.maxTotalDescriptors = 1000;
    EXPECT_EQ(computeEffectiveMaxPerImage(cfg, 0), 0);
}

TEST(ClipRows, ClampsToFirstRows)
{
    cv::Mat m(5, 32, CV_8UC1, cv::Scalar(7));
    cv::Mat out = clipRows(m, 2);
    ASSERT_EQ(out.rows, 2);
    ASSERT_EQ(out.cols, 32);
    EXPECT_TRUE(out.isContinuous());
}

TEST(ClipRows, NonPositiveKeepsAll)
{
    cv::Mat m(4, 8, CV_8UC1, cv::Scalar(3));
    cv::Mat out = clipRows(m, 0);
    EXPECT_EQ(out.rows, 4);
}

TEST(ClipRows, AtLeastAllRowsReturnsOriginalOrClone)
{
    cv::Mat m(3, 8, CV_8UC1, cv::Scalar(1));
    cv::Mat out = clipRows(m, 10);
    EXPECT_EQ(out.rows, 3);
}

TEST(ClipRows, NonContinuousFullRangeReturnsContinuousClone)
{
    cv::Mat m(10, 10, CV_8UC1, cv::Scalar(0));
    cv::Mat sub = m(cv::Rect(1, 1, 5, 5));
    ASSERT_FALSE(sub.isContinuous());
    cv::Mat out = clipRows(sub, 100);
    ASSERT_EQ(out.rows, 5);
    EXPECT_TRUE(out.isContinuous());
}
