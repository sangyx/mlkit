#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace mk;
using namespace af;

TEST(LinearModelTestCase, LinearRegressionTest)
{
    array data = utils::load_dataset<float>("../tests/data/linear_regression.txt");
    array X = data.cols(0, 1);
    array y = data.col(2);
    linear_model::LinearRegression lr = linear_model::LinearRegression(false);
    lr.fit(X, y);
    float* coef_host = lr.coef_.device<float>();
    EXPECT_NEAR(coef_host[0], 3, 0.1);
    EXPECT_NEAR(coef_host[1], 1.7, 0.1);
    EXPECT_TRUE(lr.intercept_.isempty());
}

TEST(LinearModelTestCase, LogisticRegressionTest)
{
    array data = utils::load_dataset<float>("../tests/data/logistic_regression.txt");
    array X = data.cols(0, 1);
    array y = data.col(2);
    linear_model::LogisticRegression lr = linear_model::LogisticRegression(true);
    lr.fit(X, y);
    af_print(lr.coef_);
    af_print(lr.intercept_);
    float* coef_host = lr.coef_.device<float>();
    EXPECT_NEAR(coef_host[0], 3, 0.1);
    EXPECT_NEAR(coef_host[1], 1.15, 0.1);
    float* intercept_host = lr.intercept_.device<float>();
    EXPECT_NEAR(intercept_host[0], 3, 0.1);
}