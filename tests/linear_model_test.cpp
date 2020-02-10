#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace mk;
using namespace af;

TEST(LinearModelTestCase, LinearRegressionTest)
{
    array data = utils::load_dataset<float>("../tests/data/linear_regression.txt");
    int feat = data.dims(1) - 1;
    array X = data.cols(0, feat-1);
    array y = data.col(feat);
    linear_model::LinearRegression lr = linear_model::LinearRegression(false);
    lr.fit(X, y);
    float* coef_host = lr.coef_.host<float>();
    EXPECT_NEAR(coef_host[0], 3, 0.1);
    EXPECT_NEAR(coef_host[1], 1.7, 0.1);
    EXPECT_TRUE(lr.intercept_.isempty());
    lr.score(X, y);
}

TEST(LinearModelTestCase, LogisticRegressionTest)
{
    array data = utils::load_dataset<float>("../tests/data/logistic_regression.txt");
    int feat = data.dims(1) - 1;
    array X = data.cols(0, feat-1);
    array y = data.col(feat);
    // auto [X_train, y_train, X_test, y_test] = utils::train_test_split(X, y, 0.7);
    // af_print(X_train);
    // af_print(y_train);
    linear_model::LogisticRegression lr = linear_model::LogisticRegression(true, "l2", 0.01, 0);
    lr.fit(X, y);
    EXPECT_GT(lr.score(X, y), 0.95);
    // float* coef_host = lr.coef_.host<float>();
    // float* intercept_host = lr.intercept_.host<float>();
    // EXPECT_NEAR(coef_host[0], 2.5, 0.1);
    // EXPECT_NEAR(coef_host[1], -2.1, 0.1);
    // EXPECT_NEAR(intercept_host[0], 15.3, 0.1);
}