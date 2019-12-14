#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace mk;
using namespace af;

TEST(LinearRegressionTestCase, InterceptFalseTest)
{
    array data = utils::load_dataset<float>("/home/sangyx/mlkit/tests/data/linear_regression.txt");
    array X = data.cols(0, 1);
    array y = data.col(2);
    linear_model::LinearRegression lr = linear_model::LinearRegression(false);
    lr.fit(X, y);
    float* coef_host = lr.coef_.device<float>();
    ASSERT_FLOAT_EQ(coef_host[0], 3.0077438);
    ASSERT_FLOAT_EQ(coef_host[1], 1.6953201);
    ASSERT_TRUE(lr.intercept_.isempty());
}