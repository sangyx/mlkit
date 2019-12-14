#include <gtest/gtest.h>
#include <mlkit.hpp>
#include <iostream>

using namespace mk;
using namespace af;

TEST(UtilsTestCase, LoadDataTest)
{
    array data = utils::load_dataset<float>("/home/sangyx/mlkit/tests/data/linear_regression.txt");
    ASSERT_EQ(data.dims(0), 200);
    ASSERT_EQ(data.dims(1), 3);

    float* data_host = data.row(0).device<float>();
    // 第一行数据：1.000000	0.067732	3.176513
    ASSERT_FLOAT_EQ(data_host[0], 1);
    ASSERT_FLOAT_EQ(data_host[1], 0.067732);
    ASSERT_FLOAT_EQ(data_host[2], 3.176513);
}