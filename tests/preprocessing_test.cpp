#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace mk;
using namespace af;

class PreprocessingTest: public testing::Test
{
protected:
    virtual void SetUp()
    {
        array data = utils::load_dataset<float>("../tests/data/linear_regression.txt");
        X = data.cols(0, 1);

    }

    array X;
};

TEST_F(PreprocessingTest, StandardScalerTest)
{
    preprocessing::StandardScaler ss;
    array X_ss = ss.fit_transform(X);
    for(int i = 0; i < X_ss.dims(1); ++i){
        if(ss.flag_[i])
        {
            EXPECT_NEAR(mean(X_ss.col(i), 0).scalar<float>(), 0, 1e-6);
            EXPECT_NEAR(var(X_ss.col(i), false, 0).scalar<float>(), 1, 1e-6);
        }
    }
}

TEST_F(PreprocessingTest, MinMaxScalerTest)
{
    preprocessing::MinMaxScaler mm;
    array X_mm = mm.fit_transform(X);
    for(int i = 0; i < X.dims(1); ++i)
    {
        if(mm.flag_[i])
        {
            EXPECT_NEAR(min(X_mm(span, i), 0).scalar<float>(), 0, 1e-6);
            EXPECT_NEAR(max(X_mm(span, i), 0).scalar<float>(), 1, 1e-6);
        }
    }
}
