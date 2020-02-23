#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace af;
using namespace mk;

TEST(TreeTestCase, DecisionTreeClassifierTest)
{
    array data = utils::load_dataset<float>("../tests/data/iris.txt");
    int feat = data.dims(1) - 1;
    array X = data.cols(0, feat-1);
    array y = data.col(feat);
    tree::DecisionTreeClassifier dtc;
    dtc.fit(X, y);
    float hA[] = {4.4, 2.9, 1.4, 0.2};
    array test(1, 4, hA);
    EXPECT_EQ(dtc.predict(test).scalar<float>(), 0);
}