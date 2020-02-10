#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace mk;
using namespace af;

TEST(UtilsTest, MetricsTest)
{
    float hA[] = {0, 1, 2, 3, 4, 5};
    array A(3, 2, hA);
    array B = utils::minkowski_distance(A, A, 2);
    EXPECT_EQ(B.dims(), dim4(3, 3));
    EXPECT_FLOAT_EQ(B(0, 1).scalar<float>(), 2);
    EXPECT_FLOAT_EQ(B(0, 2).scalar<float>(), 8);
}

TEST(NeighborsTest, KNeighborsClassifierTest)
{
    array data = utils::load_dataset<float>("../tests/data/knn.txt");
    array X = data.cols(0, 2);
    array y = data.col(3);
    auto [X_train, y_train, X_test, y_test] = utils::train_test_split(X, y, 0.7);
    neighbors::KNeighborsClassifier knc(5);
    knc.fit(X_train, y_train);
    EXPECT_GT(knc.score(X_test, y_test), 75);
}